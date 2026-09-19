r"""
failure_clustering.py

STEP 9: Cluster failed executions to find "failure fingerprints" --
recurring combinations of conditions that show up together in failures.

DESIGN DECISION: clustering only on proven driver features, not all 158
columns. Steps 5-8 already established which ~10 features actually
matter (host_temperature_c, seed_group, feature_flag_7, memory_alloc_mb,
thread_pool_size, voltage_v, memory_pressure_pct, cache_policy,
scheduler, timeout_ms). Clustering on all 158 columns -- most of which
are proven noise -- would let that noise dominate the distance metric
and produce meaningless clusters. This is the same "don't trust noise"
discipline from step 6, applied here.

WHAT MAKES THIS MORE THAN "Cluster 1 has 2,000 points":
For each cluster, we:
  1. Find which conditions are distinctly common in that cluster
     (compared to the overall dataset, not just compared to other
     failures -- so "Dynamic scheduler" only gets flagged if it's
     ACTUALLY over-represented, not just the most common value by chance)
  2. Turn those conditions into a rule (e.g. host_temperature_c > 65
     AND feature_flag_7 = True)
  3. Apply that rule to the FULL dataset (not just failures) to get a
     real, honest failure rate and sample size for runs matching that
     fingerprint -- this is what makes "Failure rate: 78%" meaningful
     instead of a made-up-sounding number

Usage (from backend/):
    python failure_clustering.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

RAW_PATH = "../data/raw/execution_logs.csv"
OUT_PATH = "../outputs/failure_fingerprints.csv"

# Only the features already proven to matter (steps 5-8) -- see module
# docstring for why we don't cluster on all 158 columns.
NUMERIC_CLUSTER_FEATURES = [
    "host_temperature_c", "voltage_v", "memory_alloc_mb",
    "thread_pool_size", "memory_pressure_pct", "timeout_ms", "cpu_load_pct",
]
CATEGORICAL_CLUSTER_FEATURES = ["cache_policy", "scheduler"]
BOOLEAN_CLUSTER_FEATURES = ["feature_flag_7"]

K_RANGE = range(3, 8)  # try 3 to 7 clusters, pick the best by silhouette score
TOP_N_CONDITIONS = 4    # how many distinguishing conditions to report per cluster


def build_cluster_input(failed_df: pd.DataFrame) -> pd.DataFrame:
    numeric_part = StandardScaler().fit_transform(failed_df[NUMERIC_CLUSTER_FEATURES])
    numeric_df = pd.DataFrame(numeric_part, columns=NUMERIC_CLUSTER_FEATURES, index=failed_df.index)

    cat_df = pd.get_dummies(failed_df[CATEGORICAL_CLUSTER_FEATURES], columns=CATEGORICAL_CLUSTER_FEATURES)
    bool_df = failed_df[BOOLEAN_CLUSTER_FEATURES].astype(int)

    return pd.concat([numeric_df, cat_df, bool_df], axis=1)


def find_best_k(cluster_input: pd.DataFrame) -> int:
    best_k, best_score = K_RANGE[0], -1
    print("Searching for the best number of clusters (by silhouette score):")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(cluster_input)
        score = silhouette_score(cluster_input, labels)
        print(f"  k={k}: silhouette score = {score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"Best k = {best_k}\n")
    return best_k


def find_distinguishing_conditions(cluster_df: pd.DataFrame, full_df: pd.DataFrame) -> list:
    """Return the top conditions that make this cluster distinct from the
    OVERALL dataset (not just from other failures) -- along with enough
    info to turn each into a filter rule."""
    conditions = []

    for col in NUMERIC_CLUSTER_FEATURES:
        overall_mean, overall_std = full_df[col].mean(), full_df[col].std()
        overall_median = full_df[col].median()
        cluster_mean = cluster_df[col].mean()
        z = (cluster_mean - overall_mean) / overall_std if overall_std > 0 else 0
        if abs(z) >= 0.4:  # meaningfully off-center, not just noise
            direction = "high" if z > 0 else "low"
            conditions.append({
                "feature": col, "kind": "numeric", "direction": direction,
                "cluster_mean": round(cluster_mean, 1), "threshold": round(overall_median, 1),
                "score": abs(z),
                "label": f"{col} is {direction} (avg {cluster_mean:.1f} vs dataset avg {overall_mean:.1f})",
            })

    for col in CATEGORICAL_CLUSTER_FEATURES:
        cluster_share = cluster_df[col].value_counts(normalize=True)
        overall_share = full_df[col].value_counts(normalize=True)
        for value in cluster_share.index:
            diff = cluster_share[value] - overall_share.get(value, 0)
            if diff >= 0.12:  # over-represented by at least 12 percentage points
                conditions.append({
                    "feature": col, "kind": "categorical", "value": value,
                    "score": diff,
                    "label": f"{col} = {value} ({cluster_share[value]:.0%} of cluster vs "
                             f"{overall_share.get(value, 0):.0%} overall)",
                })

    for col in BOOLEAN_CLUSTER_FEATURES:
        cluster_rate = cluster_df[col].mean()
        overall_rate = full_df[col].mean()
        diff = cluster_rate - overall_rate
        if abs(diff) >= 0.12:
            state = "True" if diff > 0 else "False"
            conditions.append({
                "feature": col, "kind": "boolean", "value": (diff > 0),
                "score": abs(diff),
                "label": f"{col} = {state} ({cluster_rate:.0%} of cluster vs {overall_rate:.0%} overall)",
            })

    conditions.sort(key=lambda c: c["score"], reverse=True)
    return conditions[:TOP_N_CONDITIONS]


def build_rule_mask(full_df: pd.DataFrame, conditions: list) -> pd.Series:
    mask = pd.Series(True, index=full_df.index)
    for cond in conditions:
        if cond["kind"] == "numeric":
            if cond["direction"] == "high":
                mask &= full_df[cond["feature"]] > cond["threshold"]
            else:
                mask &= full_df[cond["feature"]] < cond["threshold"]
        elif cond["kind"] == "categorical":
            mask &= full_df[cond["feature"]] == cond["value"]
        elif cond["kind"] == "boolean":
            mask &= full_df[cond["feature"]] == cond["value"]
    return mask


def main():
    full_df = pd.read_csv(RAW_PATH)
    failed_df = full_df[full_df["passed"] == 0].copy()
    print(f"Total runs: {len(full_df)}  |  Failed runs: {len(failed_df)} ({len(failed_df)/len(full_df):.1%})\n")

    cluster_input = build_cluster_input(failed_df)
    best_k = find_best_k(cluster_input)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    failed_df["cluster"] = km.fit_predict(cluster_input)

    fingerprint_rows = []

    for cluster_id in sorted(failed_df["cluster"].unique()):
        cluster_df = failed_df[failed_df["cluster"] == cluster_id]
        conditions = find_distinguishing_conditions(cluster_df, full_df)

        print("=" * 65)
        print(f"CLUSTER {cluster_id}  --  {len(cluster_df)} failed runs "
              f"({len(cluster_df)/len(failed_df):.1%} of all failures)")
        print("=" * 65)

        if not conditions:
            print("No strongly distinguishing conditions found -- this cluster may just be")
            print("'ordinary' failures without a clear common cause (background failure rate).\n")
            fingerprint_rows.append({
                "cluster": cluster_id, "failed_runs_in_cluster": len(cluster_df),
                "conditions": "none distinguishing", "matching_runs_total": None,
                "matching_failure_rate": None,
            })
            continue

        print("Common conditions:")
        for cond in conditions:
            print(f"  - {cond['label']}")

        rule_mask = build_rule_mask(full_df, conditions)
        n_matching = rule_mask.sum()
        failure_rate_matching = 1 - full_df.loc[rule_mask, "passed"].mean() if n_matching > 0 else float("nan")

        print(f"\nRuns matching this fingerprint (across ALL {len(full_df)} runs, pass + fail): {n_matching}")
        print(f"Failure rate for runs matching this fingerprint: {failure_rate_matching:.1%}")
        print(f"(compare to dataset baseline failure rate: {1 - full_df['passed'].mean():.1%})\n")

        fingerprint_rows.append({
            "cluster": cluster_id,
            "failed_runs_in_cluster": len(cluster_df),
            "conditions": "; ".join(c["label"] for c in conditions),
            "matching_runs_total": int(n_matching),
            "matching_failure_rate": round(failure_rate_matching, 4),
        })

    pd.DataFrame(fingerprint_rows).to_csv(OUT_PATH, index=False)
    print(f"Fingerprints saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()