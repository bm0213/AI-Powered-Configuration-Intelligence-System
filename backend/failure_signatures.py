r"""
failure_signatures.py

Closes three specific gaps found when cross-checking against the actual
SanDisk brief:

1. CONTRIBUTION-TO-TOTAL-FAILURES STATS (brief's own example phrasing):
   "Random Seed Group 8 contributes to 35% of observed failures."
   "Feature Flag X is present in 78% of failing runs."
   These are a DIFFERENT metric from "failure rate within group" (which
   is reported everywhere else in this project) -- this is "of all the
   failures that happened, what fraction trace back to X".

2. LOG COMPARISON (Q6: "What changed between successful and failed
   executions?" -- expected output explicitly includes "Log comparison").
   log_features.csv was built back in preprocessing.py specifically for
   this and has never been analyzed until now.

3. FAILURE SIGNATURES (Q5: expected output explicitly includes "Failure
   signatures", distinct from "Failure fingerprints" which step 9 already
   covers). This extends each fingerprint cluster from failure_clustering.py
   with its LOG evidence (error counts, timeout rates, etc.), not just its
   configuration conditions -- a signature is the full picture: what
   configuration AND what the logs looked like.

Usage (from backend/):
    python failure_signatures.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW_PATH = "../data/raw/execution_logs.csv"
OUT_CONTRIBUTION = "../outputs/failure_contribution_stats.csv"
OUT_LOG_COMPARISON = "../outputs/log_comparison.csv"
OUT_SIGNATURES = "../outputs/failure_signatures.csv"

LOG_COLUMNS = [
    "error_count", "warning_count", "timeout_occurred",
    "memory_error_occurred", "scheduler_error_occurred", "retry_attempts_logged",
]

# Same driver features used for clustering in failure_clustering.py --
# kept consistent so failure signatures line up with the fingerprints
# you've already presented.
NUMERIC_CLUSTER_FEATURES = [
    "host_temperature_c", "voltage_v", "memory_alloc_mb",
    "thread_pool_size", "memory_pressure_pct", "timeout_ms", "cpu_load_pct",
]
CATEGORICAL_CLUSTER_FEATURES = ["cache_policy", "scheduler"]
BOOLEAN_CLUSTER_FEATURES = ["feature_flag_7"]


# =================================================================
# PART 1: Contribution-to-total-failures stats
# =================================================================
def compute_contribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    total_fails = (df["passed"] == 0).sum()
    conditions = {
        "seed_group == 8": df["seed_group"] == 8,
        "feature_flag_7 == True": df["feature_flag_7"] == True,
        "host_temperature_c > 65 (overheating)": df["host_temperature_c"] > 65,
        "memory_alloc_mb <= 1024 AND thread_pool_size >= 16": (df["memory_alloc_mb"] <= 1024) & (df["thread_pool_size"] >= 16),
    }

    rows = []
    print("=" * 70)
    print("CONTRIBUTION TO TOTAL FAILURES")
    print("=" * 70)
    print(f"Total failures in dataset: {total_fails}\n")

    for label, mask in conditions.items():
        fails_in_condition = ((df["passed"] == 0) & mask).sum()
        pct_of_all_failures = fails_in_condition / total_fails * 100
        pct_present_given_condition = mask.mean() * 100  # what fraction of ALL runs meet this condition

        print(f"{label}")
        print(f"  Contributes to {pct_of_all_failures:.1f}% of all observed failures "
              f"({fails_in_condition} of {total_fails} failures)")
        print(f"  (This condition is present in {pct_present_given_condition:.1f}% of ALL runs, "
              f"pass or fail)\n")

        rows.append({
            "condition": label,
            "failures_matching": int(fails_in_condition),
            "total_failures": int(total_fails),
            "pct_of_all_failures": round(pct_of_all_failures, 1),
        })

    return pd.DataFrame(rows)


# =================================================================
# PART 2: Log comparison (Q6)
# =================================================================
def compute_log_comparison(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("LOG COMPARISON: passed vs failed executions (Q6)")
    print("=" * 70)

    rows = []
    for col in LOG_COLUMNS:
        if df[col].dtype == bool:
            pass_val = df.loc[df.passed == 1, col].mean() * 100
            fail_val = df.loc[df.passed == 0, col].mean() * 100
            unit = "% of runs"
        else:
            pass_val = df.loc[df.passed == 1, col].mean()
            fail_val = df.loc[df.passed == 0, col].mean()
            unit = "avg count"

        ratio = fail_val / pass_val if pass_val > 0 else float("inf")
        print(f"{col:<28} PASS: {pass_val:>7.2f} {unit}   FAIL: {fail_val:>7.2f} {unit}   "
              f"({ratio:.1f}x higher in failures)" if ratio != float("inf")
              else f"{col:<28} PASS: {pass_val:>7.2f}   FAIL: {fail_val:>7.2f}")

        rows.append({
            "log_feature": col, "pass_value": round(pass_val, 3), "fail_value": round(fail_val, 3),
            "fail_to_pass_ratio": round(ratio, 2) if ratio != float("inf") else None,
        })
    print()
    return pd.DataFrame(rows)


# =================================================================
# PART 3: Failure signatures (fingerprints + log evidence)
# =================================================================
def build_cluster_input(failed_df: pd.DataFrame) -> pd.DataFrame:
    numeric_part = StandardScaler().fit_transform(failed_df[NUMERIC_CLUSTER_FEATURES])
    numeric_df = pd.DataFrame(numeric_part, columns=NUMERIC_CLUSTER_FEATURES, index=failed_df.index)
    cat_df = pd.get_dummies(failed_df[CATEGORICAL_CLUSTER_FEATURES], columns=CATEGORICAL_CLUSTER_FEATURES)
    bool_df = failed_df[BOOLEAN_CLUSTER_FEATURES].astype(int)
    return pd.concat([numeric_df, cat_df, bool_df], axis=1)


def compute_failure_signatures(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("FAILURE SIGNATURES (fingerprints + log evidence, Q5)")
    print("=" * 70)

    failed_df = df[df["passed"] == 0].copy()
    total_fails = len(failed_df)
    cluster_input = build_cluster_input(failed_df)

    # Same k as failure_clustering.py found best (re-searching would just
    # reproduce the same result -- skip straight to k=3 for speed here)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    failed_df["cluster"] = km.fit_predict(cluster_input)

    overall_log_avg = {col: df[col].mean() for col in LOG_COLUMNS}

    rows = []
    for cluster_id in sorted(failed_df["cluster"].unique()):
        cluster_df = failed_df[failed_df["cluster"] == cluster_id]
        pct_of_all_failures = len(cluster_df) / total_fails * 100

        print(f"\nCluster {cluster_id} -- {len(cluster_df)} failed runs "
              f"({pct_of_all_failures:.1f}% of all failures)")
        print("Log signature (this cluster vs overall dataset average):")

        signature = {"cluster": cluster_id, "failed_runs": len(cluster_df),
                     "pct_of_all_failures": round(pct_of_all_failures, 1)}

        for col in LOG_COLUMNS:
            cluster_avg = cluster_df[col].mean()
            overall_avg = overall_log_avg[col]
            multiplier = cluster_avg / overall_avg if overall_avg > 0 else float("nan")
            unit = "rate" if cluster_df[col].dtype == bool else "avg"
            print(f"  {col:<28} {unit}={cluster_avg:.2f}  ({multiplier:.1f}x dataset average)")
            signature[f"{col}_avg"] = round(cluster_avg, 3)
            signature[f"{col}_vs_baseline_multiplier"] = round(multiplier, 2) if not np.isnan(multiplier) else None

        rows.append(signature)

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RAW_PATH)

    contribution_stats = compute_contribution_stats(df)
    contribution_stats.to_csv(OUT_CONTRIBUTION, index=False)

    log_comparison = compute_log_comparison(df)
    log_comparison.to_csv(OUT_LOG_COMPARISON, index=False)

    signatures = compute_failure_signatures(df)
    signatures.to_csv(OUT_SIGNATURES, index=False)

    print("\n" + "=" * 70)
    print(f"Saved: {OUT_CONTRIBUTION}")
    print(f"Saved: {OUT_LOG_COMPARISON}")
    print(f"Saved: {OUT_SIGNATURES}")


if __name__ == "__main__":
    main()