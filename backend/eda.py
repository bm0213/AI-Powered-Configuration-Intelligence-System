r"""
eda.py

STEP 4: "What happened?" -- plain descriptive statistics, no ML.

WHAT THIS SCRIPT DOES:
1. Overall summary: total executions, pass/fail counts, overall failure rate
2. For every important CATEGORICAL configuration column, a failure-rate
   breakdown by value (e.g. cache_policy: Static -> X%, Adaptive -> Y%)
3. For every important NUMERIC configuration column, a failure-rate
   breakdown by bucket (e.g. memory_alloc_mb: <=1024 -> X%, >1024 -> Y%)
4. Same breakdown for the key randomization and environment columns
   (seed_group, workload_type, node_region, etc.)
5. Saves one combined CSV with all breakdowns stacked together, plus
   prints everything to the terminal in a readable format

This directly answers the "Configuration Discovery" part of the brief
at its simplest level -- before any model or SHAP value gets involved.

Usage (from backend/):
    python eda.py
"""

import pandas as pd

RAW_PATH = "../data/raw/execution_logs.csv"
OUT_PATH = "../outputs/eda_breakdown.csv"

TARGET_COLUMN = "passed"

# Categorical config/randomization/environment columns worth breaking down
# one at a time -- the "important" ones a human would actually ask about,
# not all 90+ feature flags (those get covered by the feature importance
# model in step 5 instead, where a ranked list is more useful than 90
# separate tables).
CATEGORICAL_COLUMNS_TO_BREAK_DOWN = [
    "cache_policy",
    "scheduler",
    "compiler_opt_level",
    "workload_type",
    "traffic_pattern",
    "node_region",
    "gpu_enabled",
]

# Numeric columns worth bucketing -- (column_name, bucket_edges, bucket_labels)
NUMERIC_COLUMNS_TO_BUCKET = [
    ("memory_alloc_mb", [0, 1024, 2048, 4096, 100000], ["<=1024", "1025-2048", "2049-4096", ">4096"]),
    ("thread_pool_size", [0, 4, 8, 16, 100], ["<=4", "5-8", "9-16", ">16"]),
    ("cache_size_mb", [0, 128, 256, 512, 100000], ["<=128", "129-256", "257-512", ">512"]),
    ("timeout_ms", [0, 100, 1000, 100000], ["<=100 (aggressive)", "101-1000", ">1000"]),
    ("host_temperature_c", [0, 55, 65, 200], ["<=55 (normal)", "56-65 (warm)", ">65 (overheating)"]),
]

# Which seed group is worth calling out specifically
SEED_GROUP_COLUMN = "seed_group"


def print_overall_summary(df: pd.DataFrame):
    total = len(df)
    passed = (df[TARGET_COLUMN] == 1).sum()
    failed = (df[TARGET_COLUMN] == 0).sum()
    fail_rate = failed / total

    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total executions:      {total:,}")
    print(f"Successful executions: {passed:,} ({passed/total:.1%})")
    print(f"Failed executions:     {failed:,} ({fail_rate:.1%})")
    print(f"Overall failure rate:  {fail_rate:.1%}")
    print()

    return {"metric": "overall", "group": "all", "n": total, "failure_rate": round(fail_rate, 4)}


def breakdown_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = df.groupby(col)[TARGET_COLUMN].agg(["count", "mean"]).reset_index()
    grouped["failure_rate"] = 1 - grouped["mean"]
    grouped = grouped.rename(columns={col: "group", "count": "n"})
    grouped = grouped[["group", "n", "failure_rate"]].sort_values("failure_rate", ascending=False)
    grouped.insert(0, "metric", col)
    return grouped


def breakdown_numeric_bucketed(df: pd.DataFrame, col: str, edges: list, labels: list) -> pd.DataFrame:
    bucketed = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)
    temp = pd.DataFrame({"bucket": bucketed, "target": df[TARGET_COLUMN]})
    grouped = temp.groupby("bucket", observed=True)["target"].agg(["count", "mean"]).reset_index()
    grouped["failure_rate"] = 1 - grouped["mean"]
    grouped = grouped.rename(columns={"bucket": "group", "count": "n"})
    grouped = grouped[["group", "n", "failure_rate"]].sort_values("failure_rate", ascending=False)
    grouped.insert(0, "metric", col)
    return grouped


def print_breakdown(name: str, table: pd.DataFrame):
    print("-" * 60)
    print(f"{name} -> failure rate by group")
    print("-" * 60)
    display = table.copy()
    display["failure_rate"] = (display["failure_rate"] * 100).round(1).astype(str) + "%"
    print(display[["group", "n", "failure_rate"]].to_string(index=False))
    print()


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows\n")

    all_tables = []

    overall = print_overall_summary(df)
    all_tables.append(pd.DataFrame([overall]))

    print("#" * 60)
    print("CATEGORICAL CONFIGURATION BREAKDOWNS")
    print("#" * 60 + "\n")
    for col in CATEGORICAL_COLUMNS_TO_BREAK_DOWN:
        table = breakdown_categorical(df, col)
        print_breakdown(col, table)
        all_tables.append(table)

    print("#" * 60)
    print("NUMERIC CONFIGURATION BREAKDOWNS (bucketed)")
    print("#" * 60 + "\n")
    for col, edges, labels in NUMERIC_COLUMNS_TO_BUCKET:
        table = breakdown_numeric_bucketed(df, col, edges, labels)
        print_breakdown(col, table)
        all_tables.append(table)

    print("#" * 60)
    print("RANDOMIZATION BREAKDOWN")
    print("#" * 60 + "\n")
    seed_table = breakdown_categorical(df, SEED_GROUP_COLUMN)
    print_breakdown(SEED_GROUP_COLUMN, seed_table)
    all_tables.append(seed_table)

    # Save one combined file with every breakdown stacked together
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"All breakdowns saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()