r"""
config_explorer.py

STEP 5: "Find the best configurations" -- answers Q2 from the brief.

WHAT THIS SCRIPT DOES:
For each combination of cache_policy x scheduler x cpu_cores, calculates:
    - failure rate
    - average execution time
    - average throughput
    - average accuracy
    - average resource consumption
    - average reliability score
    - sample size (n) -- so you can see which combos have enough data
      to trust, vs. combos with only a handful of runs

Then ranks combinations by a composite "quality score" that rewards
high throughput/reliability and penalizes high failure rate, so you
can say things like:

    "Adaptive Cache + Dynamic Scheduler + 16 CPU cores
     Failure rate: 3.2%   Throughput: 912   Reliability: 94.1"

WHY cache_policy x scheduler x cpu_cores specifically?
These are the three dimensions the brief's own example calls out
("Cache Policy = Adaptive and Scheduler = Dynamic improve performance
by 22%"). You can add more dimensions (e.g. thread_pool_size) by
extending COMBO_DIMENSIONS below, but every dimension you add
multiplies the number of combinations and shrinks the sample size per
combo -- watch the 'n' column so you don't rank noise.

Usage (from backend/):
    python config_explorer.py
"""

import pandas as pd

RAW_PATH = "../data/raw/execution_logs.csv"
OUT_PATH = "../outputs/config_explorer.csv"

# The dimensions that define a "configuration combination" for this analysis.
# Change this list to explore different combos (e.g. add "thread_pool_size").
COMBO_DIMENSIONS = ["cache_policy", "scheduler", "feature_flag_7"]

MIN_SAMPLE_SIZE = 30  # combos with fewer runs than this get flagged as low-confidence


def build_config_explorer(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(COMBO_DIMENSIONS).agg(
        n=("passed", "count"),
        pass_rate=("passed", "mean"),
        avg_execution_time_ms=("execution_time_ms", "mean"),
        avg_throughput=("throughput", "mean"),
        avg_accuracy=("accuracy", "mean"),
        avg_resource_consumption_pct=("resource_consumption_pct", "mean"),
        avg_reliability_score=("reliability_score", "mean"),
    ).reset_index()

    grouped["failure_rate"] = 1 - grouped["pass_rate"]
    grouped = grouped.drop(columns=["pass_rate"])

    # Composite quality score: normalize throughput and reliability to 0-1,
    # subtract failure rate (already 0-1), so higher = better on all fronts.
    # This is a simple, explainable scoring formula -- easy to defend to judges,
    # unlike a black-box weighted model.
    thr_norm = (grouped["avg_throughput"] - grouped["avg_throughput"].min()) / (
        grouped["avg_throughput"].max() - grouped["avg_throughput"].min()
    )
    rel_norm = grouped["avg_reliability_score"] / 100

    grouped["quality_score"] = (0.4 * thr_norm + 0.4 * rel_norm - 0.2 * grouped["failure_rate"]).round(4)
    grouped["low_confidence"] = grouped["n"] < MIN_SAMPLE_SIZE

    # Round for readability
    for col in ["failure_rate"]:
        grouped[col] = (grouped[col] * 100).round(2)
    for col in ["avg_execution_time_ms", "avg_throughput", "avg_resource_consumption_pct", "avg_reliability_score"]:
        grouped[col] = grouped[col].round(1)
    grouped["avg_accuracy"] = grouped["avg_accuracy"].round(4)

    grouped = grouped.sort_values("quality_score", ascending=False).reset_index(drop=True)
    return grouped


def print_top_and_bottom(table: pd.DataFrame, top_n: int = 10, bottom_n: int = 5):
    print("=" * 70)
    print(f"TOP {top_n} CONFIGURATIONS (ranked by quality score)")
    print("=" * 70)
    for _, row in table.head(top_n).iterrows():
        combo_desc = " + ".join(f"{dim}={row[dim]}" for dim in COMBO_DIMENSIONS)
        flag = "  [LOW SAMPLE SIZE]" if row["low_confidence"] else ""
        print(f"\n{combo_desc}{flag}")
        print(f"  n={int(row['n'])}  Failure rate: {row['failure_rate']}%  "
              f"Throughput: {row['avg_throughput']}  Reliability: {row['avg_reliability_score']}  "
              f"Quality score: {row['quality_score']}")

    print("\n" + "=" * 70)
    print(f"BOTTOM {bottom_n} CONFIGURATIONS (avoid these)")
    print("=" * 70)
    for _, row in table.tail(bottom_n).iterrows():
        combo_desc = " + ".join(f"{dim}={row[dim]}" for dim in COMBO_DIMENSIONS)
        flag = "  [LOW SAMPLE SIZE]" if row["low_confidence"] else ""
        print(f"\n{combo_desc}{flag}")
        print(f"  n={int(row['n'])}  Failure rate: {row['failure_rate']}%  "
              f"Throughput: {row['avg_throughput']}  Reliability: {row['avg_reliability_score']}  "
              f"Quality score: {row['quality_score']}")
    print()


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows\n")
    print(f"Building Configuration Explorer across: {' x '.join(COMBO_DIMENSIONS)}")

    table = build_config_explorer(df)
    print(f"Found {len(table)} distinct configuration combinations\n")

    n_low_confidence = table["low_confidence"].sum()
    if n_low_confidence:
        print(f"Note: {n_low_confidence} combo(s) have fewer than {MIN_SAMPLE_SIZE} runs "
              f"-- flagged as low-confidence, treat their ranking cautiously.\n")

    print_top_and_bottom(table)

    table.to_csv(OUT_PATH, index=False)
    print(f"Full ranked table saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()