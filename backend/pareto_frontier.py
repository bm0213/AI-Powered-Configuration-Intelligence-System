r"""
pareto_frontier.py

STEP 13: Pareto frontier -- identify configurations where you can't
improve one objective (throughput) without making the other worse
(failure rate). The brief explicitly asks for "Pareto-optimal
configurations", so this directly satisfies that requirement.

DEFINITION:
A configuration is Pareto-optimal (non-dominated) if NO other
configuration has BOTH a lower-or-equal failure rate AND a
higher-or-equal throughput (with at least one strictly better).
In plain terms: every point on the frontier is a genuine trade-off --
there's no free lunch among them. Anything NOT on the frontier is
strictly worse than some other option on at least one axis without
compensating on the other, so it should never be chosen.

REUSES config_explorer.csv from step 5 -- no new modeling needed.

Usage (from backend/):
    python pareto_frontier.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPLORER_PATH = "../outputs/config_explorer.csv"
OUT_CHART = "../outputs/pareto_frontier.png"
OUT_CSV = "../outputs/pareto_optimal_configs.csv"
COMBO_DIMENSIONS = ["cache_policy", "scheduler", "feature_flag_7"]  # must match config_explorer.py


def describe_config(row: pd.Series) -> str:
    return " + ".join(f"{dim}={row[dim]}" for dim in COMBO_DIMENSIONS)


def find_pareto_optimal(df: pd.DataFrame) -> pd.DataFrame:
    """A row is Pareto-optimal if no other row dominates it: i.e. no other
    row has failure_rate <= this row's AND avg_throughput >= this row's,
    with at least one strictly better."""
    is_optimal = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            better_or_equal_fail = other["failure_rate"] <= row["failure_rate"]
            better_or_equal_throughput = other["avg_throughput"] >= row["avg_throughput"]
            strictly_better = (other["failure_rate"] < row["failure_rate"]) or \
                               (other["avg_throughput"] > row["avg_throughput"])
            if better_or_equal_fail and better_or_equal_throughput and strictly_better:
                dominated = True
                break
        is_optimal.append(not dominated)
    return df[is_optimal].copy()


def plot_frontier(all_configs: pd.DataFrame, pareto_configs: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    plt.scatter(all_configs["avg_throughput"], all_configs["failure_rate"],
                color="lightgray", label="Dominated (avoid)", s=60)
    pareto_sorted = pareto_configs.sort_values("avg_throughput")
    plt.plot(pareto_sorted["avg_throughput"], pareto_sorted["failure_rate"],
              color="#C44E52", marker="o", linewidth=2, markersize=9, label="Pareto-optimal")

    for _, row in pareto_configs.iterrows():
        plt.annotate(describe_config(row).replace(" + ", "\n"),
                     (row["avg_throughput"], row["failure_rate"]),
                     fontsize=7, xytext=(6, 6), textcoords="offset points")

    plt.xlabel("Throughput")
    plt.ylabel("Failure Rate (%)")
    plt.title("Pareto Frontier: Failure Rate vs Throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=120)
    plt.close()


def main():
    df = pd.read_csv(EXPLORER_PATH)
    eligible = df[~df["low_confidence"]].copy()
    print(f"Evaluating {len(eligible)} configurations with sufficient historical data\n")

    pareto = find_pareto_optimal(eligible)
    pareto = pareto.sort_values("avg_throughput")

    print("=" * 70)
    print(f"PARETO-OPTIMAL CONFIGURATIONS ({len(pareto)} of {len(eligible)} total)")
    print("=" * 70)
    print("These are the only configurations worth considering -- every other")
    print("configuration is strictly worse on at least one axis with no compensation.\n")

    for _, row in pareto.iterrows():
        print(f"  {describe_config(row)}")
        print(f"    Failure rate: {row['failure_rate']}%   Throughput: {row['avg_throughput']}   "
              f"Reliability: {row['avg_reliability_score']}\n")

    plot_frontier(eligible, pareto)
    pareto.to_csv(OUT_CSV, index=False)

    print(f"Chart saved to: {OUT_CHART}")
    print(f"Pareto-optimal configs saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()