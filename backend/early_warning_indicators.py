r"""
early_warning_indicators.py

Closes Q7's "Early warning indicators" expected output. The underlying
numbers already existed scattered across failure_signatures.py and
config_intelligence.py -- this consolidates them into ONE ranked,
actionable list: conditions knowable BEFORE or DURING a run that predict
elevated failure risk, ranked by how much they multiply baseline risk.

WHY "RISK RATIO" AND NOT JUST "FAILURE RATE":
A condition with a 20% failure rate sounds alarming in isolation, but
if the dataset's baseline is already 15%, that's barely a warning sign
(1.3x). Ranking by risk RATIO (condition rate / baseline rate) is what
makes this genuinely useful as an early-warning system, rather than
just restating raw numbers.

Usage (from backend/):
    python early_warning_indicators.py
"""

import pandas as pd

RAW_PATH = "../data/raw/execution_logs.csv"
OUT_PATH = "../outputs/early_warning_indicators.csv"


def main():
    df = pd.read_csv(RAW_PATH)
    baseline_failure_rate = 1 - df["passed"].mean()
    total_failures = (df["passed"] == 0).sum()

    indicators = {
        "seed_group == 8": df["seed_group"] == 8,
        "feature_flag_7 enabled": df["feature_flag_7"] == True,
        "host_temperature_c > 65 (overheating)": df["host_temperature_c"] > 65,
        "host_temperature_c > 55 (warming -- earlier warning, before full overheat)": df["host_temperature_c"] > 55,
        "memory_alloc_mb <= 1024 AND thread_pool_size >= 16": (df["memory_alloc_mb"] <= 1024) & (df["thread_pool_size"] >= 16),
        "timeout_ms <= 100 under high cpu_load_pct (>70)": (df["timeout_ms"] <= 100) & (df["cpu_load_pct"] > 70),
    }

    rows = []
    for label, mask in indicators.items():
        n_matching = mask.sum()
        failure_rate_within = 1 - df.loc[mask, "passed"].mean()
        risk_ratio = failure_rate_within / baseline_failure_rate
        pct_of_all_failures = ((df["passed"] == 0) & mask).sum() / total_failures * 100

        rows.append({
            "indicator": label,
            "n_runs_matching": int(n_matching),
            "failure_rate_when_present_pct": round(failure_rate_within * 100, 1),
            "baseline_failure_rate_pct": round(baseline_failure_rate * 100, 1),
            "risk_ratio": round(risk_ratio, 2),
            "pct_of_all_failures_explained": round(pct_of_all_failures, 1),
        })

    result = pd.DataFrame(rows).sort_values("risk_ratio", ascending=False)
    result.to_csv(OUT_PATH, index=False)

    print("=" * 80)
    print(f"EARLY WARNING INDICATORS (baseline failure rate: {baseline_failure_rate:.1%})")
    print("=" * 80)
    for _, row in result.iterrows():
        print(f"\n{row['indicator']}")
        print(f"  Failure rate when present: {row['failure_rate_when_present_pct']}% "
              f"({row['risk_ratio']}x baseline risk)")
        print(f"  Explains {row['pct_of_all_failures_explained']}% of all observed failures "
              f"(n={row['n_runs_matching']} matching runs)")

    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()