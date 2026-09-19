r"""
what_if_analysis.py

STEP 14: What-if analysis -- the decision-support layer.

Given a BASELINE configuration and a CHANGED configuration (one or more
settings modified), predict both failure probability AND throughput for
each, then state the trade-off in plain English:

    "Disabling feature_flag_7 is predicted to reduce failure probability
    by 11.3 percentage points, with an estimated 1.6% throughput reduction."

Shared logic (build_feature_vector, defaults, prediction, OOD check)
now lives in model_utils.py -- see that file's docstring for why.

Usage (from backend/):
    python what_if_analysis.py
    (edit BASELINE_SPEC / CHANGES below to test your own comparisons)
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from model_utils import compute_defaults, compute_training_ranges, predict

MODEL_PATH = "models/rf_model.joblib"
THROUGHPUT_MODEL_PATH = "models/throughput_model.joblib"
FEATURES_PATH = "../data/processed/features.csv"
OUTCOMES_PATH = "../data/processed/outcomes.csv"

# The starting point for the comparison. Edit freely.
BASELINE_SPEC = {
    "cache_policy": "Adaptive", "scheduler": "Dynamic",
    "memory_alloc_mb": 4096, "thread_pool_size": 8,
    "feature_flag_7": True, "seed_group": 2, "host_temperature_c": 45,
}

# What changes from the baseline. Only list what's DIFFERENT.
CHANGES = {"feature_flag_7": False}
CHANGE_DESCRIPTION = "Disabling feature_flag_7"


def train_or_load_throughput_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    if os.path.exists(THROUGHPUT_MODEL_PATH):
        return joblib.load(THROUGHPUT_MODEL_PATH)
    print("Training throughput regression model (one-time, ~10 seconds)...")
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, THROUGHPUT_MODEL_PATH)
    print(f"Saved to {THROUGHPUT_MODEL_PATH}\n")
    return model


def main():
    failure_model = joblib.load(MODEL_PATH)
    X = pd.read_csv(FEATURES_PATH)
    outcomes = pd.read_csv(OUTCOMES_PATH)
    feature_columns = X.columns.tolist()
    defaults = compute_defaults(X)
    training_ranges = compute_training_ranges(X)

    throughput_model = train_or_load_throughput_model(X, outcomes["throughput"])

    changed_spec = {**BASELINE_SPEC, **CHANGES}

    before = predict(BASELINE_SPEC, failure_model, throughput_model, defaults, feature_columns, training_ranges)
    after = predict(changed_spec, failure_model, throughput_model, defaults, feature_columns, training_ranges)

    print("=" * 65)
    print("WHAT-IF ANALYSIS")
    print("=" * 65)
    print("\nCurrent configuration:")
    for k, v in BASELINE_SPEC.items():
        print(f"  {k} = {v}")
    print(f"\n  Failure probability = {before['failure_probability_pct']}%  (+/- {before['confidence_std_pct']}%)")
    print(f"  Throughput          = {before['predicted_throughput']}")

    print(f"\nChange: {CHANGE_DESCRIPTION}")
    print("\nPredicted configuration:")
    for k, v in changed_spec.items():
        marker = "  <- changed" if k in CHANGES else ""
        print(f"  {k} = {v}{marker}")
    print(f"\n  Failure probability = {after['failure_probability_pct']}%  (+/- {after['confidence_std_pct']}%)")
    print(f"  Throughput          = {after['predicted_throughput']}")

    if after.get("ood_warnings") or before.get("ood_warnings"):
        print("\n*** OUT-OF-DISTRIBUTION WARNING ***")
        for w in before.get("ood_warnings", []) + after.get("ood_warnings", []):
            print(f"  {w}")

    fail_delta_pp = after["failure_probability_pct"] - before["failure_probability_pct"]
    throughput_delta_pct = (after["predicted_throughput"] - before["predicted_throughput"]) / before["predicted_throughput"] * 100

    print("\n" + "=" * 65)
    print("DECISION SUPPORT")
    print("=" * 65)
    fail_direction = "reduce" if fail_delta_pp < 0 else "increase"
    throughput_direction = "reduction" if throughput_delta_pct < 0 else "increase"
    print(f"{CHANGE_DESCRIPTION} is predicted to {fail_direction} failure probability by "
          f"{abs(fail_delta_pp):.1f} percentage points, with an estimated "
          f"{abs(throughput_delta_pct):.1f}% throughput {throughput_direction}.")


if __name__ == "__main__":
    main()