r"""
test_generalization.py

Answers a critical question: what happens when this model sees a
configuration genuinely different from anything in the training data?

TWO KINDS OF "UNSEEN DATA":
1. Unseen ROWS within the training range -- already tested by the
   80/20 train/test split in config_intelligence.py. The model has
   never seen those exact 11,000 rows, and it still scores ~76-82%
   accuracy on them. This is real evidence of generalization.

2. Genuinely NOVEL values outside the training range (extrapolation) --
   NOT tested until now. Random Forests cannot extrapolate: each tree
   is a set of learned thresholds (e.g. "is host_temperature_c > 65?"),
   and a value far beyond anything seen in training just falls into
   whatever the outermost leaf happens to be. The prediction will still
   be a number, but it may not be trustworthy.

WHAT THIS SCRIPT DOES:
  1. Reports the actual min/max range of every numeric feature the
     model was trained on -- this defines what "in-distribution" means.
  2. Tests several genuinely novel scenarios: some within range but rare
     combinations, some deliberately OUTSIDE the trained range.
  3. Adds an automatic OUT-OF-DISTRIBUTION (OOD) check: flags when a
     requested prediction falls outside the training range, so the
     system can honestly say "this input is beyond what I've learned
     from" instead of quietly returning an overconfident number.

WHAT "FINE-TUNING" MEANS HERE:
Random Forests don't fine-tune like neural networks (no gradient
updates on new data). If real SanDisk data arrives, or if certain
input ranges matter more than others, the correct fix is RETRAINING
on an expanded/updated dataset using the exact same preprocessing
pipeline already built (generate_data.py -> preprocessing.py ->
config_intelligence.py), not adjusting the existing model in place.

Usage (from backend/):
    python test_generalization.py
"""

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/rf_model.joblib"
FEATURES_PATH = "../data/processed/features.csv"

NUMERIC_FEATURES_TO_CHECK = [
    "memory_alloc_mb", "thread_pool_size", "cache_size_mb", "timeout_ms",
    "host_temperature_c", "voltage_v", "cpu_load_pct", "memory_pressure_pct",
    "ambient_humidity_pct", "seed_group",
]


def compute_defaults(X: pd.DataFrame) -> pd.Series:
    defaults = {}
    for col in X.columns:
        unique_vals = X[col].dropna().unique()
        defaults[col] = X[col].mode().iloc[0] if set(unique_vals).issubset({0, 1}) else X[col].median()
    return pd.Series(defaults)


def build_feature_vector(spec: dict, defaults: pd.Series, feature_columns: list) -> pd.DataFrame:
    vector = defaults.copy()
    for key, value in spec.items():
        if key in vector.index:
            vector[key] = int(value) if isinstance(value, bool) else value
        else:
            matching_cols = [c for c in feature_columns if c.startswith(f"{key}_")]
            if matching_cols:
                for c in matching_cols:
                    vector[c] = 0
                target_col = f"{key}_{value}"
                if target_col in vector.index:
                    vector[target_col] = 1
    return pd.DataFrame([vector])[feature_columns]


def check_out_of_distribution(spec: dict, training_ranges: dict) -> list:
    """Return a list of warnings for any specified value that falls outside
    the range the model was actually trained on."""
    warnings = []
    for key, value in spec.items():
        if key in training_ranges:
            lo, hi = training_ranges[key]
            if value < lo or value > hi:
                warnings.append(
                    f"{key}={value} is OUTSIDE the training range [{lo}, {hi}] -- "
                    f"this prediction is EXTRAPOLATION, treat it with reduced trust."
                )
    return warnings


def predict_with_confidence(model, vector: pd.DataFrame) -> tuple:
    vector_array = vector.to_numpy()
    tree_preds = np.array([t.predict_proba(vector_array)[0][0] for t in model.estimators_])
    return tree_preds.mean(), tree_preds.std()


def main():
    model = joblib.load(MODEL_PATH)
    X = pd.read_csv(FEATURES_PATH)
    feature_columns = X.columns.tolist()
    defaults = compute_defaults(X)

    # Step 1: what range did the model actually train on?
    training_ranges = {col: (X[col].min(), X[col].max()) for col in NUMERIC_FEATURES_TO_CHECK}
    print("=" * 70)
    print("TRAINING DATA RANGE (this defines 'in-distribution')")
    print("=" * 70)
    for col, (lo, hi) in training_ranges.items():
        print(f"  {col:<25} [{lo}, {hi}]")
    print()

    # Step 2: test scenarios -- mix of in-range and deliberately out-of-range
    scenarios = [
        {
            "name": "In-range, common combination (sanity check)",
            "spec": {"cache_policy": "Adaptive", "scheduler": "Dynamic", "host_temperature_c": 45, "seed_group": 2},
        },
        {
            "name": "In-range but RARE combination (edge of what model has seen)",
            "spec": {"cache_policy": "FIFO", "scheduler": "RoundRobin", "memory_alloc_mb": 512,
                     "thread_pool_size": 32, "host_temperature_c": 62, "seed_group": 8},
        },
        {
            "name": "OUT-OF-RANGE: extreme temperature never seen in training",
            "spec": {"cache_policy": "Adaptive", "scheduler": "Dynamic", "host_temperature_c": 120, "seed_group": 2},
        },
        {
            "name": "OUT-OF-RANGE: memory allocation far beyond training grid",
            "spec": {"cache_policy": "Adaptive", "scheduler": "Dynamic", "memory_alloc_mb": 32768, "seed_group": 2},
        },
    ]

    print("=" * 70)
    print("SCENARIO TESTS")
    print("=" * 70)
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"  Inputs: {scenario['spec']}")

        warnings = check_out_of_distribution(scenario["spec"], training_ranges)
        vector = build_feature_vector(scenario["spec"], defaults, feature_columns)
        mean_fail, std_fail = predict_with_confidence(model, vector)

        print(f"  Predicted failure probability: {mean_fail:.1%}  (+/- {std_fail:.1%})")
        if warnings:
            print("  ** OUT-OF-DISTRIBUTION WARNING **")
            for w in warnings:
                print(f"     - {w}")
        else:
            print("  In-distribution: prediction is within the model's learned range.")

    print("\n" + "=" * 70)
    print("TAKEAWAY")
    print("=" * 70)
    print("Notice the OUT-OF-RANGE scenarios still produce a confident-looking number --")
    print("Random Forests don't refuse to predict on novel inputs, they just extrapolate")
    print("silently using whatever leaf node the input lands in. The confidence std alone")
    print("does NOT reliably widen for out-of-range inputs (it only reflects tree DISAGREEMENT,")
    print("not distance from training data). This is why the explicit range check above matters --")
    print("it's a second, independent safety check the std can't provide on its own.")


if __name__ == "__main__":
    main()