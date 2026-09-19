r"""
predict_new_execution.py

STEP 11: Predict failure probability for a NEW (not-yet-run) execution,
given a proposed configuration -- answers the forward-looking half of Q7.

DIFFERENCE FROM failure_explainer.py (step 8):
That script explains a HISTORICAL execution that already happened.
This script predicts a HYPOTHETICAL one -- you specify some or all of
the configuration/randomization settings, and it fills in anything you
didn't specify with a sensible default (the dataset's median/most
common value), then predicts.

CONFIDENCE, NOT JUST A NUMBER:
A Random Forest is 200 individual decision trees voting. If all 200
trees agree closely, the prediction is well-supported. If they're
scattered, the model itself is uncertain about this particular
combination of settings. We report the STANDARD DEVIATION across the
200 trees' individual votes as a simple, explainable confidence measure.

Shared logic (build_feature_vector, defaults, prediction, OOD check)
now lives in model_utils.py -- see that file's docstring for why.

Usage (from backend/):
    python predict_new_execution.py
    (edit the SCENARIOS list below to test your own hypothetical configs)
"""

import joblib
import pandas as pd
import shap

from model_utils import (
    compute_defaults, compute_training_ranges, build_feature_vector, predict,
)

MODEL_PATH = "models/rf_model.joblib"
THROUGHPUT_MODEL_PATH = "models/throughput_model.joblib"
FEATURES_PATH = "../data/processed/features.csv"

TOP_N_CONTRIBUTORS = 6
BACKGROUND_SAMPLE_SIZE = 100

# Example hypothetical executions to predict. Edit these to test your own --
# any raw column can be specified; anything left out uses a sensible default.
SCENARIOS = [
    {
        "name": "Typical safe configuration",
        "spec": {
            "cache_policy": "Adaptive", "scheduler": "Dynamic",
            "memory_alloc_mb": 4096, "thread_pool_size": 8,
            "feature_flag_7": False, "seed_group": 2, "host_temperature_c": 45,
        },
    },
    {
        "name": "Same configuration, but landing on a seed_group 8 / hot node",
        "spec": {
            "cache_policy": "Adaptive", "scheduler": "Dynamic",
            "memory_alloc_mb": 4096, "thread_pool_size": 8,
            "feature_flag_7": False, "seed_group": 8, "host_temperature_c": 75,
        },
    },
    {
        "name": "Risky configuration (low memory + high threads + flag_7 on)",
        "spec": {
            "cache_policy": "FIFO", "scheduler": "Static",
            "memory_alloc_mb": 512, "thread_pool_size": 32,
            "feature_flag_7": True, "seed_group": 3, "host_temperature_c": 48,
        },
    },
    {
        "name": "Out-of-range test (deliberately extreme temperature)",
        "spec": {
            "cache_policy": "Adaptive", "scheduler": "Dynamic",
            "host_temperature_c": 120, "seed_group": 2,
        },
    },
]


def explain_prediction(model, vector: pd.DataFrame, background: pd.DataFrame, feature_columns: list):
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional", model_output="probability"
    )
    shap_values = explainer.shap_values(vector, check_additivity=False)
    fail_contributions = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0, :, 0]
    contrib_series = pd.Series(fail_contributions, index=feature_columns).sort_values(key=abs, ascending=False)
    return contrib_series.head(TOP_N_CONTRIBUTORS)


def readable_value(vector_row: pd.Series, feature_name: str) -> str:
    if "_" in feature_name and vector_row.get(feature_name) in (0, 1):
        return f"{feature_name} = {'ON' if vector_row[feature_name] == 1 else 'OFF'}"
    return f"{feature_name} = {vector_row[feature_name]}"


def main():
    failure_model = joblib.load(MODEL_PATH)
    throughput_model = joblib.load(THROUGHPUT_MODEL_PATH)
    X = pd.read_csv(FEATURES_PATH)
    feature_columns = X.columns.tolist()
    defaults = compute_defaults(X)
    training_ranges = compute_training_ranges(X)
    background = X.sample(n=min(BACKGROUND_SAMPLE_SIZE, len(X)), random_state=42)

    for scenario in SCENARIOS:
        print("=" * 65)
        print(f"SCENARIO: {scenario['name']}")
        print("=" * 65)
        print("Specified settings:")
        for k, v in scenario["spec"].items():
            print(f"  {k} = {v}")

        result = predict(scenario["spec"], failure_model, throughput_model, defaults, feature_columns, training_ranges)

        if result.get("ood_warnings"):
            print("\n*** OUT-OF-DISTRIBUTION WARNING ***")
            for w in result["ood_warnings"]:
                print(f"  {w}")

        print(f"\nPredicted failure probability: {result['failure_probability_pct']}%  "
              f"(+/- {result['confidence_std_pct']}% across 200 trees)")
        print(f"Predicted throughput: {result['predicted_throughput']}")

        std = result["confidence_std_pct"]
        if std < 10:
            confidence_note = "High confidence -- the model's trees mostly agree."
        elif std < 20:
            confidence_note = "Moderate confidence -- some disagreement among trees."
        else:
            confidence_note = "Low confidence -- this configuration is unusual or underrepresented in training data."
        print(confidence_note)

        vector = build_feature_vector(scenario["spec"], defaults, feature_columns)
        top_contributors = explain_prediction(failure_model, vector, background, feature_columns)
        print("\nTop contributors to this prediction:")
        vector_row = vector.iloc[0]
        for feature, value in top_contributors.items():
            sign = "+" if value >= 0 else "-"
            print(f"  {readable_value(vector_row, feature):<35} {sign}{abs(value)*100:5.1f}%")
        print()


if __name__ == "__main__":
    main()