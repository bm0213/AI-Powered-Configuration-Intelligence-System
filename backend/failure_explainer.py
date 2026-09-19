r"""
failure_explainer.py

STEP 8: Explain ONE specific failed execution -- answers Q5
("Why did this execution fail?").

DIFFERENCE FROM THE SHAP SUMMARY PLOT (already built in config_intelligence.py):
That plot shows which features matter ACROSS ALL 55,000 runs (global).
This script picks ONE run and shows what pushed THAT run toward failure
(local). Same underlying SHAP math, different question being answered.

OUTPUT FORMAT (matches what a dashboard "explain this failure" button
should show):

    Execution: RUN-004821
    Prediction: Failure probability = 87%
    Baseline (average run): 15%

    Contributors:
      feature_flag_7 = True         +22.4%
      host_temperature_c = 84.2     +18.1%
      seed_group = 8                +13.0%
      memory_alloc_mb = 512         + 9.4%
      cache_policy = Adaptive       - 5.2%

Usage (from backend/):
    python failure_explainer.py                  -> auto-picks the most
                                                      confidently-predicted
                                                      real failure as the example
    python failure_explainer.py RUN-004821        -> explain a specific run_id
"""

import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

MODEL_PATH = "models/rf_model.joblib"
FEATURES_PATH = "../data/processed/features.csv"
TARGET_PATH = "../data/processed/target.csv"
RAW_PATH = "../data/raw/execution_logs.csv"
OUT_CHART = "../outputs/failure_explanation.png"

TOP_N_CONTRIBUTORS = 8
BACKGROUND_SAMPLE_SIZE = 100  # small background sample keeps SHAP fast for a single-row explanation


def pick_example_run(model, X: pd.DataFrame, y: pd.Series, raw_df: pd.DataFrame, run_id: str = None):
    """Pick which run to explain: either the one requested by run_id,
    or -- if none given -- the actual failure the model is MOST confident
    about, which makes the cleanest demo example."""
    if run_id:
        matches = raw_df.index[raw_df["run_id"] == run_id].tolist()
        if not matches:
            print(f"run_id '{run_id}' not found. Falling back to auto-selection.")
            run_id = None
        else:
            idx = matches[0]
            return idx

    fail_proba = model.predict_proba(X)[:, 0]  # class 0 = FAIL (passed=0)
    actual_fails = raw_df["passed"] == 0
    # among real failures, find the one the model is most confident is a failure
    candidate_idx = fail_proba[actual_fails.values].argmax()
    idx = raw_df.index[actual_fails.values][candidate_idx]
    return idx


def readable_feature_value(raw_df: pd.DataFrame, run_idx: int, feature_name: str) -> str:
    """Turn an encoded feature name back into something readable using the
    original raw dataframe where possible (e.g. 'cache_policy_Adaptive' -> 'True',
    but showing the raw 'cache_policy' column's actual value is more useful)."""
    # One-hot encoded columns look like "cache_policy_Adaptive" -- try to
    # find the original raw column by stripping known category suffixes.
    for raw_col in raw_df.columns:
        if feature_name.startswith(raw_col + "_") and raw_col != feature_name:
            return f"{raw_col}={raw_df.loc[run_idx, raw_col]}"

    if feature_name in raw_df.columns:
        return f"{feature_name}={raw_df.loc[run_idx, feature_name]}"

    return feature_name  # fallback: just show the encoded name as-is


def explain_single_run(model, X: pd.DataFrame, raw_df: pd.DataFrame, run_idx: int):
    run_id = raw_df.loc[run_idx, "run_id"]
    actual_outcome = "FAIL" if raw_df.loc[run_idx, "passed"] == 0 else "PASS"

    background = X.sample(n=min(BACKGROUND_SAMPLE_SIZE, len(X)), random_state=42)
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional", model_output="probability"
    )

    single_row = X.loc[[run_idx]]
    shap_values = explainer.shap_values(single_row, check_additivity=False)

    # SHAP's return shape varies by version:
    #   - older versions: a list [class_0_array, class_1_array], each (n_samples, n_features)
    #   - newer versions: a single ndarray shaped (n_samples, n_features, n_classes)
    # Handle both so this script doesn't break on a version difference.
    if isinstance(shap_values, list):
        fail_contributions = shap_values[0][0]  # class 0 = FAIL, first (only) sample
    else:
        fail_contributions = shap_values[0, :, 0]  # first sample, all features, class 0 = FAIL

    expected_value = explainer.expected_value
    baseline_fail_prob = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    predicted_fail_prob = model.predict_proba(single_row)[0][0]

    contrib_series = pd.Series(fail_contributions, index=X.columns).sort_values(key=abs, ascending=False)
    top_contributors = contrib_series.head(TOP_N_CONTRIBUTORS)

    print("=" * 60)
    print(f"Execution: {run_id}  (actual outcome: {actual_outcome})")
    print("=" * 60)
    print(f"Predicted failure probability: {predicted_fail_prob:.1%}")
    print(f"Baseline (average run):        {baseline_fail_prob:.1%}")
    print()
    print("Contributors (impact on failure probability):")
    labels = []
    values = []
    for feature, value in top_contributors.items():
        readable = readable_feature_value(raw_df, run_idx, feature)
        sign = "+" if value >= 0 else "-"
        print(f"  {readable:<35} {sign}{abs(value)*100:5.1f}%")
        labels.append(readable)
        values.append(value * 100)

    # Save a horizontal bar chart -- this is the dashboard-ready visual
    plt.figure(figsize=(8, 5))
    colors = ["#C44E52" if v > 0 else "#55A868" for v in values]
    plt.barh(labels[::-1], values[::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Contribution to failure probability (percentage points)")
    plt.title(f"Why did {run_id} fail? (predicted: {predicted_fail_prob:.0%}, baseline: {baseline_fail_prob:.0%})")
    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=120)
    plt.close()
    print(f"\nSaved chart: {OUT_CHART}")


def main():
    requested_run_id = sys.argv[1] if len(sys.argv) > 1 else None

    model = joblib.load(MODEL_PATH)
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGET_PATH)["passed"]
    raw_df = pd.read_csv(RAW_PATH)

    run_idx = pick_example_run(model, X, y, raw_df, run_id=requested_run_id)
    explain_single_run(model, X, raw_df, run_idx)


if __name__ == "__main__":
    main()