r"""
model_utils.py

SHARED LOGIC used by predict_new_execution.py, what_if_analysis.py,
recommendation_engine.py, ai_copilot.py, and frontend/dashboard.py.

WHY THIS MODULE EXISTS:
Earlier in this project, build_feature_vector(), compute_defaults(),
the prediction logic, and the Wilson confidence interval calculation
were copy-pasted nearly identically into 5+ different files. That's a
real maintenance hazard -- when the probability calibration bug was
found and fixed earlier, it had to be fixed in multiple places by hand.
This module is the single source of truth: fix a bug here once, every
script and the dashboard immediately benefit.

USAGE:
    from model_utils import compute_defaults, build_feature_vector, predict, \
        wilson_confidence_interval, confidence_label, compute_training_ranges

    defaults = compute_defaults(X)
    training_ranges = compute_training_ranges(X)
    result = predict(spec, failure_model, throughput_model, defaults,
                      X.columns.tolist(), training_ranges)
"""

import math
import numpy as np
import pandas as pd

# Numeric features checked for out-of-distribution (extrapolation) warnings.
# See test_generalization.py for the analysis that established why this
# check matters -- Random Forests silently extrapolate without it.
# NOTE: consolidating this module revealed that different copies of this
# check had drifted -- predict_new_execution.py checked 10 features while
# ai_copilot.py/dashboard.py only checked 4. Using the more thorough list
# here so every caller gets the same, complete check.
_OOD_CHECK_FEATURES = [
    "memory_alloc_mb", "thread_pool_size", "cache_size_mb", "timeout_ms",
    "host_temperature_c", "voltage_v", "cpu_load_pct", "memory_pressure_pct",
    "ambient_humidity_pct", "seed_group",
]


def compute_defaults(X: pd.DataFrame) -> pd.Series:
    """A 'typical' run: median for numeric columns, most common value (mode)
    for boolean/one-hot columns. Used to fill in anything a partial config
    spec doesn't specify."""
    defaults = {}
    for col in X.columns:
        unique_vals = X[col].dropna().unique()
        defaults[col] = X[col].mode().iloc[0] if set(unique_vals).issubset({0, 1}) else X[col].median()
    return pd.Series(defaults)


def compute_training_ranges(X: pd.DataFrame) -> dict:
    """Min/max range actually seen in training for each OOD-checked feature --
    defines what 'in-distribution' means for the out-of-distribution check."""
    return {col: (float(X[col].min()), float(X[col].max()))
            for col in _OOD_CHECK_FEATURES if col in X.columns}


def build_feature_vector(spec: dict, defaults: pd.Series, feature_columns: list) -> pd.DataFrame:
    """Turn a partial config spec (e.g. {'cache_policy': 'Adaptive', 'seed_group': 2})
    into a full one-row feature vector matching the model's expected columns,
    filling anything unspecified with the dataset-typical default. Handles both
    direct numeric/boolean columns and one-hot encoded categorical columns."""
    vector = defaults.copy()
    for key, value in spec.items():
        if value is None:
            continue
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
    """Flag any specified value that falls outside the range the model was
    actually trained on -- Random Forests extrapolate silently otherwise,
    with no natural warning sign in the confidence interval alone."""
    warnings = []
    for key, value in spec.items():
        if key in training_ranges and value is not None:
            lo, hi = training_ranges[key]
            if value < lo or value > hi:
                warnings.append(
                    f"{key}={value} is OUTSIDE the training range [{lo}, {hi}] -- "
                    f"this prediction is EXTRAPOLATION, treat it with reduced trust."
                )
    return warnings


def predict(spec: dict, failure_model, throughput_model, defaults: pd.Series,
            feature_columns: list, training_ranges: dict) -> dict:
    """Predict failure probability (with confidence via tree-vote spread) and
    throughput for a hypothetical configuration, with an automatic
    out-of-distribution safety check attached."""
    vector = build_feature_vector(spec, defaults, feature_columns)
    vector_array = vector.to_numpy()
    tree_preds = np.array([t.predict_proba(vector_array)[0][0] for t in failure_model.estimators_])
    throughput = throughput_model.predict(vector)[0]

    result = {
        "failure_probability_pct": round(float(tree_preds.mean()) * 100, 1),
        "confidence_std_pct": round(float(tree_preds.std()) * 100, 1),
        "predicted_throughput": round(float(throughput), 1),
    }
    warnings = check_out_of_distribution(spec, training_ranges)
    if warnings:
        result["ood_warnings"] = warnings
    return result


def wilson_confidence_interval(p_pct: float, n: int, z: float = 1.96) -> tuple:
    """95% confidence interval for a proportion using the Wilson score method --
    more reliable than the naive normal approximation when the proportion is
    close to 0 or 1 (exactly our case: failure rates mostly 5-25%)."""
    p = p_pct / 100
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lower, upper = max(0, center - margin), min(1, center + margin)
    return round(lower * 100, 2), round(upper * 100, 2)


def confidence_label(ci_lower: float, ci_upper: float) -> str:
    """Simple High/Medium/Low label based on how wide the 95% CI is."""
    width = ci_upper - ci_lower
    if width <= 3:
        return "High"
    elif width <= 7:
        return "Medium"
    else:
        return "Low"