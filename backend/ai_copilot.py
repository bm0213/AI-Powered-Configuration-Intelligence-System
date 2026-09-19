r"""
ai_copilot.py

STEP 15: A real AI Copilot using Gemini's automatic function calling --
not a dropdown of pre-written answers.

ARCHITECTURE (matches the diagram your teammate shared):

    USER
      |
      v
    Gemini (decides which tool to call based on the question)
      |
      +--> predict_execution      (wraps predict_new_execution.py logic)
      +--> compare_configurations (wraps what_if_analysis.py logic -- single field change)
      +--> compare_named_configurations (arbitrary A vs B, multi-field, SHAP-ranked differentiators)
      +--> randomization_stats    (wraps randomization_intelligence.py results)
      +--> best_configuration     (wraps recommendation_engine.py logic)
      |
      v
    Structured result (numbers, not prose) goes back to Gemini
      |
      v
    Gemini turns it into a human-readable answer

WHY THIS IS BETTER THAN A KEYWORD-MATCHING CHATBOT:
Gemini never invents numbers -- it can ONLY answer using what these four
tools actually return, which are the same trained models and statistical
results used throughout this project. This keeps the copilot honest: a
natural-language INTERFACE to real analysis, not a new source of truth.

WHY FLAT FUNCTION SIGNATURES (not a single "config: dict" argument):
Passing plain Python functions as tools relies on Gemini's automatic
schema inference from type hints and docstrings. Dicts with arbitrary
keys don't infer into a reliable JSON schema. Flat, named, optional
parameters (cache_policy: str = None, memory_alloc_mb: int = None, ...)
infer correctly and are also easier for the LLM to fill in accurately.

SETUP:
    pip install google-genai
    Set GEMINI_API_KEY as an environment variable (see README)

Usage (from backend/):
    python ai_copilot.py
    (then type questions at the prompt, or press Enter on an empty line to quit)
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types

from model_utils import (
    compute_defaults, compute_training_ranges, build_feature_vector,
    predict as _predict_impl, wilson_confidence_interval, confidence_label,
)

load_dotenv()  # reads .env in the project root if present

MODEL_PATH = "models/rf_model.joblib"
THROUGHPUT_MODEL_PATH = "models/throughput_model.joblib"
FEATURES_PATH = "../data/processed/features.csv"
OUTCOMES_PATH = "../data/processed/outcomes.csv"
RANDOMIZATION_PATH = "../outputs/randomization_impact.csv"
CONFIG_EXPLORER_PATH = "../outputs/config_explorer.csv"

LLM_MODEL = "gemini-3.6-flash"  # fast + cheap, plenty capable for routing + explaining numbers

# ---------------------------------------------------------------
# Load everything once at startup
# ---------------------------------------------------------------
failure_model = joblib.load(MODEL_PATH)
throughput_model = joblib.load(THROUGHPUT_MODEL_PATH)
X = pd.read_csv(FEATURES_PATH)
outcomes = pd.read_csv(OUTCOMES_PATH)
randomization_impact = pd.read_csv(RANDOMIZATION_PATH)
config_explorer = pd.read_csv(CONFIG_EXPLORER_PATH)
FEATURE_COLUMNS = X.columns.tolist()

DEFAULTS = compute_defaults(X)
TRAINING_RANGES = compute_training_ranges(X)


def _build_feature_vector(spec: dict) -> pd.DataFrame:
    return build_feature_vector(spec, DEFAULTS, FEATURE_COLUMNS)


def _predict(spec: dict) -> dict:
    """Thin wrapper: keeps this file's internal name/shape (and the
    'out_of_distribution_warnings' key the system prompt already
    references) while delegating the actual logic to model_utils."""
    result = _predict_impl(spec, failure_model, throughput_model, DEFAULTS, FEATURE_COLUMNS, TRAINING_RANGES)
    if "ood_warnings" in result:
        result["out_of_distribution_warnings"] = result.pop("ood_warnings")
    return result


# =================================================================
# TOOL 1: predict_execution
# =================================================================
def predict_execution(
    cache_policy: Optional[str] = None,
    scheduler: Optional[str] = None,
    memory_alloc_mb: Optional[int] = None,
    thread_pool_size: Optional[int] = None,
    feature_flag_7: Optional[bool] = None,
    seed_group: Optional[int] = None,
    host_temperature_c: Optional[float] = None,
) -> dict:
    """Predict failure probability and throughput for ONE hypothetical execution configuration.
    Use this when the user asks about a single specific configuration.
    Any parameter left unset uses a dataset-typical default value.

    Args:
        cache_policy: One of Adaptive, Static, LRU, FIFO.
        scheduler: One of Dynamic, Static, RoundRobin.
        memory_alloc_mb: Memory allocation in megabytes, e.g. 512, 1024, 2048, 4096, 8192.
        thread_pool_size: Thread pool size, e.g. 2, 4, 8, 16, 32.
        feature_flag_7: Whether feature_flag_7 is enabled.
        seed_group: Randomization seed group bucket, 0 through 9.
        host_temperature_c: Node temperature in Celsius.
    """
    spec = {k: v for k, v in locals().items() if v is not None}
    result = _predict(spec)
    result["config_used"] = spec
    return result


# =================================================================
# TOOL 2: compare_configurations
# =================================================================
def compare_configurations(
    change_field: str,
    change_value: str,
    cache_policy: Optional[str] = None,
    scheduler: Optional[str] = None,
    memory_alloc_mb: Optional[int] = None,
    thread_pool_size: Optional[int] = None,
    feature_flag_7: Optional[bool] = None,
    seed_group: Optional[int] = None,
    host_temperature_c: Optional[float] = None,
) -> dict:
    """Compare a baseline configuration against the SAME configuration with ONE setting changed.
    Use this for 'what if I change X' questions or comparisons between two named configurations.
    The baseline fields (cache_policy, scheduler, etc.) describe the STARTING configuration --
    any left unset use dataset-typical defaults. change_field and change_value describe what
    is different in the second configuration.

    Args:
        change_field: Name of the setting being changed, e.g. "feature_flag_7", "memory_alloc_mb".
        change_value: The new value for that setting, as a string, e.g. "False", "1024", "Adaptive".
        cache_policy: Baseline cache policy: Adaptive, Static, LRU, or FIFO.
        scheduler: Baseline scheduler: Dynamic, Static, or RoundRobin.
        memory_alloc_mb: Baseline memory allocation in MB.
        thread_pool_size: Baseline thread pool size.
        feature_flag_7: Baseline state of feature_flag_7.
        seed_group: Baseline seed group, 0 through 9.
        host_temperature_c: Baseline node temperature in Celsius.
    """
    baseline_spec = {k: v for k, v in locals().items()
                      if v is not None and k not in ("change_field", "change_value")}

    # Cast change_value to the right type based on which field it targets
    if change_field in ("feature_flag_7",):
        casted_value = change_value.strip().lower() in ("true", "1", "yes", "on")
    elif change_field in ("memory_alloc_mb", "thread_pool_size", "seed_group"):
        casted_value = int(float(change_value))
    elif change_field in ("host_temperature_c",):
        casted_value = float(change_value)
    else:
        casted_value = change_value  # cache_policy, scheduler -- stay as string

    changed_spec = {**baseline_spec, change_field: casted_value}

    before = _predict(baseline_spec)
    after = _predict(changed_spec)

    fail_delta_pp = after["failure_probability_pct"] - before["failure_probability_pct"]
    throughput_delta_pct = (
        (after["predicted_throughput"] - before["predicted_throughput"])
        / before["predicted_throughput"] * 100
    )

    return {
        "before": before,
        "after": after,
        "change_applied": f"{change_field} -> {casted_value}",
        "failure_delta_percentage_points": round(fail_delta_pp, 2),
        "throughput_delta_percent": round(throughput_delta_pct, 2),
    }


# =================================================================
# TOOL 3: randomization_stats
# =================================================================
def randomization_stats(top_n: int = 5) -> dict:
    """Get which randomization or environment variables have a statistically significant
    impact on pass/fail outcomes, ranked by combined evidence across multiple statistical
    methods (mutual information, model importance, permutation importance, chi-square/p-value).
    Use this for questions about seeds, workload, timing, temperature, or 'what randomization
    factors matter'.

    Args:
        top_n: How many top-ranked variables to return.
    """
    top = randomization_impact.sort_values("combined_rank_score", ascending=False).head(top_n)
    return {
        "top_variables": top[["variable", "mutual_info", "model_importance", "p_value", "significant"]]
        .round(4).to_dict(orient="records")
    }


# =================================================================
# TOOL 4: best_configuration
# =================================================================
def best_configuration(objective: str) -> dict:
    """Get the single best REAL (historically observed) configuration for a given objective.
    Use this when the user asks for a recommendation.

    Args:
        objective: One of "minimize_failures", "maximize_throughput", or "balanced".
    """
    objective = objective.strip().lower().replace(" ", "_")
    eligible = config_explorer[~config_explorer["low_confidence"]]

    if "fail" in objective:
        best = eligible.sort_values("failure_rate", ascending=True).iloc[0]
    elif "throughput" in objective or "performance" in objective:
        best = eligible.sort_values("avg_throughput", ascending=False).iloc[0]
    else:
        best = eligible.sort_values("quality_score", ascending=False).iloc[0]

    ci_lower, ci_upper = wilson_confidence_interval(float(best["failure_rate"]), int(best["n"]))
    confidence = confidence_label(ci_lower, ci_upper)

    return {
        "cache_policy": best["cache_policy"],
        "scheduler": best["scheduler"],
        "feature_flag_7": bool(best["feature_flag_7"]),
        "failure_rate_pct": float(best["failure_rate"]),
        "failure_rate_95pct_confidence_interval": [ci_lower, ci_upper],
        "confidence_level": confidence,
        "throughput": float(best["avg_throughput"]),
        "sample_size": int(best["n"]),
    }


# =================================================================
# TOOL 5: compare_named_configurations
#
# GAP FIX: compare_configurations (tool 2) only handles "baseline plus
# ONE changed field" -- e.g. "what if I turn feature_flag_7 off". It
# can't answer "why does Configuration A fail more than Configuration B"
# when A and B differ in SEVERAL settings at once, because there's no
# single "change_field" to point at.
#
# This tool handles the general case: two FULLY ARBITRARY configurations,
# ranked by which of their differences actually drives the outcome gap.
#
# HOW THE RANKING WORKS:
# We use SHAP with configuration B as the reference point (the SHAP
# "background"), then ask SHAP to explain configuration A's prediction
# relative to that specific reference -- not relative to the whole
# dataset average like every other SHAP explanation in this project.
# This directly answers "why is A different from B" rather than "why is
# A different from typical", which is what ChatGPT's step 15 example
# ("Main factors: Feature X, Workload W7, scheduler") actually needs.
# =================================================================
def compare_named_configurations(
    a_cache_policy: Optional[str] = None, a_scheduler: Optional[str] = None,
    a_memory_alloc_mb: Optional[int] = None, a_thread_pool_size: Optional[int] = None,
    a_feature_flag_7: Optional[bool] = None, a_seed_group: Optional[int] = None,
    a_host_temperature_c: Optional[float] = None,
    b_cache_policy: Optional[str] = None, b_scheduler: Optional[str] = None,
    b_memory_alloc_mb: Optional[int] = None, b_thread_pool_size: Optional[int] = None,
    b_feature_flag_7: Optional[bool] = None, b_seed_group: Optional[int] = None,
    b_host_temperature_c: Optional[float] = None,
) -> dict:
    """Compare two NAMED configurations (Configuration A vs Configuration B) that may differ
    in MULTIPLE settings at once, and rank which specific differences are actually driving
    the failure rate gap between them. Use this for 'why does A fail more than B' or
    'compare these two configurations' questions where more than one setting differs.
    For a single changed setting, use compare_configurations instead.

    Args:
        a_cache_policy: Configuration A's cache policy.
        a_scheduler: Configuration A's scheduler.
        a_memory_alloc_mb: Configuration A's memory allocation in MB.
        a_thread_pool_size: Configuration A's thread pool size.
        a_feature_flag_7: Configuration A's feature_flag_7 state.
        a_seed_group: Configuration A's seed group, 0-9.
        a_host_temperature_c: Configuration A's node temperature in Celsius.
        b_cache_policy: Configuration B's cache policy.
        b_scheduler: Configuration B's scheduler.
        b_memory_alloc_mb: Configuration B's memory allocation in MB.
        b_thread_pool_size: Configuration B's thread pool size.
        b_feature_flag_7: Configuration B's feature_flag_7 state.
        b_seed_group: Configuration B's seed group, 0-9.
        b_host_temperature_c: Configuration B's node temperature in Celsius.
    """
    spec_a = {k[2:]: v for k, v in locals().items() if k.startswith("a_") and v is not None}
    spec_b = {k[2:]: v for k, v in locals().items() if k.startswith("b_") and v is not None}

    vector_a = _build_feature_vector(spec_a)
    vector_b = _build_feature_vector(spec_b)

    result_a = _predict(spec_a)
    result_b = _predict(spec_b)

    # Explain A's prediction USING B AS THE REFERENCE POINT -- this is what
    # makes the ranking about "why A differs from B" specifically.
    explainer = shap.TreeExplainer(
        failure_model, data=vector_b, feature_perturbation="interventional", model_output="probability"
    )
    shap_values = explainer.shap_values(vector_a, check_additivity=False)
    fail_contrib = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0, :, 0]

    contrib_series = pd.Series(fail_contrib, index=FEATURE_COLUMNS)
    # Only report features that actually differ between A and B -- a feature
    # can have a nonzero SHAP value from noise even when unchanged, so filter
    # to real differences to avoid reporting a spurious "main factor".
    differs_mask = (vector_a.iloc[0] != vector_b.iloc[0])
    real_differentiators = contrib_series[differs_mask].sort_values(key=abs, ascending=False).head(6)

    return {
        "configuration_a": {"spec": spec_a, **result_a},
        "configuration_b": {"spec": spec_b, **result_b},
        "failure_rate_gap_percentage_points": round(
            result_a["failure_probability_pct"] - result_b["failure_probability_pct"], 2
        ),
        "main_differentiating_factors": [
            {"feature": feat, "contribution_to_a_vs_b_gap_pct": round(val * 100, 2)}
            for feat, val in real_differentiators.items()
        ],
    }


SYSTEM_PROMPT = (
    "You are a configuration intelligence copilot for a hardware/software validation system. "
    "You answer questions using ONLY the tool results you get back -- never invent numbers. "
    "Explain results in plain, concise language an engineer would find useful, citing the "
    "specific numbers from the tool result. "
    "IMPORTANT: if a tool result includes 'out_of_distribution_warnings', you MUST mention "
    "this prominently and clearly at the start of your answer -- it means the requested "
    "configuration includes a value the model was never trained on, so the prediction may be "
    "unreliable despite looking confident. Do not bury this warning at the end or omit it."
)


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set the GEMINI_API_KEY environment variable first.")
        print('PowerShell: $env:GEMINI_API_KEY = "your-key-here"')
        return

    client = genai.Client()  # reads GEMINI_API_KEY automatically

    # Google's SDK recommends using the Chat interface (not models.generate_content
    # directly) when relying on automatic function calling. This also gives us
    # multi-turn memory for free -- follow-up questions can reference earlier ones.
    chat = client.chats.create(
        model=LLM_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[predict_execution, compare_configurations, compare_named_configurations,
                   randomization_stats, best_configuration],
        ),
    )

    print("AI Copilot ready. Ask a question (empty line to quit).\n")
    print("Example questions:")
    print("  - Why does feature_flag_7 increase failures?")
    print("  - What happens if I disable feature_flag_7 on an Adaptive+Dynamic config?")
    print("  - Why does Configuration A (Adaptive, Dynamic, flag7=True, seed_group=8) fail more "
          "than Configuration B (Static, Static, flag7=False, seed_group=1)?")
    print("  - What randomization factors matter most?")
    print("  - Recommend a configuration that minimizes failures.\n")

    while True:
        question = input("> ").strip()
        if not question:
            break

        try:
            response = chat.send_message(question)
            print(f"\n{response.text}\n")
        except Exception as e:
            error_text = str(e)
            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                print("\nRate limit reached. Gemini's free tier allows only a few requests "
                      "per minute. Wait about a minute and try again.\n")
            else:
                print(f"\nSomething went wrong calling Gemini: {error_text}\n")


if __name__ == "__main__":
    main()