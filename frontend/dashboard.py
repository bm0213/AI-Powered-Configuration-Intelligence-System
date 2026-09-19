"""
dashboard.py -- Configuration Intelligence Dashboard (final version)

Wires together every analytics step, every brief-alignment gap fix, and
a REAL Gemini-powered AI Copilot with live tool-calling (not a dropdown
of pre-written answers).

Run with:
    streamlit run dashboard.py
Then open the local URL it prints (usually http://localhost:8501).

Requires GEMINI_API_KEY set as an environment variable for the
AI Copilot and Executive Summary pages -- every other page works without it.
"""

import os
import sys
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root if present -- lets GEMINI_API_KEY
               # work without manually setting it in every new terminal session
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # ensures relative paths (../data, ../outputs,
                                                         # ../backend) resolve correctly regardless of
                                                         # where streamlit is launched from -- fixes
                                                         # Streamlit Community Cloud, which always runs
                                                         # from the repo root, not this file's folder
# backend/ is a sibling directory, not a real Python package -- this makes
# model_utils.py importable from here without restructuring the whole
# project into a formal installable package (which would be overkill at
# this scale). See backend/model_utils.py for why this module exists.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
# The sibling backend directory is added above at runtime; the inline directive
# also prevents static analyzers from flagging this intentional local import.
from model_utils import (  # pyright: ignore[reportMissingImports]
    compute_defaults as _compute_defaults,
    compute_training_ranges,
    build_feature_vector as _build_feature_vector,
    predict as _predict_impl,
    wilson_confidence_interval as wilson_ci,
    confidence_label,
)

st.set_page_config(page_title="Configuration Intelligence Dashboard", layout="wide")

# ---------------------------------------------------------------
# Data loading (cached so the app doesn't reload files on every click)
# ---------------------------------------------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    return joblib.load("../backend/models/rf_model.joblib")

@st.cache_resource
def load_throughput_model():
    return joblib.load("../backend/models/throughput_model.joblib")

@st.cache_data
def load_features():
    return pd.read_csv("../data/processed/features.csv")

@st.cache_data
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def safe_load_csv(path, label):
    try:
        return load_csv(path)
    except FileNotFoundError:
        st.warning(f"'{path}' not found. Run the script that generates {label} first.")
        return None

raw_df = load_csv("../data/raw/execution_logs.csv")
feature_importance = load_csv("../outputs/feature_importance.csv")
randomization_impact = load_csv("../outputs/randomization_impact.csv")
config_explorer = load_csv("../outputs/config_explorer.csv")
fingerprints = load_csv("../outputs/failure_fingerprints.csv")
repeatability = load_csv("../outputs/repeatability_scores.csv")
model = load_model()
throughput_model = load_throughput_model()
X_full = load_features()

# Newer, gap-fix outputs -- loaded defensively in case a script hasn't been run yet
correlation_matrix_df = safe_load_csv("../outputs/correlation_matrix.csv", "correlation_matrix.py")
trend_df = safe_load_csv("../outputs/trend_analysis.csv", "trend_analysis.py")
signatures_df = safe_load_csv("../outputs/failure_signatures.csv", "failure_signatures.py")
log_comparison_df = safe_load_csv("../outputs/log_comparison.csv", "failure_signatures.py")
contribution_df = safe_load_csv("../outputs/failure_contribution_stats.csv", "failure_signatures.py")
early_warning_df = safe_load_csv("../outputs/early_warning_indicators.csv", "early_warning_indicators.py")
pareto_df = safe_load_csv("../outputs/pareto_optimal_configs.csv", "pareto_frontier.py")

TOTAL_RUNS = len(raw_df)
FAIL_RATE = 1 - raw_df["passed"].mean()
FEATURE_COLUMNS = X_full.columns.tolist()

# ---------------------------------------------------------------
# Shared helpers -- delegating to backend/model_utils.py instead of
# maintaining a separate local copy of this logic. See that file's
# docstring for why this consolidation happened.
#
# PERFORMANCE NOTE: DEFAULTS and TRAINING_RANGES used to be computed with
# st.cache_data(_compute_defaults)(X_full) -- passing the full 55,000-row
# dataframe as an ARGUMENT to a cached call. Streamlit's cache has to hash
# every argument to check for a cache hit, and it does this on EVERY
# script rerun (i.e. every single click, since Streamlit reruns the whole
# script top-to-bottom on each interaction). Hashing a 55k-row dataframe
# on every click was a real, measurable source of the "dashboard feels
# slow" complaint. Fixed by wrapping this in a cached function that takes
# NO arguments -- Streamlit only needs to check "has this function been
# called before", which is nearly instant, instead of re-hashing the data
# every time.
# ---------------------------------------------------------------
@st.cache_data
def _load_defaults_and_ranges():
    return _compute_defaults(X_full), compute_training_ranges(X_full)

DEFAULTS, TRAINING_RANGES = _load_defaults_and_ranges()

# PERFORMANCE NOTE: the Predict New Execution page used to rebuild a fresh
# background sample AND a fresh shap.TreeExplainer on EVERY click of the
# Predict button -- even though the background sample is deterministic
# (fixed random_state) and never actually changes. TreeExplainer
# construction with interventional feature_perturbation has real
# overhead (it processes the background data against the full tree
# structure), so rebuilding it repeatedly was pure waste. st.cache_resource
# is the right tool here (not st.cache_data) because we're caching a live
# object, not serializable data.
@st.cache_resource
def get_shap_explainer():
    background = X_full.sample(n=100, random_state=42)
    return shap.TreeExplainer(model, data=background, feature_perturbation="interventional", model_output="probability")

def build_feature_vector(spec: dict) -> pd.DataFrame:
    return _build_feature_vector(spec, DEFAULTS, FEATURE_COLUMNS)

def predict(spec: dict) -> dict:
    return _predict_impl(spec, model, throughput_model, DEFAULTS, FEATURE_COLUMNS, TRAINING_RANGES)

# ---------------------------------------------------------------
# AI Copilot tool functions (mirrors backend/ai_copilot.py, adapted to
# reuse objects already loaded above instead of re-loading from disk --
# avoids the backend/ vs frontend/ relative-path mismatch)
# ---------------------------------------------------------------
def tool_predict_execution(cache_policy: Optional[str] = None, scheduler: Optional[str] = None,
                            memory_alloc_mb: Optional[int] = None, thread_pool_size: Optional[int] = None,
                            feature_flag_7: Optional[bool] = None, seed_group: Optional[int] = None,
                            host_temperature_c: Optional[float] = None):
    """Predict failure probability and throughput for ONE hypothetical execution configuration.
    Use this when the user asks about a single specific configuration.

    Args:
        cache_policy: One of Adaptive, Static, LRU, FIFO.
        scheduler: One of Dynamic, Static, RoundRobin.
        memory_alloc_mb: Memory allocation in MB.
        thread_pool_size: Thread pool size.
        feature_flag_7: Whether feature_flag_7 is enabled.
        seed_group: Seed group bucket, 0-9.
        host_temperature_c: Node temperature in Celsius.
    """
    spec = {k: v for k, v in locals().items() if v is not None}
    result = predict(spec)
    result["config_used"] = spec
    return result

def tool_compare_configurations(change_field: str, change_value: str, cache_policy: Optional[str] = None,
                                 scheduler: Optional[str] = None, memory_alloc_mb: Optional[int] = None,
                                 thread_pool_size: Optional[int] = None, feature_flag_7: Optional[bool] = None,
                                 seed_group: Optional[int] = None, host_temperature_c: Optional[float] = None):
    """Compare a baseline configuration against the same configuration with ONE setting changed.

    Args:
        change_field: Name of the setting being changed.
        change_value: The new value, as a string.
        cache_policy: Baseline cache policy.
        scheduler: Baseline scheduler.
        memory_alloc_mb: Baseline memory allocation in MB.
        thread_pool_size: Baseline thread pool size.
        feature_flag_7: Baseline feature_flag_7 state.
        seed_group: Baseline seed group.
        host_temperature_c: Baseline node temperature.
    """
    baseline_spec = {k: v for k, v in locals().items()
                      if v is not None and k not in ("change_field", "change_value")}
    if change_field == "feature_flag_7":
        casted = change_value.strip().lower() in ("true", "1", "yes", "on")
    elif change_field in ("memory_alloc_mb", "thread_pool_size", "seed_group"):
        casted = int(float(change_value))
    elif change_field == "host_temperature_c":
        casted = float(change_value)
    else:
        casted = change_value

    before = predict(baseline_spec)
    after = predict({**baseline_spec, change_field: casted})
    return {
        "before": before, "after": after,
        "change_applied": f"{change_field} -> {casted}",
        "failure_delta_percentage_points": round(after["failure_probability_pct"] - before["failure_probability_pct"], 2),
        "throughput_delta_percent": round((after["predicted_throughput"] - before["predicted_throughput"]) / before["predicted_throughput"] * 100, 2),
    }

def tool_randomization_stats(top_n: int = 5):
    """Get which randomization/environment variables have a statistically significant impact
    on pass/fail, ranked by combined evidence across multiple statistical methods.

    Args:
        top_n: How many top-ranked variables to return.
    """
    top = randomization_impact.sort_values("combined_rank_score", ascending=False).head(top_n)
    return {"top_variables": top[["variable", "mutual_info", "model_importance", "p_value", "significant"]]
            .round(4).to_dict(orient="records")}

def tool_best_configuration(objective: str):
    """Get the single best real (historically observed) configuration for a given objective.

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
    ci_lower, ci_upper = wilson_ci(float(best["failure_rate"]), int(best["n"]))
    return {
        "cache_policy": best["cache_policy"], "scheduler": best["scheduler"],
        "feature_flag_7": bool(best["feature_flag_7"]), "failure_rate_pct": float(best["failure_rate"]),
        "failure_rate_95pct_confidence_interval": [ci_lower, ci_upper],
        "confidence_level": confidence_label(ci_lower, ci_upper),
        "throughput": float(best["avg_throughput"]), "sample_size": int(best["n"]),
    }

def tool_compare_named_configurations(
    a_cache_policy: Optional[str] = None, a_scheduler: Optional[str] = None,
    a_memory_alloc_mb: Optional[int] = None, a_thread_pool_size: Optional[int] = None,
    a_feature_flag_7: Optional[bool] = None, a_seed_group: Optional[int] = None,
    a_host_temperature_c: Optional[float] = None,
    b_cache_policy: Optional[str] = None, b_scheduler: Optional[str] = None,
    b_memory_alloc_mb: Optional[int] = None, b_thread_pool_size: Optional[int] = None,
    b_feature_flag_7: Optional[bool] = None, b_seed_group: Optional[int] = None,
    b_host_temperature_c: Optional[float] = None,
):
    """Compare two named configurations (A vs B) that may differ in MULTIPLE settings at once,
    ranked by which differences actually drive the failure rate gap.

    Args:
        a_cache_policy: Configuration A's cache policy.
        a_scheduler: Configuration A's scheduler.
        a_memory_alloc_mb: Configuration A's memory allocation.
        a_thread_pool_size: Configuration A's thread pool size.
        a_feature_flag_7: Configuration A's feature_flag_7 state.
        a_seed_group: Configuration A's seed group.
        a_host_temperature_c: Configuration A's temperature.
        b_cache_policy: Configuration B's cache policy.
        b_scheduler: Configuration B's scheduler.
        b_memory_alloc_mb: Configuration B's memory allocation.
        b_thread_pool_size: Configuration B's thread pool size.
        b_feature_flag_7: Configuration B's feature_flag_7 state.
        b_seed_group: Configuration B's seed group.
        b_host_temperature_c: Configuration B's temperature.
    """
    spec_a = {k[2:]: v for k, v in locals().items() if k.startswith("a_") and v is not None}
    spec_b = {k[2:]: v for k, v in locals().items() if k.startswith("b_") and v is not None}
    vector_a, vector_b = build_feature_vector(spec_a), build_feature_vector(spec_b)
    result_a, result_b = predict(spec_a), predict(spec_b)

    explainer = shap.TreeExplainer(model, data=vector_b, feature_perturbation="interventional", model_output="probability")
    shap_values = explainer.shap_values(vector_a, check_additivity=False)
    fail_contrib = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0, :, 0]
    contrib_series = pd.Series(fail_contrib, index=FEATURE_COLUMNS)
    differs_mask = (vector_a.iloc[0] != vector_b.iloc[0])
    top_diff = contrib_series[differs_mask].sort_values(key=abs, ascending=False).head(6)

    return {
        "configuration_a": {"spec": spec_a, **result_a},
        "configuration_b": {"spec": spec_b, **result_b},
        "failure_rate_gap_percentage_points": round(result_a["failure_probability_pct"] - result_b["failure_probability_pct"], 2),
        "main_differentiating_factors": [
            {"feature": f, "contribution_pct": round(v * 100, 2)} for f, v in top_diff.items()
        ],
    }

COPILOT_TOOLS = [tool_predict_execution, tool_compare_configurations, tool_compare_named_configurations,
                 tool_randomization_stats, tool_best_configuration]

COPILOT_SYSTEM_PROMPT = (
    "You are a configuration intelligence copilot for a hardware/software validation system. "
    "You answer questions using ONLY the tool results you get back -- never invent numbers. "
    "Explain results in plain, concise language an engineer would find useful, citing specific "
    "numbers. IMPORTANT: if a tool result includes 'ood_warnings', you MUST mention this "
    "prominently as a low-confidence/extrapolation warning before the rest of your answer."
)

# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("Configuration Intelligence")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Configuration Explorer (Q2)",
        "Pareto Frontier (Q2)",
        "Feature Importance (Q1)",
        "Randomization + Correlation (Q3)",
        "Failure Signatures (Q5)",
        "Log Comparison (Q6)",
        "Repeatability (Q4)",
        "Trend Analysis",
        "Early Warning Indicators (Q7)",
        "Predict New Execution (Q7)",
        "What-If Analysis (Q6)",
        "Recommendation Engine (Q8)",
        "Executive Summary",
        "AI Copilot",
    ],
)

# =================================================================
# PAGE: Overview
# =================================================================
if page == "Overview":
    st.title("AI-Powered Configuration Intelligence")
    st.caption("SanDisk Hackathon -- Configuration Intelligence Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Executions", f"{TOTAL_RUNS:,}")
    col2.metric("Overall Failure Rate", f"{FAIL_RATE:.1%}")
    col3.metric("Prediction Features", "158")

    st.divider()
    st.subheader("Key findings at a glance")
    st.markdown("""
    - **Root cause identified:** `seed_group=8` runs land on hotter, higher-voltage nodes,
      causing a **2.8x higher failure rate** than baseline -- confirmed across 4 independent
      statistical methods AND a correlation matrix check.
    - **Not random noise:** in 99% of tested configurations, seed_group 8 causes a median
      **8.1x failure rate spike**, regardless of configuration.
    - **Real trade-off found:** `feature_flag_7` raises throughput ~12% but roughly doubles
      the failure rate -- confirmed as a genuine Pareto-optimal trade-off (only 2 of 24
      configurations are worth considering).
    - **Log evidence confirms root causes:** failed runs show 9x more errors, 10x more
      scheduler errors, 8x more memory errors than passed runs.
    - Every finding is backed by a saved, reproducible analysis script and cross-checked
      against the original SanDisk brief question by question.
    """)

    st.subheader("Pass rate by seed group")
    st.bar_chart(raw_df.groupby("seed_group")["passed"].mean())

# =================================================================
# PAGE: Configuration Explorer
# =================================================================
elif page == "Configuration Explorer (Q2)":
    st.title("Q2: Which configuration combinations consistently yield the best results?")
    sort_col = st.selectbox("Sort by", ["quality_score", "failure_rate", "avg_throughput", "avg_reliability_score"])
    ascending = sort_col == "failure_rate"
    st.dataframe(config_explorer.sort_values(sort_col, ascending=ascending), width='stretch')
    st.caption("Quality score = 0.4 x normalized throughput + 0.4 x normalized reliability - 0.2 x failure rate. "
               "Rows flagged low_confidence=True have fewer than 30 historical runs.")

# =================================================================
# PAGE: Pareto Frontier
# =================================================================
elif page == "Pareto Frontier (Q2)":
    st.title("Q2: Pareto-Optimal Configurations")
    st.caption("Configurations where you cannot improve throughput without increasing failure rate, or vice versa.")
    st.image("../outputs/pareto_frontier.png", width='stretch')
    if pareto_df is not None:
        eligible = config_explorer[~config_explorer["low_confidence"]]
        st.markdown(f"**{len(pareto_df)} of {len(eligible)} configurations are Pareto-optimal.** "
                    f"Every other configuration is strictly dominated and should never be chosen.")
        st.dataframe(pareto_df, width='stretch')

# =================================================================
# PAGE: Feature Importance
# =================================================================
elif page == "Feature Importance (Q1)":
    st.title("Q1: What configuration settings most strongly influence success or failure?")
    top15 = feature_importance.head(15)
    st.bar_chart(top15.set_index("feature")["importance_pct"])
    st.dataframe(top15[["feature", "importance_pct"]], width='stretch')
    st.image("../outputs/shap_summary.png", width='stretch', caption="SHAP summary: direction and magnitude of each feature's effect")
    st.caption("Model: Random Forest (200 trees), trained on 158 configuration/randomization/environment features.")

# =================================================================
# PAGE: Randomization + Correlation
# =================================================================
elif page == "Randomization + Correlation (Q3)":
    st.title("Q3: Which randomization parameters have the highest impact on outcomes?")
    top15_rand = randomization_impact.sort_values("combined_rank_score", ascending=False).head(15)
    st.bar_chart(top15_rand.set_index("variable")["combined_rank_score"])
    st.dataframe(top15_rand, width='stretch')
    st.info("Only `seed_group` shows a statistically significant relationship with outcome (p<0.0001). "
            "All other randomization variables were tested and ruled out across 4 independent methods.")

    st.subheader("Correlation Matrix")
    if correlation_matrix_df is not None:
        st.image("../outputs/correlation_matrix.png", width='stretch')
        st.caption("Even among the top 15 ranked variables, only seed_group shows meaningful correlation "
                   "with outcome -- everything else is near-zero, confirming it's noise, not a hidden signal.")

# =================================================================
# PAGE: Failure Signatures
# =================================================================
elif page == "Failure Signatures (Q5)":
    st.title("Q5: What factors contributed to observed failures?")
    st.caption("Failure fingerprints (configuration conditions) enriched with log evidence.")

    for _, row in fingerprints.iterrows():
        if row["conditions"] == "none distinguishing":
            continue
        with st.container(border=True):
            st.markdown(f"**Cluster {int(row['cluster'])}** -- {int(row['failed_runs_in_cluster'])} failed runs")
            st.markdown(f"Configuration conditions: {row['conditions']}")
            if pd.notna(row["matching_runs_total"]):
                st.markdown(f"**Matching the full dataset:** {int(row['matching_runs_total'])} runs, "
                            f"**{row['matching_failure_rate']:.1%} failure rate** (vs {FAIL_RATE:.1%} baseline)")

            if signatures_df is not None:
                sig_row = signatures_df[signatures_df["cluster"] == row["cluster"]]
                if len(sig_row):
                    sig_row = sig_row.iloc[0]
                    log_cols = [c for c in signatures_df.columns if c.endswith("_avg")]
                    st.markdown("**Log signature (vs dataset average):**")
                    log_summary = ", ".join(
                        f"{c.replace('_avg','')}: {sig_row[c]:.2f} ({sig_row[c.replace('_avg','_vs_baseline_multiplier')]:.1f}x)"
                        for c in log_cols
                    )
                    st.caption(log_summary)

    if contribution_df is not None:
        st.subheader("Contribution to total failures")
        st.dataframe(contribution_df, width='stretch')

# =================================================================
# PAGE: Log Comparison
# =================================================================
elif page == "Log Comparison (Q6)":
    st.title("Q6: What changed between successful and failed executions?")
    st.caption("Log-derived signals compared between passed and failed runs.")
    if log_comparison_df is not None:
        st.dataframe(log_comparison_df, width='stretch')
        chart_df = log_comparison_df.set_index("log_feature")[["pass_value", "fail_value"]]
        st.bar_chart(chart_df)
        st.info("Failed runs consistently show far more errors, warnings, timeouts, and retries -- "
                "strong, independent evidence backing the configuration-based root causes found elsewhere.")

# =================================================================
# PAGE: Repeatability
# =================================================================
elif page == "Repeatability (Q4)":
    st.title("Q4: Are failures deterministic or random?")
    weighted_avg = np.average(repeatability["repeatability_score"], weights=repeatability["total_n"])
    pct_seed8_worst = (repeatability["worst_seed_group"] == 8).mean()
    col1, col2 = st.columns(2)
    col1.metric("Overall Repeatability Score", f"{weighted_avg:.3f}")
    col2.metric("Configs where seed_group 8 is worst", f"{pct_seed8_worst:.0%}")
    st.warning("The aggregate score alone is misleading. Outcomes ARE deterministic once you know "
               "the seed group -- but NOT deterministic from configuration alone. seed_group 8 "
               "(traced to node temperature) overrides configuration choice in nearly every case.")
    st.dataframe(repeatability, width='stretch')

# =================================================================
# PAGE: Trend Analysis
# =================================================================
elif page == "Trend Analysis":
    st.title("Trend Analysis")
    st.caption("Failure rate over run sequence (proxy for time, since this dataset has no real timestamp).")
    if trend_df is not None:
        st.line_chart(trend_df.set_index("bucket")["failure_rate"])
        st.image("../outputs/trend_analysis.png", width='stretch')
        st.info("No significant trend detected (p>0.05) -- expected, since this dataset was generated "
                "without time-based drift. This confirms the analysis is working correctly.")

# =================================================================
# PAGE: Early Warning Indicators
# =================================================================
elif page == "Early Warning Indicators (Q7)":
    st.title("Q7: Early Warning Indicators")
    st.caption("Conditions ranked by how much they multiply baseline failure risk.")
    if early_warning_df is not None:
        for _, row in early_warning_df.sort_values("risk_ratio", ascending=False).iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['indicator']}**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Risk Ratio", f"{row['risk_ratio']}x")
                col2.metric("Failure Rate When Present", f"{row['failure_rate_when_present_pct']}%")
                col3.metric("% of All Failures Explained", f"{row['pct_of_all_failures_explained']}%")

# =================================================================
# PAGE: Predict New Execution
# =================================================================
elif page == "Predict New Execution (Q7)":
    st.title("Q7: Predict failure probability for a new execution")
    col1, col2, col3 = st.columns(3)
    with col1:
        cache_policy = st.selectbox("Cache Policy", ["Adaptive", "Static", "LRU", "FIFO"])
        scheduler = st.selectbox("Scheduler", ["Dynamic", "Static", "RoundRobin"])
    with col2:
        memory_alloc_mb = st.select_slider("Memory Allocation (MB)", [512, 1024, 2048, 4096, 8192], value=4096)
        thread_pool_size = st.select_slider("Thread Pool Size", [2, 4, 8, 16, 32], value=8)
    with col3:
        feature_flag_7 = st.checkbox("Feature Flag 7 enabled", value=False)
        seed_group = st.slider("Seed Group", 0, 9, 2)
        host_temperature_c = st.slider("Host Temperature (C)", 20, 130, 45)

    if st.button("Predict", type="primary"):
        spec = {"cache_policy": cache_policy, "scheduler": scheduler, "memory_alloc_mb": memory_alloc_mb,
                "thread_pool_size": thread_pool_size, "feature_flag_7": feature_flag_7,
                "seed_group": seed_group, "host_temperature_c": host_temperature_c}
        result = predict(spec)

        if result.get("ood_warnings"):
            for w in result["ood_warnings"]:
                st.error(f"OUT-OF-DISTRIBUTION WARNING: {w}")

        st.metric("Predicted Failure Probability", f"{result['failure_probability_pct']}%",
                   delta=f"+/- {result['confidence_std_pct']}%")

        vector = build_feature_vector(spec)
        explainer = get_shap_explainer()
        shap_values = explainer.shap_values(vector, check_additivity=False)
        fail_contrib = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0, :, 0]
        contrib_series = pd.Series(fail_contrib, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=False).head(6)
        st.subheader("Top contributors")
        st.bar_chart(contrib_series)

# =================================================================
# PAGE: What-If Analysis
# =================================================================
elif page == "What-If Analysis (Q6)":
    st.title("What-If Analysis")
    st.caption("Compare a baseline configuration against the same configuration with one setting changed.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline")
        b_cache_policy = st.selectbox("Cache Policy", ["Adaptive", "Static", "LRU", "FIFO"], key="wcache")
        b_scheduler = st.selectbox("Scheduler", ["Dynamic", "Static", "RoundRobin"], key="wsched")
        b_feature_flag_7 = st.checkbox("Feature Flag 7 enabled", value=True, key="wflag")
        b_seed_group = st.slider("Seed Group", 0, 9, 2, key="wseed")
    with col2:
        st.subheader("Change")
        change_field = st.selectbox("Setting to change", ["feature_flag_7", "cache_policy", "scheduler", "seed_group"])
        change_value = st.text_input("New value", value="False")

    if st.button("Compare", type="primary"):
        baseline_spec = {"cache_policy": b_cache_policy, "scheduler": b_scheduler,
                          "feature_flag_7": b_feature_flag_7, "seed_group": b_seed_group}
        if change_field == "feature_flag_7":
            casted = change_value.strip().lower() in ("true", "1", "yes", "on")
        elif change_field == "seed_group":
            casted = int(change_value)
        else:
            casted = change_value

        before = predict(baseline_spec)
        after = predict({**baseline_spec, change_field: casted})
        fail_delta = after["failure_probability_pct"] - before["failure_probability_pct"]
        throughput_delta = (after["predicted_throughput"] - before["predicted_throughput"]) / before["predicted_throughput"] * 100

        col1, col2 = st.columns(2)
        col1.metric("Failure Probability (Before)", f"{before['failure_probability_pct']}%")
        col2.metric("Failure Probability (After)", f"{after['failure_probability_pct']}%", delta=f"{fail_delta:.1f} pp", delta_color="inverse")
        col1.metric("Throughput (Before)", f"{before['predicted_throughput']}")
        col2.metric("Throughput (After)", f"{after['predicted_throughput']}", delta=f"{throughput_delta:.1f}%")

        direction = "reduce" if fail_delta < 0 else "increase"
        tdirection = "reduction" if throughput_delta < 0 else "increase"
        st.success(f"Changing {change_field} to {casted} is predicted to {direction} failure probability by "
                   f"{abs(fail_delta):.1f} percentage points, with an estimated {abs(throughput_delta):.1f}% throughput {tdirection}.")

# =================================================================
# PAGE: Recommendation Engine
# =================================================================
elif page == "Recommendation Engine (Q8)":
    st.title("Q8: What configuration should be used for the next run?")
    objective = st.radio("Choose your objective",
                          ["Minimize failures", "Maximize throughput", "Balance performance + reliability"],
                          horizontal=True)
    objective_key = {"Minimize failures": "minimize_failures", "Maximize throughput": "maximize_throughput",
                      "Balance performance + reliability": "balanced"}[objective]
    best = tool_best_configuration(objective_key)

    st.success(f"Recommended: cache_policy={best['cache_policy']} + scheduler={best['scheduler']} "
               f"+ feature_flag_7={best['feature_flag_7']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Failure Rate", f"{best['failure_rate_pct']}%")
    col2.metric("95% CI", f"{best['failure_rate_95pct_confidence_interval'][0]}-{best['failure_rate_95pct_confidence_interval'][1]}%")
    col3.metric("Confidence", best["confidence_level"])
    col4.metric("Throughput", f"{best['throughput']}")
    st.caption(f"Based on {best['sample_size']} historical runs matching this configuration.")

# =================================================================
# PAGE: Executive Summary
# =================================================================
elif page == "Executive Summary":
    st.title("Executive Summary")
    st.caption("GenAI-generated summary of all analysis findings.")
    try:
        summary_text = load_text("../outputs/executive_summary.md")
        st.markdown(summary_text)
        st.caption("Generated by Gemini from structured analysis results -- see genai_executive_summary.py "
                   "to regenerate with the latest data.")
    except FileNotFoundError:
        st.warning("No summary found yet. Run `python genai_executive_summary.py` from the backend folder first.")

# =================================================================
# PAGE: AI Copilot (REAL -- Gemini with live tool-calling)
# =================================================================
elif page == "AI Copilot":
    st.title("AI Copilot")
    st.caption("Ask a question in plain English. Gemini decides which analysis tool to call, "
               "runs it against the real trained models, and explains the result.")

    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set. Set it as an environment variable and restart the dashboard.")
    else:
        from google import genai
        from google.genai import types

        if "gemini_chat" not in st.session_state:
            client = genai.Client()
            st.session_state.gemini_chat = client.chats.create(
                model="gemini-3.6-flash",
                config=types.GenerateContentConfig(
                    system_instruction=COPILOT_SYSTEM_PROMPT,
                    tools=COPILOT_TOOLS,
                ),
            )
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.caption("Try: \"Why does feature_flag_7 increase failures?\" or "
                   "\"Recommend a configuration that minimizes failures.\"")

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.gemini_chat.send_message(prompt)
                        answer = response.text
                    except Exception as e:
                        error_text = str(e)
                        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                            answer = (
                                "**Rate limit reached.** Gemini's free tier allows only a "
                                "few requests per minute. Please wait about a minute and "
                                "try again -- this isn't a bug, just a quota pause."
                            )
                        else:
                            answer = f"**Something went wrong calling Gemini:** {error_text}"
                        st.error(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.stop()
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})