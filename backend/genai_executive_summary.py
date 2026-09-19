r"""
genai_executive_summary.py

Closes the bonus feature: "GenAI Summaries -- Create executive-level
summaries from thousands of log files."

HONEST SCOPING NOTE: this doesn't re-read the raw 55,000 execution logs
directly (that would be enormously wasteful and pointless -- an LLM
reading 55,000 rows just to say what a groupby already computed exactly
and reproducibly is worse, not better). Instead, it feeds Gemini the
AGGREGATE FINDINGS from every prior analysis step -- the same structured,
validated numbers used throughout this project. This is the same
"structured result -> LLM -> human-readable answer" pattern the AI
Copilot uses (step 15), applied to internal summary generation instead
of interactive Q&A. This is both more honest (no invented numbers) and
more practical (an executive summary should be fast to generate,
not require re-processing the entire dataset every time).

Usage (from backend/):
    python genai_executive_summary.py
"""

import os
import pandas as pd
from google import genai

FEATURE_IMPORTANCE_PATH = "../outputs/feature_importance.csv"
RANDOMIZATION_IMPACT_PATH = "../outputs/randomization_impact.csv"
CONFIG_EXPLORER_PATH = "../outputs/config_explorer.csv"
PARETO_PATH = "../outputs/pareto_optimal_configs.csv"
FINGERPRINTS_PATH = "../outputs/failure_fingerprints.csv"
REPEATABILITY_PATH = "../outputs/repeatability_scores.csv"
CONTRIBUTION_PATH = "../outputs/failure_contribution_stats.csv"
LOG_COMPARISON_PATH = "../outputs/log_comparison.csv"
RAW_PATH = "../data/raw/execution_logs.csv"

OUT_PATH = "../outputs/executive_summary.md"
LLM_MODEL = "gemini-3.6-flash"


def gather_facts() -> str:
    """Pull the key numbers out of every analysis step's saved output into
    a compact fact sheet -- this is what Gemini will summarize. Every
    number here traces back to a specific script already run and verified
    earlier in this project."""
    raw_df = pd.read_csv(RAW_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    randomization_impact = pd.read_csv(RANDOMIZATION_IMPACT_PATH)
    config_explorer = pd.read_csv(CONFIG_EXPLORER_PATH)
    pareto = pd.read_csv(PARETO_PATH)
    fingerprints = pd.read_csv(FINGERPRINTS_PATH)
    repeatability = pd.read_csv(REPEATABILITY_PATH)
    contribution = pd.read_csv(CONTRIBUTION_PATH)
    log_comparison = pd.read_csv(LOG_COMPARISON_PATH)

    total_runs = len(raw_df)
    failure_rate = 1 - raw_df["passed"].mean()
    top_features = feature_importance.head(5)[["feature", "importance_pct"]].to_string(index=False)
    top_random_var = randomization_impact.sort_values("combined_rank_score", ascending=False).iloc[0]
    eligible_configs = config_explorer[~config_explorer["low_confidence"]]
    safest_config = eligible_configs.sort_values("failure_rate", ascending=True).iloc[0]
    fastest_config = eligible_configs.sort_values("avg_throughput", ascending=False).iloc[0]
    n_pareto = len(pareto)
    n_total_configs = len(config_explorer)
    top_fingerprint = fingerprints.iloc[fingerprints["failed_runs_in_cluster"].idxmax()]
    weighted_repeatability = (repeatability["repeatability_score"] * repeatability["total_n"]).sum() / repeatability["total_n"].sum()

    facts = f"""
DATASET: {total_runs:,} total execution runs, {failure_rate:.1%} overall failure rate, 158 legitimate
prediction features (configuration + randomization + environment) plus 6 log-derived signals.

TOP 5 FAILURE-DRIVING FEATURES (Random Forest + SHAP):
{top_features}

RANDOMIZATION: only 1 of 51 tested randomization variables is statistically significant --
'{top_random_var['variable']}' (p={top_random_var['p_value']:.4f}). All others, including
workload type and traffic pattern, were tested and ruled out as noise.

BEST CONFIGURATIONS FOUND (a genuine trade-off, both are Pareto-optimal -- neither one
strictly dominates the other):
  - Lowest failure risk: {safest_config['cache_policy']} cache + {safest_config['scheduler']} scheduler
    + feature_flag_7={safest_config['feature_flag_7']} -> {safest_config['failure_rate']}% failure rate,
    {safest_config['avg_throughput']} throughput ({int(safest_config['n'])} historical runs)
  - Highest throughput: {fastest_config['cache_policy']} cache + {fastest_config['scheduler']} scheduler
    + feature_flag_7={fastest_config['feature_flag_7']} -> {fastest_config['failure_rate']}% failure rate,
    {fastest_config['avg_throughput']} throughput ({int(fastest_config['n'])} historical runs)

PARETO FRONTIER: only {n_pareto} of {n_total_configs} tested configurations are actually
worth considering -- every other configuration is strictly dominated on both failure rate
and throughput simultaneously.

LARGEST FAILURE CLUSTER: {int(top_fingerprint['failed_runs_in_cluster'])} failed runs share
this fingerprint: {top_fingerprint['conditions']}. Runs matching this pattern show a
{top_fingerprint['matching_failure_rate']:.1%} failure rate across the full dataset.

REPEATABILITY: overall weighted repeatability score is {weighted_repeatability:.3f} (1.0 = fully
deterministic), but this average hides a real, large effect: seed_group 8 specifically causes
a median 8x+ failure rate spike versus other seed groups, in nearly all tested configurations.

CONTRIBUTION TO TOTAL FAILURES:
{contribution.to_string(index=False)}

LOG EVIDENCE (failed vs passed runs):
{log_comparison.to_string(index=False)}
"""
    return facts.strip()


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set the GEMINI_API_KEY environment variable first.")
        print('PowerShell: $env:GEMINI_API_KEY = "your-key-here"')
        return

    facts = gather_facts()
    print("--- Facts gathered from all analysis steps ---")
    print(facts)
    print("\n--- Generating executive summary with Gemini ---\n")

    client = genai.Client()
    prompt = (
        "You are writing an executive summary for a hardware/software validation team's "
        "leadership, based on an AI-powered analysis of execution logs. Use ONLY the facts "
        "below -- do not invent numbers. Write 3-4 short paragraphs: (1) the headline finding, "
        "(2) the root cause story, (3) the actionable recommendation -- note there is a genuine "
        "trade-off between the two best configurations found, don't present just one as 'the' "
        "answer, (4) one sentence on confidence/limitations. Keep it concise and non-technical "
        "enough for a director-level audience, but precise with numbers.\n\n"
        f"FACTS:\n{facts}"
    )

    chat = client.chats.create(model=LLM_MODEL)
    response = chat.send_message(prompt)
    summary = response.text

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# Executive Summary\n\n")
        f.write(summary)

    print(summary)
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()