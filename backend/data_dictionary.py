r"""
data_dictionary.py

STEP 1 of the pipeline: understand what you actually have before
touching any ML. Reads your execution logs CSV and produces a
column-by-column inventory: data type, missing values, unique value
count, sample values, and a best-guess category (Configuration,
Randomization, Environment, Performance, Outcome/Target, ID, or
Unknown) based on the column name.

The auto-tagging is a heuristic, not ground truth -- it will get most
columns right and leave some as "Unknown" or a guess you disagree
with. That's expected with 158 columns. Open the output CSV, skim it,
and manually fix any category that's wrong. THAT corrected file
becomes your real data dictionary and the reference every later
script uses.

Usage:
    python data_dictionary.py execution_logs.csv
"""

import sys
import pandas as pd

# ---------------------------------------------------------------
# Keyword hints used to guess a column's category from its name.
# Extend these lists as you look at your real column names --
# this is meant to get you 70-80% of the way there automatically.
# ---------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "ID": ["run_id", "execution_id", "test_id", "uuid"],
    "Target": ["result", "status", "outcome", "pass", "fail", "label"],
    # Log/Trace is checked BEFORE Configuration on purpose: columns like
    # "timeout_occurred" or "scheduler_error_occurred" contain a config-like
    # word (timeout, scheduler) but are actually log-derived outcomes.
    # Checking "occurred"/"error" first means they get classified correctly.
    "Log/Trace": ["log", "trace", "error_code", "exception", "warning", "message", "error", "occurred"],
    "Randomization": [
        "seed", "rand", "jitter", "workload", "traffic", "shuffle",
        "sample_rate", "timing_variation", "perturbation",
    ],
    "Environment": [
        "temperature", "temp", "humidity", "voltage", "env_", "host", "node",
        "region", "hardware", "cpu_load", "memory_pressure",
    ],
    "Performance": [
        "time_ms", "latency", "throughput", "duration", "execution_time",
        "utilization", "cpu_usage", "memory_usage", "iops", "bandwidth",
        "accuracy", "resource_consumption", "reliability_score",
    ],
    "Configuration": [
        "cache", "scheduler", "policy", "compiler", "flag", "thread",
        "batch", "compression", "encryption", "timeout", "retry",
        "mode", "config", "setting", "alloc", "cores", "pool", "gpu",
    ],
}


def guess_category(col_name: str) -> str:
    name = col_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return category
    return "Unknown"


def build_data_dictionary(csv_path: str) -> pd.DataFrame:
    # nrows=None loads everything; if the file is huge and this is slow,
    # you can test faster first with nrows=5000 while iterating on categories
    df = pd.read_csv(csv_path)

    rows = []
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        n_missing = series.isna().sum()
        pct_missing = round(100 * n_missing / len(df), 2)

        # crude dtype bucket: numeric vs categorical vs boolean
        if pd.api.types.is_bool_dtype(series):
            dtype_bucket = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            dtype_bucket = "numerical"
        else:
            dtype_bucket = "categorical"

        sample_values = series.dropna().unique()[:5]
        sample_str = ", ".join(str(v) for v in sample_values)

        rows.append({
            "column": col,
            "dtype": dtype_bucket,
            "unique_values": n_unique,
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "sample_values": sample_str,
            "guessed_category": guess_category(col),
        })

    inventory = pd.DataFrame(rows)
    return inventory, df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_dictionary.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    inventory, df = build_data_dictionary(csv_path)

    out_path = "../data/processed/data_dictionary.csv"
    inventory.to_csv(out_path, index=False)

    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {csv_path}")
    print(f"Data dictionary written to {out_path}\n")

    print("Category breakdown (auto-guessed -- verify manually):")
    print(inventory["guessed_category"].value_counts().to_string())

    n_unknown = (inventory["guessed_category"] == "Unknown").sum()
    if n_unknown:
        print(f"\n{n_unknown} column(s) tagged 'Unknown' -- open data_dictionary.csv and "
              f"assign these by hand. These are the ones the keyword list didn't recognize.")

    n_high_missing = (inventory["missing_pct"] > 20).sum()
    if n_high_missing:
        print(f"\n{n_high_missing} column(s) have >20% missing values -- flag these for "
              f"the cleaning step (step 3), you'll need to decide impute vs drop per column.")