r"""
preprocessing.py

STEPS 2-3 of the pipeline: clean the data and prepare it for modeling.

WHAT THIS SCRIPT DOES:
1. Loads execution_logs.csv and the data_dictionary.csv from step 1
2. Checks for missing values and duplicate rows
3. Splits the 158 columns into three groups:
      - FEATURES:  Configuration + Randomization + Environment columns
                    (things you know BEFORE a run happens)
      - TARGET:    the 'passed' column (what we want to predict)
      - OUTCOMES:  Performance columns like throughput, execution_time_ms
                    (things you only know AFTER a run -- kept aside for
                    step 5 "config intelligence", NOT used to predict pass/fail)
4. Encodes categorical/boolean features into numbers a model can use
5. Saves three clean files to data/processed/

WHY SEPARATE FEATURES FROM OUTCOMES?
If you let a failure-prediction model see 'throughput' or 'execution_time_ms',
it would partly be cheating -- those numbers are only known AFTER the run
finishes, same time you already know if it passed or failed. A model that
uses them looks great in testing and is useless for real prediction.

Usage (from backend/):
    python preprocessing.py
"""

import pandas as pd
import numpy as np

RAW_PATH = "../data/raw/execution_logs.csv"
DICT_PATH = "../data/processed/data_dictionary.csv"

OUT_FEATURES = "../data/processed/features.csv"
OUT_TARGET = "../data/processed/target.csv"
OUT_OUTCOMES = "../data/processed/outcomes.csv"
OUT_LOG_FEATURES = "../data/processed/log_features.csv"
OUT_CLEANED_FULL = "../data/processed/cleaned_data.csv"

# Columns that are technically "Randomization" or informational but should
# NOT be used as model features -- too high-cardinality (would just be
# memorized) or redundant with a better version of the same signal.
EXCLUDE_FROM_FEATURES = ["random_seed"]

ID_COLUMN = "run_id"
TARGET_COLUMN = "passed"


def load_data():
    df = pd.read_csv(RAW_PATH)
    data_dict = pd.read_csv(DICT_PATH)
    return df, data_dict


def check_data_quality(df: pd.DataFrame):
    print("--- Data quality check ---")
    n_dupes = df.duplicated().sum()
    print(f"Duplicate rows: {n_dupes}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(f"Columns with missing values ({len(missing)}):")
        print(missing.to_string())
    print()
    return n_dupes, missing


def split_columns(data_dict: pd.DataFrame):
    """Use the data dictionary from step 1 to decide which columns go where."""
    config_cols = data_dict.loc[data_dict.guessed_category == "Configuration", "column"].tolist()
    rand_cols = data_dict.loc[data_dict.guessed_category == "Randomization", "column"].tolist()
    env_cols = data_dict.loc[data_dict.guessed_category == "Environment", "column"].tolist()
    perf_cols = data_dict.loc[data_dict.guessed_category == "Performance", "column"].tolist()
    log_cols = data_dict.loc[data_dict.guessed_category == "Log/Trace", "column"].tolist()

    feature_cols = [c for c in (config_cols + rand_cols + env_cols) if c not in EXCLUDE_FROM_FEATURES]

    print("--- Column split (from data dictionary) ---")
    print(f"Feature columns (Configuration + Randomization + Environment): {len(feature_cols)}")
    print(f"  (excluded as too high-cardinality: {EXCLUDE_FROM_FEATURES})")
    print(f"Outcome columns (Performance, kept aside for step 5): {len(perf_cols)}")
    print(f"Log-derived columns (kept aside for step 7 root-cause, NOT used as prediction "
          f"features -- they're a consequence of failure, not a cause you'd know in advance): {len(log_cols)}")
    print(f"Target column: {TARGET_COLUMN}")
    print()

    return feature_cols, perf_cols, log_cols


def encode_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    X = df[feature_cols].copy()

    # 1. Convert boolean columns (True/False) to 0/1
    bool_cols = X.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # 2. One-hot encode remaining categorical (string) columns
    #    e.g. cache_policy=Adaptive -> cache_policy_Adaptive=1, others=0
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"Boolean columns converted to 0/1: {len(bool_cols)}")
    print(f"Categorical columns one-hot encoded: {cat_cols}")

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # get_dummies can produce True/False dtype columns on newer pandas -- normalize to int
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)

    print(f"Feature matrix shape after encoding: {X.shape}")
    print()
    return X


def main():
    df, data_dict = load_data()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns\n")

    n_dupes, missing = check_data_quality(df)

    # Drop exact duplicate rows if any exist
    if n_dupes > 0:
        df = df.drop_duplicates()
        print(f"Dropped {n_dupes} duplicate rows. New shape: {df.shape}\n")

    # Simple missing-value handling: numeric -> median, categorical -> mode
    # (Only runs if missing values actually exist -- your synthetic data
    # likely has none, but real SanDisk logs probably will.)
    if len(missing) > 0:
        for col in missing.index:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
            else:
                fill_value = df[col].mode().iloc[0]
            df[col] = df[col].fillna(fill_value)
        print(f"Filled missing values in {len(missing)} column(s) using median/mode.\n")

    feature_cols, outcome_cols, log_cols = split_columns(data_dict)

    X = encode_features(df, feature_cols)
    y = df[[TARGET_COLUMN]].copy()
    outcomes = df[outcome_cols].copy()
    log_features = df[log_cols].copy()
    # Convert log-derived booleans to 0/1 too, for consistency
    for col in log_features.select_dtypes(include="bool").columns:
        log_features[col] = log_features[col].astype(int)

    # Save everything
    X.to_csv(OUT_FEATURES, index=False)
    y.to_csv(OUT_TARGET, index=False)
    outcomes.to_csv(OUT_OUTCOMES, index=False)
    log_features.to_csv(OUT_LOG_FEATURES, index=False)

    # Also save one combined "cleaned" file (features + target + outcomes + logs + id)
    # -- useful for EDA/plotting/root-cause analysis, even though the pass/fail
    # model only ever trains on X/y
    cleaned_full = pd.concat([df[[ID_COLUMN]], X, y, outcomes, log_features], axis=1)
    cleaned_full.to_csv(OUT_CLEANED_FULL, index=False)

    print("--- Done ---")
    print(f"Features saved to:      {OUT_FEATURES}  (shape {X.shape})")
    print(f"Target saved to:        {OUT_TARGET}  (shape {y.shape})")
    print(f"Outcomes saved to:      {OUT_OUTCOMES}  (shape {outcomes.shape})")
    print(f"Log features saved to:  {OUT_LOG_FEATURES}  (shape {log_features.shape})")
    print(f"Full cleaned file:      {OUT_CLEANED_FULL}  (shape {cleaned_full.shape})")
    print(f"\nOverall pass rate in cleaned data: {y[TARGET_COLUMN].mean():.1%}")


if __name__ == "__main__":
    main()