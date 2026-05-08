"""
UFC 328 Prediction — Script 02: Feature Engineering
====================================================
Reads:  data/mw_raw_combined.csv
Writes: data/mw_processed.csv  (model-ready)
        models/feature_columns.json

Builds differential features: Fighter A stat - Fighter B stat
Target: winner (1 = Fighter A wins, 0 = Fighter B wins)
"""

import os
import json
import pandas as pd

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

STAT_COLS = ["slpm", "str_acc", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]


def load_data():
    path = os.path.join(DATA_DIR, "mw_raw_combined.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}\nRun 01_scrape_data.py first.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows | columns: {list(df.columns[:10])} ...")
    return df


def _diff(row, a_col, b_col):
    return (
        pd.to_numeric(row.get(a_col), errors="coerce") -
        pd.to_numeric(row.get(b_col), errors="coerce")
    )


def build_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    a_prefix = "a_" if any(c.startswith("a_") for c in df.columns) else "r_"
    b_prefix = "b_"

    for _, row in df.iterrows():
        feat = {}

        for stat in STAT_COLS:
            a_col = f"{a_prefix}{stat}"
            b_col = f"{b_prefix}{stat}"
            if a_col in df.columns and b_col in df.columns:
                feat[f"{stat}_diff"] = _diff(row, a_col, b_col)

        # win streak
        for col in ["current_win_streak", "win_streak", "fighter_win_streak"]:
            a_col = f"{a_prefix}{col}"
            b_col = f"{b_prefix}{col}"
            if a_col in df.columns and b_col in df.columns:
                feat["win_streak_diff"] = _diff(row, a_col, b_col)
                break

        # finish rate
        for col in ["finish_rate", "finish_pct"]:
            a_col = f"{a_prefix}{col}"
            b_col = f"{b_prefix}{col}"
            if a_col in df.columns and b_col in df.columns:
                feat["finish_rate_diff"] = _diff(row, a_col, b_col)
                break

        # age
        for col in ["age", "fighter_age"]:
            a_col = f"{a_prefix}{col}"
            b_col = f"{b_prefix}{col}"
            if a_col in df.columns and b_col in df.columns:
                feat["age_diff"] = _diff(row, a_col, b_col)
                break

        # reach
        for col in ["reach", "reach_in_cm", "reach_cms"]:
            a_col = f"{a_prefix}{col}"
            b_col = f"{b_prefix}{col}"
            if a_col in df.columns and b_col in df.columns:
                feat["reach_diff"] = _diff(row, a_col, b_col)
                break

        # days since last fight
        for col in ["days_since_last"]:
            a_col = f"{a_prefix}{col}"
            b_col = f"{b_prefix}{col}"
            if a_col in df.columns and b_col in df.columns:
                feat["days_diff"] = _diff(row, a_col, b_col)
                break

        feat["title_fight"] = int(
            str(row.get("title_fight", "0")).strip().lower() in ["1", "true", "yes"]
        )
        feat["rounds"]           = pd.to_numeric(row.get("no_of_rounds", 3), errors="coerce")
        feat["weight_class_lbs"] = pd.to_numeric(row.get("weight_class_lbs", 185), errors="coerce")
        feat["date"]             = row.get("date")

        winner_val = str(row.get("winner", "")).strip().lower()
        if winner_val in ["blue", "b", "fighter_b", "0"]:
            feat["winner"] = 0
        elif winner_val in ["red", "a", "fighter_a", "1"]:
            feat["winner"] = 1
        else:
            continue

        rows.append(feat)

    result = pd.DataFrame(rows)
    non_feature = ["winner", "date"]
    feature_cols_only = [c for c in result.columns if c not in non_feature]
    result = result.dropna(subset=feature_cols_only)
    print(f"Feature matrix: {result.shape[0]} rows x {result.shape[1]} columns")
    return result


def save_outputs(df: pd.DataFrame):
    out_path = os.path.join(DATA_DIR, "mw_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset -> {out_path}")

    feature_cols = [c for c in df.columns if c not in ("winner", "date")]
    col_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(col_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature columns -> {col_path}")
    return feature_cols


if __name__ == "__main__":
    print("=" * 55)
    print("UFC 328 — Step 2: Feature Engineering")
    print("=" * 55)
    df_raw = load_data()
    df_processed = build_differential_features(df_raw)
    feature_cols = save_outputs(df_processed)
    print(f"\nFeatures built: {feature_cols}")
    print("\nDone. Run 03_train_model.py next.")

