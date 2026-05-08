"""
UFC 328 Prediction — Script 01: Data Acquisition
================================================
Source: UFC.csv (Kaggle rajeevw/ufcdata) — 8337 fights with career stats + winner

win_streak and finish_rate are computed chronologically from full fight history
so stats reflect what was true BEFORE each fight (no data leakage).

Output: data/mw_raw_combined.csv
"""

import os
import sys
from collections import defaultdict

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
UFC_CSV  = os.path.join(DATA_DIR, "UFC.csv")

FINISH_METHODS = {"KO/TKO", "KO", "TKO", "SUB", "SUBMISSION"}


# ─────────────────────────────────────────────
# STEP 1 — Load
# ─────────────────────────────────────────────
def load_ufc_csv():
    if not os.path.exists(UFC_CSV):
        print(f"[FAIL] UFC.csv not found at {UFC_CSV}")
        sys.exit(1)

    print("[1/4] Loading UFC.csv ...")
    df = pd.read_csv(UFC_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"      {len(df)} total fights loaded")
    return df


# ─────────────────────────────────────────────
# STEP 2 — Compute win_streak & finish_rate
#           per fighter, per fight (no leakage)
# ─────────────────────────────────────────────
def compute_running_stats(df):
    """Walk fights in chronological order. Before each fight, record each
    fighter's current streak and finish_rate. Then update after the result."""
    print("[2/4] Computing win_streak and finish_rate per fight (chronological) ...")

    wins           = defaultdict(int)
    finishes       = defaultdict(int)
    streak         = defaultdict(int)
    last_fight_date = {}  # fighter_id -> pd.Timestamp of last fight

    r_streak_col      = []
    b_streak_col      = []
    r_finish_rate_col = []
    b_finish_rate_col = []
    r_days_col        = []
    b_days_col        = []

    for _, row in df.iterrows():
        r_id   = row.get("r_id", "")
        b_id   = row.get("b_id", "")
        winner = str(row.get("winner", "")).strip()
        r_name = str(row.get("r_name", "")).strip()
        b_name = str(row.get("b_name", "")).strip()
        method = str(row.get("method", "")).strip().upper()
        fight_date = row.get("date")

        # Record stats BEFORE this fight
        r_streak_col.append(streak[r_id])
        b_streak_col.append(streak[b_id])
        r_finish_rate_col.append(finishes[r_id] / wins[r_id] if wins[r_id] > 0 else 0.0)
        b_finish_rate_col.append(finishes[b_id] / wins[b_id] if wins[b_id] > 0 else 0.0)

        def _days_since(fid):
            if fid in last_fight_date and pd.notna(fight_date):
                return (pd.Timestamp(fight_date) - last_fight_date[fid]).days
            return 0

        r_days_col.append(_days_since(r_id))
        b_days_col.append(_days_since(b_id))

        # Update stats AFTER this fight
        is_finish = any(s in method for s in FINISH_METHODS)
        if winner == r_name:
            wins[r_id]     += 1
            finishes[r_id] += int(is_finish)
            streak[r_id]   += 1
            streak[b_id]    = 0
        elif winner == b_name:
            wins[b_id]     += 1
            finishes[b_id] += int(is_finish)
            streak[b_id]   += 1
            streak[r_id]    = 0
        # no update on draws / no-contest

        if pd.notna(fight_date):
            last_fight_date[r_id] = pd.Timestamp(fight_date)
            last_fight_date[b_id] = pd.Timestamp(fight_date)

    df = df.copy()
    df["r_win_streak"]       = r_streak_col
    df["b_win_streak"]       = b_streak_col
    df["r_finish_rate"]      = r_finish_rate_col
    df["b_finish_rate"]      = b_finish_rate_col
    df["r_days_since_last"]  = r_days_col
    df["b_days_since_last"]  = b_days_col

    print(f"      Done. Sample r_win_streak range: "
          f"{df['r_win_streak'].min()}-{df['r_win_streak'].max()}")
    return df


# Weight class label -> numeric weight (lbs) used as a feature
WEIGHT_CLASS_LBS = {
    "welterweight":     170,
    "middleweight":     185,
    "light heavyweight": 205,
}

# ─────────────────────────────────────────────
# STEP 3 — Filter to WW + MW + LHW, build rows
# ─────────────────────────────────────────────
def build_mw_training_rows(df):
    print("[3/4] Filtering to male WW/MW/LHW (2016+) and building rows ...")

    div = df["division"].str.lower().fillna("")
    mw = df[
        (div.isin(["welterweight", "middleweight", "light heavyweight"])) &
        (df["date"] >= "2016-01-01")
    ].copy().reset_index(drop=True)

    print(f"      {len(mw)} WW+MW+LHW fights found")

    # Resolve winner to red/blue side
    mw["winner_side"] = mw.apply(
        lambda r: "a" if str(r.get("winner", "")).strip() == str(r.get("r_name", "")).strip()
                  else ("b" if str(r.get("winner", "")).strip() == str(r.get("b_name", "")).strip()
                  else None),
        axis=1
    )
    mw = mw.dropna(subset=["winner_side"]).reset_index(drop=True)
    print(f"      {len(mw)} rows with resolved winner (across WW/MW/LHW)")

    # Age at fight time
    for corner in ["r", "b"]:
        mw[f"{corner}_age"] = (
            mw["date"] - pd.to_datetime(mw[f"{corner}_dob"], errors="coerce")
        ).dt.days / 365.25

    # Reach is already stored in cm in UFC.csv — no conversion needed
    for corner in ["r", "b"]:
        mw[f"{corner}_reach_cm"] = pd.to_numeric(mw[f"{corner}_reach"], errors="coerce")

    # UFC.csv column name aliases
    col_map = {
        "slpm":    "splm",       # UFC.csv typo: splm not slpm
        "str_acc": "str_acc",
        "str_def": "str_def",
        "td_avg":  "td_avg",
        "td_acc":  "td_avg_acc",
        "td_def":  "td_def",
        "sub_avg": "sub_avg",
    }

    rows = []
    for _, row in mw.iterrows():
        feat = {}

        for our_name, ufc_name in col_map.items():
            feat[f"a_{our_name}"] = pd.to_numeric(row.get(f"r_{ufc_name}"), errors="coerce")
            feat[f"b_{our_name}"] = pd.to_numeric(row.get(f"b_{ufc_name}"), errors="coerce")

        feat["a_age"]              = row.get("r_age")
        feat["b_age"]              = row.get("b_age")
        feat["a_reach"]            = row.get("r_reach_cm")
        feat["b_reach"]            = row.get("b_reach_cm")
        feat["a_current_win_streak"] = row.get("r_win_streak", 0)
        feat["b_current_win_streak"] = row.get("b_win_streak", 0)
        feat["a_finish_rate"]      = row.get("r_finish_rate", 0.0)
        feat["b_finish_rate"]      = row.get("b_finish_rate", 0.0)
        feat["title_fight"]          = 1 if str(row.get("title_fight", "")).strip().lower() in ("true", "1") else 0
        feat["no_of_rounds"]         = pd.to_numeric(row.get("total_rounds", 3), errors="coerce")
        feat["a_days_since_last"]    = row.get("r_days_since_last", 0)
        feat["b_days_since_last"]    = row.get("b_days_since_last", 0)
        feat["date"]                 = row.get("date")
        feat["winner"]               = row["winner_side"]
        div_label                    = str(row.get("division", "middleweight")).strip().lower()
        feat["weight_class_lbs"]     = WEIGHT_CLASS_LBS.get(div_label, 185)

        rows.append(feat)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# STEP 4 — Save
# ─────────────────────────────────────────────
def save(df):
    print("[4/4] Saving ...")
    df = df.dropna(subset=["a_slpm", "b_slpm", "winner"])
    out = os.path.join(DATA_DIR, "mw_raw_combined.csv")
    df.to_csv(out, index=False)
    print(f"      Saved {len(df)} rows -> {out}")
    print(f"      Winner split: {df['winner'].value_counts().to_dict()}")
    new_cols = ["a_current_win_streak", "b_current_win_streak", "a_finish_rate", "b_finish_rate"]
    print(f"      New features: {df[new_cols].describe().round(3).to_string()}")


if __name__ == "__main__":
    print("=" * 55)
    print("UFC 328 - Step 1: Data Acquisition (UFC.csv)")
    print("=" * 55)
    df_all = load_ufc_csv()
    df_all = compute_running_stats(df_all)
    df_mw  = build_mw_training_rows(df_all)
    save(df_mw)
    print("\nDone. Run 02_feature_engineering.py next.")
