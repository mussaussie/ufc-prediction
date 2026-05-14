"""
UFC 328 Weight-Cut Sensitivity Analysis — Script 05
====================================================
Reads:  models/best_ufc_model.joblib
        models/feature_columns.json
Writes: outputs/prediction_weight_cut.txt

Applies a physiological penalty to fighter stats based on reported weight cuts:
  Chimaev   : ~46 lb cut (210+ lbs → 185 lbs) — extreme
  Strickland: ~15 lb cut (~200 lbs → 185 lbs) — normal for MW

Penalty logic:
  - Grappling output (td_avg, sub_avg, finish_rate): proportional to cut severity
  - Striking output (slpm): minor penalty for extreme cuts only
  - Defence and accuracy: less affected (motor patterns hold under fatigue)

No retraining — same saved model, modified inference row.
"""

import os
import json
import joblib
import pandas as pd

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Reported weight cuts ───────────────────────────────────────────────────────
CHIMAEV_CUT_LBS    = 46   # media-reported (walk-around ~210-215 → 185)
STRICKLAND_CUT_LBS = 15   # estimated (walks ~198-200 → 185)

# ── Raw stats (same as 04_predict.py) ─────────────────────────────────────────
#                       Chimaev   Strickland
CHIMAEV    = dict(slpm=5.24, str_acc=0.60, str_def=0.63,
                  td_avg=5.29, td_acc=0.55, td_def=0.64,
                  sub_avg=1.80, win_streak=12, finish_rate=0.80,
                  age=32.3, days_since_last=266, reach=178)

STRICKLAND = dict(slpm=7.25, str_acc=0.61, str_def=0.61,
                  td_avg=0.43, td_acc=0.42, td_def=0.62,
                  sub_avg=0.10, win_streak=1, finish_rate=0.35,
                  age=35.2, days_since_last=77, reach=178)


def cut_penalty(cut_lbs: float) -> dict:
    """Return per-stat penalty multipliers based on weight cut size."""
    # Thresholds: ≤15 = normal | 16-30 = notable | 31-45 = severe | 46+ = extreme
    if cut_lbs <= 15:
        return dict(grappling=0.97, striking=0.99, defence=1.00)
    elif cut_lbs <= 30:
        return dict(grappling=0.90, striking=0.96, defence=0.99)
    elif cut_lbs <= 45:
        return dict(grappling=0.82, striking=0.93, defence=0.98)
    else:  # 46+
        return dict(grappling=0.72, striking=0.90, defence=0.97)


def apply_penalty(stats: dict, cut_lbs: float) -> dict:
    p = cut_penalty(cut_lbs)
    return dict(
        slpm        = stats["slpm"]        * p["striking"],
        str_acc     = stats["str_acc"]     * p["defence"],
        str_def     = stats["str_def"]     * p["defence"],
        td_avg      = stats["td_avg"]      * p["grappling"],
        td_acc      = stats["td_acc"]      * p["grappling"],
        td_def      = stats["td_def"]      * p["defence"],
        sub_avg     = stats["sub_avg"]     * p["grappling"],
        win_streak  = stats["win_streak"],   # momentum is pre-fight, not cut-affected
        finish_rate = stats["finish_rate"] * p["grappling"],
        age         = stats["age"],
        days_since_last = stats["days_since_last"],
        reach       = stats["reach"],
    )


def build_inference_row(a: dict, b: dict) -> dict:
    return {
        "slpm_diff":          a["slpm"]         - b["slpm"],
        "str_acc_diff":       a["str_acc"]       - b["str_acc"],
        "str_def_diff":       a["str_def"]       - b["str_def"],
        "td_avg_diff":        a["td_avg"]        - b["td_avg"],
        "td_acc_diff":        a["td_acc"]        - b["td_acc"],
        "td_def_diff":        a["td_def"]        - b["td_def"],
        "sub_avg_diff":       a["sub_avg"]       - b["sub_avg"],
        "win_streak_diff":    a["win_streak"]    - b["win_streak"],
        "finish_rate_diff":   a["finish_rate"]   - b["finish_rate"],
        "age_diff":           a["age"]           - b["age"],
        "days_diff":          a["days_since_last"] - b["days_since_last"],
        "reach_diff":         a["reach"]         - b["reach"],
        "title_fight":        1,
        "rounds":             5,
        "weight_class_lbs":   185,
    }


def predict(model, feature_cols, row: dict) -> tuple[float, float]:
    x = {k: row.get(k, 0) for k in feature_cols}
    X = pd.DataFrame([x])[feature_cols]
    prob = model.predict_proba(X.values)[0]
    return prob[1], prob[0]   # chimaev_prob, strickland_prob


if __name__ == "__main__":
    print("=" * 60)
    print("UFC 328 — Weight-Cut Sensitivity Analysis")
    print("=" * 60)

    model_path = os.path.join(MODEL_DIR, "best_ufc_model.joblib")
    feat_path  = os.path.join(MODEL_DIR, "feature_columns.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Run 03_train_model.py first.")
    model = joblib.load(model_path)
    with open(feat_path) as f:
        feature_cols = json.load(f)

    # ── Baseline (no penalty) ──────────────────────────────────────────────────
    row_base = build_inference_row(CHIMAEV, STRICKLAND)
    c_base, s_base = predict(model, feature_cols, row_base)

    # ── With weight-cut penalty ────────────────────────────────────────────────
    chimaev_adj    = apply_penalty(CHIMAEV,    CHIMAEV_CUT_LBS)
    strickland_adj = apply_penalty(STRICKLAND, STRICKLAND_CUT_LBS)
    row_adj = build_inference_row(chimaev_adj, strickland_adj)
    c_adj, s_adj = predict(model, feature_cols, row_adj)

    p_chi = cut_penalty(CHIMAEV_CUT_LBS)
    p_str = cut_penalty(STRICKLAND_CUT_LBS)

    output = f"""
============================================================
  UFC 328 — WEIGHT-CUT SENSITIVITY ANALYSIS
  Khamzat Chimaev vs Sean Strickland (MW Title)
  May 9, 2026 | Prudential Center, Newark NJ
============================================================

  REPORTED CUTS
  Chimaev    : {CHIMAEV_CUT_LBS} lbs  (~210+ lbs to 185 lbs) -- EXTREME
  Strickland : {STRICKLAND_CUT_LBS} lbs  (~200 lbs to 185 lbs) -- Normal for MW

------------------------------------------------------------
  BASELINE PREDICTION (no weight-cut adjustment)
  Chimaev    : {c_base*100:.1f}%
  Strickland : {s_base*100:.1f}%
  Winner     : {"Chimaev" if c_base > 0.5 else "Strickland"}

------------------------------------------------------------
  WEIGHT-CUT ADJUSTED PREDICTION
  Chimaev    : {c_adj*100:.1f}%
  Strickland : {s_adj*100:.1f}%
  Winner     : {"Chimaev" if c_adj > 0.5 else "Strickland"}

------------------------------------------------------------
  PENALTY APPLIED (Chimaev — {CHIMAEV_CUT_LBS} lb cut)
  Grappling output  : -{(1 - p_chi["grappling"])*100:.0f}%  (TD avg, sub avg, finish rate)
  Striking output   : -{(1 - p_chi["striking"])*100:.0f}%   (SLpM)
  Defence/accuracy  : -{(1 - p_chi["defence"])*100:.0f}%   (str acc, str def, TD def)

  PENALTY APPLIED (Strickland — {STRICKLAND_CUT_LBS} lb cut)
  Grappling output  : -{(1 - p_str["grappling"])*100:.0f}%
  Striking output   : -{(1 - p_str["striking"])*100:.0f}%
  Defence/accuracy  : -{(1 - p_str["defence"])*100:.0f}%

------------------------------------------------------------
  KEY STAT ADJUSTMENTS
  {"Stat":<20} {"Chimaev raw":>12} {"Chimaev adj":>12} {"Strickland adj":>15}
  {"-"*62}
  {"td_avg":<20} {CHIMAEV["td_avg"]:>12.2f} {chimaev_adj["td_avg"]:>12.2f} {strickland_adj["td_avg"]:>15.2f}
  {"sub_avg":<20} {CHIMAEV["sub_avg"]:>12.2f} {chimaev_adj["sub_avg"]:>12.2f} {strickland_adj["sub_avg"]:>15.2f}
  {"finish_rate":<20} {CHIMAEV["finish_rate"]:>12.2f} {chimaev_adj["finish_rate"]:>12.2f} {strickland_adj["finish_rate"]:>15.2f}
  {"slpm":<20} {CHIMAEV["slpm"]:>12.2f} {chimaev_adj["slpm"]:>12.2f} {strickland_adj["slpm"]:>15.2f}

------------------------------------------------------------
  ACTUAL FIGHT RESULT (May 9, 2026)
  Winner     : Sean Strickland
  Key factor : Chimaev's extreme weight cut impaired
               grappling output — his primary weapon

  MODEL VERDICT: {"Correctly flips to Strickland" if c_adj < 0.5 else "Still picks Chimaev (penalty not enough to flip)"}
============================================================
"""
    print(output)

    out_path = os.path.join(OUTPUT_DIR, "prediction_weight_cut.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Saved -> {out_path}")
