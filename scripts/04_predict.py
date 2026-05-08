"""
UFC 328 Prediction — Script 04: Final Prediction
=================================================
Reads:  models/xgb_ufc_model.joblib
        models/feature_columns.json
Writes: outputs/prediction_result.txt

Runs inference for Chimaev vs Strickland.
Fighter A = Khamzat Chimaev | Fighter B = Sean Strickland
Differential = Chimaev stat - Strickland stat
"""

import os
import json
import joblib
import pandas as pd

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stats sourced from ufcstats.com via web search, May 2026
# Chimaev: career stats updated through UFC 319 (du Plessis, Aug 2025)
# Strickland: career stats updated through UFC Fight Night (Hernandez, Feb 2026)
#
#                       Chimaev   Strickland
# SLpM                    5.24       7.25    (Strickland +2.01 - high volume)
# Str Acc                 0.60       0.61    (essentially equal)
# Str Def                 0.63       0.61
# TD Avg/15min            5.29       0.43    (Chimaev +4.86 - elite grappler)
# TD Acc                  0.55       0.42
# TD Def                  0.64       0.62    (now equal - Strickland dropped 74->62%)
# Sub Avg/15min           1.80       0.10    (Chimaev +1.70 - major sub threat)
# Win Streak                12          1
# Finish Rate             0.80       0.35    (80% = 12 finishes in 15 wins)
# Age (May 9 2026)        32.3       35.2
# Days since last fight    266         77    (Chimaev last fought Aug 2025, Strickland Feb 2026)

INFERENCE_ROW = {
    "slpm_diff":          5.24  - 7.25,
    "str_acc_diff":       0.60  - 0.61,
    "str_def_diff":       0.63  - 0.61,
    "td_avg_diff":        5.29  - 0.43,
    "td_acc_diff":        0.55  - 0.42,
    "td_def_diff":        0.64  - 0.62,
    "sub_avg_diff":       1.80  - 0.10,
    "win_streak_diff":    12    - 1,
    "finish_rate_diff":   0.80  - 0.35,
    "age_diff":           32.3  - 35.2,
    "days_diff":          266   - 77,
    "reach_diff":         0,
    "title_fight":        1,
    "rounds":             5,
    "weight_class_lbs":   185,
}


def load_model_and_features():
    model_path = os.path.join(MODEL_DIR, "best_ufc_model.joblib")
    feat_path  = os.path.join(MODEL_DIR, "feature_columns.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun 03_train_model.py first to generate best_ufc_model.joblib.")
    model = joblib.load(model_path)
    with open(feat_path) as f:
        feature_cols = json.load(f)
    return model, feature_cols


def run_prediction(model, feature_cols):
    row = {k: INFERENCE_ROW.get(k, 0) for k in feature_cols}
    X_pred = pd.DataFrame([row])[feature_cols]
    prob = model.predict_proba(X_pred.values)[0]
    chimaev_prob    = prob[1]
    strickland_prob = prob[0]
    return chimaev_prob, strickland_prob


def format_and_save(chimaev_prob, strickland_prob):
    winner     = "Khamzat Chimaev" if chimaev_prob > 0.5 else "Sean Strickland"
    confidence = max(chimaev_prob, strickland_prob) * 100

    output = f"""
======================================================
     UFC 328 — FIGHT PREDICTION RESULT
     Khamzat Chimaev vs Sean Strickland (MW Title)
     May 9, 2026 | Prudential Center, Newark NJ
======================================================

  PREDICTED WINNER : {winner}

  Chimaev Win Probability    : {chimaev_prob*100:.1f}%
  Strickland Win Probability : {strickland_prob*100:.1f}%
  Model Confidence           : {confidence:.1f}%

------------------------------------------------------
  BENCHMARK COMPARISON
  Elo Model      : Chimaev 79.0%  | Strickland 21.0%
  Betting Odds   : Chimaev ~85%   | Strickland ~15%
  Our Model (LR) : Chimaev {chimaev_prob*100:.1f}%  | Strickland {strickland_prob*100:.1f}%
------------------------------------------------------
  KEY FACTORS DRIVING PREDICTION
  TD Avg Diff    : {INFERENCE_ROW['td_avg_diff']:+.2f}  (Chimaev elite grappler)
  Sub Avg Diff   : {INFERENCE_ROW['sub_avg_diff']:+.2f}  (Chimaev major sub threat)
  SLpM Diff      : {INFERENCE_ROW['slpm_diff']:+.2f}  (Strickland higher volume)
  Win Streak     : {INFERENCE_ROW['win_streak_diff']:+.0f}   (Chimaev 12-0 streak)
  Finish Rate    : {INFERENCE_ROW['finish_rate_diff']:+.2f}  (Chimaev 80% vs Strickland 35%)
  Age Diff       : {INFERENCE_ROW['age_diff']:+.1f}  (Chimaev 2.9yr younger)
======================================================
"""
    print(output)
    out_path = os.path.join(OUTPUT_DIR, "prediction_result.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"Result saved -> {out_path}")


if __name__ == "__main__":
    print("=" * 55)
    print("UFC 328 — Step 4: Chimaev vs Strickland Prediction")
    print("=" * 55)
    model, feature_cols = load_model_and_features()
    chimaev_prob, strickland_prob = run_prediction(model, feature_cols)
    format_and_save(chimaev_prob, strickland_prob)

