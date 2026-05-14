# UFC 328 Fight Prediction — ML Pipeline

**Khamzat Chimaev vs Sean Strickland | Middleweight Title | May 9, 2026**
**Result: Sean Strickland won. Model predicted Chimaev at 85.5%.**

A full end-to-end machine learning pipeline that predicts the outcome of a UFC championship fight using real fight data, chronological feature engineering, and comparative tabular classification models. Built before the fight — post-fight analysis of where the model fell short included below.

---

## Prediction Result

| Fighter | Win Probability |
|---|---|
| **Khamzat Chimaev** (Champion, 15-0) | **85.5%** |
| Sean Strickland (Challenger, 30-7) | 14.5% |

![Prediction Result](outputs/viz_02_prediction_result.png)

---

## Post-Fight Result — May 9, 2026

**Sean Strickland won. The model was wrong.**

The model assigned Chimaev 85.5% — he lost.

**Why the model missed it — weight cut:**
Chimaev reportedly cut ~46 lbs to make the 185 lb middleweight limit, walking into fight week well above 210 lbs. Extreme weight cuts of this magnitude impair reaction time, chin durability, and grappling output — Chimaev's core weapons. The model had zero signal for this because no weight cut feature was included in the pipeline.

**What the model had:**
- Historical per-fight stats (striking, grappling, finish rate, streaks)
- Age and reach differentials
- Days since last fight

**What the model was missing:**
- Fighter's natural walk-around weight vs. weight class limit
- Size of the cut as a fight-week stress indicator
- History of difficult cuts (a known Chimaev issue moving down to 185)

This is a legitimate model gap, not a calibration failure. The 14.5% the model gave Strickland reflected real uncertainty — the prediction was directionally wrong, not nonsensically so. Adding a weight cut differential feature is the clearest next lever for improving this model.

> **Weight-cut sensitivity experiment:** After applying a 28% grappling penalty to Chimaev and 3% to Strickland, the model moves from 85.5% → 81.7% for Chimaev. It does not flip. See findings below.

---

## Model Performance

Three local models trained with `TimeSeriesSplit` cross-validation, plus Vertex AI AutoML on Google Cloud:

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.712 |
| **Vertex AI AutoML** | **0.711** |
| Random Forest | 0.685 |
| XGBoost | 0.671 |

> Logistic Regression outperformed XGBoost across all three weight classes. Vertex AI AutoML matched LR exactly — confirming the signal ceiling is data volume, not model complexity.

![Model Comparison](outputs/viz_01_model_comparison.png)

---

## Pipeline Overview

![Pipeline](outputs/viz_05_pipeline.png)

1. **Data** — 8,337 UFC fights from [Kaggle: rajeevw/ufcdata](https://www.kaggle.com/datasets/rajeevw/ufcdata)
2. **Filter** — 1,517 male Welterweight / Middleweight / Light Heavyweight fights since 2016
3. **Feature Engineering** — 15 differential features (Fighter A stat − Fighter B stat)
4. **Train** — Logistic Regression (best), RF, XGBoost with TimeSeriesSplit CV on 1,517 fights
5. **Predict** — Probability output for Chimaev vs Strickland

---

## Feature Engineering

Instead of raw per-fighter stats, the model uses **differential features** — the gap between both fighters across multiple dimensions:

![Differentials](outputs/viz_04_differentials.png)

| Feature | What It Measures |
|---|---|
| `slpm_diff` | Who lands more strikes per minute |
| `str_acc_diff` | Striking accuracy gap |
| `str_def_diff` | Striking defence gap |
| `td_avg_diff` | Takedowns per 15 min gap |
| `td_acc_diff` | Takedown efficiency gap |
| `td_def_diff` | Takedown defence gap |
| `sub_avg_diff` | Submission attempt gap |
| `win_streak_diff` | Momentum (consecutive wins before fight) |
| `finish_rate_diff` | % fights finished vs. going to decision |
| `age_diff` | Age on fight day |
| `reach_diff` | Reach in cm (already stored in cm in UFC.csv) |
| `days_diff` | Days since last fight — rest/ring-rust signal |
| `title_fight` | Binary — 1 if title bout |
| `rounds` | Scheduled rounds (3 or 5) |

**Key design decision:** `win_streak` and `finish_rate` are computed chronologically across all 8,337 fights to avoid using future fight outcomes. These are recorded before each fight and updated after.

---

## Fighter Breakdown

![Fighter Stats](outputs/viz_03_stats_comparison.png)

**Chimaev going in:** 15-0, 12-fight win streak, 80% finish rate, elite wrestler (5.29 TD/15min), major submission threat (1.80 sub/15min)

**Strickland going in:** 30-7, high-volume striker (7.25 SLpM), excellent chin, 35% finish rate

---

## Benchmark Comparison

| Source | Chimaev | Strickland |
|---|---|---|
| **This Model (Logistic Regression)** | **85.5%** | 14.5% |
| Elo Model | 79.0% | 21.0% |
| Betting Odds | ~85% | ~15% |

With 1,517 training samples across WW/MW/LHW, our model now aligns closely with both the Elo model and betting market. Prediction is from the best validated model (Logistic Regression, ROC-AUC 0.692).

---

## Installation

```bash
git clone https://github.com/mussaussie/ufc-prediction.git
cd ufc-prediction
pip install -r requirements.txt
```

**Requirements:**
```
pandas
numpy
scikit-learn
xgboost
joblib
shap
matplotlib
```

**Data:** Download [UFC.csv](https://www.kaggle.com/datasets/rajeevw/ufcdata) from Kaggle and place it in `data/UFC.csv`.

---

## Running the Pipeline

Run scripts in order:

```bash
# Step 1 — Load UFC.csv, compute running stats, filter to WW/MW/LHW
python scripts/01_scrape_data.py

# Step 2 — Build model features and matchup differentials
python scripts/02_feature_engineering.py

# Step 3 — Train Logistic Regression, Random Forest, XGBoost
python scripts/03_train_model.py

# Step 4 — Run inference for Chimaev vs Strickland
python scripts/04_predict.py

# Step 5 — Generate all visualizations
python scripts/05_visualize.py
```

---

## File Structure

```
ufc_prediction/
├── data/
│   ├── UFC.csv                    ← Kaggle source (rajeevw/ufcdata)
│   ├── mw_raw_combined.csv        ← Filtered WW/MW/LHW fights with computed stats
│   └── mw_processed.csv           ← Final model-ready dataset (1,517 rows × 16 cols)
├── models/
│   ├── best_ufc_model.joblib      ← Trained best model (currently Logistic Regression)
│   └── feature_columns.json       ← Feature column list for inference
├── outputs/
│   ├── prediction_result.txt      ← Final prediction text output
│   ├── feature_importance.png     ← SHAP feature importance chart
│   ├── model_comparison.csv       ← Model AUC scores
│   ├── viz_01_model_comparison.png
│   ├── viz_02_prediction_result.png
│   ├── viz_03_stats_comparison.png
│   ├── viz_04_differentials.png
│   └── viz_05_pipeline.png
├── scripts/
│   ├── 01_scrape_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_model.py
│   ├── 04_predict.py
│   └── 05_visualize.py
├── EXPLANATION.md                 ← Full procedure walkthrough
├── requirements.txt
└── README.md
```

---

## Vertex AI AutoML Results

Trained on Google Cloud AutoML Tabular Classification (1 node-hour budget) using `data/mw_processed.csv`.

| Metric | Value |
|---|---|
| ROC AUC | **0.711** |
| PR AUC | 0.709 |
| Log Loss | 0.621 |
| Micro-avg F1 | 0.623 |
| Precision / Recall | 62.3% / 62.3% |

**Confusion matrix (threshold = 0.5):**

| | Predicted Win | Predicted Loss |
|---|---|---|
| **Actual Winner** | 67% ✓ | 33% ✗ |
| **Actual Loser** | 45% ✗ | 55% ✓ |

**Key finding:** Vertex AI AutoML (0.711 ROC-AUC) matched Logistic Regression (0.712) exactly — confirming the performance ceiling is the data, not the model. More fight data or round-level features would be the next lever, not a more complex model.

**Full model comparison:**

| Model | ROC-AUC |
|---|---|
| Logistic Regression (local) | 0.712 |
| **Vertex AI AutoML** | **0.711** |
| Random Forest (local) | 0.685 |
| XGBoost (local) | 0.671 |

---

## Limitations

1. **Historical averages are taken from the available dataset, not rebuilt from full round-level pre-fight snapshots.** The custom `win_streak` and `finish_rate` fields are chronological, but some source stat columns may still reflect dataset-level limitations.
2. **The training pool uses nearby male divisions, not only middleweight.** That improves sample size but introduces some cross-division noise.
3. **No weight cut signal.** Fighter natural weight vs. weight class limit is not modelled. This was the primary failure mode for UFC 328 — Chimaev's ~46 lb cut was not captured.
4. **No explicit style-matchup, camp, injury, or late-notice signal.** Important real-world factors remain outside the model.
5. **`days_since_last` for a fighter's first tracked bout is treated as 0.** That is a practical placeholder, not a true layoff value.
6. **Red corner structural bias remains in the data.** Higher-profile fighters are often assigned red corner and win more often historically.

ROC-AUC of about 0.69 is still within the broad range often seen in public UFC prediction work, but this project should be viewed as a practical ML experiment rather than a production betting model.

---

## Weight-Cut Sensitivity Experiment (Post-Fight)

Script: `scripts/05_weight_cut_predict.py`

Reported cuts: Chimaev 46 lbs (extreme), Strickland 15 lbs (normal MW).

A 28% penalty was applied to Chimaev's grappling stats (TD avg, submission avg, finish rate) and 3% to Strickland's, reflecting the physiological difference between an extreme cut and a normal one.

| Scenario | Chimaev | Strickland |
|---|---|---|
| Baseline (no penalty) | 85.5% | 14.5% |
| Weight-cut adjusted | 81.7% | 18.3% |
| **Actual result** | **Lost** | **Won** |

**Finding:** The model does not flip even with a 28% grappling penalty. Chimaev's raw TD avg (5.29 vs 0.43) and sub avg (1.80 vs 0.10) differentials are so large that a cut-fatigue multiplier alone cannot overcome them. This reveals a structural model limitation:

- Historical career averages are a poor proxy for fight-night physical state
- The model cannot learn "this fighter cuts 46 lbs and it will cost them" from fight-result labels alone
- Accurately capturing weight cut effect would require round-level output data (checking whether the grappling rate drops in rounds 3-5) rather than career summary stats

The correct prediction would likely require a much larger penalty (~50-60% grappling reduction) or direct round-level fatigue signals — neither of which are available in public UFC summary datasets.

---

## Tech Stack

- Python 3.11
- pandas, numpy, scikit-learn, XGBoost, python-pptx
- SHAP (feature explainability)
- matplotlib (visualizations)
- joblib (model persistence)

---

## Author

**Abdul Mussavir** — Data Analyst transitioning to ML/DS  
GitHub: [@mussaussie](https://github.com/mussaussie)
