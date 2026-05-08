"""
UFC 328 Prediction — Script 03: Train Model
============================================
Reads:  data/mw_processed.csv
Writes: models/best_ufc_model.joblib
        outputs/feature_importance.png
        outputs/model_comparison.csv

Trains 3 models (LogReg, RandomForest, XGBoost), saves best by ROC-AUC.
Validation: TimeSeriesSplit (chronological — prevents fighter-level leakage).
"""

import os
import json
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    path = os.path.join(DATA_DIR, "mw_processed.csv")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values
    y = df["winner"].values
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features (sorted chronologically)")
    return X, y, feature_cols


def train_and_evaluate(X, y):
    # TimeSeriesSplit preserves chronological order — prevents same-fighter leakage across folds
    cv = TimeSeriesSplit(n_splits=5)
    results = []
    candidates = []  # (model_object, auc_mean)

    print("\n[1/3] Logistic Regression...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_scores = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"    ROC-AUC: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")
    results.append({"Model": "Logistic Regression", "ROC-AUC": lr_scores.mean(), "Std": lr_scores.std()})
    candidates.append((lr_pipe, lr_scores.mean()))

    print("[2/3] Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print(f"    ROC-AUC: {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")
    results.append({"Model": "Random Forest", "ROC-AUC": rf_scores.mean(), "Std": rf_scores.std()})
    candidates.append((rf, rf_scores.mean()))

    print("[3/3] XGBoost...")
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
        xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc")
        print(f"    ROC-AUC: {xgb_scores.mean():.4f} +/- {xgb_scores.std():.4f}")
        results.append({"Model": "XGBoost", "ROC-AUC": xgb_scores.mean(), "Std": xgb_scores.std()})
        candidates.append((xgb, xgb_scores.mean()))
    except ImportError:
        print("    XGBoost not installed. Run: pip install xgboost")

    best_model, best_auc = max(candidates, key=lambda x: x[1])
    best_name = results[[r["ROC-AUC"] for r in results].index(best_auc)]["Model"]
    print(f"\nBest model: {best_name} (ROC-AUC {best_auc:.4f})")
    return best_model, results


def save_model(model, X, y, feature_cols):
    print("\nFitting final model on full data...")
    model.fit(X, y)
    path = os.path.join(MODEL_DIR, "best_ufc_model.joblib")
    joblib.dump(model, path)
    print(f"Model saved -> {path}")

    try:
        import shap
        print("Generating SHAP feature importance...")
        # TreeExplainer for RF/XGB; LinearExplainer for LR Pipeline
        from sklearn.pipeline import Pipeline as SKPipeline
        if isinstance(model, SKPipeline):
            clf = model.named_steps["clf"]
            X_scaled = model.named_steps["scaler"].transform(X)
            explainer = shap.LinearExplainer(clf, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            X_plot = X_scaled
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            X_plot = X
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_plot, feature_names=feature_cols, plot_type="bar", show=False)
        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Feature importance saved -> {fig_path}")
    except ImportError:
        print("SHAP not installed (optional). Run: pip install shap")
    except Exception as e:
        print(f"SHAP skipped: {e}")


def save_comparison(results):
    df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    benchmarks = pd.DataFrame([
        {"Model": "Elo Model (benchmark)",      "ROC-AUC": None, "Std": None},
        {"Model": "Betting Odds (benchmark)",   "ROC-AUC": None, "Std": None},
    ])
    df = pd.concat([df, benchmarks], ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)
    print(f"\nComparison saved -> {path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    print("=" * 55)
    print("UFC 328 — Step 3: Model Training")
    print("=" * 55)
    X, y, feature_cols = load_data()
    best_model, results = train_and_evaluate(X, y)
    save_model(best_model, X, y, feature_cols)
    save_comparison(results)
    print("\nDone. Run 04_predict.py for final prediction.")

