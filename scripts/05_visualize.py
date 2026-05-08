"""
UFC 328 Prediction — Script 05: Visualizations
===============================================
Generates publication-ready charts for GitHub, LinkedIn, and reports.

Outputs:
  outputs/viz_01_model_comparison.png
  outputs/viz_02_prediction_result.png
  outputs/viz_03_stats_comparison.png
  outputs/viz_04_differentials.png
  outputs/viz_05_data_pipeline.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
RED   = "#C0392B"
BLUE  = "#2471A3"
GOLD  = "#F39C12"
DARK  = "#1A1A2E"
GREY  = "#95A5A6"
LIGHT = "#F8F9FA"
GREEN = "#27AE60"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   DARK,
    "axes.edgecolor":   GREY,
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "grid.color":       "#2C2C4A",
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
})


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
def viz_model_comparison():
    models = ["Logistic Regression\n(Best — saved)", "Random\nForest", "XGBoost", "Elo Model\n(Benchmark)", "Betting Odds\n(Benchmark)"]
    scores = [0.692, 0.659, 0.649, None, None]
    colors = [GREEN, BLUE, BLUE, GOLD, GOLD]
    labels = ["0.692", "0.659", "0.649", "~0.75*", "~0.80*"]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK)

    bars = ax.barh(models, [s if s else 0.75 for s in scores], color=colors,
                   height=0.55, alpha=0.85, edgecolor="white", linewidth=0.3)

    # Hatching for benchmarks
    bars[3].set_hatch("//")
    bars[4].set_hatch("//")
    bars[3].set_alpha(0.5)
    bars[4].set_alpha(0.5)

    for i, (bar, label) in enumerate(zip(bars, labels)):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=12, fontweight="bold",
                color=GOLD if i >= 3 else "white")

    ax.axvline(0.5, color=GREY, linestyle="--", alpha=0.5, label="Random baseline (0.50)")
    ax.set_xlim(0.45, 0.87)
    ax.set_xlabel("ROC-AUC Score", fontsize=12)
    ax.set_title("UFC 328 Prediction Model Performance\nTraining: 1,517 real UFC fights — WW/MW/LHW (2016-2024) | TimeSeriesSplit CV",
                 fontsize=14, fontweight="bold", pad=15, color="white")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    note = "* Benchmark AUC estimated from published Elo / odds accuracy in literature"
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, color=GREY, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(OUTPUT_DIR, "viz_01_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2: Prediction Result Donut
# ─────────────────────────────────────────────────────────────────────────────
def viz_prediction_result():
    chimaev    = 85.5
    strickland = 14.5

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    sizes  = [chimaev, strickland]
    colors = [RED, BLUE]
    wedge_props = dict(width=0.45, edgecolor=DARK, linewidth=3)
    wedges, _ = ax.pie(sizes, colors=colors, wedgeprops=wedge_props,
                       startangle=90, counterclock=False)

    # Centre text
    ax.text(0, 0.12, "PREDICTED", ha="center", fontsize=11, color=GREY)
    ax.text(0, -0.08, "WINNER", ha="center", fontsize=11, color=GREY)
    ax.text(0, -0.38, "Khamzat\nChimaev", ha="center", fontsize=16,
            fontweight="bold", color=RED)

    # Labels
    ax.text(-0.65, 0.65, f"CHIMAEV\n{chimaev}%", ha="center", fontsize=15,
            fontweight="bold", color=RED)
    ax.text(0.65, -0.65, f"STRICKLAND\n{strickland}%", ha="center", fontsize=15,
            fontweight="bold", color=BLUE)

    ax.set_title("UFC 328 Win Probability\nKhamzat Chimaev vs Sean Strickland — May 9, 2026",
                 fontsize=13, fontweight="bold", pad=20, color="white")

    # Benchmark band
    comparison = [
        ("Our Best (LR)",  85.5, GREEN),
        ("Elo Model",      79.0, GOLD),
        ("Betting Odds",   84.6, GREY),
    ]
    y_pos = -1.25
    for label, val, col in comparison:
        ax.annotate(f"{label}: Chimaev {val}%", xy=(0, y_pos),
                    ha="center", fontsize=10, color=col)
        y_pos -= 0.18

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "viz_02_prediction_result.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3: Fighter Stats Head-to-Head
# ─────────────────────────────────────────────────────────────────────────────
def viz_stats_comparison():
    categories = ["SLpM", "Str Acc%", "Str Def%", "TD Avg\n/15min", "TD Def%", "Sub Avg\n/15min", "Finish\nRate%", "Win\nStreak"]
    chimaev    = [5.24,  60,  63,  5.29,  64,  1.80,  80,  12]
    strickland = [7.25,  61,  61,  0.43,  62,  0.10,  35,   1]

    # Normalise to 0-1 for display (each stat has a known max)
    maxvals = [10, 100, 100, 8, 100, 3, 100, 15]
    c_norm = [c/m*100 for c, m in zip(chimaev, maxvals)]
    s_norm = [s/m*100 for s, m in zip(strickland, maxvals)]

    x = np.arange(len(categories))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2.5, 1]})
    fig.patch.set_facecolor(DARK)

    ax = axes[0]
    ax.set_facecolor(DARK)
    bars1 = ax.bar(x - width/2, c_norm, width, label="Chimaev", color=RED,   alpha=0.85, edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + width/2, s_norm, width, label="Strickland", color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)

    # Annotate with raw values
    for bar, val in zip(bars1, chimaev):
        label = f"{val:.2f}" if isinstance(val, float) and val < 10 else f"{int(val)}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                label, ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")
    for bar, val in zip(bars2, strickland):
        label = f"{val:.2f}" if isinstance(val, float) and val < 10 else f"{int(val)}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                label, ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Normalised Score (% of max)", fontsize=11)
    ax.set_title("Fighter Stats Comparison\n(values shown above bars are raw ufcstats.com career averages)",
                 fontsize=13, fontweight="bold", color="white")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    # Right panel: record summary
    ax2 = axes[1]
    ax2.set_facecolor(DARK)
    ax2.axis("off")

    records = [
        ("CHIMAEV", "15-0", "12 KO/SUB", "MW Champion", RED),
        ("STRICKLAND", "30-7", "Mainly Dec", "Ex-Champion", BLUE),
    ]
    y = 0.85
    for name, record, finishes, title, col in records:
        ax2.text(0.5, y,       name,     ha="center", fontsize=14, fontweight="bold", color=col, transform=ax2.transAxes)
        ax2.text(0.5, y-0.07,  record,   ha="center", fontsize=18, fontweight="bold", color="white", transform=ax2.transAxes)
        ax2.text(0.5, y-0.13,  finishes, ha="center", fontsize=10, color=GREY, transform=ax2.transAxes)
        ax2.text(0.5, y-0.19,  title,    ha="center", fontsize=10, color=GOLD, transform=ax2.transAxes)
        y -= 0.38

    ax2.set_title("Records", fontsize=13, fontweight="bold", color="white")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "viz_03_stats_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4: Differential Features (what the model actually sees)
# ─────────────────────────────────────────────────────────────────────────────
def viz_differentials():
    diffs = {
        "SLpM (Strikes/min)":     5.24 - 7.25,
        "Str Accuracy %":         0.60 - 0.61,
        "Str Defence %":          0.63 - 0.61,
        "TD Avg /15min":          5.29 - 0.43,
        "TD Accuracy %":          0.55 - 0.42,
        "TD Defence %":           0.64 - 0.62,
        "Sub Avg /15min":         1.80 - 0.10,
        "Win Streak":             12   - 1,
        "Finish Rate %":          0.80 - 0.35,
        "Age (yrs, -=younger)":   32.3 - 35.2,
        "Days Since Last Fight":  266  - 77,
    }

    labels = list(diffs.keys())
    values = list(diffs.values())
    colors = [GREEN if v > 0 else RED for v in values]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3, height=0.6)

    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + (0.05 if val >= 0 else -0.05)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=10, fontweight="bold",
                color=GREEN if val >= 0 else RED)

    ax.axvline(0, color="white", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Chimaev - Strickland  (positive = Chimaev advantage)", fontsize=11)
    ax.set_title("Feature Differentials: What the Model Sees\n(Chimaev minus Strickland for each stat)",
                 fontsize=13, fontweight="bold", pad=15, color="white")

    green_patch = mpatches.Patch(color=GREEN, alpha=0.85, label="Chimaev advantage")
    red_patch   = mpatches.Patch(color=RED,   alpha=0.85, label="Strickland advantage")
    ax.legend(handles=[green_patch, red_patch], fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "viz_04_differentials.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 5: Data Pipeline Diagram
# ─────────────────────────────────────────────────────────────────────────────
def viz_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)

    steps = [
        (1.2,  2.5, "UFC.csv\n8,337 fights\n(Kaggle)", BLUE,   "DATA"),
        (3.8,  2.5, "Filter\nWW/MW/LHW\n1,517 fights", BLUE,   "FILTER"),
        (6.4,  2.5, "Feature\nEngineering\n15 differentials", GOLD,   "FEATURES"),
        (9.0,  2.5, "Logistic Reg.\nBest Model\nAUC 0.692", GREEN,  "MODEL"),
        (11.6, 2.5, "Prediction\nChimaev\n85.5%", RED,    "OUTPUT"),
    ]

    for i, (x, y, label, col, tag) in enumerate(steps):
        fancy = FancyBboxPatch((x-1.0, y-1.1), 2.0, 2.2,
                               boxstyle="round,pad=0.15",
                               facecolor=col, alpha=0.25,
                               edgecolor=col, linewidth=2)
        ax.add_patch(fancy)
        ax.text(x, y+0.6, tag,    ha="center", fontsize=8,  color=col,     fontweight="bold")
        ax.text(x, y-0.1, label,  ha="center", fontsize=9,  color="white", va="center",
                multialignment="center")

        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i+1][0]-1.05, y), xytext=(x+1.05, y),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.8))

    ax.text(7, 4.6, "UFC 328 ML Prediction Pipeline", ha="center",
            fontsize=15, fontweight="bold", color="white")
    ax.text(7, 0.3,
            "win_streak & finish_rate computed chronologically (no data leakage) | "
            "Best model: Logistic Regression ROC-AUC 0.692 | TimeSeriesSplit CV | 1,517 fights WW/MW/LHW",
            ha="center", fontsize=8.5, color=GREY, style="italic")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "viz_05_pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("UFC 328 - Step 5: Generating Visualizations")
    print("=" * 55)
    viz_model_comparison()
    viz_prediction_result()
    viz_stats_comparison()
    viz_differentials()
    viz_pipeline()
    print("\nAll visualizations saved to outputs/")
