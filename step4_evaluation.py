"""
Step 4: Benchmarking & Evaluation
==================================
- 4A: Compare Model A (Baseline) vs Model B (Hybrid) — Accuracy, Precision, Recall
- 4B: Mathematical proof of which model wins
- 4C: Feature importance analysis (identify best sentiment window)
- 4D: Visualizations (dashboard + sentiment decay deep-dive)
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD SAVED MODELS AND DATA
# ============================================================

def load_step3_results():
    """Load pickled model results and data from Step 3."""
    results_path = PROJECT_ROOT / "step3_results.pkl"
    data_path = PROJECT_ROOT / "step3_data.pkl"

    if not results_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(
            "Step 3 output not found. Run step3_model_training.py first."
        )

    with open(results_path, "rb") as f:
        results = pickle.load(f)
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    return results, data


# ============================================================
# 4A. MODEL PERFORMANCE COMPARISON
# ============================================================

def compare_models(results):
    """Build and print a comparison table of all trained models."""
    print("\n" + "=" * 70)
    print("4A. MODEL PERFORMANCE COMPARISON")
    print("=" * 70)

    rows = []
    for key, res in results.items():
        rows.append({
            "Model": res["model_name"],
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1 Score": res["f1"],
        })

    comp_df = pd.DataFrame(rows)
    print("\n  Full Comparison Table:")
    print(comp_df.to_string(index=False))

    return comp_df


# ============================================================
# 4B. MATHEMATICAL COMPARISON: BASELINE vs HYBRID
# ============================================================

def mathematical_comparison(results):
    """
    Compare best baseline (A_*) vs best hybrid (B_*) on all 4 metrics.
    Returns best_baseline, best_hybrid, and number of hybrid wins.
    """
    print("\n" + "=" * 70)
    print("4B. MATHEMATICAL COMPARISON: BASELINE vs. HYBRID")
    print("=" * 70)

    # Separate baseline and hybrid models
    baseline_models = {k: v for k, v in results.items() if k.startswith("A_")}
    hybrid_models = {k: v for k, v in results.items() if k.startswith("B_")}

    # Find best by F1, then accuracy as tiebreaker
    best_baseline_key = max(
        baseline_models.keys(),
        key=lambda k: (baseline_models[k]["f1"], baseline_models[k]["accuracy"]),
    )
    best_hybrid_key = max(
        hybrid_models.keys(),
        key=lambda k: (hybrid_models[k]["f1"], hybrid_models[k]["accuracy"]),
    )

    best_baseline = baseline_models[best_baseline_key]
    best_hybrid = hybrid_models[best_hybrid_key]

    print(f"\n  Best Baseline: {best_baseline['model_name']}")
    print(f"  Best Hybrid:   {best_hybrid['model_name']}")

    # Head-to-head comparison
    metrics = ["accuracy", "precision", "recall", "f1"]
    print(f"\n  {'Metric':<12} {'Baseline':>10} {'Hybrid':>10} {'Delta':>10} {'Winner':>18}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 18}")

    hybrid_wins = 0
    for m in metrics:
        b_val = best_baseline[m]
        h_val = best_hybrid[m]
        delta = h_val - b_val
        winner = "HYBRID" if delta > 0 else ("BASELINE" if delta < 0 else "TIE")
        if delta > 0:
            hybrid_wins += 1
        print(f"  {m.capitalize():<12} {b_val:>10.4f} {h_val:>10.4f} {delta:>+10.4f} {winner:>18}")

    # Verdict
    print(f"\n  VERDICT: Hybrid Model wins on {hybrid_wins}/4 metrics.")
    if hybrid_wins >= 3:
        print("  CONCLUSION: Model B (Hybrid) OUTPERFORMS Model A (Baseline).")
        print("  Adding sentiment decay features IMPROVES prediction accuracy.")
    elif hybrid_wins >= 2:
        print("  CONCLUSION: Model B (Hybrid) shows MARGINAL improvement over Baseline.")
        print("  Sentiment features add some predictive value.")
    else:
        print("  CONCLUSION: Comparison inconclusive for this data split.")
        print("  More historical news data would provide a stronger comparison.")

    return best_baseline, best_hybrid, hybrid_wins


# ============================================================
# 4C. FEATURE IMPORTANCE ANALYSIS
# ============================================================

def categorize_feature(name, sentiment_features):
    """Categorize a feature into its group for analysis."""
    if name in sentiment_features:
        if "Sent_Mean_MA" in name or "Sent_Ratio_MA" in name:
            return "Sentiment Decay"
        elif "Articles_MA" in name:
            return "News Volume"
        elif "Momentum" in name:
            return "Sentiment Momentum"
        else:
            return "Sentiment Base"
    return "Technical"


def analyze_feature_importance(results, data):
    """
    Extract feature importances from Hybrid Random Forest model.
    Identify which sentiment lookback window has the highest impact.
    """
    print("\n" + "=" * 70)
    print("4C. FEATURE IMPORTANCE ANALYSIS (Model B - Hybrid Random Forest)")
    print("=" * 70)

    hybrid_features = data["hybrid_features"]
    sentiment_features = data["sentiment_features"]

    # Use Random Forest for interpretable feature importance
    hybrid_rf_model = results["B_RF"]["model"]
    importances = hybrid_rf_model.feature_importances_

    # Build feature importance DataFrame
    feat_imp_df = pd.DataFrame({
        "Feature": hybrid_features,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    feat_imp_df["Category"] = feat_imp_df["Feature"].apply(
        lambda name: categorize_feature(name, sentiment_features)
    )

    # Print top 20
    print("\n  Top 20 Features by Importance:")
    print(f"  {'Rank':<5} {'Feature':<25} {'Importance':<12} {'Category':<20}")
    print(f"  {'-' * 5} {'-' * 25} {'-' * 12} {'-' * 20}")

    for rank, (_, row) in enumerate(feat_imp_df.head(20).iterrows(), 1):
        print(f"  {rank:<5} {row['Feature']:<25} {row['Importance']:<12.4f} {row['Category']:<20}")

    # Aggregate by category
    cat_importance = feat_imp_df.groupby("Category")["Importance"].sum().sort_values(ascending=False)
    total_importance = cat_importance.sum()

    print(f"\n  Importance by Category:")
    for cat, imp in cat_importance.items():
        print(f"    {cat:<25} {imp:.4f} ({imp / total_importance:.1%})")

    # Check if sentiment momentum is zero (expected for sparse data)
    mom_imp = float(cat_importance.get("Sentiment Momentum", 0.0))
    if mom_imp < 1e-8:
        merged_path = PROJECT_ROOT / "step1_merged_data.csv"
        if merged_path.is_file():
            mdf = pd.read_csv(merged_path, index_col="Date", parse_dates=True)
            first_news = pd.Timestamp(mdf.index.min())
            test_start = pd.Timestamp(data["test_df"].index.min())
            train_end = test_start - pd.Timedelta(days=1)
            if train_end < first_news:
                print(
                    f"\n  NOTE: Sentiment Momentum importance is ~0 because the train window "
                    f"ends ({train_end.date()}) before first news ({first_news.date()}). "
                    f"Pre-news rows use a linear ramp, so momentum is nearly constant. "
                    f"This is expected for this data split."
                )

    # Best sentiment decay window
    decay_features = feat_imp_df[feat_imp_df["Category"] == "Sentiment Decay"]
    if not decay_features.empty:
        best_decay = decay_features.iloc[0]
        print(f"\n  BEST Sentiment Lookback Window: {best_decay['Feature']}")
        print(f"    Importance: {best_decay['Importance']:.4f}")

        print(f"\n  Sentiment Decay Feature Ranking:")
        for _, row in decay_features.iterrows():
            bar = "#" * int(row["Importance"] * 200)
            print(f"    {row['Feature']:<22} {row['Importance']:.4f}  {bar}")

    return feat_imp_df, cat_importance


# ============================================================
# 4D. VISUALIZATIONS
# ============================================================

# Color scheme
COLORS = {
    "Technical": "#2196F3",           # Blue
    "Sentiment Base": "#FF9800",      # Orange
    "Sentiment Decay": "#4CAF50",     # Green
    "News Volume": "#9C27B0",         # Purple
    "Sentiment Momentum": "#F44336",  # Red
    "Baseline": "#78909C",            # Gray-Blue
    "Hybrid": "#00BCD4",              # Cyan
}

METRIC_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]


def plot_dashboard(results, data, best_baseline, best_hybrid, feat_imp_df, cat_importance):
    """Generate the full model evaluation dashboard (6 subplots)."""
    print("\n" + "=" * 70)
    print("4D. GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        "BTC/USD Hybrid Sentiment-Technical Trading System\nModel Evaluation Dashboard",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # ── Plot 1: Model Comparison Bar Chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_model_comparison(ax1, results)

    # ── Plot 2: Baseline vs Hybrid Head-to-Head ──
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_head_to_head(ax2, best_baseline, best_hybrid)

    # ── Plot 3: Top 20 Feature Importances ──
    ax3 = fig.add_subplot(gs[1, :])
    _plot_feature_importance(ax3, feat_imp_df)

    # ── Plot 4: Importance by Category (Pie Chart) ──
    ax4 = fig.add_subplot(gs[2, 0])
    _plot_category_pie(ax4, cat_importance)

    # ── Plot 5: Sentiment Decay Window Comparison ──
    ax5 = fig.add_subplot(gs[2, 1])
    _plot_decay_windows(ax5, feat_imp_df)

    # ── Plot 6: Prediction Accuracy Timeline ──
    ax6 = fig.add_subplot(gs[3, :])
    _plot_prediction_timeline(ax6, results, data)

    # Save
    path = OUTPUT_DIR / "btc_model_evaluation_dashboard.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  Dashboard saved: {path.name}")

    # ── Figure 2: Sentiment Decay Deep Dive ──
    _plot_sentiment_deep_dive(feat_imp_df, cat_importance)


def _plot_model_comparison(ax, results):
    """Bar chart comparing all models on all metrics."""
    chart_order = [
        ("A_XGB", "A-XGB\n(Tech)"), ("A_RF", "A-RF\n(Tech)"),
        ("B_XGB", "B-XGB\n(Hybrid)"), ("B_RF", "B-RF\n(Hybrid)"),
    ]
    chart_pairs = [(k, lbl) for k, lbl in chart_order if k in results]
    model_keys = [k for k, _ in chart_pairs]
    model_labels = [lbl for _, lbl in chart_pairs]
    metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(model_labels))
    width = 0.18

    for i, metric in enumerate(metrics):
        vals = [results[k][metric] for k in model_keys]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                      color=METRIC_COLORS[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)


def _plot_head_to_head(ax, best_baseline, best_hybrid):
    """Head-to-head comparison of best baseline vs best hybrid."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    baseline_vals = [best_baseline[m] for m in metrics]
    hybrid_vals = [best_hybrid[m] for m in metrics]

    x = np.arange(len(metric_names))
    bars_b = ax.bar(x - 0.2, baseline_vals, 0.35, label="Baseline (Tech Only)",
                    color=COLORS["Baseline"], edgecolor="white")
    bars_h = ax.bar(x + 0.2, hybrid_vals, 0.35, label="Hybrid (Tech + Sent)",
                    color=COLORS["Hybrid"], edgecolor="white")

    for bar, val in zip(bars_b, baseline_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars_h, hybrid_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Head-to-Head: Best Baseline vs Best Hybrid", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)


def _plot_feature_importance(ax, feat_imp_df):
    """Top 20 feature importances, color-coded by category."""
    top20 = feat_imp_df.head(20).iloc[::-1]
    colors = [COLORS.get(cat, "gray") for cat in top20["Category"]]

    ax.barh(range(len(top20)), top20["Importance"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["Feature"], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 20 Feature Importances (Model B - Hybrid Random Forest)",
                 fontweight="bold", fontsize=12)

    legend_patches = [mpatches.Patch(color=c, label=cat) for cat, c in COLORS.items()
                      if cat not in ("Baseline", "Hybrid")]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.3)

    for i, (_, row) in enumerate(top20.iterrows()):
        ax.text(row["Importance"] + 0.001, i, f'{row["Importance"]:.4f}', va="center", fontsize=8)


def _plot_category_pie(ax, cat_importance):
    """Pie chart of feature importance by category."""
    cat_colors = [COLORS.get(cat, "gray") for cat in cat_importance.index]
    wedges, texts, autotexts = ax.pie(
        cat_importance.values, labels=cat_importance.index,
        autopct="%1.1f%%", colors=cat_colors,
        startangle=90, textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    ax.set_title("Feature Importance by Category", fontweight="bold", fontsize=12)


def _plot_decay_windows(ax, feat_imp_df):
    """Bar chart comparing sentiment decay lookback windows."""
    sent_mean_mas = feat_imp_df[feat_imp_df["Feature"].str.startswith("Sent_Mean_MA")]
    sent_ratio_mas = feat_imp_df[feat_imp_df["Feature"].str.startswith("Sent_Ratio_MA")]

    if sent_mean_mas.empty:
        ax.text(0.5, 0.5, "No sentiment decay data", ha="center", va="center")
        return

    windows = [int(f.split("MA")[1]) for f in sent_mean_mas["Feature"]]
    mean_imps = sent_mean_mas["Importance"].values
    ratio_imps = (sent_ratio_mas["Importance"].values
                  if not sent_ratio_mas.empty else np.zeros(len(windows)))

    x = np.arange(len(windows))
    ax.bar(x - 0.2, mean_imps, 0.35, label="Sentiment Mean",
           color=COLORS["Sentiment Decay"], edgecolor="white")
    ax.bar(x + 0.2, ratio_imps, 0.35, label="Sentiment Ratio",
           color=COLORS["Sentiment Base"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}-day" for w in windows])
    ax.set_xlabel("Lookback Window")
    ax.set_ylabel("Feature Importance")
    ax.set_title("Sentiment Decay: Which Lookback Window Matters Most?",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Highlight best window
    best_idx = np.argmax(mean_imps + ratio_imps)
    ax.annotate(
        f"Best: {windows[best_idx]}-day",
        xy=(best_idx, max(mean_imps[best_idx], ratio_imps[best_idx])),
        xytext=(best_idx + 0.5, max(mean_imps[best_idx], ratio_imps[best_idx]) + 0.01),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=11, fontweight="bold", color="red",
    )


def _plot_prediction_timeline(ax, results, data):
    """Scatter plot showing actual vs predicted direction over time."""
    test_dates = data["test_df"].index
    y_true = data["y_test"].values
    y_pred_hybrid = results["B_RF"]["y_pred"]
    y_pred_baseline = results["A_RF"]["y_pred"]

    correct_hybrid = (y_true == y_pred_hybrid).astype(int)
    correct_baseline = (y_true == y_pred_baseline).astype(int)

    for i in range(len(test_dates)):
        # Actual direction
        color_actual = "#4CAF50" if y_true[i] == 1 else "#F44336"
        ax.scatter(i, 2.0, color=color_actual, s=200, zorder=5, marker="s")

        # Baseline prediction
        color_base = "#4CAF50" if y_pred_baseline[i] == 1 else "#F44336"
        border_base = "gold" if correct_baseline[i] else "black"
        lw_base = 3 if correct_baseline[i] else 1
        ax.scatter(i, 1.0, color=color_base, s=200, zorder=5, marker="o",
                   edgecolors=border_base, linewidths=lw_base)

        # Hybrid prediction
        color_hyb = "#4CAF50" if y_pred_hybrid[i] == 1 else "#F44336"
        border_hyb = "gold" if correct_hybrid[i] else "black"
        lw_hyb = 3 if correct_hybrid[i] else 1
        ax.scatter(i, 0.0, color=color_hyb, s=200, zorder=5, marker="D",
                   edgecolors=border_hyb, linewidths=lw_hyb)

    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["Hybrid\nPrediction", "Baseline\nPrediction", "Actual\nDirection"], fontsize=10)

    date_labels = [d.strftime("%m/%d") for d in test_dates]
    tick_stride = max(1, len(test_dates) // 40)
    ax.set_xticks(np.arange(len(test_dates))[::tick_stride])
    ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), tick_stride)],
                       rotation=45, fontsize=9)
    ax.set_title("Prediction Timeline: Actual vs Predicted (Green=Up, Red=Down, Gold Border=Correct)",
                 fontweight="bold", fontsize=12)
    ax.set_xlim(-0.5, len(test_dates) - 0.5)
    ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50", markersize=12, label="Up Day"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F44336", markersize=12, label="Down Day"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=12,
               markeredgecolor="gold", markeredgewidth=2, label="Correct Prediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)


def _plot_sentiment_deep_dive(feat_imp_df, cat_importance):
    """Separate figure: sentiment feature analysis."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Sentiment Decay Analysis: Impact of Lookback Windows on BTC Price Prediction",
                 fontsize=14, fontweight="bold")

    # Left: All sentiment features ranked
    sent_categories = ["Sentiment Decay", "Sentiment Base", "News Volume", "Sentiment Momentum"]
    sent_features_only = feat_imp_df[feat_imp_df["Category"].isin(sent_categories)].iloc[::-1]

    colors = [COLORS.get(cat, "gray") for cat in sent_features_only["Category"]]
    ax_left.barh(range(len(sent_features_only)), sent_features_only["Importance"],
                 color=colors, edgecolor="white", height=0.7)
    ax_left.set_yticks(range(len(sent_features_only)))
    ax_left.set_yticklabels(sent_features_only["Feature"], fontsize=8)
    ax_left.set_xlabel("Feature Importance")
    ax_left.set_title("All Sentiment Features Ranked", fontweight="bold")
    ax_left.grid(axis="x", alpha=0.3)

    # Right: Technical vs Sentiment aggregate
    tech_total = feat_imp_df[feat_imp_df["Category"] == "Technical"]["Importance"].sum()
    sent_total = feat_imp_df[feat_imp_df["Category"] != "Technical"]["Importance"].sum()
    grand_total = tech_total + sent_total

    bars = ax_right.bar(
        ["Technical\nIndicators", "Sentiment\n(All Combined)"],
        [tech_total, sent_total],
        color=[COLORS["Technical"], COLORS["Sentiment Decay"]],
        edgecolor="white", width=0.5,
    )
    for bar, val in zip(bars, [tech_total, sent_total]):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}\n({val / grand_total:.1%})",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax_right.set_ylabel("Total Feature Importance")
    ax_right.set_title("Technical vs Sentiment: Aggregate Importance", fontweight="bold")
    ax_right.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "btc_sentiment_decay_analysis.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  Sentiment analysis saved: {path.name}")


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(results, data, best_baseline, best_hybrid,
                        hybrid_wins, feat_imp_df, cat_importance):
    """Print the final project summary."""
    technical_features = data["technical_features"]
    sentiment_features = data["sentiment_features"]
    hybrid_features = data["hybrid_features"]
    test_rows = len(data["test_df"])

    tech_total = feat_imp_df[feat_imp_df["Category"] == "Technical"]["Importance"].sum()
    sent_total = feat_imp_df[feat_imp_df["Category"] != "Technical"]["Importance"].sum()
    grand_total = tech_total + sent_total

    decay_features = feat_imp_df[feat_imp_df["Category"] == "Sentiment Decay"]
    best_window = decay_features.iloc[0]["Feature"] if not decay_features.empty else "N/A"

    print("\n" + "=" * 70)
    print("FINAL PROJECT SUMMARY")
    print("=" * 70)
    print(f"""
  DATASET:
    Test period rows:     {test_rows}
    Hybrid feature count: {len(hybrid_features)} ({len(technical_features)} technical + {len(sentiment_features)} sentiment)

  MODELS TRAINED:
    Model A (Baseline):   XGBoost & Random Forest — Technical indicators only
    Model B (Hybrid):     XGBoost & Random Forest — Technical + Sentiment Decay

  KEY FINDINGS:
    Best Baseline:  {best_baseline['model_name']}
      Accuracy={best_baseline['accuracy']:.1%}, F1={best_baseline['f1']:.4f}

    Best Hybrid:    {best_hybrid['model_name']}
      Accuracy={best_hybrid['accuracy']:.1%}, F1={best_hybrid['f1']:.4f}

    Hybrid wins on {hybrid_wins}/4 metrics (Accuracy, Precision, Recall, F1)

  SENTIMENT IMPACT:
    Technical importance:  {tech_total:.3f} ({tech_total / grand_total:.1%})
    Sentiment importance:  {sent_total:.3f} ({sent_total / grand_total:.1%})
    Best decay window:     {best_window}

  OUTPUT FILES:
    btc_model_evaluation_dashboard.png  — Full evaluation dashboard
    btc_sentiment_decay_analysis.png    — Sentiment decay deep-dive

  NOTE: Hybrid value depends on time-varying news; sparse daily news
  reduces sentiment signal until Phase 1 covers more distinct dates.
""")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 4: BENCHMARKING & EVALUATION")
    print("=" * 70)

    # Load data
    results, data = load_step3_results()
    y_test = data["y_test"]

    # 4A: Model comparison
    comp_df = compare_models(results)

    # 4B: Mathematical comparison
    best_baseline, best_hybrid, hybrid_wins = mathematical_comparison(results)

    # 4C: Feature importance
    feat_imp_df, cat_importance = analyze_feature_importance(results, data)

    # 4D: Visualizations
    plot_dashboard(results, data, best_baseline, best_hybrid, feat_imp_df, cat_importance)

    # Final summary
    print_final_summary(results, data, best_baseline, best_hybrid,
                        hybrid_wins, feat_imp_df, cat_importance)

    print("=" * 70)
    print("STEP 4 COMPLETE!")
    print("=" * 70)
