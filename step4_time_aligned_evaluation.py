"""
Step 4 — Time-Aligned Evaluation: Apples-to-Apples Model Comparison
=====================================================================
Author  : Senior Quantitative Analyst
Purpose : Definitively prove whether adding News Sentiment features
          improves predictive performance over pure Technical Indicators,
          when evaluated under strictly identical timeframes.

Experimental Design
───────────────────
  DATA CONTEXT
  • 5 years of Technical Indicators (full BTC/USD OHLCV history).
  • ~35 days of News Sentiment (recent window only).

  UNIVERSAL TEST SET  ("The Arena")
  • The last TEST_DAYS rows of the dataset.
  • Every model is scored on this EXACT same slice — no exceptions.

  MODEL A   — Long-Term Baseline
    Train  : ALL historical rows before the test set (~5 years).
    Features: 21 Technical Indicators ONLY.

  MODEL AA  — Short-Term Baseline
    Train  : The NEWS_TRAIN_DAYS rows immediately before the test set.
    Features: 21 Technical Indicators ONLY.

  MODEL B   — Short-Term Hybrid
    Train  : The EXACT same NEWS_TRAIN_DAYS rows as Model AA.
    Features: 21 Technical + 27 Sentiment features (48 total).

  KEY COMPARISONS
  ───────────────
  Model A  vs Model AA  →  Does more training data compensate for recency?
  Model B  vs Model AA  →  Does Sentiment add a measurable edge when the
                           training window is held constant?

  CONTROLS
  • All models use RandomForestClassifier(max_depth=3, random_state=42).
  • StandardScaler fitted ONLY on each model's own training set.
  • Chronological integrity preserved — no future leakage.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent

# Import project utilities
from step1_data_ingestion import load_btc_price_data
from step2_feature_engineering import (
    add_sentiment_decay_features,
    add_technical_indicators,
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                              ║
# ╚══════════════════════════════════════════════════════════════╝

TEST_DAYS = 10          # Rows in the Universal Test Set
NEWS_TRAIN_DAYS = 25    # Rows for the short-term training window


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. FEATURE DEFINITIONS                                     ║
# ╚══════════════════════════════════════════════════════════════╝

TECHNICAL_FEATURES = [
    # Trend (8)
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    # Bollinger Bands (5)
    "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "BB_PctB",
    # Volatility (1)
    "ATR_14",
    # Volume (1)
    "OBV",
    # Price-derived (6)
    "Daily_Return", "Log_Return", "Price_Range", "Close_to_SMA20",
    "Volume", "Change_Pct",
]

SENTIMENT_FEATURES = [
    # Base sentiment (7)
    "Daily_Sentiment_Mean", "Daily_Sentiment_Std",
    "Positive_Count", "Negative_Count", "Neutral_Count",
    "Total_Articles", "Sentiment_Ratio",
    # Sentiment Mean decay — rolling windows (6)
    "Sent_Mean_MA1", "Sent_Mean_MA3", "Sent_Mean_MA5",
    "Sent_Mean_MA7", "Sent_Mean_MA14", "Sent_Mean_MA20",
    # Sentiment Ratio decay — rolling windows (6)
    "Sent_Ratio_MA1", "Sent_Ratio_MA3", "Sent_Ratio_MA5",
    "Sent_Ratio_MA7", "Sent_Ratio_MA14", "Sent_Ratio_MA20",
    # News volume decay (6)
    "Articles_MA1", "Articles_MA3", "Articles_MA5",
    "Articles_MA7", "Articles_MA14", "Articles_MA20",
    # Sentiment momentum (2)
    "Sent_Momentum_3d", "Sent_Momentum_7d",
]


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. DATA CONSTRUCTION                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def build_unified_dataset() -> pd.DataFrame:
    """
    Build one master DataFrame that contains:
      • 5-year BTC price data
      • 21 Technical indicator columns  (computed on full history)
      • 27 Sentiment + decay columns    (populated only for news window)
      • Binary target: 1 = next-day UP, 0 = next-day DOWN

    Sentiment columns are LEFT-JOINED, so most rows will have NaN
    sentiment — this is intentional.  Model A uses only tech columns
    and trains on ALL rows.  Models AA / B use only the recent rows
    where real sentiment data exists.
    """
    btc_path = PROJECT_ROOT / "Bitcoin Historical Data 5year.csv"
    merged_path = PROJECT_ROOT / "step1_merged_data.csv"

    if not btc_path.is_file():
        sys.exit(f"ERROR: {btc_path} not found.")
    if not merged_path.is_file():
        sys.exit(f"ERROR: {merged_path} not found. Run step1 first.")

    # ── Load 5-year price history ──
    price_df = load_btc_price_data(btc_path)
    print(f"  Raw price rows   : {len(price_df)}")

    # ── Compute technical indicators on FULL history ──
    tech_df = add_technical_indicators(price_df)

    # ── Load sentiment data (short recent window) ──
    merged = pd.read_csv(merged_path, index_col="Date", parse_dates=True).sort_index()

    sentiment_base_cols = [
        "Daily_Sentiment_Mean", "Daily_Sentiment_Std",
        "Positive_Count", "Negative_Count", "Neutral_Count",
        "Total_Articles", "Sentiment_Ratio",
    ]

    news_start = merged.index.min()
    news_end = merged.index.max()
    print(f"  News coverage     : {news_start.date()} to {news_end.date()} "
          f"({len(merged)} days)")

    # ── LEFT join: keep all price rows, add sentiment where available ──
    df = tech_df.join(merged[sentiment_base_cols], how="left")

    # ── Add sentiment decay features (rolling windows fill forward) ──
    # Fill sentiment NaN with 0 BEFORE computing rolling windows
    # so decay features are 0 outside the news window (not NaN).
    df[sentiment_base_cols] = df[sentiment_base_cols].fillna(0.0)
    df = add_sentiment_decay_features(df)

    # ── Drop rows with NaN from technical rolling windows (first ~50) ──
    df = df.dropna()

    # ── Create binary target: next-day direction ──
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
    df = df.dropna(subset=["Next_Close"])

    print(f"  Final dataset     : {len(df)} rows  x  {len(df.columns)} cols")
    print(f"  Date range        : {df.index.min().date()} to {df.index.max().date()}")

    return df, news_start


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. TRAIN / EVALUATE A SINGLE MODEL                        ║
# ╚══════════════════════════════════════════════════════════════╝

def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_label: str,
) -> dict:
    """
    Standardise → Train RandomForest(max_depth=3) → Predict → Score.
    Returns a dict with all metrics.
    """
    # ── Standardise (fit on train ONLY) ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Train ──
    clf = RandomForestClassifier(max_depth=3, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    # ── Predict ──
    y_pred = clf.predict(X_test_s)

    # ── Metrics ──
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    return {
        "label":     model_label,
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "y_pred":    y_pred,
        "model":     clf,
        "scaler":    scaler,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. PRINT BENCHMARKING TABLE                               ║
# ╚══════════════════════════════════════════════════════════════╝

def print_benchmarking_table(results: dict) -> None:
    """Print a clean, consolidated comparison table."""

    print("\n")
    print("┌" + "─" * 88 + "┐")
    print("│" + "CONSOLIDATED BENCHMARKING TABLE".center(88) + "│")
    print("├" + "─" * 88 + "┤")

    header = (f"│  {'Model':<42} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>10} {'F1-Score':>10}  │")
    print(header)
    print("├" + "─" * 88 + "┤")

    for key in ["A", "AA", "B"]:
        r = results[key]
        row = (f"│  {r['label']:<42} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
               f"{r['recall']:>10.4f} {r['f1']:>10.4f}  │")
        print(row)

    print("└" + "─" * 88 + "┘")


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. QUANTITATIVE VERDICT                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def print_quantitative_verdict(results: dict) -> None:
    """
    Auto-generate the final verdict by comparing F1-Scores.
    Answers two questions:
      1. Model A  vs Model AA  → Long-term vs short-term training data?
      2. Model B  vs Model AA  → Does sentiment add a measurable edge?
    """
    f1_a  = results["A"]["f1"]
    f1_aa = results["AA"]["f1"]
    f1_b  = results["B"]["f1"]

    acc_a  = results["A"]["accuracy"]
    acc_aa = results["AA"]["accuracy"]
    acc_b  = results["B"]["accuracy"]

    delta_a_vs_aa  = f1_a - f1_aa
    delta_b_vs_aa  = f1_b - f1_aa

    print("\n")
    print("╔" + "═" * 88 + "╗")
    print("║" + "QUANTITATIVE VERDICT".center(88) + "║")
    print("╠" + "═" * 88 + "╣")

    # ── Question 1: Long-term vs Short-term ──
    print("║" + "".center(88) + "║")
    print("║" + "  QUESTION 1: Does MORE training data help?".ljust(88) + "║")
    print("║" + f"  Model A  (5yr, Tech-Only)   →  F1 = {f1_a:.4f}   Acc = {acc_a:.4f}".ljust(88) + "║")
    print("║" + f"  Model AA (25-row, Tech-Only) →  F1 = {f1_aa:.4f}   Acc = {acc_aa:.4f}".ljust(88) + "║")
    print("║" + f"  Delta F1 (A − AA)           →  {delta_a_vs_aa:+.4f}".ljust(88) + "║")

    if abs(delta_a_vs_aa) < 0.01:
        verdict_1 = "INCONCLUSIVE — difference < 1% (within noise)"
    elif delta_a_vs_aa > 0:
        verdict_1 = f"YES — Long-term training WINS by {delta_a_vs_aa:+.4f} F1"
    else:
        verdict_1 = f"NO — Short-term training is BETTER by {abs(delta_a_vs_aa):.4f} F1"

    print("║" + f"  ▸ Answer: {verdict_1}".ljust(88) + "║")

    # ── Question 2: Sentiment edge ──
    print("║" + "".center(88) + "║")
    print("║" + "  QUESTION 2: Does SENTIMENT add a measurable edge?".ljust(88) + "║")
    print("║" + f"  Model AA (Tech-Only)        →  F1 = {f1_aa:.4f}   Acc = {acc_aa:.4f}".ljust(88) + "║")
    print("║" + f"  Model B  (Tech+Sentiment)   →  F1 = {f1_b:.4f}   Acc = {acc_b:.4f}".ljust(88) + "║")
    print("║" + f"  Delta F1 (B − AA)           →  {delta_b_vs_aa:+.4f}".ljust(88) + "║")

    if abs(delta_b_vs_aa) < 0.01:
        verdict_2 = "INCONCLUSIVE — difference < 1% (within noise)"
    elif delta_b_vs_aa > 0:
        verdict_2 = f"YES — Sentiment IMPROVES F1 by {delta_b_vs_aa:+.4f}"
    else:
        verdict_2 = f"NO — Sentiment HURTS F1 by {abs(delta_b_vs_aa):.4f}"

    print("║" + f"  ▸ Answer: {verdict_2}".ljust(88) + "║")

    # ── Final summary ──
    print("║" + "".center(88) + "║")
    print("╠" + "═" * 88 + "╣")

    best_key = max(results, key=lambda k: results[k]["f1"])
    best = results[best_key]
    print("║" + f"  BEST MODEL: {best['label']}".ljust(88) + "║")
    print("║" + f"  F1 = {best['f1']:.4f}  |  Accuracy = {best['accuracy']:.4f}  |  "
                f"Precision = {best['precision']:.4f}  |  Recall = {best['recall']:.4f}".ljust(88) + "║")
    print("║" + "".center(88) + "║")
    print("╚" + "═" * 88 + "╝")


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN EXECUTION                                            ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    print("=" * 90)
    print("STEP 4 — TIME-ALIGNED EVALUATION".center(90))
    print("Apples-to-Apples Model Comparison".center(90))
    print("=" * 90)

    # ──────────────────────────────────────────────────────────
    #  PHASE 1: Build the unified dataset
    # ──────────────────────────────────────────────────────────
    print("\n[PHASE 1] Building unified dataset (5yr price + sentiment)...")
    df, news_start_date = build_unified_dataset()

    total_rows = len(df)

    # Validate we have enough rows
    required_rows = TEST_DAYS + NEWS_TRAIN_DAYS
    if total_rows < required_rows:
        sys.exit(f"ERROR: Need at least {required_rows} rows, but only have {total_rows}.")

    # ──────────────────────────────────────────────────────────
    #  PHASE 2: Carve the Universal Test Set
    # ──────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print(f"[PHASE 2] Defining Universal Test Set (last {TEST_DAYS} rows)")
    print("─" * 90)

    # Indices
    test_start_idx = total_rows - TEST_DAYS
    news_train_start_idx = test_start_idx - NEWS_TRAIN_DAYS

    # ── Universal Test Set ──
    test_set = df.iloc[test_start_idx:]

    print(f"  Total rows in dataset : {total_rows}")
    print(f"  Test set rows         : {len(test_set)}")
    print(f"  Test set date range   : {test_set.index.min().date()} → "
          f"{test_set.index.max().date()}")
    print(f"  Test target dist      : UP={int(test_set['Target'].sum())}  "
          f"DOWN={int(len(test_set) - test_set['Target'].sum())}")

    y_test = test_set["Target"]

    # ──────────────────────────────────────────────────────────
    #  PHASE 3: Define training slices for each model
    # ──────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("[PHASE 3] Carving training sets")
    print("─" * 90)

    # Filter feature lists to columns that actually exist
    tech_features = [f for f in TECHNICAL_FEATURES if f in df.columns]
    sent_features = [f for f in SENTIMENT_FEATURES if f in df.columns]
    hybrid_features = tech_features + sent_features

    print(f"  Technical features    : {len(tech_features)}")
    print(f"  Sentiment features    : {len(sent_features)}")
    print(f"  Hybrid features       : {len(hybrid_features)}")

    # ── Model A: Long-term (all rows before test set) ──
    train_a = df.iloc[:test_start_idx]
    X_train_a = train_a[tech_features]
    y_train_a = train_a["Target"]

    print(f"\n  Model A  (Long-Term, Tech-Only):")
    print(f"    Train rows    : {len(X_train_a)}")
    print(f"    Date range    : {train_a.index.min().date()} → {train_a.index.max().date()}")
    print(f"    Target dist   : UP={int(y_train_a.sum())}  "
          f"DOWN={int(len(y_train_a) - y_train_a.sum())}")

    # ── Model AA: Short-term, tech only (25 rows before test set) ──
    train_aa = df.iloc[news_train_start_idx:test_start_idx]
    X_train_aa = train_aa[tech_features]
    y_train_aa = train_aa["Target"]

    print(f"\n  Model AA (Short-Term, Tech-Only):")
    print(f"    Train rows    : {len(X_train_aa)}")
    print(f"    Date range    : {train_aa.index.min().date()} → {train_aa.index.max().date()}")
    print(f"    Target dist   : UP={int(y_train_aa.sum())}  "
          f"DOWN={int(len(y_train_aa) - y_train_aa.sum())}")

    # ── Model B: Short-term, hybrid (SAME 25 rows, tech + sentiment) ──
    X_train_b = train_aa[hybrid_features]
    y_train_b = y_train_aa  # Exact same target as AA

    print(f"\n  Model B  (Short-Term, Tech+Sentiment):")
    print(f"    Train rows    : {len(X_train_b)}")
    print(f"    Date range    : {train_aa.index.min().date()} → {train_aa.index.max().date()}")
    print(f"    Features      : {len(hybrid_features)} "
          f"(+{len(sent_features)} sentiment vs AA)")

    # ── Test feature slices (from the same universal test set) ──
    X_test_tech   = test_set[tech_features]
    X_test_hybrid = test_set[hybrid_features]

    # ──────────────────────────────────────────────────────────
    #  PHASE 4: Train & evaluate all three models
    # ──────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("[PHASE 4] Training & evaluating on Universal Test Set")
    print("─" * 90)

    results = {}

    # ── MODEL A ──
    print("\n  ▶ Model A  — Long-Term Baseline (Tech-Only)")
    results["A"] = train_and_evaluate(
        X_train_a, X_test_tech, y_train_a, y_test,
        "Model A  — RF (5yr, Tech-Only)",
    )
    print(f"    Acc={results['A']['accuracy']:.4f}  "
          f"Prec={results['A']['precision']:.4f}  "
          f"Rec={results['A']['recall']:.4f}  "
          f"F1={results['A']['f1']:.4f}")

    # ── MODEL AA ──
    print(f"\n  ▶ Model AA — Short-Term Baseline ({NEWS_TRAIN_DAYS}-row, Tech-Only)")
    results["AA"] = train_and_evaluate(
        X_train_aa, X_test_tech, y_train_aa, y_test,
        f"Model AA — RF ({NEWS_TRAIN_DAYS}-row, Tech-Only)",
    )
    print(f"    Acc={results['AA']['accuracy']:.4f}  "
          f"Prec={results['AA']['precision']:.4f}  "
          f"Rec={results['AA']['recall']:.4f}  "
          f"F1={results['AA']['f1']:.4f}")

    # ── MODEL B ──
    print(f"\n  ▶ Model B  — Short-Term Hybrid ({NEWS_TRAIN_DAYS}-row, Tech+Sentiment)")
    results["B"] = train_and_evaluate(
        X_train_b, X_test_hybrid, y_train_b, y_test,
        f"Model B  — RF ({NEWS_TRAIN_DAYS}-row, Tech+Sent)",
    )
    print(f"    Acc={results['B']['accuracy']:.4f}  "
          f"Prec={results['B']['precision']:.4f}  "
          f"Rec={results['B']['recall']:.4f}  "
          f"F1={results['B']['f1']:.4f}")

    # ──────────────────────────────────────────────────────────
    #  PHASE 5: Classification reports (detailed)
    # ──────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("[PHASE 5] Detailed Classification Reports")
    print("─" * 90)

    for key in ["A", "AA", "B"]:
        r = results[key]
        print(f"\n  {r['label']}")
        print(classification_report(
            y_test, r["y_pred"],
            target_names=["Down (0)", "Up (1)"],
            zero_division=0,
        ))

    # ──────────────────────────────────────────────────────────
    #  PHASE 6: Consolidated benchmarking table
    # ──────────────────────────────────────────────────────────
    print_benchmarking_table(results)

    # ──────────────────────────────────────────────────────────
    #  PHASE 7: Quantitative verdict
    # ──────────────────────────────────────────────────────────
    print_quantitative_verdict(results)

    print("\n" + "=" * 90)
    print("STEP 4 — TIME-ALIGNED EVALUATION COMPLETE".center(90))
    print("=" * 90)
