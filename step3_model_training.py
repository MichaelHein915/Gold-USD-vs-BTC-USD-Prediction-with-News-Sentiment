"""
Step 3: Model Training (Baseline vs. Hybrid)
==============================================
- Target: next-day price direction (UP=1, DOWN=0)
- Time-series split: 80% train / 20% test
- Model A: Technical indicators only
- Model B: Hybrid (Technical + Sentiment)
- Algorithms: XGBoost + Random Forest
"""
from __future__ import annotations

import pickle
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

from step1_data_ingestion import load_btc_price_data
from step2_feature_engineering import (
    add_sentiment_decay_features,
    add_technical_indicators,
    handle_missing_values,
)


# ============================================================
# 3A. TARGET VARIABLE
# ============================================================

def create_target(df):
    """
    Define binary target:
        target = 1 if next_day_close > current_close (UP)
        target = 0 otherwise (DOWN)
    """
    df = df.copy()
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
    df = df.dropna(subset=["Next_Close"])

    total = len(df)
    up_count = int((df["Target"] == 1).sum())
    down_count = total - up_count

    print(f"  Target distribution:")
    print(f"    Up (1):   {up_count} days ({up_count / total:.1%})")
    print(f"    Down (0): {down_count} days ({down_count / total:.1%})")

    return df


# ============================================================
# 3B. BUILD TRAINING DATAFRAME
# ============================================================

def build_training_dataframe():
    """
    Build the full training DataFrame:
    1. Load 5-year BTC price history
    2. Add technical indicators (full history for proper lookback)
    3. Merge with Step 1 sentiment data (forward-fill + fill zeros)
    4. Add sentiment decay features
    5. Handle missing values
    """
    merged_path = PROJECT_ROOT / "step1_merged_data.csv"
    btc_path = PROJECT_ROOT / "Bitcoin Historical Data 5year.csv"

    if not merged_path.is_file():
        print(f"ERROR: {merged_path} not found. Run Phase 1 first.", file=sys.stderr)
        sys.exit(1)
    if not btc_path.is_file():
        print(f"ERROR: {btc_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Load sentiment data from Step 1
    merged = pd.read_csv(merged_path, index_col="Date", parse_dates=True).sort_index()

    sentiment_cols = [
        "Daily_Sentiment_Mean", "Daily_Sentiment_Std",
        "Positive_Count", "Negative_Count", "Neutral_Count",
        "Total_Articles", "Sentiment_Ratio",
    ]
    for col in sentiment_cols:
        if col not in merged.columns:
            print(f"ERROR: missing column '{col}' in merged data", file=sys.stderr)
            sys.exit(1)

    # Load full price history and compute technical indicators
    price_df = load_btc_price_data(btc_path)
    price_with_tech = add_technical_indicators(price_df)

    # Left join: keep all price rows, add sentiment where available
    df = price_with_tech.join(merged[sentiment_cols], how="left")

    # Create linear sentiment ramp for pre-news period
    # (avoids all-zero sentiment in training data)
    first_news_date = merged.index.min()
    first_news_values = merged.loc[first_news_date]
    pre_news_mask = df.index < first_news_date
    num_pre_news = int(pre_news_mask.sum())

    if num_pre_news > 0:
        for col in sentiment_cols:
            target_value = float(first_news_values[col])
            ramp_values = np.linspace(0.0, target_value, num_pre_news + 2)[1:-1]
            df.loc[pre_news_mask, col] = ramp_values
        print(f"  Sentiment ramp: {num_pre_news} rows before {first_news_date.date()}")

    # Forward-fill remaining gaps, then fill any leftover NaN with 0
    df[sentiment_cols] = df[sentiment_cols].ffill()
    df[sentiment_cols] = df[sentiment_cols].fillna(0.0)

    # Add sentiment decay features
    df = add_sentiment_decay_features(df)

    # Handle missing values
    df = handle_missing_values(df)

    print(f"  Built training frame: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ============================================================
# 3C. FEATURE SETS
# ============================================================

def get_feature_sets(df):
    """
    Define feature groups:
    - Technical features (21): RSI, MACD, SMA, EMA, Bollinger, ATR, OBV, etc.
    - Sentiment features (27): base sentiment + decay windows + momentum
    - Hybrid features: all combined
    """
    technical_features = [
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "BB_PctB",
        "ATR_14", "OBV",
        "Daily_Return", "Log_Return", "Price_Range", "Close_to_SMA20",
        "Volume", "Change_Pct",
    ]

    sentiment_features = [
        # Base sentiment
        "Daily_Sentiment_Mean", "Daily_Sentiment_Std",
        "Positive_Count", "Negative_Count", "Neutral_Count",
        "Total_Articles", "Sentiment_Ratio",
        # Sentiment decay (rolling windows: 1, 3, 5, 7, 14, 20 days)
        "Sent_Mean_MA1", "Sent_Mean_MA3", "Sent_Mean_MA5",
        "Sent_Mean_MA7", "Sent_Mean_MA14", "Sent_Mean_MA20",
        "Sent_Ratio_MA1", "Sent_Ratio_MA3", "Sent_Ratio_MA5",
        "Sent_Ratio_MA7", "Sent_Ratio_MA14", "Sent_Ratio_MA20",
        # News volume decay
        "Articles_MA1", "Articles_MA3", "Articles_MA5",
        "Articles_MA7", "Articles_MA14", "Articles_MA20",
        # Sentiment momentum
        "Sent_Momentum_3d", "Sent_Momentum_7d",
    ]

    # Only keep features that exist in the DataFrame
    technical_features = [f for f in technical_features if f in df.columns]
    sentiment_features = [f for f in sentiment_features if f in df.columns]
    hybrid_features = technical_features + sentiment_features

    print(f"  Technical features: {len(technical_features)}")
    print(f"  Sentiment features: {len(sentiment_features)}")
    print(f"  Hybrid features:    {len(hybrid_features)}")

    return technical_features, sentiment_features, hybrid_features


# ============================================================
# 3D. CHRONOLOGICAL SPLIT (80/20)
# ============================================================

def chronological_split(df, features, target_col="Target", test_ratio=0.2):
    """
    Time-series split preserving chronological order.
    80% train / 20% test (no shuffling).
    """
    total = len(df)
    if total < 2:
        raise ValueError("Need at least 2 rows for splitting.")

    split_idx = max(1, int(total * (1 - test_ratio)))
    if split_idx >= total:
        split_idx = total - 1

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    print(f"  Train: {len(X_train)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  Test:  {len(X_test)} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")
    print(f"  Train target: Up={int(y_train.sum())}, Down={int(len(y_train) - y_train.sum())}")
    print(f"  Test target:  Up={int(y_test.sum())}, Down={int(len(y_test) - y_test.sum())}")

    return X_train, X_test, y_train, y_test, train_df, test_df


# ============================================================
# 3E. TRAIN AND EVALUATE
# ============================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model):
    """
    Train a model with StandardScaler and evaluate on test set.
    Returns dict with model, scaler, and all metrics.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  {model_name} Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Down (0)", "Up (1)"], zero_division=0))

    return {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: MODEL TRAINING (BASELINE vs. HYBRID)")
    print("=" * 60)

    # Build full training DataFrame
    print("\n  Building features (full history + sentiment)...")
    df = build_training_dataframe()
    print(f"\n  Dataset before target: {df.shape}")

    # Create target variable
    print("\n" + "-" * 60)
    print("[3A] Creating target variable...")
    df = create_target(df)
    print(f"  After target: {df.shape}")

    if len(df) < 10:
        print("\n  WARNING: Very few samples — metrics will be noisy.", file=sys.stderr)

    # Define feature sets
    print("\n" + "-" * 60)
    print("[3B] Defining feature sets...")
    technical_features, sentiment_features, hybrid_features = get_feature_sets(df)

    # Chronological split
    print("\n" + "-" * 60)
    print("[3C] Chronological split 80/20...")
    X_train_full, X_test_full, y_train, y_test, train_df, test_df = \
        chronological_split(df, hybrid_features, test_ratio=0.2)

    X_train_tech = X_train_full[technical_features]
    X_test_tech = X_test_full[technical_features]

    # Check for XGBoost
    print("\n" + "-" * 60)
    print("[3D] Training models...")
    try:
        from xgboost import XGBClassifier
        xgb_available = True
        print("  XGBoost: available")
    except ImportError:
        xgb_available = False
        print("  XGBoost: not installed (using Random Forest only)")

    results = {}

    # ── MODEL A: Technical Only ──
    print("\n" + "=" * 40)
    print("MODEL A: TECHNICAL ONLY")
    print("=" * 40)

    if xgb_available:
        results["A_XGB"] = train_and_evaluate(
            X_train_tech, X_test_tech, y_train, y_test,
            "Model A - XGBoost (Technical Only)",
            XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, eval_metric="logloss", verbosity=0
            ),
        )

    results["A_RF"] = train_and_evaluate(
        X_train_tech, X_test_tech, y_train, y_test,
        "Model A - Random Forest (Technical Only)",
        RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_split=3,
            random_state=42, n_jobs=-1
        ),
    )

    # ── MODEL B: Hybrid (Technical + Sentiment) ──
    print("\n" + "=" * 40)
    print("MODEL B: HYBRID (TECHNICAL + SENTIMENT)")
    print("=" * 40)

    if xgb_available:
        results["B_XGB"] = train_and_evaluate(
            X_train_full, X_test_full, y_train, y_test,
            "Model B - XGBoost (Technical + Sentiment)",
            XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, eval_metric="logloss", verbosity=0
            ),
        )

    results["B_RF"] = train_and_evaluate(
        X_train_full, X_test_full, y_train, y_test,
        "Model B - Random Forest (Technical + Sentiment)",
        RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_split=3,
            random_state=42, n_jobs=-1
        ),
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, res in results.items():
        print(f"  {key}: acc={res['accuracy']:.4f}  prec={res['precision']:.4f}  "
              f"rec={res['recall']:.4f}  f1={res['f1']:.4f}")

    # Save results
    results_path = PROJECT_ROOT / "step3_results.pkl"
    data_path = PROJECT_ROOT / "step3_data.pkl"

    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    with open(data_path, "wb") as f:
        pickle.dump({
            "technical_features": technical_features,
            "sentiment_features": sentiment_features,
            "hybrid_features": hybrid_features,
            "X_test_full": X_test_full,
            "X_test_tech": X_test_tech,
            "y_test": y_test,
            "test_df": test_df,
        }, f)

    print(f"\n  Saved: {results_path}")
    print(f"  Saved: {data_path}")
    print("\nSTEP 3 COMPLETE!")
