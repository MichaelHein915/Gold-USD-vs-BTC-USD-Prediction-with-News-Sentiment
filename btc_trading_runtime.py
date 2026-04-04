"""
Phases 5-8: Trading Runtime
============================
- Phase 5: Trading Logic (BUY / SELL / HOLD signals)
- Phase 6: Actionable Output (prediction report)
- Phase 7: Paper Trading Simulation
- Phase 8: Live Prediction Function (predict_latest)

Stop-Loss : Take-Profit = 1 : 2  (SL = 2%, TP = 4%)
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Risk Parameters (1:2 ratio) ──────────────────────────────
STOP_LOSS_PCT = 0.02   # -2%
TAKE_PROFIT_PCT = 0.04  # +4%


# ============================================================
# HELPER: Load saved model + data from Step 3
# ============================================================

def load_model_pack():
    """Load pickled model results and data from Step 3."""
    results_path = PROJECT_ROOT / "step3_results.pkl"
    data_path = PROJECT_ROOT / "step3_data.pkl"

    if not results_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Run step3_model_training.py first.")

    with open(results_path, "rb") as f:
        results = pickle.load(f)
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    return results, data


def pick_best_hybrid_model(results):
    """Select best hybrid model: prefer XGBoost, fallback to Random Forest."""
    if "B_XGB" in results:
        return results["B_XGB"]
    return results["B_RF"]


# ============================================================
# PHASE 5: TRADING LOGIC
# ============================================================

def build_features_dataframe():
    """Load full feature dataset by rebuilding from Step 3."""
    from step3_model_training import build_training_dataframe
    return build_training_dataframe()


def add_model_predictions(df, results, data):
    """Add model predictions and probability columns to the DataFrame."""
    hybrid_features = data["hybrid_features"]
    best_model = pick_best_hybrid_model(results)

    model = best_model["model"]
    scaler = best_model["scaler"]

    X_scaled = scaler.transform(df[hybrid_features])

    output_df = df.copy()
    output_df["Model_Pred"] = model.predict(X_scaled)
    output_df["Model_Proba_Up"] = model.predict_proba(X_scaled)[:, 1]

    return output_df


def add_trading_columns(df):
    """
    Generate trading signals based on model predictions + technical rules.

    Entry Rules (BUY):
        - Model predicts UP (1)
        - RSI_14 < 70 (not overbought)
        - Price > SMA_20 (above trend)

    Exit Rules (SELL):
        - Model predicts DOWN (0)
        - OR RSI_14 > 70 (overbought)

    Otherwise: HOLD
    """
    df = df.copy()
    prediction = df["Model_Pred"].astype(int)

    # Trend direction
    df["Trend"] = np.where(prediction == 1, "Bullish", "Bearish")

    # BUY signal: prediction UP + RSI not overbought + price above SMA_20
    buy_condition = (
        (prediction == 1)
        & (df["RSI_14"] < 70)
        & (df["Close"] > df["SMA_20"])
    )

    # SELL signal: prediction DOWN or RSI overbought
    sell_condition = (prediction == 0) | (df["RSI_14"] > 70)

    # Assign signals (SELL takes priority over HOLD)
    df["Signal"] = np.where(sell_condition, "SELL",
                   np.where(buy_condition, "BUY", "HOLD"))

    return df


# ============================================================
# PHASE 6: ACTIONABLE OUTPUT
# ============================================================

def format_actionable_report(row, stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT):
    """
    Generate human-readable trading report for the latest data point.

    Example output:
        Prediction: UP
        Trend: Bullish
        Confidence: 0.67

        Signal: BUY
        Entry Price: 42,500.00
        Exit Rule:
          - Stop-loss: -2% (sell if price drops 2%)
          - Take-profit: +4% (sell if price rises 4%)
          - Or model flips to DOWN / RSI > 70
    """
    prediction = int(row["Model_Pred"])
    direction = "UP" if prediction == 1 else "DOWN"

    # Confidence = probability of the predicted direction
    prob_up = float(row["Model_Proba_Up"])
    confidence = prob_up if prediction == 1 else (1.0 - prob_up)

    entry_price = float(row["Close"])

    report = "\n".join([
        f"Prediction: {direction}",
        f"Trend:      {row['Trend']}",
        f"Confidence: {confidence:.2f}",
        "",
        f"Signal:      {row['Signal']}",
        f"Entry Price: {entry_price:,.2f}",
        "Exit Rule:",
        f"  - Stop-loss:   -{stop_loss * 100:.0f}% (sell if price drops to {entry_price * (1 - stop_loss):,.2f})",
        f"  - Take-profit: +{take_profit * 100:.0f}% (sell if price rises to {entry_price * (1 + take_profit):,.2f})",
        f"  - Or model flips to DOWN / RSI > 70",
    ])

    return report


# ============================================================
# PHASE 7: PAPER TRADING SIMULATION
# ============================================================

def run_paper_trading(df, initial_cash=10000.0,
                      stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT):
    """
    Simulate paper trading using BUY/SELL signals.

    Rules:
        - BUY when signal = BUY and no position held
        - SELL when signal = SELL and position held
        - Auto-exit on stop-loss (-2%) or take-profit (+4%)

    Tracks: total return, win rate, number of trades.
    """
    cash = initial_cash
    shares = 0.0
    entry_price = None
    trades = []
    wins = 0
    losses = 0

    for date, row in df.iterrows():
        price = float(row["Close"])
        signal = row["Signal"]

        # ── Check stop-loss and take-profit if we have a position ──
        if shares > 0 and entry_price is not None:

            # Stop-loss hit
            if price <= entry_price * (1 - stop_loss):
                cash = shares * price
                trade_return = (price - entry_price) / entry_price
                trades.append({
                    "date": date, "side": "SELL",
                    "price": price, "reason": "stop_loss",
                    "return_pct": trade_return * 100
                })
                losses += 1
                shares = 0.0
                entry_price = None
                continue

            # Take-profit hit
            if price >= entry_price * (1 + take_profit):
                cash = shares * price
                trade_return = (price - entry_price) / entry_price
                trades.append({
                    "date": date, "side": "SELL",
                    "price": price, "reason": "take_profit",
                    "return_pct": trade_return * 100
                })
                wins += 1
                shares = 0.0
                entry_price = None
                continue

        # ── Execute BUY signal ──
        if shares == 0 and signal == "BUY":
            shares = cash / price
            entry_price = price
            cash = 0.0
            trades.append({
                "date": date, "side": "BUY",
                "price": price, "reason": "signal"
            })

        # ── Execute SELL signal ──
        elif shares > 0 and signal == "SELL":
            cash = shares * price
            trade_return = (price - entry_price) / entry_price if entry_price else 0.0
            trades.append({
                "date": date, "side": "SELL",
                "price": price, "reason": "signal",
                "return_pct": trade_return * 100
            })
            if trade_return > 0:
                wins += 1
            else:
                losses += 1
            shares = 0.0
            entry_price = None

    # Final portfolio value
    final_value = cash + shares * float(df["Close"].iloc[-1])
    num_buys = sum(1 for t in trades if t["side"] == "BUY")
    total_closed = wins + losses

    return {
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": round((final_value - initial_cash) / initial_cash * 100, 2),
        "num_trades": num_buys,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total_closed * 100, 1) if total_closed else 0.0,
        "trades": trades,
    }


# ============================================================
# PHASE 8: LIVE PREDICTION FUNCTION
# ============================================================

def predict_latest(refresh_news=True, api_key=None):
    """
    Fetch latest data, recompute features, and output prediction + signal.

    Parameters:
        refresh_news: If True, re-run Phase 1 to fetch latest news
        api_key:      NewsAPI key (or set NEWSAPI_KEY env var)

    Returns:
        Dictionary with prediction, trend, signal, confidence, entry price,
        and formatted report text.
    """
    # Optionally refresh news data
    if refresh_news:
        from step1_data_ingestion import run_phase1
        key = api_key or os.environ.get("NEWSAPI_KEY", "").strip()
        if not key:
            raise ValueError("NEWSAPI_KEY required when refresh_news=True")
        run_phase1(api_key=key)

    # Load model and build features
    results, data = load_model_pack()
    df = build_features_dataframe()
    df = add_model_predictions(df, results, data)
    df = add_trading_columns(df)

    # Get the latest row
    latest = df.iloc[-1]
    prediction = int(latest["Model_Pred"])
    prob_up = float(latest["Model_Proba_Up"])
    confidence = prob_up if prediction == 1 else (1.0 - prob_up)

    return {
        "date": latest.name,
        "prediction": "UP" if prediction == 1 else "DOWN",
        "trend": str(latest["Trend"]),
        "signal": str(latest["Signal"]),
        "confidence": round(confidence, 4),
        "entry_price": float(latest["Close"]),
        "report_text": format_actionable_report(latest),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASES 5-8: TRADING RUNTIME")
    print(f"  Stop-Loss: {STOP_LOSS_PCT*100:.0f}%  |  Take-Profit: {TAKE_PROFIT_PCT*100:.0f}%  |  Ratio: 1:2")
    print("=" * 60)

    # Load model
    results, data = load_model_pack()

    # Build features + predictions + signals
    df = build_features_dataframe()
    df = add_model_predictions(df, results, data)
    df = add_trading_columns(df)

    # Phase 6: Actionable report
    print("\n[Phase 6] Latest Actionable Insight:\n")
    print(format_actionable_report(df.iloc[-1]))

    # Phase 7: Paper trading simulation
    paper = run_paper_trading(df)
    print("\n" + "-" * 60)
    print("[Phase 7] Paper Trading Simulation:")
    print(f"  Initial Cash:  ${paper['initial_cash']:,.2f}")
    print(f"  Final Value:   ${paper['final_value']:,.2f}")
    print(f"  Total Return:  {paper['total_return_pct']:+.2f}%")
    print(f"  Trades:        {paper['num_trades']}")
    print(f"  Wins / Losses: {paper['wins']} / {paper['losses']}")
    print(f"  Win Rate:      {paper['win_rate']:.1f}%")

    # Phase 8: predict_latest (without news refresh)
    print("\n" + "-" * 60)
    print("[Phase 8] predict_latest(refresh_news=False):\n")
    latest = predict_latest(refresh_news=False)
    print(latest["report_text"])

    print("\n" + "=" * 60)
    print("TRADING RUNTIME COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
