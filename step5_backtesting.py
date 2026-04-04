"""
Step 5: Quantitative Backtesting
=================================
- Simulate trading on the TEST set using model predictions
- Apply BUY/SELL signals with Stop-Loss (-2%) and Take-Profit (+4%) → ratio 1:2
- Track:  Total Return, Win Rate, Sharpe Ratio, Max Drawdown
- Plot:   Equity curve over time
- Compare: Strategy vs Buy-and-Hold BTC

Uses tuned models from step3b if available, otherwise falls back to step3.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Risk Parameters (1:2 ratio) ──────────────────────────────
STOP_LOSS_PCT = 0.02    # -2%
TAKE_PROFIT_PCT = 0.04  # +4%
INITIAL_CAPITAL = 10_000.0
TRADING_DAYS_PER_YEAR = 365


# ============================================================
# LOAD MODEL (tuned > untuned fallback)
# ============================================================

def load_best_model():
    """
    Load the best available model.
    Priority: step3b tuned results > step3 untuned results.
    Returns (results_dict, data_dict, source_label).
    """
    tuned_results_path = PROJECT_ROOT / "step3b_tuned_results.pkl"
    tuned_data_path = PROJECT_ROOT / "step3b_tuned_data.pkl"
    untuned_results_path = PROJECT_ROOT / "step3_results.pkl"
    untuned_data_path = PROJECT_ROOT / "step3_data.pkl"

    # Try tuned first
    if tuned_results_path.is_file() and tuned_data_path.is_file():
        with open(tuned_results_path, "rb") as f:
            results = pickle.load(f)
        with open(tuned_data_path, "rb") as f:
            data = pickle.load(f)
        return results, data, "TUNED (step3b)"

    # Fall back to untuned
    if untuned_results_path.is_file() and untuned_data_path.is_file():
        with open(untuned_results_path, "rb") as f:
            results = pickle.load(f)
        with open(untuned_data_path, "rb") as f:
            data = pickle.load(f)
        return results, data, "UNTUNED (step3)"

    raise FileNotFoundError(
        "No model found. Run step3_model_training.py or step3b_hyperparameter_tuning.py first."
    )


def pick_best_hybrid(results):
    """Select the best hybrid model by F1, then accuracy."""
    hybrid_keys = [k for k in results if k.startswith("B_")]
    if not hybrid_keys:
        raise ValueError("No hybrid (B_*) models found in results.")

    best_key = max(hybrid_keys,
                   key=lambda k: (results[k]["f1"], results[k]["accuracy"]))
    return results[best_key]


# ============================================================
# BUILD BACKTEST DATAFRAME
# ============================================================

def build_backtest_dataframe(results, data):
    """
    Build a DataFrame with model predictions and trading signals
    for the TEST period only.
    """
    from step3_model_training import build_training_dataframe, create_target

    # Rebuild full feature DataFrame
    df = build_training_dataframe()
    df = create_target(df)

    # Get test period
    test_df = data["test_df"]
    hybrid_features = data["hybrid_features"]
    test_start = test_df.index.min()
    test_end = test_df.index.max()

    # Slice to test period
    backtest_df = df.loc[test_start:test_end].copy()

    # Add model predictions
    best_model_result = pick_best_hybrid(results)
    model = best_model_result["model"]
    scaler = best_model_result["scaler"]

    X_scaled = scaler.transform(backtest_df[hybrid_features])
    backtest_df["Model_Pred"] = model.predict(X_scaled)
    backtest_df["Model_Proba_Up"] = model.predict_proba(X_scaled)[:, 1]

    # Add trading signals
    prediction = backtest_df["Model_Pred"].astype(int)
    backtest_df["Trend"] = np.where(prediction == 1, "Bullish", "Bearish")

    buy_condition = (
        (prediction == 1)
        & (backtest_df["RSI_14"] < 70)
        & (backtest_df["Close"] > backtest_df["SMA_20"])
    )
    sell_condition = (prediction == 0) | (backtest_df["RSI_14"] > 70)
    backtest_df["Signal"] = np.where(sell_condition, "SELL",
                            np.where(buy_condition, "BUY", "HOLD"))

    return backtest_df, best_model_result["model_name"]


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def run_backtest(df, initial_capital=INITIAL_CAPITAL,
                 stop_loss=STOP_LOSS_PCT, take_profit=TAKE_PROFIT_PCT):
    """
    Simulate trading on the backtest DataFrame.

    Rules:
        BUY:  when Signal = BUY and no position held
        SELL: when Signal = SELL, or stop-loss/take-profit hit
        SL:TP = 1:2  (2% : 4%)

    Returns:
        equity_curve: pd.Series of daily portfolio value (indexed by date)
        trades:       list of trade dicts (date, side, price, reason, return_pct)
        summary:      dict with Total Return, Win Rate, Sharpe, Max Drawdown
    """
    cash = initial_capital
    shares = 0.0
    entry_price = None

    # Track daily portfolio value
    dates = []
    equity_values = []
    trades = []
    wins = 0
    losses = 0

    for date, row in df.iterrows():
        price = float(row["Close"])
        signal = row["Signal"]

        # ── Check stop-loss / take-profit ──
        if shares > 0 and entry_price is not None:

            # Stop-loss hit (-2%)
            if price <= entry_price * (1 - stop_loss):
                cash = shares * price
                trade_return = (price - entry_price) / entry_price
                trades.append({
                    "date": date, "side": "SELL",
                    "price": price, "reason": "stop_loss",
                    "return_pct": trade_return * 100,
                })
                losses += 1
                shares = 0.0
                entry_price = None
                dates.append(date)
                equity_values.append(cash)
                continue

            # Take-profit hit (+4%)
            if price >= entry_price * (1 + take_profit):
                cash = shares * price
                trade_return = (price - entry_price) / entry_price
                trades.append({
                    "date": date, "side": "SELL",
                    "price": price, "reason": "take_profit",
                    "return_pct": trade_return * 100,
                })
                wins += 1
                shares = 0.0
                entry_price = None
                dates.append(date)
                equity_values.append(cash)
                continue

        # ── Execute BUY ──
        if shares == 0 and signal == "BUY":
            shares = cash / price
            entry_price = price
            cash = 0.0
            trades.append({
                "date": date, "side": "BUY",
                "price": price, "reason": "signal",
            })

        # ── Execute SELL ──
        elif shares > 0 and signal == "SELL":
            cash = shares * price
            trade_return = (price - entry_price) / entry_price if entry_price else 0.0
            trades.append({
                "date": date, "side": "SELL",
                "price": price, "reason": "signal",
                "return_pct": trade_return * 100,
            })
            if trade_return > 0:
                wins += 1
            else:
                losses += 1
            shares = 0.0
            entry_price = None

        # ── Record daily equity ──
        portfolio_value = cash + shares * price
        dates.append(date)
        equity_values.append(portfolio_value)

    # Build equity curve
    equity_curve = pd.Series(equity_values, index=dates, name="Strategy")

    # Build trade log
    total_closed = wins + losses

    return equity_curve, trades, {
        "wins": wins,
        "losses": losses,
        "total_closed": total_closed,
    }


# ============================================================
# BUY-AND-HOLD BENCHMARK
# ============================================================

def compute_buy_and_hold(df, initial_capital=INITIAL_CAPITAL):
    """
    Compute Buy-and-Hold equity curve.
    Buy BTC at first day's close, hold until last day.
    """
    prices = df["Close"].values
    first_price = prices[0]
    shares = initial_capital / first_price

    equity = shares * df["Close"]
    equity.name = "Buy_and_Hold"

    return equity


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def compute_daily_returns(equity_curve):
    """Compute daily percentage returns from equity curve."""
    return equity_curve.pct_change().dropna()


def compute_total_return(equity_curve):
    """Total return as a percentage."""
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def compute_sharpe_ratio(daily_returns, trading_days=TRADING_DAYS_PER_YEAR):
    """
    Annualized Sharpe Ratio (assuming risk-free rate = 0).

    Formula:
        Sharpe = (mean_daily_return / std_daily_return) * sqrt(trading_days)
    """
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days)


def compute_max_drawdown(equity_curve):
    """
    Maximum Drawdown: largest peak-to-trough decline.

    Formula:
        running_max = cumulative max of equity
        drawdown    = (equity - running_max) / running_max
        max_dd      = min(drawdown)
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    return max_dd * 100  # as percentage


def compute_win_rate(wins, total_closed):
    """Win rate as a percentage."""
    if total_closed == 0:
        return 0.0
    return (wins / total_closed) * 100


def build_performance_summary(equity_curve, trade_stats, label="Strategy"):
    """
    Build a complete performance summary dict.
    """
    daily_returns = compute_daily_returns(equity_curve)

    return {
        "label": label,
        "initial_capital": equity_curve.iloc[0],
        "final_value": round(equity_curve.iloc[-1], 2),
        "total_return_pct": round(compute_total_return(equity_curve), 2),
        "win_rate_pct": round(compute_win_rate(
            trade_stats["wins"], trade_stats["total_closed"]
        ), 1),
        "sharpe_ratio": round(compute_sharpe_ratio(daily_returns), 4),
        "max_drawdown_pct": round(compute_max_drawdown(equity_curve), 2),
        "num_trades": trade_stats["total_closed"],
        "wins": trade_stats["wins"],
        "losses": trade_stats["losses"],
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_performance_comparison(strategy_summary, bnh_summary):
    """Print side-by-side comparison of Strategy vs Buy-and-Hold."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: STRATEGY vs BUY-AND-HOLD")
    print("=" * 70)

    metrics = [
        ("Initial Capital",  "initial_capital",    "${:>12,.2f}"),
        ("Final Value",      "final_value",        "${:>12,.2f}"),
        ("Total Return",     "total_return_pct",   "{:>12.2f}%"),
        ("Win Rate",         "win_rate_pct",       "{:>12.1f}%"),
        ("Sharpe Ratio",     "sharpe_ratio",       "{:>13.4f}"),
        ("Max Drawdown",     "max_drawdown_pct",   "{:>12.2f}%"),
        ("Total Trades",     "num_trades",         "{:>13d}"),
        ("Wins",             "wins",               "{:>13d}"),
        ("Losses",           "losses",             "{:>13d}"),
    ]

    print(f"\n  {'Metric':<20} {'Strategy':>15} {'Buy & Hold':>15}")
    print(f"  {'-' * 20} {'-' * 15} {'-' * 15}")

    for label, key, fmt in metrics:
        s_val = strategy_summary.get(key, "—")
        b_val = bnh_summary.get(key, "—")

        s_str = fmt.format(s_val) if isinstance(s_val, (int, float)) else str(s_val)
        b_str = fmt.format(b_val) if isinstance(b_val, (int, float)) else "—"

        print(f"  {label:<20} {s_str:>15} {b_str:>15}")

    # Verdict
    s_ret = strategy_summary["total_return_pct"]
    b_ret = bnh_summary["total_return_pct"]
    diff = s_ret - b_ret

    print(f"\n  {'RETURN DIFFERENCE':<20} {diff:>+15.2f}%")
    if diff > 0:
        print(f"  VERDICT: Strategy OUTPERFORMS Buy-and-Hold by {diff:+.2f}%")
    elif diff < 0:
        print(f"  VERDICT: Buy-and-Hold outperforms Strategy by {abs(diff):.2f}%")
    else:
        print(f"  VERDICT: Strategy matches Buy-and-Hold")


def print_trade_log(trades, max_trades=20):
    """Print the trade log (first N trades)."""
    print("\n" + "-" * 70)
    print(f"TRADE LOG (showing first {min(len(trades), max_trades)} of {len(trades)} trades)")
    print("-" * 70)

    print(f"  {'#':<4} {'Date':<12} {'Side':<6} {'Price':>12} {'Reason':<12} {'Return':>8}")
    print(f"  {'-' * 4} {'-' * 12} {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 8}")

    for i, trade in enumerate(trades[:max_trades], 1):
        date_str = trade["date"].strftime("%Y-%m-%d") if hasattr(trade["date"], "strftime") else str(trade["date"])[:10]
        ret_str = f"{trade.get('return_pct', 0):+.2f}%" if "return_pct" in trade else "—"
        print(f"  {i:<4} {date_str:<12} {trade['side']:<6} "
              f"${trade['price']:>11,.2f} {trade['reason']:<12} {ret_str:>8}")


# ============================================================
# PLOTTING
# ============================================================

def plot_equity_curve(strategy_equity, bnh_equity, strategy_summary, bnh_summary):
    """
    Plot equity curve: Strategy vs Buy-and-Hold.
    Also plots drawdown below.
    """
    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        "Quantitative Backtesting: Strategy vs Buy-and-Hold",
        fontsize=16, fontweight="bold", y=0.95,
    )

    # ── Top: Equity Curve ──
    ax_equity.plot(strategy_equity.index, strategy_equity.values,
                   color="#2196F3", linewidth=2, label="Strategy (ML Signals)")
    ax_equity.plot(bnh_equity.index, bnh_equity.values,
                   color="#FF9800", linewidth=2, linestyle="--", label="Buy & Hold BTC")

    # Shade the gap
    ax_equity.fill_between(
        strategy_equity.index,
        strategy_equity.values,
        bnh_equity.reindex(strategy_equity.index).values,
        alpha=0.1, color="#2196F3",
    )

    ax_equity.axhline(y=INITIAL_CAPITAL, color="gray", linestyle=":", alpha=0.5,
                      label=f"Initial: ${INITIAL_CAPITAL:,.0f}")
    ax_equity.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax_equity.set_title("Equity Curve", fontweight="bold", fontsize=13)
    ax_equity.legend(fontsize=10, loc="upper left")
    ax_equity.grid(alpha=0.3)

    # Add performance text box
    textstr = (
        f"Strategy:     {strategy_summary['total_return_pct']:+.2f}%  |  "
        f"Sharpe: {strategy_summary['sharpe_ratio']:.2f}  |  "
        f"MaxDD: {strategy_summary['max_drawdown_pct']:.2f}%\n"
        f"Buy & Hold:  {bnh_summary['total_return_pct']:+.2f}%  |  "
        f"Sharpe: {bnh_summary['sharpe_ratio']:.2f}  |  "
        f"MaxDD: {bnh_summary['max_drawdown_pct']:.2f}%"
    )
    ax_equity.text(
        0.02, 0.05, textstr, transform=ax_equity.transAxes,
        fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # ── Bottom: Drawdown ──
    running_max = strategy_equity.cummax()
    drawdown = (strategy_equity - running_max) / running_max * 100

    ax_drawdown.fill_between(drawdown.index, drawdown.values, 0,
                             color="#F44336", alpha=0.4)
    ax_drawdown.plot(drawdown.index, drawdown.values,
                     color="#F44336", linewidth=1)
    ax_drawdown.set_ylabel("Drawdown (%)", fontsize=12)
    ax_drawdown.set_xlabel("Date", fontsize=12)
    ax_drawdown.set_title("Strategy Drawdown", fontweight="bold", fontsize=13)
    ax_drawdown.grid(alpha=0.3)

    # Format x-axis dates
    ax_drawdown.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_drawdown.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_drawdown.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    path = OUTPUT_DIR / "btc_backtest_equity_curve.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Equity curve saved: {path.name}")

    return path


def plot_trade_returns(trades):
    """Plot histogram of individual trade returns."""
    sell_trades = [t for t in trades if t["side"] == "SELL" and "return_pct" in t]
    if not sell_trades:
        print("  No closed trades to plot")
        return None

    returns = [t["return_pct"] for t in sell_trades]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in returns]
    ax.bar(range(len(returns)), returns, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Trade #", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.set_title("Individual Trade Returns (Green = Win, Red = Loss)",
                 fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Add stats
    avg_return = np.mean(returns)
    ax.axhline(y=avg_return, color="#2196F3", linestyle="--", alpha=0.7,
               label=f"Avg Return: {avg_return:+.2f}%")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = OUTPUT_DIR / "btc_backtest_trade_returns.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Trade returns saved: {path.name}")

    return path


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print("STEP 5: QUANTITATIVE BACKTESTING")
    print(f"  SL: {STOP_LOSS_PCT*100:.0f}%  |  TP: {TAKE_PROFIT_PCT*100:.0f}%  |  Ratio: 1:2")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print("=" * 70)

    # ── Load model ──
    print("\n[5A] Loading best model...")
    results, data, source = load_best_model()
    print(f"  Source: {source}")

    # ── Build backtest DataFrame ──
    print("\n[5B] Building backtest DataFrame (test period only)...")
    backtest_df, model_name = build_backtest_dataframe(results, data)
    print(f"  Model: {model_name}")
    print(f"  Test period: {backtest_df.index.min().date()} to {backtest_df.index.max().date()}")
    print(f"  Rows: {len(backtest_df)}")

    # Signal distribution
    signal_counts = backtest_df["Signal"].value_counts()
    print(f"  Signals: {signal_counts.to_dict()}")

    # ── Run Backtest ──
    print("\n" + "-" * 70)
    print("[5C] Running backtest simulation...")
    strategy_equity, trades, trade_stats = run_backtest(backtest_df)

    # ── Buy-and-Hold Benchmark ──
    print("\n[5D] Computing Buy-and-Hold benchmark...")
    bnh_equity = compute_buy_and_hold(backtest_df)

    # Align indices
    bnh_equity = bnh_equity.reindex(strategy_equity.index, method="ffill")

    # ── Performance Summaries ──
    strategy_summary = build_performance_summary(strategy_equity, trade_stats, "Strategy")

    bnh_trade_stats = {
        "wins": 1 if bnh_equity.iloc[-1] > bnh_equity.iloc[0] else 0,
        "losses": 0 if bnh_equity.iloc[-1] > bnh_equity.iloc[0] else 1,
        "total_closed": 1,
    }
    bnh_summary = build_performance_summary(bnh_equity, bnh_trade_stats, "Buy & Hold")

    # ── Print Results ──
    print_performance_comparison(strategy_summary, bnh_summary)
    print_trade_log(trades)

    # ── Plot ──
    print("\n" + "-" * 70)
    print("[5E] Generating charts...")
    plot_equity_curve(strategy_equity, bnh_equity, strategy_summary, bnh_summary)
    plot_trade_returns(trades)

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"""
  Model Used:      {model_name}
  Source:           {source}
  Test Period:      {backtest_df.index.min().date()} to {backtest_df.index.max().date()}
  Risk Params:     SL={STOP_LOSS_PCT*100:.0f}% / TP={TAKE_PROFIT_PCT*100:.0f}% (1:2)

  STRATEGY:
    Total Return:  {strategy_summary['total_return_pct']:+.2f}%
    Win Rate:      {strategy_summary['win_rate_pct']:.1f}%
    Sharpe Ratio:  {strategy_summary['sharpe_ratio']:.4f}
    Max Drawdown:  {strategy_summary['max_drawdown_pct']:.2f}%

  BUY & HOLD:
    Total Return:  {bnh_summary['total_return_pct']:+.2f}%
    Sharpe Ratio:  {bnh_summary['sharpe_ratio']:.4f}
    Max Drawdown:  {bnh_summary['max_drawdown_pct']:.2f}%

  OUTPUT FILES:
    btc_backtest_equity_curve.png   — Equity curve + drawdown
    btc_backtest_trade_returns.png  — Individual trade returns
""")
    print("=" * 70)
    print("STEP 5 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
