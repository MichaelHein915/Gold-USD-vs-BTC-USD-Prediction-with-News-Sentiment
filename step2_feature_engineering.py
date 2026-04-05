"""
Step 2: Feature Engineering (Technical + Sentiment Decay)
=========================================================
- Calculate technical indicators on FULL price history (1,857 rows)
- Then merge with sentiment data
- Calculate Sentiment Decay rolling averages (1, 3, 5, 7, 14, 20 days)
- Handle missing values
"""

from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 2A. TECHNICAL INDICATORS
# ============================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, MACD Signal, and MACD Histogram"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Bollinger Bands: upper, middle (SMA), lower, and bandwidth"""
    sma = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = sma + (std_dev * rolling_std)
    lower = sma - (std_dev * rolling_std)
    bandwidth = (upper - lower) / sma
    pct_b = (series - lower) / (upper - lower)  # %B indicator
    return upper, sma, lower, bandwidth, pct_b


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (ATR) - volatility indicator"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV)"""
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv


def add_technical_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on the full price DataFrame.
    This uses the complete 5-year history for proper lookback.
    """
    df = price_df.copy()

    # --- Trend Indicators ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # --- RSI ---
    df['RSI_14'] = compute_rsi(df['Close'], period=14)

    # --- MACD ---
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = compute_macd(df['Close'])

    # --- Bollinger Bands ---
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['BB_Width'], df['BB_PctB'] = \
        compute_bollinger_bands(df['Close'])

    # --- Volatility ---
    df['ATR_14'] = compute_atr(df['High'], df['Low'], df['Close'], period=14)

    # --- Volume ---
    df['OBV'] = compute_obv(df['Close'], df['Volume'])

    # --- Price-derived features ---
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']  # normalized range
    df['Close_to_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']  # distance from SMA

    print(f"  Technical indicators computed on {len(df)} rows")
    print(f"  New columns added: {len(df.columns) - len(price_df.columns)}")

    # List all technical feature columns
    tech_cols = [c for c in df.columns if c not in price_df.columns]
    print(f"  Technical features: {tech_cols}")

    return df


# ============================================================
# 2B. SENTIMENT DECAY FEATURES
# ============================================================

def add_sentiment_decay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling averages of sentiment over multiple lookback windows.
    Windows: 1, 3, 5, 7, 14, 20 days.
    Applied to: Daily_Sentiment_Mean, Sentiment_Ratio, Total_Articles
    """
    windows = [1, 3, 5, 7, 14, 20]

    # Sentiment Mean rolling decay
    for w in windows:
        df[f'Sent_Mean_MA{w}'] = df['Daily_Sentiment_Mean'].rolling(window=w, min_periods=1).mean()

    # Sentiment Ratio rolling decay
    for w in windows:
        df[f'Sent_Ratio_MA{w}'] = df['Sentiment_Ratio'].rolling(window=w, min_periods=1).mean()

    # Article count rolling (captures news volume/attention)
    for w in windows:
        df[f'Articles_MA{w}'] = df['Total_Articles'].rolling(window=w, min_periods=1).mean()

    # Sentiment momentum: change in sentiment over time
    df['Sent_Momentum_3d'] = df['Daily_Sentiment_Mean'] - df['Daily_Sentiment_Mean'].shift(3)
    df['Sent_Momentum_7d'] = df['Daily_Sentiment_Mean'] - df['Daily_Sentiment_Mean'].shift(7)
    # No prior days -> NaN from shift; fill 0 so sparse news dates keep rows after dropna
    df['Sent_Momentum_3d'] = df['Sent_Momentum_3d'].fillna(0)
    df['Sent_Momentum_7d'] = df['Sent_Momentum_7d'].fillna(0)

    decay_cols = [c for c in df.columns if 'Sent_' in c or 'Articles_MA' in c]
    print(f"  Sentiment decay features added: {len(decay_cols)}")
    print(f"  Features: {decay_cols}")

    return df


# ============================================================
# 2C. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN values from rolling calculations and merging.
    Strategy: Drop rows with NaN (from beginning of rolling windows).
    """
    before_shape = df.shape

    # Check NaN counts per column
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]

    if len(nan_cols) > 0:
        print(f"\n  Columns with NaN values:")
        for col, count in nan_cols.items():
            print(f"    {col}: {count} NaN")

    # Drop rows with any NaN
    df = df.dropna()

    after_shape = df.shape
    print(f"\n  Shape before NaN handling: {before_shape}")
    print(f"  Shape after NaN handling:  {after_shape}")
    print(f"  Rows dropped: {before_shape[0] - after_shape[0]}")

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    # Load the full price data (for technical indicators with proper lookback)
    from step1_data_ingestion import load_btc_price_data

    BTC_CSV_PATH = PROJECT_ROOT / "Bitcoin Historical Data 5year.csv"
    MERGED_CSV_PATH = PROJECT_ROOT / "step1_merged_data.csv"

    # 2A. Compute technical indicators on FULL 5-year price history
    print("\n[2A] Computing technical indicators on full price history...")
    price_df = load_btc_price_data(BTC_CSV_PATH)
    price_with_tech = add_technical_indicators(price_df)

    # Load the merged sentiment data from Step 1
    print("\n" + "-" * 60)
    print("[2A+] Re-merging: technical indicators + sentiment data...")
    merged_df = pd.read_csv(MERGED_CSV_PATH, index_col='Date', parse_dates=True)

    # Get sentiment columns from the merged data
    sentiment_cols = ['Daily_Sentiment_Mean', 'Daily_Sentiment_Std',
                      'Positive_Count', 'Negative_Count', 'Neutral_Count',
                      'Total_Articles', 'Sentiment_Ratio']
    sentiment_data = merged_df[sentiment_cols]

    # Inner join: price+tech with sentiment (keeps only dates with both)
    df = price_with_tech.join(sentiment_data, how='inner')
    print(f"  After re-merge with sentiment: {df.shape}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    # 2B. Add sentiment decay features
    print("\n" + "-" * 60)
    print("[2B] Adding Sentiment Decay features (rolling windows: 1,3,5,7,14,20 days)...")
    df = add_sentiment_decay_features(df)

    # 2C. Handle missing values
    print("\n" + "-" * 60)
    print("[2C] Handling missing values...")
    df = handle_missing_values(df)

    # Final summary
    print("\n" + "-" * 60)
    print("FINAL FEATURE-ENGINEERED DATASET:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"\n  All columns ({len(df.columns)}):")

    # Group columns by type for clarity
    price_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Change_Pct']
    tech_cols = [c for c in df.columns if c not in price_cols
                 and c not in sentiment_cols
                 and 'Sent_' not in c and 'Articles_MA' not in c]
    sent_base = [c for c in df.columns if c in sentiment_cols]
    sent_decay = [c for c in df.columns if 'Sent_' in c or 'Articles_MA' in c]

    print(f"\n  Price columns ({len(price_cols)}): {price_cols}")
    print(f"  Technical columns ({len(tech_cols)}): {tech_cols}")
    print(f"  Sentiment base ({len(sent_base)}): {sent_base}")
    print(f"  Sentiment decay ({len(sent_decay)}): {sent_decay}")

    print(f"\n  Dataset preview (last 5 rows):")
    print(df.tail().to_string())

    # Save for Step 3
    output_path = PROJECT_ROOT / "step2_features.csv"
    df.to_csv(output_path)
    print(f"\n  Feature-engineered data saved to: {output_path}")
    if len(df) < 30:
        print(
            "\n  NOTE: Few rows after merge — expand Phase 1 news coverage (more dates/articles) "
            "for robust training in Steps 3–4."
        )

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE!")
    print("=" * 60)
