"""
Step 1: Data Ingestion & Merging (Phase 1)
- BTC OHLCV from CSV, NewsAPI + JSON, VADER -1/0/+1, impact weights, daily merge.
Requires: NEWSAPI_KEY, packages: pandas numpy requests nltk
"""
from __future__ import annotations
import os
import sys
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
IMPACT_BY_CATEGORY = {"Regulatory": 1.5, "Macro": 1.3, "Technical": 1.0, "Social": 0.7}

def load_btc_price_data(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)
    def parse_volume(vol_str):
        vol_str = str(vol_str).strip()
        if vol_str in ("-", "nan") or pd.isna(vol_str):
            return np.nan
        if "K" in vol_str: return float(vol_str.replace("K", "")) * 1_000
        if "M" in vol_str: return float(vol_str.replace("M", "")) * 1_000_000
        if "B" in vol_str: return float(vol_str.replace("B", "")) * 1_000_000_000
        return float(vol_str)
    df["Volume"] = df["Vol."].apply(parse_volume)
    df["Change_Pct"] = df["Change %"].str.replace("%", "", regex=False).astype(float)
    df = df.rename(columns={"Price": "Close"})
    df = df[["Date", "Close", "Open", "High", "Low", "Volume", "Change_Pct"]]
    df = df.sort_values("Date").reset_index(drop=True)
    return df.set_index("Date")

def _classify_impact_category(text: str) -> str:
    t = (text or "").lower()
    regulatory_kw = ("sec ", "sec,", "regulation", "regulatory", "cftc", "lawsuit", "ban ", "legal ", "court", "policy", "compliance", " etf", "spot bitcoin", "approval", "reject", "enforcement", "sanction")
    macro_kw = ("fed ", "federal reserve", "inflation", "cpi ", "interest rate", "recession", "dollar", "gdp", "unemployment", "macro", "treasury", "yield", "rate cut", "rate hike", "jobs report", "payroll")
    technical_kw = ("chart", "support", "resistance", "breakout", "rsi", "moving average", "trading", "technical", "price target", "candlestick", "pattern", "oversold", "overbought", "fibonacci", "trend line")
    social_kw = ("elon", "musk", "twitter", "x.com", "reddit", "tweet", "viral", "social", "meme", "community", "influencer")
    for kw in regulatory_kw:
        if kw in t: return "Regulatory"
    for kw in macro_kw:
        if kw in t: return "Macro"
    for kw in technical_kw:
        if kw in t: return "Technical"
    for kw in social_kw:
        if kw in t: return "Social"
    return "Technical"

def fetch_news_newsapi(api_key, date_from, date_to, query="Bitcoin OR BTC OR cryptocurrency", language="en", page_size=100, max_pages=1):
    base = "https://newsapi.org/v2/everything"
    rows = []
    for page in range(1, max_pages + 1):
        params = {"q": query, "from": date_from.strftime("%Y-%m-%d"), "to": date_to.strftime("%Y-%m-%d"), "language": language, "sortBy": "publishedAt", "pageSize": page_size, "page": page, "apiKey": api_key}
        r = requests.get(base, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"NewsAPI error {r.status_code}: {r.text[:500]}")
        data = r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"NewsAPI status: {data}")
        articles = data.get("articles") or []
        for a in articles:
            title = (a.get("title") or "").strip()
            desc = (a.get("description") or "").strip()
            published = a.get("publishedAt")
            if not published or not title:
                continue
            pub_dt = pd.to_datetime(published, utc=True).tz_convert(None).normalize()
            text_for_class = f"{title} {desc}"
            rows.append({"Date": pub_dt, "Headline": title, "Description": desc, "Source": (a.get("source") or {}).get("name", ""), "URL": a.get("url", ""), "Impact_Category": _classify_impact_category(text_for_class)})
        total = data.get("totalResults", 0)
        if len(articles) < page_size or page * page_size >= total:
            break
    if not rows:
        return pd.DataFrame(columns=["Date", "Headline", "Description", "Source", "URL", "Impact_Category"])
    out = pd.DataFrame(rows)
    return out.drop_duplicates(subset=["Date", "Headline"]).reset_index(drop=True)


def load_local_news_csv(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw["Date"] = pd.to_datetime(raw["date"], format="mixed").dt.normalize()
    raw = raw.rename(columns={"headline": "Headline"})
    raw["Headline"] = raw["Headline"].astype(str)
    raw["Description"] = ""
    raw["Source"] = "local_csv"
    raw["URL"] = ""
    raw["Impact_Category"] = raw["Headline"].map(lambda h: _classify_impact_category(str(h)))
    return raw[["Date", "Headline", "Description", "Source", "URL", "Impact_Category"]].drop_duplicates(subset=["Date", "Headline"]).reset_index(drop=True)


def _ensure_vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

def score_sentiment_discrete(news_df: pd.DataFrame) -> pd.DataFrame:
    _ensure_vader()
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    def discrete_label(text: str) -> int:
        if pd.isna(text) or str(text).strip() == "":
            return 0
        c = sia.polarity_scores(str(text))["compound"]
        if c >= 0.05: return 1
        if c <= -0.05: return -1
        return 0
    news_df = news_df.copy()
    desc = news_df.get("Description", pd.Series([""] * len(news_df))).fillna("")
    combined = news_df["Headline"].fillna("") + " " + desc
    news_df["Sentiment_Code"] = combined.map(discrete_label)
    news_df["Impact_Amplitude"] = news_df["Impact_Category"].map(IMPACT_BY_CATEGORY)
    news_df["Weighted_Score"] = news_df["Sentiment_Code"] * news_df["Impact_Amplitude"]
    return news_df

def aggregate_and_merge(price_df, news_df):
    if news_df.empty:
        raise ValueError("No news articles — cannot merge.")
    daily = news_df.groupby("Date").agg(
        Daily_Sentiment_Mean=("Weighted_Score", "mean"),
        Daily_Sentiment_Std=("Weighted_Score", "std"),
        Positive_Count=("Sentiment_Code", lambda s: (s == 1).sum()),
        Negative_Count=("Sentiment_Code", lambda s: (s == -1).sum()),
        Neutral_Count=("Sentiment_Code", lambda s: (s == 0).sum()),
        Total_Articles=("Sentiment_Code", "count"),
    )
    daily["Daily_Sentiment_Std"] = daily["Daily_Sentiment_Std"].fillna(0.0)
    daily["Sentiment_Ratio"] = (daily["Positive_Count"] - daily["Negative_Count"]) / daily["Total_Articles"].replace(0, 1)
    merged_df = price_df.join(daily, how="inner")
    print(f"\n  Daily sentiment records: {len(daily)}")
    print(f"  Date range (news): {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"\n  Merged dataset shape: {merged_df.shape}")
    print(f"  Merged date range: {merged_df.index.min().date()} to {merged_df.index.max().date()}")
    print(f"\n  Columns: {list(merged_df.columns)}")
    print(f"\n  First 5 rows:\n{merged_df.head().to_string()}")
    print(f"\n  Last 5 rows:\n{merged_df.tail().to_string()}")
    return merged_df

def run_phase1(btc_csv=None, output_csv=None, api_key=None):
    btc_csv = btc_csv or (PROJECT_ROOT / "Bitcoin Historical Data 5year.csv")
    output_csv = output_csv or (PROJECT_ROOT / "step1_merged_data.csv")
    api_key = api_key or os.environ.get("NEWSAPI_KEY", "").strip()
    if not api_key:
        print("ERROR: Set NEWSAPI_KEY", file=sys.stderr)
        sys.exit(1)
    if not Path(btc_csv).is_file():
        print(f"ERROR: BTC CSV not found: {btc_csv}", file=sys.stderr)
        sys.exit(1)
    print("=" * 60)
    print("STEP 1: DATA INGESTION (NewsAPI + VADER + Impact)")
    print("=" * 60)
    print("\n[1A] Loading BTC/USD price data...")
    price_df = load_btc_price_data(btc_csv)
    print(f"  Shape: {price_df.shape}")
    print(f"  Date range: {price_df.index.min().date()} to {price_df.index.max().date()}")
    date_to = price_df.index.max().normalize()
    lookback_days = 7
    date_from = max(price_df.index.min().normalize(), date_to - timedelta(days=lookback_days - 1))
    print("\n" + "-" * 60)
    print(f"[1B] NewsAPI {date_from.date()} .. {date_to.date()} (~{lookback_days}d) ...")
    news_df = fetch_news_newsapi(api_key, date_from, date_to)
    print(f"  Articles from NewsAPI: {len(news_df)}")
    extra_path = PROJECT_ROOT / "news_data_batch1.csv"
    if extra_path.is_file():
        csv_part = load_local_news_csv(extra_path)
        news_df = pd.concat([news_df, csv_part], ignore_index=True)
        news_df = news_df.drop_duplicates(subset=["Date", "Headline"]).reset_index(drop=True)
        print(f"  Merged local CSV: +{len(csv_part)} articles -> {len(news_df)} total")
    if news_df.empty:
        print("ERROR: No news articles", file=sys.stderr)
        sys.exit(1)
    print("\n" + "-" * 60)
    print("[1C] VADER sentiment + impact...")
    news_df = score_sentiment_discrete(news_df)
    print("  Sentiment_Code:", news_df["Sentiment_Code"].value_counts().to_dict())
    print("  Impact_Category:", news_df["Impact_Category"].value_counts().to_dict())
    print("\n" + "-" * 60)
    print("[1D] Merge...")
    merged_df = aggregate_and_merge(price_df, news_df)
    merged_df.to_csv(output_csv)
    if len(merged_df) < 5:
        print("\n  NOTE: Few merged rows — NewsAPI developer tier limits (e.g. 100 articles, date window). Paid tier or daily news archives yield longer history.")
    print(f"\n  Saved: {output_csv}")
    print("STEP 1 COMPLETE!")
    return merged_df

if __name__ == "__main__":
    run_phase1()
