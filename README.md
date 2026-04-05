# BTC Hybrid Sentiment-Technical Trading System

> **Research Question:** Does News Sentiment provide a measurable predictive edge over Technical Indicators alone for BTC/USD direction forecasting?

---

## The Problem

Most hybrid trading models compare a "technical-only" model trained on years of price history against a "hybrid" model trained on just weeks of sentiment data. This creates an unfair comparison: any performance difference could be caused by the **different training window sizes**, not by the sentiment features themselves.

In this project, we have **5 years** of BTC/USD price data but only **~35 days** of news sentiment data. A naive comparison would be scientifically invalid.

## The Solution: Time-Aligned Evaluation

We solve this with a controlled experiment using a **Universal Test Set** and **three models** that isolate sentiment as the only variable:

| Model | Training Window | Features | Purpose |
|-------|----------------|----------|---------|
| **Model A** | ~5 years (full history) | Technical only | Long-term baseline |
| **Model AA** | 25 days (short window) | Technical only | **Controlled baseline** |
| **Model B** | 25 days (short window) | Technical + Sentiment | Hybrid candidate |

**The key insight:** By comparing **Model B vs Model AA**, we hold the training window constant and isolate sentiment as the *only* difference. Any performance gap is directly attributable to the news sentiment features.

All three models are evaluated on the **same 10-day Universal Test Set** (the last 10 trading days), ensuring an apples-to-apples comparison.

---

## Project Architecture

```
BTC_Hybrid_Trading_System/
│
├── run_pipeline.py                    # CLI entrypoint (run any step or full pipeline)
│
├── step1_data_ingestion.py            # Phase 1: NewsAPI + VADER sentiment + merge
├── step2_feature_engineering.py       # Phase 2: Technical indicators + sentiment decay
├── step3_model_training.py            # Phase 3: Baseline model training (RF + XGBoost)
├── step3b_hyperparameter_tuning.py    # Phase 3b: RandomizedSearchCV + TimeSeriesSplit
├── step4_time_aligned_evaluation.py   # Phase 4a: Time-aligned 3-model comparison (A/AA/B)
├── step4_evaluation.py                # Phase 4b: Visualization dashboard + analysis
├── btc_trading_runtime.py             # Phase 5-8: Live trading simulation
├── hybrid_model_pick.py               # Shared hybrid (B_*) selection: F1, then accuracy
├── step5_backtesting.py               # Backtesting: Equity curve, Sharpe, vs Buy-and-Hold
│
├── Bitcoin Historical Data 5year.csv  # Source: BTC/USD OHLCV (5 years)
├── news_data_batch1.csv               # Source: Local news headlines archive
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
└── outputs/                           # Generated charts and dashboards (auto-created)
```

---

## Methodology

### Phase 1 — Data Ingestion (`step1_data_ingestion.py`)
- Loads 5 years of BTC/USD OHLCV data from CSV
- Fetches recent crypto news from NewsAPI
- Scores each headline with VADER sentiment (-1 / 0 / +1)
- Applies impact weights by category (Regulatory: 1.5x, Macro: 1.3x, Technical: 1.0x, Social: 0.7x)
- Merges price + sentiment into a single daily dataset

### Phase 2 — Feature Engineering (`step2_feature_engineering.py`)
- Computes 21 technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, etc.)
- Engineers 27 sentiment-derived features (decay, momentum, rolling stats)
- Creates binary target: `Target = 1` if next-day Close > today's Close

### Phase 3 — Model Training (`step3_model_training.py`)
- Trains baseline and hybrid classifiers (RandomForest, XGBoost)
- Chronological train/test split (no shuffling, no future leakage)
- Saves trained models and evaluation metrics

### Phase 3b — Hyperparameter Tuning (`step3b_hyperparameter_tuning.py`)
- RandomizedSearchCV with TimeSeriesSplit cross-validation
- Tunes both RandomForest and XGBoost hyperparameters
- Respects temporal ordering throughout

### Phase 4a — Time-Aligned Evaluation (`step4_time_aligned_evaluation.py`)
**This is the core scientific evaluation.** It implements the 3-model comparison:
- Builds a unified dataset via LEFT join (preserves full 5yr history)
- Carves a Universal Test Set (last 10 days)
- Trains all 3 models with `RandomForestClassifier(max_depth=3, random_state=42)`
- StandardScaler fitted only on each model's own training data (no leakage)
- Outputs a consolidated benchmarking table and automatic **Quantitative Verdict**

### Phase 4b — Visualization Dashboard (`step4_evaluation.py`)
- Generates ROC curves, confusion matrices, and feature importance charts
- Produces sentiment decay analysis plots
- Saves all visualizations to `outputs/`

### Phase 5-8 — Trading Runtime (`btc_trading_runtime.py`)
- Simulates live trading with position management
- Implements signal generation from trained models

### Backtesting (`step5_backtesting.py`)
- Full backtest with equity curve visualization
- Computes Sharpe ratio, max drawdown, and win rate
- Compares strategy returns vs Buy-and-Hold benchmark

---

## How to Run

### Prerequisites

```
Python 3.9+  (see pinned versions in requirements.txt)
```

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your NewsAPI Key

**PowerShell:**
```powershell
$env:NEWSAPI_KEY = "your_api_key_here"
```

**CMD:**
```cmd
set NEWSAPI_KEY=your_api_key_here
```

**Linux / macOS:**
```bash
export NEWSAPI_KEY="your_api_key_here"
```

> Get a free key at [https://newsapi.org](https://newsapi.org)

### 3. Run the Pipeline

By default, `all` / `full` **skip hyperparameter tuning (step3b)** for a faster run. Backtests then use **step 3** models (or existing step3b pickles if you already ran tuning). Add **`--tune`** to run step3b as well.

**Full pipeline (includes data ingestion):**
```bash
python run_pipeline.py full
```

**Same, with hyperparameter tuning (slow):**
```bash
python run_pipeline.py full --tune
```

**Skip data ingestion (if you already have `step1_merged_data.csv`):**
```bash
python run_pipeline.py all
```

**With tuning:**
```bash
python run_pipeline.py all --tune
```

**Run individual steps:**
```bash
python run_pipeline.py phase1      # Data ingestion
python run_pipeline.py phase2      # Feature engineering
python run_pipeline.py phase3      # Model training
python run_pipeline.py tuning      # Hyperparameter tuning
python run_pipeline.py aligned     # Time-aligned evaluation (3-model comparison)
python run_pipeline.py phase4      # Visualization dashboard
python run_pipeline.py trading     # Trading simulation
python run_pipeline.py backtest    # Backtesting
```

### 4. Run the Scientific Evaluation Directly

```bash
python step4_time_aligned_evaluation.py
```

### 5. Repo hygiene (optional)

- **Do not commit** `.venv/` — use `requirements.txt` and a local venv.
- If `.venv` was ever committed, install [git-filter-repo](https://github.com/newren/git-filter-repo) and **rewrite history** (coordinate with anyone who cloned the repo):

```bash
git filter-repo --path .venv --invert-paths --force
git push origin main --force
```

`--force` overwrites the remote branch; teammates must re-clone or hard-reset.

---

## Final Results & Verdict

> **Paste the output from `python step4_time_aligned_evaluation.py` below:**

```
(Run the evaluation and paste results here)
```

<!--
Expected output format:

┌─────────────────────────────────────────────────────────────┐
│                   BENCHMARKING TABLE                        │
├──────────┬──────────┬───────────┬────────┬─────────────────┤
│ Model    │ Accuracy │ Precision │ Recall │ F1-Score        │
├──────────┼──────────┼───────────┼────────┼─────────────────┤
│ Model A  │  0.XXXX  │  0.XXXX   │ 0.XXXX │ 0.XXXX         │
│ Model AA │  0.XXXX  │  0.XXXX   │ 0.XXXX │ 0.XXXX         │
│ Model B  │  0.XXXX  │  0.XXXX   │ 0.XXXX │ 0.XXXX         │
└──────────┴──────────┴───────────┴────────┴─────────────────┘

QUANTITATIVE VERDICT:
  Q1: Does training window length matter? → ...
  Q2: Does sentiment add predictive value? → ...
-->

---

## Technical Details

**Two evaluation tracks (on purpose):**

- **`step4_time_aligned_evaluation.py`** — Last **10** rows = universal test set; Models A, AA, and B all use `RandomForestClassifier(max_depth=3, random_state=42)`; sentiment outside the news window is **zero-filled** in that script’s unified dataset. Use this for the controlled “sentiment vs technical-only” comparison.
- **Step 3 / step 3b, `phase4` dashboards, backtest, and trading runtime** — **80/20** chronological split; Random Forest (deeper, more trees) and optional XGBoost; sentiment **before** the first news date uses a **linear ramp** in `build_training_dataframe()`. **Backtests and trading load the step 3 (or 3b) pickles only** — not the A/AA/B models from the time-aligned script.

**Hybrid used for backtest + trading:** Among `B_XGB` and `B_RF`, the code picks **highest F1**, then accuracy (`hybrid_model_pick.py`), so runtime matches the backtest.

**Shared methodology:** `StandardScaler` fit on training data only; chronological splits; VADER-based sentiment with category weights.

---

## License

This project is for educational and research purposes.
