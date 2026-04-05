#!/usr/bin/env python3
"""
CLI entrypoint for BTC Hybrid Trading System pipeline.

Commands:
    phase1    — Data ingestion (NewsAPI + VADER + merge)
    phase2    — Feature engineering (technical + sentiment decay)
    phase3    — Model training (baseline vs hybrid)
    tuning    — Hyperparameter tuning (RandomizedSearchCV + TimeSeriesSplit)
    aligned   — Time-aligned evaluation (Model A vs AA vs B)
    phase4    — Evaluation & visualization dashboard
    trading   — Trading runtime (phases 5-8)
    backtest  — Quantitative backtesting (equity curve, Sharpe, vs Buy-and-Hold)
    all       — Run phase2 → phase3 → aligned → phase4 → trading → backtest (no tuning)
    full      — Run phase1 first, then same as 'all'

Options:
    --tune    — With 'all' or 'full' only: insert step3b after phase3 (slow).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Default pipeline: skip step3b (fast). Use `all --tune` or `full --tune` to include tuning.
PIPELINE_SCRIPTS_FAST = [
    "step2_feature_engineering.py",
    "step3_model_training.py",
    "step4_time_aligned_evaluation.py",
    "step4_evaluation.py",
    "btc_trading_runtime.py",
    "step5_backtesting.py",
]

STEP3B_SCRIPT = "step3b_hyperparameter_tuning.py"


def pipeline_scripts(include_tuning: bool) -> list[str]:
    if not include_tuning:
        return list(PIPELINE_SCRIPTS_FAST)
    out: list[str] = []
    for name in PIPELINE_SCRIPTS_FAST:
        out.append(name)
        if name == "step3_model_training.py":
            out.append(STEP3B_SCRIPT)
    return out

# Map command names to script files
COMMAND_MAP = {
    "phase1":   "step1_data_ingestion.py",
    "phase2":   "step2_feature_engineering.py",
    "phase3":   "step3_model_training.py",
    "tuning":   "step3b_hyperparameter_tuning.py",
    "aligned":  "step4_time_aligned_evaluation.py",
    "phase4":   "step4_evaluation.py",
    "trading":  "btc_trading_runtime.py",
    "backtest": "step5_backtesting.py",
}


def run_script(name: str) -> int:
    """Run a Python script and return its exit code."""
    path = ROOT / name
    if not path.is_file():
        print(f"Missing: {path}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(path)], cwd=str(ROOT))


def run_full_pipeline(
    include_phase1: bool = False,
    *,
    include_tuning: bool = False,
) -> int:
    """Run the complete pipeline in order."""
    if include_phase1:
        print("=" * 60)
        print("step1_data_ingestion.py")
        print("=" * 60)
        rc = run_script("step1_data_ingestion.py")
        if rc != 0:
            print(f"Stopped at phase1: exit {rc}", file=sys.stderr)
            return rc

    scripts = pipeline_scripts(include_tuning)
    if include_tuning:
        print("\n  (--tune) Including step3b_hyperparameter_tuning.py\n")
    for script in scripts:
        print("\n" + "=" * 60)
        print(script)
        print("=" * 60)
        rc = run_script(script)
        if rc != 0:
            print(f"Stopped at {script}: exit {rc}", file=sys.stderr)
            return rc

    print("\n" + "=" * 60)
    print("FULL PIPELINE COMPLETE!")
    print("=" * 60)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="BTC Hybrid Trading System pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "command",
        choices=list(COMMAND_MAP.keys()) + ["all", "full"],
        help="Pipeline step to run (or 'all'/'full' for the complete pipeline)",
    )
    ap.add_argument(
        "--tune",
        action="store_true",
        help="With 'all' or 'full' only: run step3b hyperparameter tuning (slow).",
    )
    args = ap.parse_args()

    # Single step
    if args.command in COMMAND_MAP:
        if args.tune:
            print(
                "Note: --tune applies only to 'all' or 'full'; ignoring.",
                file=sys.stderr,
            )
        return run_script(COMMAND_MAP[args.command])

    # Full pipeline
    if args.command == "full":
        return run_full_pipeline(include_phase1=True, include_tuning=args.tune)

    # All (skip phase1)
    return run_full_pipeline(include_phase1=False, include_tuning=args.tune)


if __name__ == "__main__":
    sys.exit(main())
