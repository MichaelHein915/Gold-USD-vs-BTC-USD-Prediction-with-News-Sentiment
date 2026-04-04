"""
Step 3B: Hyperparameter Tuning
===============================
- Use RandomizedSearchCV with TimeSeriesSplit for proper time-series CV
- Tune: n_estimators, max_depth, min_samples_split, learning_rate
- Tune both Model A (Technical) and Model B (Hybrid)
- Select best model based on cross-validation F1 score
- Save tuned models for backtesting (step5)
"""
from __future__ import annotations

import pickle
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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent

from step3_model_training import (
    build_training_dataframe,
    chronological_split,
    create_target,
    get_feature_sets,
)


# ============================================================
# HYPERPARAMETER SEARCH SPACES
# ============================================================

# Random Forest search space
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

# XGBoost search space
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

# Number of random combinations to try
N_ITER = 30

# Number of CV folds for TimeSeriesSplit
N_SPLITS = 5


# ============================================================
# TUNING FUNCTION
# ============================================================

def tune_model(X_train, y_train, model, param_grid, model_name,
               n_iter=N_ITER, n_splits=N_SPLITS):
    """
    Tune a model using RandomizedSearchCV with TimeSeriesSplit.

    Parameters:
        X_train:    Scaled training features
        y_train:    Training labels
        model:      Base estimator (RF or XGB)
        param_grid: Dict of parameter distributions
        model_name: Display name for logging
        n_iter:     Number of random combinations to try
        n_splits:   Number of CV folds

    Returns:
        best_model:  Fitted model with best parameters
        best_params: Dict of best hyperparameters
        cv_results:  Full CV results DataFrame
    """
    print(f"\n  Tuning {model_name}...")
    print(f"    Search space: {n_iter} random combinations")
    print(f"    CV strategy:  TimeSeriesSplit with {n_splits} folds")

    # TimeSeriesSplit respects chronological order
    time_series_cv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=time_series_cv,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = search.best_score_

    # Build results DataFrame
    cv_results = pd.DataFrame(search.cv_results_)

    print(f"\n    Best CV F1 Score: {best_cv_score:.4f}")
    print(f"    Best Parameters:")
    for param, value in sorted(best_params.items()):
        print(f"      {param}: {value}")

    return best_model, best_params, cv_results


# ============================================================
# EVALUATE TUNED MODEL ON TEST SET
# ============================================================

def evaluate_on_test(model, scaler, X_test, y_test, model_name):
    """
    Evaluate a tuned model on the held-out test set.
    Returns dict with model, scaler, and all metrics.
    """
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  {model_name} — Test Set Results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"\n    Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Down (0)", "Up (1)"],
        zero_division=0,
    ))

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
# COMPARE TUNED vs UNTUNED
# ============================================================

def compare_tuned_vs_untuned(tuned_results, untuned_results):
    """
    Print side-by-side comparison of tuned vs untuned models.
    Returns the overall best model key and result dict.
    """
    print("\n" + "=" * 70)
    print("TUNED vs. UNTUNED COMPARISON")
    print("=" * 70)

    metrics = ["accuracy", "precision", "recall", "f1"]
    header = f"  {'Model':<45} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print(header)
    print("  " + "-" * 73)

    all_results = {}

    # Print untuned
    for key, res in untuned_results.items():
        label = f"[Untuned] {res['model_name']}"
        if len(label) > 44:
            label = label[:44]
        print(f"  {label:<45} {res['accuracy']:>7.4f} {res['precision']:>7.4f} "
              f"{res['recall']:>7.4f} {res['f1']:>7.4f}")
        all_results[f"untuned_{key}"] = res

    print("  " + "-" * 73)

    # Print tuned
    for key, res in tuned_results.items():
        label = f"[TUNED]  {res['model_name']}"
        if len(label) > 44:
            label = label[:44]
        print(f"  {label:<45} {res['accuracy']:>7.4f} {res['precision']:>7.4f} "
              f"{res['recall']:>7.4f} {res['f1']:>7.4f}")
        all_results[f"tuned_{key}"] = res

    # Find the overall best by F1
    best_key = max(all_results.keys(),
                   key=lambda k: (all_results[k]["f1"], all_results[k]["accuracy"]))
    best = all_results[best_key]

    print(f"\n  OVERALL BEST: {best['model_name']}")
    print(f"    F1={best['f1']:.4f}  Accuracy={best['accuracy']:.4f}")
    is_tuned = best_key.startswith("tuned_")
    print(f"    Source: {'TUNED' if is_tuned else 'UNTUNED'}")

    return best_key, best


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 70)
    print("STEP 3B: HYPERPARAMETER TUNING")
    print("=" * 70)

    # ── Load data ──
    print("\n[3B-1] Building training data...")
    df = build_training_dataframe()
    df = create_target(df)

    print("\n[3B-2] Feature sets...")
    technical_features, sentiment_features, hybrid_features = get_feature_sets(df)

    print("\n[3B-3] Chronological split 80/20...")
    X_train_full, X_test_full, y_train, y_test, train_df, test_df = \
        chronological_split(df, hybrid_features, test_ratio=0.2)

    X_train_tech = X_train_full[technical_features]
    X_test_tech = X_test_full[technical_features]

    # ── Scale features ──
    scaler_tech = StandardScaler()
    X_train_tech_scaled = scaler_tech.fit_transform(X_train_tech)

    scaler_hybrid = StandardScaler()
    X_train_hybrid_scaled = scaler_hybrid.fit_transform(X_train_full)

    # ── Check XGBoost ──
    try:
        from xgboost import XGBClassifier
        xgb_available = True
        print("\n  XGBoost: available")
    except ImportError:
        xgb_available = False
        print("\n  XGBoost: not installed (tuning RF only)")

    # ── Load untuned results for comparison ──
    untuned_path = PROJECT_ROOT / "step3_results.pkl"
    if untuned_path.is_file():
        with open(untuned_path, "rb") as f:
            untuned_results = pickle.load(f)
        print(f"  Loaded untuned results from step3 for comparison")
    else:
        untuned_results = {}
        print("  No untuned results found (step3 not run yet)")

    # ================================================================
    # TUNE MODEL A: TECHNICAL ONLY
    # ================================================================
    print("\n" + "=" * 70)
    print("TUNING MODEL A: TECHNICAL ONLY")
    print("=" * 70)

    tuned_results = {}

    # ── Tune Random Forest (Technical) ──
    best_rf_tech, best_rf_tech_params, _ = tune_model(
        X_train_tech_scaled, y_train,
        RandomForestClassifier(random_state=42, n_jobs=-1),
        RF_PARAM_GRID,
        "Model A - Random Forest (Technical)",
    )
    tuned_results["A_RF"] = evaluate_on_test(
        best_rf_tech, scaler_tech, X_test_tech, y_test,
        "Model A - Tuned RF (Technical)",
    )

    # ── Tune XGBoost (Technical) ──
    if xgb_available:
        best_xgb_tech, best_xgb_tech_params, _ = tune_model(
            X_train_tech_scaled, y_train,
            XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            XGB_PARAM_GRID,
            "Model A - XGBoost (Technical)",
        )
        tuned_results["A_XGB"] = evaluate_on_test(
            best_xgb_tech, scaler_tech, X_test_tech, y_test,
            "Model A - Tuned XGB (Technical)",
        )

    # ================================================================
    # TUNE MODEL B: HYBRID (TECHNICAL + SENTIMENT)
    # ================================================================
    print("\n" + "=" * 70)
    print("TUNING MODEL B: HYBRID (TECHNICAL + SENTIMENT)")
    print("=" * 70)

    # ── Tune Random Forest (Hybrid) ──
    best_rf_hybrid, best_rf_hybrid_params, _ = tune_model(
        X_train_hybrid_scaled, y_train,
        RandomForestClassifier(random_state=42, n_jobs=-1),
        RF_PARAM_GRID,
        "Model B - Random Forest (Hybrid)",
    )
    tuned_results["B_RF"] = evaluate_on_test(
        best_rf_hybrid, scaler_hybrid, X_test_full, y_test,
        "Model B - Tuned RF (Hybrid)",
    )

    # ── Tune XGBoost (Hybrid) ──
    if xgb_available:
        best_xgb_hybrid, best_xgb_hybrid_params, _ = tune_model(
            X_train_hybrid_scaled, y_train,
            XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            XGB_PARAM_GRID,
            "Model B - XGBoost (Hybrid)",
        )
        tuned_results["B_XGB"] = evaluate_on_test(
            best_xgb_hybrid, scaler_hybrid, X_test_full, y_test,
            "Model B - Tuned XGB (Hybrid)",
        )

    # ================================================================
    # COMPARE TUNED vs UNTUNED
    # ================================================================
    if untuned_results:
        best_key, best_result = compare_tuned_vs_untuned(tuned_results, untuned_results)
    else:
        print("\n  Skipping comparison (no untuned results)")

    # ================================================================
    # SAVE TUNED RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("SAVING TUNED MODELS")
    print("=" * 70)

    tuned_results_path = PROJECT_ROOT / "step3b_tuned_results.pkl"
    tuned_data_path = PROJECT_ROOT / "step3b_tuned_data.pkl"

    with open(tuned_results_path, "wb") as f:
        pickle.dump(tuned_results, f)

    with open(tuned_data_path, "wb") as f:
        pickle.dump({
            "technical_features": technical_features,
            "sentiment_features": sentiment_features,
            "hybrid_features": hybrid_features,
            "X_test_full": X_test_full,
            "X_test_tech": X_test_tech,
            "y_test": y_test,
            "test_df": test_df,
        }, f)

    print(f"  Saved: {tuned_results_path}")
    print(f"  Saved: {tuned_data_path}")

    # ── Print best parameters for reference ──
    print("\n" + "-" * 70)
    print("BEST HYPERPARAMETERS (for reference):")
    print("-" * 70)

    for key, res in tuned_results.items():
        model = res["model"]
        params = model.get_params()
        print(f"\n  {res['model_name']}:")

        # Only print the tuned params (not all sklearn defaults)
        relevant_params = ["n_estimators", "max_depth", "min_samples_split",
                           "min_samples_leaf", "max_features",
                           "learning_rate", "min_child_weight",
                           "subsample", "colsample_bytree"]
        for p in relevant_params:
            if p in params:
                print(f"    {p}: {params[p]}")

    print("\n" + "=" * 70)
    print("STEP 3B COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
