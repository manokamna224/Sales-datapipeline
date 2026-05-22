"""
Step 6 — Random Forest Revenue Prediction
Trains a Random Forest regressor to predict order revenue using scikit-learn.
Covers:
  • Feature selection & encoding
  • Train/test split
  • Cross-validation (5-fold)
  • Hyperparameter tuning (GridSearchCV)
  • Feature importance analysis
  • Model persistence (joblib)
  • Evaluation metrics: RMSE, MAE, R²
Output: outputs/feature_importance.png, models/revenue_model.joblib
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

CLEAN_PATH = "data/cleaned_sales_data.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/revenue_model.joblib"
OUTPUT_DIR = "outputs"
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load cleaned data and engineer ML-ready features."""
    df = pd.read_csv(path, parse_dates=["order_date"])
    print(f"✅ Loaded {len(df):,} rows")

    # ── Feature engineering ───────────────────────────────────────────────────
    df["month"]       = df["order_date"].dt.month
    df["quarter"]     = df["order_date"].dt.quarter
    df["day_of_week"] = df["order_date"].dt.dayofweek   # 0=Mon … 6=Sun
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["year"]        = df["order_date"].dt.year

    # Encode categoricals with LabelEncoder
    cat_cols = ["region", "product", "payment_method", "order_status", "sales_rep"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # ── Feature matrix ────────────────────────────────────────────────────────
    feature_cols = [
        # Numeric
        "quantity", "unit_price", "discount",
        # Date-derived
        "month", "quarter", "day_of_week", "is_weekend", "year",
        # Encoded categoricals
        "region_enc", "product_enc", "payment_method_enc",
        "order_status_enc", "sales_rep_enc",
    ]

    X = df[feature_cols].copy()
    y = df["revenue"]

    print(f"   Features : {X.shape[1]}")
    print(f"   Target   : revenue  (mean={y.mean():.2f}, std={y.std():.2f})")
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"\n   Train set : {len(X_train):,} rows")
    print(f"   Test set  : {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATION (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_baseline(X_train, y_train) -> None:
    print("\n── Cross-Validation (5-fold, baseline RF) ───────────────")
    # n_jobs=1 avoids joblib multiprocessing warning flood on Python 3.14
    baseline = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=1)
    cv_scores = cross_val_score(
        baseline, X_train, y_train,
        cv=5, scoring="r2", n_jobs=1
    )
    rmse_scores = np.sqrt(-cross_val_score(
        baseline, X_train, y_train,
        cv=5, scoring="neg_mean_squared_error", n_jobs=1
    ))
    print(f"   R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   RMSE : {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(X_train, y_train) -> RandomForestRegressor:
    print("\n── Hyperparameter Tuning (GridSearchCV) ─────────────────")

    # Reduced grid — 12 combos × 3 folds = 36 fits (was 144)
    # n_jobs=1 avoids joblib multiprocessing warning flood on Python 3.14
    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 10],
        "min_samples_split": [2, 5],
        "max_features":      ["sqrt", "log2"],
    }

    rf = RandomForestRegressor(random_state=SEED, n_jobs=1)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best = grid_search.best_estimator_
    print(f"   Best params : {grid_search.best_params_}")
    print(f"   Best CV R²  : {grid_search.best_score_:.4f}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    print("\n── Test Set Evaluation ──────────────────────────────────")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100

    metrics = {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE (%)": mape}
    for name, val in metrics.items():
        print(f"   {name:<10}: {val:.4f}")

    return metrics, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols: list[str]) -> None:
    print("\n── Feature Importance ───────────────────────────────────")
    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
          .sort_values("importance", ascending=True)
    )

    palette = sns.color_palette("Blues_r", len(fi_df))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
    fig.suptitle("Random Forest — Revenue Prediction", fontsize=15, fontweight="bold")

    # ── Left: Feature importance bar chart ───────────────────────────────────
    ax = axes[0]
    bars = ax.barh(fi_df["feature"], fi_df["importance"],
                   color=palette, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, fi_df["importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_title("Feature Importance (Mean Decrease Impurity)", pad=10)
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, fi_df["importance"].max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.4)

    # ── Right: Actual vs Predicted scatter ───────────────────────────────────
    ax2 = axes[1]
    # Load test predictions stored in outer scope via closure trick
    # (passed via global for simplicity)
    y_test_vals = _eval_cache.get("y_test")
    y_pred_vals = _eval_cache.get("y_pred")

    if y_test_vals is not None and y_pred_vals is not None:
        ax2.scatter(y_test_vals, y_pred_vals, alpha=0.4, s=18,
                    color="#2196F3", edgecolors="none")
        lims = [
            min(y_test_vals.min(), y_pred_vals.min()),
            max(y_test_vals.max(), y_pred_vals.max()),
        ]
        ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax2.set_xlabel("Actual Revenue ($)")
        ax2.set_ylabel("Predicted Revenue ($)")
        ax2.set_title("Actual vs Predicted Revenue", pad=10)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.legend(fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)

        r2 = _eval_cache.get("r2", 0)
        ax2.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax2.transAxes,
                 fontsize=10, color="#E91E63", fontweight="bold")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"   💾 Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, feature_cols: list[str]) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {"model": model, "feature_cols": feature_cols}
    joblib.dump(payload, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")


def load_model():
    """Utility: load saved model for inference."""
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_cols"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# Cache for sharing eval results between functions
_eval_cache: dict = {}

if __name__ == "__main__":
    print("=" * 60)
    print("  Sales Data Pipeline — Step 6: ML Revenue Prediction")
    print("=" * 60)

    X, y, feature_cols = load_and_prepare(CLEAN_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    cross_validate_baseline(X_train, y_train)
    best_model = tune_model(X_train, y_train)
    metrics, y_pred = evaluate_model(best_model, X_test, y_test)

    # Store for plotting
    _eval_cache["y_test"] = y_test.values
    _eval_cache["y_pred"] = y_pred
    _eval_cache["r2"]     = metrics["R²"]

    plot_feature_importance(best_model, feature_cols)
    save_model(best_model, feature_cols)

    # ── Sample prediction ─────────────────────────────────────────────────────
    print("\n── Sample Prediction ────────────────────────────────────")
    sample = X_test.iloc[:3].copy()
    preds = best_model.predict(sample)
    actuals = y_test.iloc[:3].values
    for i, (pred, actual) in enumerate(zip(preds, actuals)):
        print(f"   Order {i+1}: Predicted=${pred:,.2f}  |  Actual=${actual:,.2f}  "
              f"|  Error={abs(pred-actual):,.2f}")

    print("\n✅ Step 6 complete!")
