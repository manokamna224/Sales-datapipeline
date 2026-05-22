"""
Internal runner — injects user site-packages into sys.path before
importing any project module. Run this instead of the step files directly
if packages are installed in the user site-packages location.

Usage:
    python _run.py step2
    python _run.py step3
    python _run.py step5
    python _run.py step6
    python _run.py all          # steps 2,3,5,6 (skips PySpark)
"""
import sys
import os
import warnings

# Suppress all warnings globally (including sklearn joblib worker warnings)
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Ensure user site-packages are on the path
USER_PKGS = r"C:\Users\manok\AppData\Roaming\Python\Python314\site-packages"
if USER_PKGS not in sys.path:
    sys.path.insert(0, USER_PKGS)

# Also add the sales_pipeline directory itself
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Set non-interactive matplotlib backend BEFORE any module imports it
import matplotlib
matplotlib.use("Agg")

import argparse

def run_step2():
    import step2_generate_data as m
    df = m.generate_sales_data()
    m.save_data(df, "data/raw_sales_data.csv")

def run_step3():
    import step3_pandas_pipeline as m
    df_raw = m.load_data(m.RAW_PATH)
    df_clean = m.clean_data(df_raw)
    df_feat = m.engineer_features(df_clean)
    aggs = m.aggregate_data(df_feat)
    m.save_outputs(df_feat, aggs)
    print("\n📊 Region summary:")
    print(aggs["region"].to_string(index=False))

def run_step5():
    import step5_dashboard as m
    m.build_dashboard()

def run_step6():
    import step6_ml_model as m
    X, y, feature_cols = m.load_and_prepare(m.CLEAN_PATH)
    X_train, X_test, y_train, y_test = m.split_data(X, y)
    m.cross_validate_baseline(X_train, y_train)
    best_model = m.tune_model(X_train, y_train)
    metrics, y_pred = m.evaluate_model(best_model, X_test, y_test)
    r2_key = [k for k in metrics if "R" in k][0]
    m._eval_cache.update({"y_test": y_test.values, "y_pred": y_pred, "r2": metrics[r2_key]})
    m.plot_feature_importance(best_model, feature_cols)
    m.save_model(best_model, feature_cols)

STEPS = {
    "step2": run_step2,
    "step3": run_step3,
    "step5": run_step5,
    "step6": run_step6,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sales Data Pipeline runner",
        epilog="If no step is given, all steps are run automatically.",
    )
    parser.add_argument(
        "step",
        nargs="?",                          # makes the argument optional
        default="all",                      # default when nothing is passed
        choices=list(STEPS.keys()) + ["all"],
        help="Step to run: step2 | step3 | step5 | step6 | all (default: all)",
    )
    args = parser.parse_args()

    import time
    steps = list(STEPS.keys()) if args.step == "all" else [args.step]
    for step in steps:
        print(f"\n{'='*60}\n  ▶  Running {step}\n{'='*60}")
        t = time.time()
        STEPS[step]()
        print(f"\n  ✅ {step} done in {time.time()-t:.1f}s")
