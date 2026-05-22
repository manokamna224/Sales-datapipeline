"""
run_pipeline.py — Execute all 6 steps in sequence.
Usage:
    python run_pipeline.py              # run all steps
    python run_pipeline.py --steps 2 3  # run specific steps
    python run_pipeline.py --skip 4     # skip PySpark step
"""

import argparse
import importlib
import sys
import time
import traceback


STEPS = {
    1: ("step1_setup",          "Environment Setup"),
    2: ("step2_generate_data",  "Data Generation"),
    3: ("step3_pandas_pipeline","Pandas Pipeline"),
    4: ("step4_pyspark_pipeline","PySpark Pipeline"),
    5: ("step5_dashboard",      "Dashboard"),
    6: ("step6_ml_model",       "ML Model"),
}


def run_step(step_num: int) -> bool:
    module_name, label = STEPS[step_num]
    print(f"\n{'='*60}")
    print(f"  ▶  Step {step_num}: {label}")
    print(f"{'='*60}")
    start = time.time()
    try:
        mod = importlib.import_module(module_name)
        # Each step runs its logic in __main__ block;
        # for programmatic use we call the primary function directly.
        fn_map = {
            1: lambda: mod.install_packages(mod.REQUIRED_PACKAGES),
            2: lambda: mod.save_data(mod.generate_sales_data(), "data/raw_sales_data.csv"),
            3: lambda: _run_step3(mod),
            4: lambda: _run_step4(mod),
            5: lambda: mod.build_dashboard(),
            6: lambda: _run_step6(mod),
        }
        fn_map[step_num]()
        elapsed = time.time() - start
        print(f"\n  ✅ Step {step_num} completed in {elapsed:.1f}s")
        return True
    except Exception as exc:
        print(f"\n  ❌ Step {step_num} failed: {exc}")
        traceback.print_exc()
        return False


def _run_step3(mod):
    df_raw = mod.load_data(mod.RAW_PATH)
    df_clean = mod.clean_data(df_raw)
    df_feat = mod.engineer_features(df_clean)
    aggs = mod.aggregate_data(df_feat)
    mod.save_outputs(df_feat, aggs)


def _run_step4(mod):
    spark = mod.create_spark_session()
    df = mod.load_data(spark, mod.CLEAN_CSV)
    df = mod.transform_data(df)
    df = mod.apply_window_functions(df)
    aggs = mod.aggregate_data(df)
    mod.write_parquet(df, aggs, mod.PARQUET_OUT)
    spark.stop()


def _run_step6(mod):
    X, y, feature_cols = mod.load_and_prepare(mod.CLEAN_PATH)
    X_train, X_test, y_train, y_test = mod.split_data(X, y)
    mod.cross_validate_baseline(X_train, y_train)
    best_model = mod.tune_model(X_train, y_train)
    metrics, y_pred = mod.evaluate_model(best_model, X_test, y_test)
    mod._eval_cache.update({"y_test": y_test.values, "y_pred": y_pred, "r2": metrics["R²"]})
    mod.plot_feature_importance(best_model, feature_cols)
    mod.save_model(best_model, feature_cols)


def main():
    parser = argparse.ArgumentParser(description="Sales Data Pipeline Runner")
    parser.add_argument("--steps", nargs="+", type=int,
                        help="Steps to run (e.g. --steps 2 3 5)")
    parser.add_argument("--skip", nargs="+", type=int, default=[],
                        help="Steps to skip (e.g. --skip 4)")
    args = parser.parse_args()

    steps_to_run = sorted(args.steps) if args.steps else list(STEPS.keys())
    steps_to_run = [s for s in steps_to_run if s not in args.skip]

    print("\n" + "=" * 60)
    print("  🚀 Sales Data Pipeline — Full Run")
    print(f"  Steps: {steps_to_run}")
    print("=" * 60)

    results = {}
    total_start = time.time()
    for step in steps_to_run:
        results[step] = run_step(step)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  📊 Pipeline Summary")
    print("=" * 60)
    for step, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  Step {step} — {STEPS[step][1]:<25} {status}")
    print(f"\n  Total time: {total_elapsed:.1f}s")

    failed = [s for s, ok in results.items() if not ok]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
