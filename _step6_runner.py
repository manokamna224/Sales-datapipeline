import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'C:\Users\manok\AppData\Roaming\Python\Python314\site-packages')

import matplotlib
matplotlib.use('Agg')
import step6_ml_model as m

X, y, feature_cols = m.load_and_prepare(m.CLEAN_PATH)
X_train, X_test, y_train, y_test = m.split_data(X, y)
m.cross_validate_baseline(X_train, y_train)
best_model = m.tune_model(X_train, y_train)
metrics, y_pred = m.evaluate_model(best_model, X_test, y_test)

r2_key = [k for k in metrics if 'R' in k][0]
m._eval_cache.update({'y_test': y_test.values, 'y_pred': y_pred, 'r2': metrics[r2_key]})
m.plot_feature_importance(best_model, feature_cols)
m.save_model(best_model, feature_cols)

print()
print("Sample predictions (5 orders):")
print(f"  {'Order':<8} {'Predicted':>12} {'Actual':>12} {'Error':>10}")
print("  " + "-" * 46)
sample = X_test.iloc[:5]
preds = best_model.predict(sample)
actuals = y_test.iloc[:5].values
for i, (p, a) in enumerate(zip(preds, actuals)):
    print(f"  Order {i+1:<3}  ${p:>10,.2f}  ${a:>10,.2f}  ${abs(p-a):>8,.2f}")

print()
print("Step 6 complete!")
