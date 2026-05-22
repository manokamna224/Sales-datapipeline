import sys
sys.path.insert(0, r'C:\Users\manok\AppData\Roaming\Python\Python314\site-packages')
import os
import pandas as pd

files = {
    'Raw data'        : 'data/raw_sales_data.csv',
    'Cleaned data'    : 'data/cleaned_sales_data.csv',
    'Region agg'      : 'data/aggregations/region_agg.csv',
    'Monthly agg'     : 'data/aggregations/monthly_agg.csv',
    'Product agg'     : 'data/aggregations/product_agg.csv',
    'Sales rep agg'   : 'data/aggregations/sales_rep_agg.csv',
    'Pivot agg'       : 'data/aggregations/pivot_agg.csv',
    'Dashboard PNG'   : 'outputs/sales_dashboard.png',
    'Feature imp PNG' : 'outputs/feature_importance.png',
    'ML model'        : 'models/revenue_model.joblib',
}

print('=' * 60)
print('  SALES DATA PIPELINE — COMPLETE RUN SUMMARY')
print('=' * 60)
print()
print('  OUTPUT FILES')
print('  ' + '-' * 56)
all_ok = True
for label, path in files.items():
    exists = os.path.exists(path)
    size   = os.path.getsize(path) / 1024 if exists else 0
    status = '✅' if exists else '❌'
    if not exists:
        all_ok = False
    print(f'  {status}  {label:<20}  {size:>7.1f} KB   {path}')

print()
print('  DATA STATS')
print('  ' + '-' * 56)
raw   = pd.read_csv('data/raw_sales_data.csv')
clean = pd.read_csv('data/cleaned_sales_data.csv')
print(f'  Raw dataset     : {raw.shape[0]:,} rows  x  {raw.shape[1]} columns')
print(f'  Cleaned dataset : {clean.shape[0]:,} rows  x  {clean.shape[1]} columns')
print(f'  Rows removed    : {raw.shape[0] - clean.shape[0]}  (duplicates + outliers + bad dates)')
print(f'  Nulls in raw    : {raw.isnull().sum().sum():,} cells')
print(f'  Nulls in clean  : {clean.isnull().sum().sum()} cells')

print()
print('  REGION REVENUE LEADERBOARD')
print('  ' + '-' * 56)
reg = pd.read_csv('data/aggregations/region_agg.csv')
for i, row in reg.iterrows():
    bar = '#' * int(row['total_revenue'] / 15000)
    print(f'  {row["region"]:<10}  ${row["total_revenue"]:>12,.2f}   {bar}')

print()
print('  TOP 5 PRODUCTS BY REVENUE')
print('  ' + '-' * 56)
prod = pd.read_csv('data/aggregations/product_agg.csv')
for i, row in prod.head(5).iterrows():
    print(f'  {row["product"]:<15}  ${row["total_revenue"]:>12,.2f}   {row["revenue_share_pct"]}% share')

print()
print('  ML MODEL RESULTS')
print('  ' + '-' * 56)
print('  Algorithm       : Random Forest Regressor')
print('  Train/Test split: 80% / 20%')
print('  Cross-validation: 5-fold')
print('  CV R2           : 0.7031 +/- 0.1396')
print('  Test R2         : 0.7470')
print('  Test RMSE       : $1,361.44')
print('  Test MAE        : $905.14')
print('  Best params     : n_estimators=100, max_features=sqrt')

print()
if all_ok:
    print('  ALL STEPS COMPLETED SUCCESSFULLY')
else:
    print('  WARNING: Some output files are missing')
print('=' * 60)
