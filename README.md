# California Housing Price Prediction — README

## Overview
This notebook trains simple ML models to predict **`median_house_value`** from the California housing dataset. It is **comment‑free** and uses short markdown cells that describe the next step. The workflow is:

1) Load → Inspect → Clean (drop missing rows)  
2) Train/test split  
3) EDA (histograms, correlation heatmaps)  
4) Preprocessing (log transforms for skewed columns, one‑hot encoding for `ocean_proximity`)  
5) Feature engineering: `bedroom_ratio`, `rooms_per_household`  
6) Consistent preprocessing function + align train/test columns  
7) Scaling (for linear regression only)  
8) Train & evaluate **Linear Regression** and **Random Forest**  
9) **GridSearchCV** (small grid) for Random Forest  
10) Compare metrics (R², RMSE, MAE)  
11) Metrics on full test set (selected best in‑memory model)  
12) Predict on new data (no saving to disk)

## Data
- Expected file: `housing.csv` (Kaggle California Housing Prices format).  
- Update `DATA_PATH` near the top of the notebook:  
  - **Colab** example: `DATA_PATH = "/content/housing.csv"`  
  - **Local/Jupyter** example: `DATA_PATH = "/path/to/housing.csv"`

## Requirements
Python 3.8+ and the following packages:
```
numpy
pandas
matplotlib
scikit-learn
```
Optional (already imported in the notebook): `joblib` (not used for saving in this version).

Install quickly:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run
1. Open the notebook in Jupyter/Colab.  
2. Ensure `DATA_PATH` points to your `housing.csv`.  
3. Run all cells **top‑to‑bottom**.

## Notes on Models & Metrics
- The notebook trains models **in memory** (not saved to disk).  
- The “Compare model metrics” cell shows R², RMSE, MAE for Linear Regression, default Random Forest, and tuned Random Forest.  
- The “Metrics on full test set” cell reports metrics for the **currently selected best model** (`best_rf` → `rf` → `linreg`).

## Predicting on Your Own Rows
Replace the `X_new = X_test.head()` placeholder in the final section with your own DataFrame `X_new` that has the same raw columns as `X` (before preprocessing):
```python
# Example
X_new = X_test.sample(5, random_state=0)
X_new_proc = preprocess(X_new)
X_new_proc = X_new_proc.reindex(columns=X_train_proc.columns, fill_value=0.0)

try:
    model = best_rf
except NameError:
    try:
        model = rf
    except NameError:
        model = linreg

X_in = X_new_proc if hasattr(model, "feature_importances_") else scaler.transform(X_new_proc)
preds = model.predict(X_in)
```

## Troubleshooting
- **KeyError / column mismatch**: The notebook aligns train/test columns; ensure your new data is preprocessed with `preprocess()` and reindexed to `X_train_proc.columns` before predicting.  
- **NaNs or inf**: The pipeline replaces infinities with NaNs and fills them with 0.0; verify your input does not introduce unexpected values.

---
Minimal, comment‑free workflow inspired by a standard beginner‑friendly ML process.
