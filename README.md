# California Housing Price Prediction

## Overview
This notebook trains a Machine Learning model to predict **`median_house_value`** from the California housing dataset. The workflow is:

1) Load → Inspect → Clean (drop missing rows)  
2) Train/test split
3) EDA (histograms, correlation heatmaps)  
4) Preprocessing (log transforms for skewed columns, one‑hot encoding for `ocean_proximity`)  
5) Feature engineering: `bedroom_ratio`, `rooms_per_household`  
6) Consistent preprocessing function + align train/test columns  
7) Scaling (for linear regression only)  
8) Train & evaluate **Linear Regression** and **Random Forest**  
9) **GridSearchCV** for Random Forest  
10) Compare metrics (R², RMSE, MAE)  
11) Metrics on full test set 
12) Predict on new data


## Notes on Models & Metrics
- The notebook trains models **in memory** (not saved to disk).  
- The “Compare model metrics” cell shows R², RMSE, and MAE for Linear Regression, default Random Forest, and tuned Random Forest.  
- The “Metrics on full test set” cell reports metrics for the **currently selected best model** (`best_rf` → `rf` → `linreg`).
