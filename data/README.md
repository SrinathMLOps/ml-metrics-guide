# Data Directory

This directory contains sample datasets used in the ML metrics demonstrations.

## Synthetic Data

The notebooks generate synthetic datasets using scikit-learn:

- **Classification**: `make_classification()` - Binary classification with imbalanced classes
- **Regression**: `make_regression()` - Linear regression with noise
- **Clustering**: `make_blobs()` - Clustered data points
- **Recommendation**: Random user-item interaction matrices

## Adding Real Data

To use your own datasets:

1. Place CSV files in this directory
2. Update the notebook to load your data instead of synthetic data
3. Ensure proper preprocessing for each metric type

## Data Format

Expected formats for each task:
- **Classification**: Features (X) and binary/multiclass labels (y)
- **Regression**: Features (X) and continuous targets (y)  
- **Clustering**: Features (X) only
- **Recommendation**: User-item interaction matrix or ratings