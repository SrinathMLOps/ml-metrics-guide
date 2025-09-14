# ML Metrics Reference

## Classification Metrics

### Accuracy
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Range**: [0, 1]
- **Best**: 1 (higher is better)
- **Use case**: Balanced datasets

### Precision
- **Formula**: TP / (TP + FP)
- **Range**: [0, 1] 
- **Best**: 1 (higher is better)
- **Use case**: When false positives are costly

### Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Range**: [0, 1]
- **Best**: 1 (higher is better)
- **Use case**: When false negatives are costly

### F1-Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: [0, 1]
- **Best**: 1 (higher is better)
- **Use case**: Balance between precision and recall

### ROC-AUC
- **Formula**: Area under ROC curve (TPR vs FPR)
- **Range**: [0, 1]
- **Best**: 1 (higher is better)
- **Use case**: Binary classification with probability scores

## Regression Metrics

### Mean Absolute Error (MAE)
- **Formula**: (1/n) Σ |y - ŷ|
- **Range**: [0, ∞)
- **Best**: 0 (lower is better)
- **Use case**: Robust to outliers

### Mean Squared Error (MSE)
- **Formula**: (1/n) Σ (y - ŷ)²
- **Range**: [0, ∞)
- **Best**: 0 (lower is better)
- **Use case**: Penalizes large errors more

### Root Mean Squared Error (RMSE)
- **Formula**: √MSE
- **Range**: [0, ∞)
- **Best**: 0 (lower is better)
- **Use case**: Same units as target variable

### R-squared (R²)
- **Formula**: 1 - SS_res/SS_tot
- **Range**: (-∞, 1]
- **Best**: 1 (higher is better)
- **Use case**: Proportion of variance explained

## Clustering Metrics

### Silhouette Score
- **Formula**: (b - a) / max(a, b)
- **Range**: [-1, 1]
- **Best**: 1 (higher is better)
- **Use case**: Cluster cohesion and separation

### Davies-Bouldin Index
- **Formula**: Average similarity measure
- **Range**: [0, ∞)
- **Best**: 0 (lower is better)
- **Use case**: Cluster compactness and separation

## Recommendation Metrics

### Precision@K
- **Formula**: #relevant@K / K
- **Range**: [0, 1]
- **Best**: 1 (higher is better)
- **Use case**: Quality of top-K recommendations

### Recall@K
- **Formula**: #relevant@K / #all_relevant
- **Range**: [0, 1]
- **Best**: 1 (higher is better)
- **Use case**: Coverage of relevant items in top-K