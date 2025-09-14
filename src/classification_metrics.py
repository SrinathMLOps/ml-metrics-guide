"""Classification metrics utilities."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_prob: Predicted probabilities (optional, for ROC-AUC)
        
    Returns:
        dict: Dictionary of metric scores
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics