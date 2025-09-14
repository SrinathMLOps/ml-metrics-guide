"""Tests for classification metrics."""

import numpy as np
import pytest
from src.classification_metrics import calculate_classification_metrics


def test_classification_metrics():
    """Test classification metrics calculation."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.4, 0.2, 0.9])
    
    metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1