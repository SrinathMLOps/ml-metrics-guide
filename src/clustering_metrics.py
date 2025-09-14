"""Clustering metrics utilities."""

from sklearn.metrics import silhouette_score, davies_bouldin_score


def calculate_clustering_metrics(X, labels):
    """Calculate clustering evaluation metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        dict: Dictionary of metric scores
    """
    return {
        'silhouette_score': silhouette_score(X, labels),
        'davies_bouldin_score': davies_bouldin_score(X, labels)
    }