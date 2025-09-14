"""Recommendation system metrics utilities."""

import numpy as np


def precision_at_k(relevance, scores, k=5):
    """Calculate Precision@K for a single user.
    
    Args:
        relevance: Binary relevance array
        scores: Prediction scores
        k: Number of top items to consider
        
    Returns:
        float: Precision@K score
    """
    top_k_indices = np.argsort(-scores)[:k]
    return relevance[top_k_indices].sum() / k


def recall_at_k(relevance, scores, k=5):
    """Calculate Recall@K for a single user.
    
    Args:
        relevance: Binary relevance array
        scores: Prediction scores  
        k: Number of top items to consider
        
    Returns:
        float: Recall@K score
    """
    top_k_indices = np.argsort(-scores)[:k]
    total_relevant = relevance.sum()
    return relevance[top_k_indices].sum() / total_relevant if total_relevant > 0 else 0


def calculate_recommendation_metrics(relevance_matrix, scores_matrix, k_values=[3, 5, 10]):
    """Calculate recommendation metrics for multiple users.
    
    Args:
        relevance_matrix: Binary relevance matrix (users x items)
        scores_matrix: Prediction scores matrix (users x items)
        k_values: List of K values to evaluate
        
    Returns:
        dict: Dictionary of averaged metrics for each K
    """
    n_users = relevance_matrix.shape[0]
    results = {}
    
    for k in k_values:
        precisions = [precision_at_k(relevance_matrix[u], scores_matrix[u], k) 
                     for u in range(n_users)]
        recalls = [recall_at_k(relevance_matrix[u], scores_matrix[u], k) 
                  for u in range(n_users)]
        
        results[f'precision_at_{k}'] = np.mean(precisions)
        results[f'recall_at_{k}'] = np.mean(recalls)
    
    return results