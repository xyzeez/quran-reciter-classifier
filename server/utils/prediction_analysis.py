"""
Utilities for analyzing prediction reliability in the Quran reciter classifier.
"""
import numpy as np
from typing import Dict, Union

def calculate_distances(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distances between input features and class centroids.
    
    Args:
        features: Input features array of shape (n_features,)
        centroids: Class centroids array of shape (n_classes, n_features)
        
    Returns:
        Array of distances to each centroid
    """
    # Ensure features are 2D for broadcasting
    if features.ndim == 1:
        features = features.reshape(1, -1)
        
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((features - centroids) ** 2, axis=1))
    return distances

def analyze_prediction_reliability(
    probabilities: np.ndarray,
    distances: np.ndarray,
    thresholds: Dict[str, float],
    prediction: str,
    min_prob_diff: float = 0.1
) -> Dict[str, Union[bool, float]]:
    """
    Analyze prediction reliability based on multiple criteria.
    
    Args:
        probabilities: Array of predicted class probabilities
        distances: Array of distances to class centroids
        thresholds: Dictionary containing threshold values:
            - 'min_probability': Minimum acceptable probability
            - 'max_distance': Maximum acceptable distance
        prediction: Predicted class name
        min_prob_diff: Minimum required difference between top probabilities
        
    Returns:
        Dictionary containing:
            - is_reliable: Boolean indicating if prediction meets reliability criteria
            - max_probability: Highest prediction probability
            - prob_difference: Difference between top two probabilities
            - min_distance: Minimum distance to any centroid
    """
    # Get maximum probability and its index
    max_prob = np.max(probabilities)
    max_prob_idx = np.argmax(probabilities)
    
    # Get second highest probability
    probabilities_without_max = np.delete(probabilities, max_prob_idx)
    second_max_prob = np.max(probabilities_without_max)
    
    # Calculate probability difference
    prob_difference = max_prob - second_max_prob
    
    # Get minimum distance
    min_distance = np.min(distances)
    
    # Check reliability criteria
    is_reliable = (
        max_prob >= thresholds['min_probability'] and
        prob_difference >= min_prob_diff and
        min_distance <= thresholds['max_distance']
    )
    
    return {
        'is_reliable': is_reliable,
        'max_probability': float(max_prob),
        'prob_difference': float(prob_difference),
        'min_distance': float(min_distance)
    } 