import numpy as np
import logging
from config import *

logger = logging.getLogger(__name__)


def calculate_centroids(features, labels):
    """
    Calculate centroids for each class in the feature space.

    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Class labels

    Returns:
        dict: Mapping of class labels to their centroids
    """
    unique_labels = np.unique(labels)
    centroids = {}

    for label in unique_labels:
        class_features = features[labels == label]
        centroids[label] = np.mean(class_features, axis=0)

    return centroids


def calculate_intra_class_thresholds(features, labels, centroids, percentile=95):
    """
    Calculate distance thresholds for each class based on training data.

    Args:
        features (np.ndarray): Feature matrix
        labels (np.ndarray): Class labels
        centroids (dict): Class centroids
        percentile (int): Percentile for threshold calculation

    Returns:
        dict: Mapping of class labels to their distance thresholds
    """
    thresholds = {}

    for label in centroids.keys():
        class_features = features[labels == label]
        centroid = centroids[label]

        # Calculate distances from each sample to its class centroid
        distances = np.linalg.norm(class_features - centroid, axis=1)

        # Set threshold at specified percentile
        thresholds[label] = np.percentile(distances, percentile)

    return thresholds


def calculate_distances(features, centroids):
    """
    Calculate distances from a sample to all class centroids.

    Args:
        features (np.ndarray): Feature vector or matrix
        centroids (dict): Class centroids

    Returns:
        dict: Distances to each class centroid
    """
    distances = {}
    features = features.flatten()  # Flatten the features array

    for label, centroid in centroids.items():
        # Calculate Euclidean distance
        dist = np.linalg.norm(features - centroid)
        distances[label] = [dist]  # Keep the list format for compatibility

    return distances


def analyze_prediction_reliability(probabilities, distances, thresholds, pred_class):
    """
    Analyze prediction reliability using both probability and distance metrics.

    Args:
        probabilities (np.ndarray): Prediction probabilities
        distances (dict): Distances to each class centroid
        thresholds (dict): Distance thresholds for each class
        pred_class (str): Predicted class label

    Returns:
        dict: Reliability analysis information
    """
    sorted_indices = np.argsort(probabilities)[::-1]
    top_confidence = probabilities[sorted_indices[0]]
    second_confidence = probabilities[sorted_indices[1]]
    confidence_diff = top_confidence - second_confidence

    # Get distance metrics
    distance_to_pred = distances[pred_class][0]
    threshold_pred = thresholds[pred_class]
    distance_ratio = distance_to_pred / threshold_pred

    is_reliable = (
        top_confidence >= CONFIDENCE_THRESHOLD and
        second_confidence < SECONDARY_CONFIDENCE_THRESHOLD and
        confidence_diff >= MAX_CONFIDENCE_DIFF and
        distance_ratio <= 1.0  # Distance within threshold
    )

    reliability_info = {
        'is_reliable': is_reliable,
        'top_confidence': top_confidence,
        'second_confidence': second_confidence,
        'confidence_diff': confidence_diff,
        'distance_ratio': distance_ratio,
        'distance': distance_to_pred,
        'threshold': threshold_pred,
        'failure_reasons': []
    }

    if top_confidence < CONFIDENCE_THRESHOLD:
        reliability_info['failure_reasons'].append(
            f"Main confidence too low ({top_confidence:.2%})")
    if second_confidence >= SECONDARY_CONFIDENCE_THRESHOLD:
        reliability_info['failure_reasons'].append(
            f"Secondary prediction too strong ({second_confidence:.2%})")
    if confidence_diff < MAX_CONFIDENCE_DIFF:
        reliability_info['failure_reasons'].append(
            f"Not enough distinction between predictions (diff: {confidence_diff:.2%})")
    if distance_ratio > 1.0:
        reliability_info['failure_reasons'].append(
            f"Distance ({distance_to_pred:.2f}) exceeds threshold ({threshold_pred:.2f})")

    return reliability_info


def analyze_distances(distances, thresholds, probabilities):
    """
    Analyze distances and probabilities to determine if a sample belongs to known classes.

    Args:
        distances (dict): Distances to each class centroid
        thresholds (dict): Distance thresholds for each class
        probabilities (np.ndarray): Prediction probabilities

    Returns:
        tuple: (is_known, closest_class, analysis_info)
    """
    # Find the predicted class (highest probability)
    pred_class = list(distances.keys())[np.argmax(probabilities)]
    pred_distance = distances[pred_class][0]
    pred_threshold = thresholds[pred_class]

    # Check if the sample is within the threshold
    is_within_threshold = pred_distance <= pred_threshold

    # Get the ratio of distance to threshold
    distance_ratio = pred_distance / pred_threshold

    analysis = {
        'predicted_class': pred_class,
        'distance': float(pred_distance),
        'threshold': float(pred_threshold),
        'distance_ratio': float(distance_ratio),
        'is_within_threshold': bool(is_within_threshold)
    }

    return is_within_threshold, pred_class, analysis
