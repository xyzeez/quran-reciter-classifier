"""
Prediction utilities for the server.
"""
import numpy as np
from src.models import load_model
from src.utils import calculate_distances, analyze_prediction_reliability
from server.config import MODEL_DIR, LATEST_MODEL_SYMLINK, TOP_N_PREDICTIONS


def find_latest_model():
    """
    Find the latest model by checking timestamps of model directories.

    Returns:
        Path: Path to the latest model file, or None if no models found
    """
    try:
        # Get all directories in the models folder that match the timestamp pattern
        model_dirs = [d for d in MODEL_DIR.iterdir() if d.is_dir() and
                      len(d.name) >= 15 and d.name[:8].isdigit() and
                      d.name[8] == '_' and d.name[9:15].isdigit()]

        if not model_dirs:
            return None

        # Sort by name (timestamp) in reverse order
        latest_dir = sorted(model_dirs, key=lambda x: x.name, reverse=True)[0]

        # Find model file in the latest directory
        model_files = list(latest_dir.glob('model_*.joblib'))
        if not model_files:
            return None

        return model_files[0]

    except Exception:
        return None


def load_latest_model():
    """
    Load the latest trained model.

    Returns:
        tuple: (model, error_message)
    """
    try:
        # First try using the symlink
        if LATEST_MODEL_SYMLINK.exists():
            try:
                model = load_model(str(LATEST_MODEL_SYMLINK))
                return model, None
            except Exception:
                pass  # If symlink fails, try finding latest model

        # Find latest model by timestamp
        model_path = find_latest_model()
        if model_path is None:
            return None, "No trained models found in models directory"

        model = load_model(str(model_path))
        return model, None

    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def get_predictions(model, features):
    """
    Get predictions for the given features.

    Args:
        model: Trained model
        features: Extracted features

    Returns:
        dict: Prediction results
    """
    try:
        # Ensure features are properly shaped (2D array)
        if features is None:
            raise ValueError("Features array is None")

        # Reshape if needed
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get prediction and probabilities
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Calculate distances
        distances = calculate_distances(features, model.centroids)

        # Analyze reliability
        reliability = analyze_prediction_reliability(
            probabilities, distances, model.thresholds, prediction)

        # Get top N predictions with confidence scores
        sorted_indices = np.argsort(probabilities)[::-1][:TOP_N_PREDICTIONS]
        top_predictions = [
            {
                "reciter": str(model.classes_[idx]),
                "confidence": float(probabilities[idx])
            }
            for idx in sorted_indices
        ]

        # Convert numpy/boolean types to Python native types
        result = {
            "predicted_reciter": "Unknown" if not reliability['is_reliable'] else str(prediction),
            "is_reliable": bool(reliability['is_reliable']),
            "top_predictions": top_predictions,
            "failure_reasons": [str(reason) for reason in reliability['failure_reasons']]
        }

        # Add additional metrics if available
        if 'top_confidence' in reliability:
            result["confidence"] = float(reliability['top_confidence'])
        if 'distance_ratio' in reliability:
            result["distance_ratio"] = float(reliability['distance_ratio'])

        return result

    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")
