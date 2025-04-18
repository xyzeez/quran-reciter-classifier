"""
Prediction utilities for the server.
"""
import numpy as np
import json
from pathlib import Path
from src.models import load_model
from src.utils import calculate_distances, analyze_prediction_reliability
from server.config import MODEL_DIR, LATEST_MODEL_SYMLINK, TOP_N_PREDICTIONS, MODEL_ID

# Load reciters data
def load_reciters_data():
    """Load reciters information from reciters.json"""
    try:
        with open('data/reciters.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading reciters data: {str(e)}")

# Load reciters data at module level
RECITERS_DATA = load_reciters_data()

def get_reciter_info(reciter_name):
    """Get reciter information from reciters.json"""
    # Try to find an exact match first
    if reciter_name in RECITERS_DATA:
        info = RECITERS_DATA[reciter_name]
        server_url = info['servers'][0] if isinstance(info['servers'], list) else info['servers']
        return {
            'nationality': info['nationality'],
            'serverUrl': server_url,
            'flagUrl': info['flagUrl'],
            'imageUrl': info['imageUrl']
        }
    
    # If no exact match, try case-insensitive match
    for name, info in RECITERS_DATA.items():
        if name.lower() == reciter_name.lower():
            server_url = info['servers'][0] if isinstance(info['servers'], list) else info['servers']
            return {
                'nationality': info['nationality'],
                'serverUrl': server_url,
                'flagUrl': info['flagUrl'],
                'imageUrl': info['imageUrl']
            }
    
    # If still no match, return default values
    return {
        'nationality': 'Unknown',
        'serverUrl': '',
        'flagUrl': '',
        'imageUrl': ''
    }

def find_model_by_id(model_id):
    """
    Find a model by its ID (timestamp directory).

    Args:
        model_id (str): Model ID (e.g., '20240306_152417_train')

    Returns:
        Path: Path to the model file, or None if not found
    """
    try:
        model_dir = MODEL_DIR / model_id
        if not model_dir.exists() or not model_dir.is_dir():
            return None

        model_files = list(model_dir.glob('model_*.joblib'))
        return model_files[0] if model_files else None

    except Exception:
        return None


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
    Load the model based on priority:
    1. Specific model ID from config
    2. Latest model symlink
    3. Latest model by timestamp

    Returns:
        tuple: (model, error_message)
    """
    try:
        # 1. Try loading specific model ID from config
        if MODEL_ID:
            model_path = find_model_by_id(MODEL_ID)
            if model_path:
                try:
                    model = load_model(str(model_path))
                    return model, None
                except Exception as e:
                    return None, f"Error loading model {MODEL_ID}: {str(e)}"
            else:
                return None, f"Model with ID {MODEL_ID} not found"

        # 2. Try using the symlink
        if LATEST_MODEL_SYMLINK.exists():
            try:
                model = load_model(str(LATEST_MODEL_SYMLINK))
                return model, None
            except Exception:
                pass  # If symlink fails, try finding latest model

        # 3. Find latest model by timestamp
        model_path = find_latest_model()
        if model_path is None:
            return None, "No trained models found in models directory"

        model = load_model(str(model_path))
        return model, None

    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def get_predictions(model, features, show_unreliable_predictions=False):
    """
    Get predictions for the given features.

    Args:
        model: Trained model
        features: Extracted features
        show_unreliable_predictions: If True, will show top predictions even when prediction is unreliable
                                   (useful for testing/development)

    Returns:
        dict: Prediction results containing:
            - reliable: boolean indicating if prediction is reliable
            - main_prediction: details of the main prediction (only if reliable)
            - top_predictions: array of other predictions (if reliable or show_unreliable_predictions is True)
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
        
        # Format the main prediction
        reciter_name = str(prediction)
        reciter_info = get_reciter_info(reciter_name)
        main_prediction = {
            "name": reciter_name,
            "confidence": float(probabilities[sorted_indices[0]] * 100),  # Convert to percentage
            "nationality": reciter_info['nationality'],
            "serverUrl": reciter_info['serverUrl'],
            "flagUrl": reciter_info['flagUrl'],
            "imageUrl": reciter_info['imageUrl']
        }

        # Format the top predictions (including the main prediction)
        top_predictions = []
        for idx in sorted_indices:
            reciter_name = str(model.classes_[idx])
            confidence = float(probabilities[idx] * 100)  # Convert to percentage
            reciter_info = get_reciter_info(reciter_name)
            top_predictions.append({
                "name": reciter_name,
                "confidence": confidence,
                "nationality": reciter_info['nationality'],
                "serverUrl": reciter_info['serverUrl'],
                "flagUrl": reciter_info['flagUrl'],
                "imageUrl": reciter_info['imageUrl']
            })

        # If prediction is not reliable and we're not showing unreliable predictions
        if not reliability['is_reliable'] and not show_unreliable_predictions:
            return {
                "reliable": False
            }
        
        # If prediction is not reliable but we're showing unreliable predictions (dev mode)
        if not reliability['is_reliable'] and show_unreliable_predictions:
            return {
                "reliable": False,
                "top_predictions": top_predictions
            }

        # Return the full response format for reliable predictions
        result = {
            "reliable": bool(reliability['is_reliable']),  # Ensure it's a Python bool
            "main_prediction": main_prediction,
            "top_predictions": top_predictions
        }

        return result

    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")
