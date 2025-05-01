"""
Prediction utilities for the Quran reciter classifier.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from src.models.model_factory import load_model
from src.features import extract_features
from .prediction_analysis import calculate_distances, analyze_prediction_reliability
from server.config import (
    MODEL_DIR,
    MODEL_ID,
    LATEST_MODEL_SYMLINK,
    TOP_N_PREDICTIONS,
    CONFIDENCE_THRESHOLD
)

# Configure logging
logger = logging.getLogger(__name__)

# Global model instance
_model = None
_centroids = None

def find_model_path() -> Optional[Path]:
    """
    Find the appropriate model file path based on config.
    
    Returns:
        Path: Path to model file or None if not found
    """
    models_dir = Path(MODEL_DIR)
    
    try:
        # Try specific model ID first
        if MODEL_ID:
            model_dir = models_dir / MODEL_ID
            if model_dir.exists():
                model_files = list(model_dir.glob('model_*.joblib'))
                if model_files:
                    return model_files[0]
                logger.warning(f"No model file found in {model_dir}")
            logger.warning(f"Model directory not found at {model_dir}, trying latest")
            
        # Try latest symlink
        latest_link = models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                target_dir = latest_link.resolve()
            else:
                target_dir = latest_link
                
            model_files = list(target_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            logger.warning(f"No model file found in {target_dir}")
            
        # Find most recent model directory
        model_dirs = [d for d in models_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_train')]
        if model_dirs:
            latest_dir = max(model_dirs, key=lambda d: d.stat().st_mtime)
            model_files = list(latest_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            logger.warning(f"No model file found in {latest_dir}")
            
        logger.error("No valid model file found in any location")
        return None
        
    except Exception as e:
        logger.error(f"Error finding model path: {str(e)}")
        return None

def get_model():
    """
    Get the loaded model instance, loading it if necessary.
    
    Returns:
        Model instance or None if loading fails
    """
    global _model
    if _model is None:
        try:
            # Find and load model file
            model_path = find_model_path()
            if model_path is None:
                return None
                
            logger.info(f"Loading model from {model_path}")
            _model = load_model(model_path)
            
            # Initialize centroids if model has them
            if hasattr(_model, 'centroids'):
                global _centroids
                _centroids = _model.centroids
                logger.info("Initialized centroids from model")
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
            
    return _model

def update_centroids(features: np.ndarray, labels: List[str]) -> None:
    """
    Update class centroids based on provided features.
    
    Args:
        features: Array of features of shape (n_samples, n_features)
        labels: List of corresponding class labels
    """
    global _centroids
    try:
        unique_labels = np.unique(labels)
        n_features = features.shape[1]
        _centroids = np.zeros((len(unique_labels), n_features))
        
        for i, label in enumerate(unique_labels):
            label_features = features[np.array(labels) == label]
            _centroids[i] = np.mean(label_features, axis=0)
            
        logger.info("Successfully updated class centroids")
        
        # Update model centroids if possible
        model = get_model()
        if model is not None and hasattr(model, 'centroids'):
            model.centroids = _centroids
            logger.info("Updated model centroids")
            
    except Exception as e:
        logger.error(f"Failed to update centroids: {str(e)}")

def get_predictions(
    audio_data: np.ndarray,
    sample_rate: int,
    top_k: int = TOP_N_PREDICTIONS,
    min_confidence: float = CONFIDENCE_THRESHOLD,
    reliability_thresholds: Optional[Dict[str, float]] = None
) -> Tuple[List[Dict[str, Union[str, float]]], bool]:
    """
    Get model predictions with reliability analysis.
    
    Args:
        audio_data: Audio data array
        sample_rate: Audio sample rate
        top_k: Number of top predictions to return
        min_confidence: Minimum confidence threshold for predictions
        reliability_thresholds: Dictionary of thresholds for reliability analysis:
            - 'min_probability': Minimum acceptable probability
            - 'max_distance': Maximum acceptable distance to centroid
            
    Returns:
        Tuple containing:
        - List of dictionaries with predictions and scores
        - Boolean indicating if prediction is reliable
    """
    model = get_model()
    if model is None:
        logger.error("No model available for prediction")
        return [], False
        
    try:
        # Extract features
        logger.info("Extracting features from audio")
        features = extract_features(audio_data, sample_rate)
        if features is None:
            logger.error("Feature extraction failed")
            return [], False
            
        # Reshape features based on model type
        model_info = model.get_model_info()
        if model_info['model_type'].lower() == 'blstm':
            # For BLSTM, keep temporal dimension
            features = features.reshape(1, -1, features.shape[-1])
        else:
            # For other models (e.g. Random Forest), flatten temporal dimension
            features = features.reshape(1, -1)
            
        logger.info(f"Features shape after reshaping: {features.shape}")
        
        # Get model predictions
        probabilities = model.predict_proba(features)
        if probabilities.ndim > 2:
            # For sequence models, take mean over time
            probabilities = np.mean(probabilities, axis=1)
            
        # Get indices of top k predictions
        top_indices = np.argsort(probabilities[0])[-top_k:][::-1]
        
        # Filter predictions above minimum confidence
        predictions = []
        for idx in top_indices:
            confidence = probabilities[0][idx]
            if confidence >= min_confidence:
                predictions.append({
                    'reciter': model.classes_[idx],
                    'confidence': float(confidence)
                })
                
        if not predictions:
            logger.warning("No predictions above confidence threshold")
            return [], False
            
        # Get model thresholds if not provided
        if reliability_thresholds is None and hasattr(model, 'thresholds'):
            reliability_thresholds = model.thresholds
            
        # Perform reliability analysis if thresholds provided
        is_reliable = True
        if reliability_thresholds and hasattr(model, 'centroids'):
            # For sequence models, use mean features for distance calculation
            if features.ndim > 2:
                distance_features = np.mean(features, axis=1)
            else:
                distance_features = features
                
            distances = calculate_distances(distance_features, model.centroids)
            analysis = analyze_prediction_reliability(
                probabilities[0],
                distances,
                reliability_thresholds,
                predictions[0]['reciter']
            )
            is_reliable = analysis['is_reliable']
            
            # Add reliability metrics to top prediction
            predictions[0].update({
                'max_probability': analysis['max_probability'],
                'prob_difference': analysis['prob_difference'],
                'min_distance': analysis['min_distance']
            })
            
        return predictions, is_reliable
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return [], False 