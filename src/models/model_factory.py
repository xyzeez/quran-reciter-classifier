"""
Factory for creating and loading Quran reciter classifier models.
Supports multiple model types and handles automatic type detection.
"""
import logging
from pathlib import Path
import os
from src.models.random_forest import RandomForestModel
from src.models.blstm_model import BLSTMModel
from config import MODEL_TYPE

logger = logging.getLogger(__name__)


def create_model(model_type=None):
    """
    Create a new model instance.

    Args:
        model_type: Type of model to create ('random_forest' or 'blstm')
                   Uses MODEL_TYPE from config if not specified

    Returns:
        New model instance

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type is None:
        model_type = MODEL_TYPE

    model_type = model_type.lower()

    if model_type == 'random_forest':
        logger.info("Creating Random Forest model")
        return RandomForestModel()
    elif model_type == 'blstm':
        logger.info("Creating Bidirectional LSTM model")
        return BLSTMModel()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_latest_model(model_dir):
    """
    Find most recently modified model file.

    Args:
        model_dir: Directory containing model files

    Returns:
        Path to latest model file

    Raises:
        FileNotFoundError: If no model files exist
    """
    model_dir = Path(model_dir)
    model_files = list(model_dir.glob('model_*.joblib'))

    if not model_files:
        raise FileNotFoundError('No model files found in the models directory')

    return max(model_files, key=os.path.getctime)


def detect_model_type_from_filename(model_path):
    """
    Infer model type from filename.

    Args:
        model_path: Path to model file

    Returns:
        Detected model type or default from config
    """
    filename = str(model_path).lower()

    if "blstm" in filename:
        return "blstm"
    elif "random_forest" in filename or "randomforest" in filename:
        return "random_forest"

    logger.warning(f"Could not detect model type from: {model_path}")
    logger.warning(f"Using default type: {MODEL_TYPE}")
    return MODEL_TYPE


def load_model(model_path=None):
    """
    Load a saved model with automatic type detection.

    Args:
        model_path: Path to model file
                   Loads latest model if not specified

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model loading fails for all supported types
    """
    from config import MODEL_OUTPUT_DIR

    if model_path is None:
        model_dir = Path(MODEL_OUTPUT_DIR)
        model_path = get_latest_model(model_dir)
        logger.info(f"Loading latest model: {model_path}")
    else:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model_type = detect_model_type_from_filename(model_path)

    # Try loading with detected type first, then fallback
    if model_type == "blstm":
        logger.info("Loading as BLSTM model")
        try:
            return BLSTMModel.load(model_path)
        except Exception as e:
            logger.error(f"Error loading as BLSTM model: {str(e)}")
            logger.info("Trying RandomForest as fallback...")
            try:
                return RandomForestModel.load(model_path)
            except Exception as e2:
                logger.error(f"Error loading as RandomForest model: {str(e2)}")
                raise ValueError(
                    f"Failed to load {model_path} as either type: {str(e)}, {str(e2)}")
    else:
        logger.info("Loading as RandomForest model")
        try:
            return RandomForestModel.load(model_path)
        except Exception as e:
            logger.error(f"Error loading as RandomForest model: {str(e)}")
            logger.info("Trying BLSTM as fallback...")
            try:
                return BLSTMModel.load(model_path)
            except Exception as e2:
                logger.error(f"Error loading as BLSTM model: {str(e2)}")
                raise ValueError(
                    f"Failed to load {model_path} as either type: {str(e)}, {str(e2)}")
