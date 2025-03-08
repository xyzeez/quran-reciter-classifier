"""
Factory for creating model instances.
"""
import logging
from pathlib import Path
import os
import joblib

from src.models.random_forest import RandomForestModel
from src.models.blstm_model import BLSTMModel
from config import MODEL_TYPE

logger = logging.getLogger(__name__)


def create_model(model_type=None):
    """
    Create a model instance based on the specified model type.

    Args:
        model_type (str, optional): Model type to create.
            If None, will use the MODEL_TYPE from config.

    Returns:
        Model instance

    Raises:
        ValueError: If the model type is not supported
    """
    # Use config model type if not specified
    if model_type is None:
        model_type = MODEL_TYPE

    # Convert to lowercase for case-insensitive comparison
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
    Find the latest model file in the models directory.

    Args:
        model_dir (str or Path): Directory containing model files

    Returns:
        Path: Path to the latest model file

    Raises:
        FileNotFoundError: If no model files are found
    """
    model_dir = Path(model_dir)
    model_files = list(model_dir.glob('model_*.joblib'))

    if not model_files:
        raise FileNotFoundError('No model files found in the models directory')

    return max(model_files, key=os.path.getctime)


def detect_model_type_from_filename(model_path):
    """
    Detect the model type from the filename.
    
    Args:
        model_path (str or Path): Path to the model file
        
    Returns:
        str: Detected model type (lowercase)
    """
    filename = str(model_path).lower()
    
    # Check for specific model types in the filename
    if "blstm" in filename:
        return "blstm"
    elif "random_forest" in filename or "randomforest" in filename:
        return "random_forest"
    # Add more model type checks here as they are implemented
    
    # Default to config model type if detection fails
    logger.warning(f"Could not detect model type from filename: {model_path}")
    logger.warning(f"Using default model type: {MODEL_TYPE}")
    return MODEL_TYPE


def load_model(model_path=None):
    """
    Load a model from file with improved model type detection.

    Args:
        model_path (str, optional): Path to the model file.
            If None, will load the latest model from the MODEL_OUTPUT_DIR.

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If the model file is not found
    """
    from config import MODEL_OUTPUT_DIR

    if model_path is None:
        # Load the latest model
        model_dir = Path(MODEL_OUTPUT_DIR)
        model_path = get_latest_model(model_dir)
        logger.info(f"Loading latest model: {model_path}")
    else:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    # Detect model type from filename
    model_type = detect_model_type_from_filename(model_path)
    
    # Try to load using the detected model type
    if model_type == "blstm":
        logger.info("Loading as BLSTM model")
        try:
            return BLSTMModel.load(model_path)
        except Exception as e:
            logger.error(f"Error loading as BLSTM model: {str(e)}")
            logger.info("Trying to load as RandomForest model as fallback...")
            try:
                return RandomForestModel.load(model_path)
            except Exception as e2:
                logger.error(f"Error loading as RandomForest model: {str(e2)}")
                raise ValueError(f"Failed to load model {model_path} as either BLSTM or RandomForest: {str(e)}, {str(e2)}")
    else:
        # Default to RandomForest
        logger.info("Loading as RandomForest model")
        try:
            return RandomForestModel.load(model_path)
        except Exception as e:
            logger.error(f"Error loading as RandomForest model: {str(e)}")
            # Try BLSTM as a fallback
            logger.info("Trying to load as BLSTM model as fallback...")
            try:
                return BLSTMModel.load(model_path)
            except Exception as e2:
                logger.error(f"Error loading as BLSTM model: {str(e2)}")
                raise ValueError(f"Failed to load model {model_path} as either RandomForest or BLSTM: {str(e)}, {str(e2)}")