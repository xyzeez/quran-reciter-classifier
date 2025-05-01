"""
Factory for creating Ayah identification model instances.
"""
import logging
from pathlib import Path
from src.models.whisper_ayah_model import WhisperAyahModel

logger = logging.getLogger(__name__)

# Default model type
DEFAULT_AYAH_MODEL = "whisper"

def create_ayah_model(model_type: str = None):
    """
    Create an Ayah identification model instance.
    
    Args:
        model_type (str, optional): Type of model to create.
            If None, uses DEFAULT_AYAH_MODEL.
            
    Returns:
        BaseAyahModel: Model instance
        
    Raises:
        ValueError: If model type is not supported
    """
    # Use default if not specified
    if model_type is None:
        model_type = DEFAULT_AYAH_MODEL
    
    # Convert to lowercase for case-insensitive comparison
    model_type = model_type.lower()
    
    if model_type == "whisper":
        logger.info("Creating Whisper Ayah model")
        return WhisperAyahModel()
    else:
        raise ValueError(f"Unsupported Ayah model type: {model_type}")

def load_ayah_model(model_type: str = None, model_path: Path = None):
    """
    Create and load an Ayah identification model.
    
    Args:
        model_type (str, optional): Type of model to load
        model_path (Path, optional): Path to model files
            
    Returns:
        BaseAyahModel: Loaded model instance
    """
    # Create model instance
    model = create_ayah_model(model_type)
    
    # Load model if path provided
    if model_path is not None:
        model.load(model_path)
    else:
        # Load default/pretrained model
        model.load()
    
    return model 