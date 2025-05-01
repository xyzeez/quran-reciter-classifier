"""
Factory for creating verse identification models.
Supports multiple model types with consistent interface.
"""
import logging
from pathlib import Path
from src.models.whisper_ayah_model import WhisperAyahModel

logger = logging.getLogger(__name__)

DEFAULT_AYAH_MODEL = "whisper"  # Default model type

def create_ayah_model(model_type: str = None):
    """
    Create a verse identification model.
    
    Args:
        model_type: Model type to create ('whisper')
                   Uses DEFAULT_AYAH_MODEL if None
            
    Returns:
        New model instance
        
    Raises:
        ValueError: If model_type not supported
    """
    if model_type is None:
        model_type = DEFAULT_AYAH_MODEL
    
    model_type = model_type.lower()
    
    if model_type == "whisper":
        logger.info("Creating Whisper Ayah model")
        return WhisperAyahModel()
    else:
        raise ValueError(f"Unsupported Ayah model type: {model_type}")

def load_ayah_model(model_type: str = None, model_path: Path = None):
    """
    Create and load a verse identification model.
    
    Args:
        model_type: Model type to load
        model_path: Path to saved model
            
    Returns:
        Loaded model instance
    """
    model = create_ayah_model(model_type)
    
    if model_path is not None:
        model.load(model_path)
    else:
        model.load()  # Load pretrained
    
    return model 