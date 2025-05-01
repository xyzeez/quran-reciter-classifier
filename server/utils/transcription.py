"""
Audio transcription utilities using Ayah identification models.
"""
import logging
import numpy as np
from typing import Optional
from src.models.ayah_model_factory import load_ayah_model

logger = logging.getLogger(__name__)

# Initialize model
_model = None

def get_model():
    """Get the loaded model instance, loading it if necessary."""
    global _model
    if _model is None:
        _model = load_model()
    return _model

def load_model():
    """
    Load the Ayah identification model.
    This should be called once when the server starts.
    """
    try:
        logger.info("Loading Ayah identification model...")
        model = load_ayah_model()
        logger.info("Ayah identification model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading Ayah identification model: {str(e)}")
        raise

def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
    """
    Transcribe audio data using the Ayah identification model.
    
    Args:
        audio_data: Audio time series
        sample_rate: Sampling rate of audio
        
    Returns:
        str: Transcribed text or None on failure
    """
    try:
        # Get model instance
        model = get_model()
        
        # Transcribe audio
        transcription = model.transcribe(audio_data, sample_rate)
        return transcription
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise 