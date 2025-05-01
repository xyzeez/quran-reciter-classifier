"""
Audio transcription utilities using Ayah identification models.
"""
import logging
from pathlib import Path
import librosa
from src.models.ayah_model_factory import load_ayah_model

# Configure logging
logger = logging.getLogger(__name__)

# Global model instance
_model = None

def load_model() -> bool:
    """
    Load the Ayah identification model.
    This should be called once when the server starts.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global _model
    try:
        if _model is None:
            logger.info("Loading Ayah identification model...")
            _model = load_ayah_model()
            logger.info("Ayah identification model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading Ayah identification model: {str(e)}")
        return False

def transcribe_audio(audio_data, sample_rate):
    """
    Transcribe audio using the loaded model.

    Args:
        audio_data (numpy.ndarray): Audio data array
        sample_rate (int): Sample rate of the audio

    Returns:
        str: Transcribed text or None if transcription fails
    """
    try:
        # Ensure model is loaded
        if not load_model():
            raise RuntimeError("Failed to load transcription model")

        # Transcribe audio
        transcription = _model.transcribe(audio_data, sample_rate)
        return transcription.strip()

    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None 