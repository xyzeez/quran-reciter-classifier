import librosa
import numpy as np
import noisereduce as nr
import torch
import logging
from config import *
from src.data.loader import load

# Setup logging
logger = logging.getLogger(__name__)

# Determine device for computations
device = torch.device("cuda" if torch.cuda.is_available()
                      and USE_GPU else "cpu")


def preprocess_audio_with_logic(file_path: str, mode: str = "train") -> tuple:
    """
    Preprocess an audio file.

    Args:
        file_path (str): Path to the audio file.
        mode (str): Mode to run in ('train' or 'test')

    Returns:
        Tuple[np.ndarray, int]: Preprocessed audio data and sample rate.
    """
    try:
        # Load the audio file
        audio_data, sr = load(file_path)
        if audio_data is None or sr is None:
            logger.error(f"Failed to load audio file: {file_path}")
            return None, None

        # Convert to tensor if using GPU
        if device.type == "cuda":
            audio_data = torch.from_numpy(audio_data).to(device)

        # Use the entire audio without skipping
        if device.type == "cuda":
            processed_audio = audio_data.cpu().numpy()
        else:
            processed_audio = audio_data

        return processed_audio, sr
    except Exception as e:
        logger.error(f"[ERROR] {file_path}: {str(e)}")
        return None, None


def preprocess_audio(audio_data: np.ndarray, rate: int) -> tuple:
    """
    Enhanced audio preprocessing pipeline.

    Args:
        audio_data (np.ndarray): Raw audio data.
        rate (int): Sample rate of the audio data.

    Returns:
        Tuple[np.ndarray, int]: Preprocessed audio data and new sample rate.
    """
    try:
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Audio data is empty or None.")

        # Convert to tensor if using GPU
        if device.type == "cuda":
            audio_tensor = torch.from_numpy(audio_data).to(device)

        # Noise reduction
        audio_data = nr.reduce_noise(y=audio_data, sr=rate)

        # Normalize amplitude
        audio_data = librosa.util.normalize(audio_data)

        # Trim silence with custom parameters
        audio_data, _ = librosa.effects.trim(audio_data, top_db=30)

        # Resample to default sample rate if necessary
        if rate != DEFAULT_SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=rate,
                target_sr=DEFAULT_SAMPLE_RATE
            )
            rate = DEFAULT_SAMPLE_RATE

        if not np.isfinite(audio_data).all():
            raise ValueError(
                "Audio data contains non-finite values after preprocessing.")

        return audio_data, rate
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return None, None
