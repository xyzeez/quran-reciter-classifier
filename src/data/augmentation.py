import librosa
import numpy as np
import logging
from config import *

logger = logging.getLogger(__name__)


def augment_audio(audio_data: np.ndarray, sr: int) -> list:
    """
    Enhanced audio data augmentation with error handling.

    Args:
        audio_data (np.ndarray): Input audio data.
        sr (int): Sample rate of the audio data.

    Returns:
        list: List of augmented audio samples including the original.
    """
    if not np.isfinite(audio_data).all() or len(audio_data) == 0:
        raise ValueError(
            "Invalid audio data: contains NaNs, infinities, or is empty.")

    augmentations = []

    # Original audio
    augmentations.append(audio_data)

    try:
        # Pitch shifting
        for steps in PITCH_STEPS:
            pitched = librosa.effects.pitch_shift(
                y=audio_data, sr=sr, n_steps=steps)
            augmentations.append(pitched)
    except Exception as e:
        logger.warning(f"Pitch shift augmentation failed: {str(e)}")

    try:
        # Time stretching
        for rate in TIME_STRETCH_RATES:
            stretched = librosa.effects.time_stretch(y=audio_data, rate=rate)
            augmentations.append(stretched)
    except Exception as e:
        logger.warning(f"Time stretch augmentation failed: {str(e)}")

    # Noise addition
    try:
        noise = np.random.normal(0, NOISE_FACTOR, audio_data.shape)
        noisy_audio = audio_data + noise
        augmentations.append(noisy_audio)
    except Exception as e:
        logger.warning(f"Noise augmentation failed: {str(e)}")

    # Volume adjustment
    try:
        volume_adjusted = librosa.util.normalize(audio_data) * VOLUME_ADJUST
        augmentations.append(volume_adjusted)
    except Exception as e:
        logger.warning(f"Volume adjustment failed: {str(e)}")

    return augmentations
