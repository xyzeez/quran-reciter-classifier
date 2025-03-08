import librosa
import numpy as np
import noisereduce as nr
import torch
import warnings
import logging
from pathlib import Path
import soundfile as sf
from config import *
from src.data.loader import load

# Setup logging
logger = logging.getLogger(__name__)

# Determine device for computations
device = torch.device("cuda" if torch.cuda.is_available()
                      and USE_GPU else "cpu")


def preprocess_audio_with_logic(file_path: str,
                                base_threshold: int = MINIMUM_DURATION,
                                skip_start: int = SKIP_START,
                                skip_end: int = SKIP_END) -> tuple:
    """
    Preprocess an audio file with smart duration-based filtering.

    Args:
        file_path (str): Path to the audio file.
        base_threshold (int): Minimum acceptable duration (seconds).
        skip_start (int): Duration to skip from start (seconds).
        skip_end (int): Duration to skip from end (seconds).

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

        # Calculate duration
        file_duration = len(audio_data) / sr

        if file_duration < base_threshold:
            logger.warning(
                f"[REJECTED] {file_path}: Too short ({file_duration:.2f}s)")
            return None, None

        # Smart duration handling
        # Use percentage threshold from config
        start_time = min(skip_start, file_duration *
                         DURATION_PERCENTAGE_THRESHOLD)
        end_time = file_duration - \
            min(skip_end, file_duration * DURATION_PERCENTAGE_THRESHOLD)

        usable_duration = end_time - start_time
        if usable_duration < MIN_USABLE_DURATION:
            logger.warning(
                f"[REJECTED] {file_path}: Usable duration too short ({usable_duration:.2f}s)")
            return None, None

        # Extract usable portion
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        if device.type == "cuda":
            processed_audio = audio_data[start_sample:end_sample].cpu().numpy()
        else:
            processed_audio = audio_data[start_sample:end_sample]

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
