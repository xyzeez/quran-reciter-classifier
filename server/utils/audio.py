"""
Server-side audio processing utilities for loading and validating audio files.
Handles duration constraints and feature extraction for both reciter and verse identification.
"""
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Union, BinaryIO, Optional

# Audio constraints
DEFAULT_SAMPLE_RATE = 22050
AYAH_MIN_DURATION = 1.0  # seconds
AYAH_MAX_DURATION = 10.0  # seconds
RECITER_MIN_DURATION = 5.0  # seconds
RECITER_MAX_DURATION = 30.0  # seconds

def process_audio_file(
    file: Union[str, Path, BinaryIO], 
    for_ayah: bool = True,
    sample_rate: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Union[int, str]]:
    """
    Load and validate audio for analysis, enforcing duration constraints.
    
    Args:
        file: Audio file path or file-like object
        for_ayah: Use verse constraints if True, reciter constraints if False
        sample_rate: Target sample rate, defaults to 22050 Hz
        
    Returns:
        (audio_data, sample_rate) or (None, error_message)
    """
    try:
        # Set constraints based on mode
        min_duration = AYAH_MIN_DURATION if for_ayah else RECITER_MIN_DURATION
        max_duration = AYAH_MAX_DURATION if for_ayah else RECITER_MAX_DURATION
        target_sr = sample_rate or DEFAULT_SAMPLE_RATE

        # Load audio file
        y, sr = librosa.load(file, sr=None)

        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Check duration
        duration = len(y) / sr
        if duration < min_duration:
            return None, f"Audio too short. Minimum duration: {min_duration} seconds"
        
        if duration > max_duration:
            # Truncate to max duration
            samples = int(max_duration * sr)
            y = y[:samples]

        return y, sr

    except Exception as e:
        return None, f"Error processing audio file: {str(e)}"

def extract_features(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract mel spectrogram features from audio data.
    
    Args:
        audio_data: Audio time series
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Log-scaled mel spectrogram features
    """
    try:
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=128,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
        
    except Exception as e:
        raise RuntimeError(f"Error extracting features: {str(e)}") 