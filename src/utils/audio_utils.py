"""
Audio processing utilities for loading, validating, and extracting features from audio files.
Supports various audio formats and handles both file paths and file-like objects.
"""
import io
import logging
import librosa
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Union, BinaryIO, Optional
from pydub import AudioSegment
from config import USE_GPU

# Configure logging
logger = logging.getLogger(__name__)

# Use GPU if available and enabled in config
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

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
    Load and validate an audio file for analysis.
    
    Args:
        file: Audio file path or file-like object
        for_ayah: Use ayah duration limits if True, reciter limits if False
        sample_rate: Target sample rate (default: 22050 Hz)
        
    Returns:
        (audio_data, sample_rate) or (None, error_message)
    """
    try:
        # Set duration limits based on task
        min_duration = AYAH_MIN_DURATION if for_ayah else RECITER_MIN_DURATION
        max_duration = AYAH_MAX_DURATION if for_ayah else RECITER_MAX_DURATION
        target_sr = sample_rate or DEFAULT_SAMPLE_RATE

        # Try librosa first, fallback to pydub for other formats
        try:
            y, sr = librosa.load(file, sr=None)
        except Exception as e:
            logger.debug(f"Direct loading failed, trying pydub: {e}")
            
            if isinstance(file, (str, Path)):
                audio = AudioSegment.from_file(file)
            else:
                file.seek(0)
                audio = AudioSegment.from_file(file)
            
            # Convert to normalized numpy array
            samples = np.array(audio.get_array_of_samples())
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            
            y = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            sr = audio.frame_rate

        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Validate duration
        duration = len(y) / sr
        if duration < min_duration:
            return None, f"Audio too short. Minimum duration: {min_duration} seconds"
        
        if duration > max_duration:
            samples = int(max_duration * sr)
            y = y[:samples]
            logger.debug(f"Audio truncated to {max_duration} seconds")

        return librosa.util.normalize(y), sr

    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return None, f"Error processing audio file: {str(e)}"

def extract_features(
    audio_data: np.ndarray, 
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: int = 8000
) -> np.ndarray:
    """
    Extract log-scaled mel spectrogram features from audio.
    
    Args:
        audio_data: Audio time series
        sample_rate: Audio sample rate
        n_mels: Number of mel frequency bands
        n_fft: FFT window size
        hop_length: Samples between frames
        fmax: Maximum frequency
        
    Returns:
        Log-scaled mel spectrogram features
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmax=fmax
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Add batch dimension if needed
        if len(mel_spec_db.shape) < 3:
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
            
        return mel_spec_db
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise RuntimeError(f"Error extracting features: {str(e)}")

def prepare_audio_batch(features: np.ndarray) -> torch.Tensor:
    """
    Convert audio features to PyTorch tensor for model input.
    
    Args:
        features: Mel spectrogram features
        
    Returns:
        Model-ready tensor on appropriate device
    """
    try:
        tensor = torch.from_numpy(features).float()
        
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(DEVICE)
        
    except Exception as e:
        logger.error(f"Error preparing audio batch: {str(e)}")
        raise RuntimeError(f"Error preparing audio batch: {str(e)}") 