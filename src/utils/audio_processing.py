"""
Common audio processing utilities for the Quran Reciter Classifier system.
"""
import librosa
import numpy as np
from pathlib import Path

def process_audio_file(file, for_ayah=True):
    """Process audio file for analysis.
    
    Args:
        file: File-like object or path to audio file
        for_ayah: If True, use ayah-specific constraints, else use reciter constraints
    
    Returns:
        tuple: (audio_data, sample_rate) or (None, error_message) on failure
    """
    try:
        # Load audio file
        if isinstance(file, (str, Path)):
            y, sr = librosa.load(file, sr=None)
        else:
            y, sr = librosa.load(file, sr=None)

        # Resample if needed
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050

        # Apply duration constraints
        if for_ayah:
            min_duration = 1.0  # seconds
            max_duration = 10.0  # seconds
        else:
            min_duration = 5.0  # seconds
            max_duration = 30.0  # seconds

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