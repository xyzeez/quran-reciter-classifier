"""
Audio processing utilities for the server.
"""
import librosa
from pydub import AudioSegment
import io
import torch
from config import USE_GPU
from server.config import (
    MIN_AUDIO_DURATION, MAX_AUDIO_DURATION,
    AYAH_MIN_DURATION, AYAH_MAX_DURATION,
    SAMPLE_RATE
)
from src.features.extractors import extract_features as src_extract_features

# Determine device for computations
device = torch.device("cuda" if torch.cuda.is_available()
                      and USE_GPU else "cpu")


def process_audio_file(audio_file, for_ayah=False):
    """
    Process audio file for prediction.

    Args:
        audio_file: File-like object containing audio data
        for_ayah: Whether this is for ayah identification (different duration constraints)

    Returns:
        tuple: (processed_audio, sample_rate) or (None, error_message)
    """
    try:
        # Read audio data from file-like object
        audio_segment = AudioSegment.from_file(audio_file)

        # Convert to wav format in memory
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)

        # Load audio using librosa
        audio_data, sr = librosa.load(wav_io, sr=SAMPLE_RATE)

        # Get duration
        duration = len(audio_data) / sr

        # Use appropriate duration constraints
        if for_ayah:
            min_duration = AYAH_MIN_DURATION
            max_duration = AYAH_MAX_DURATION
        else:
            min_duration = MIN_AUDIO_DURATION
            max_duration = MAX_AUDIO_DURATION

        # Check minimum duration
        if duration < min_duration:
            return None, f"Audio duration ({duration:.1f}s) is less than minimum required ({min_duration}s)"

        # Handle maximum duration
        if duration > max_duration:
            # Calculate samples for max duration
            max_samples = int(max_duration * sr)
            # Take the first max_duration seconds
            audio_data = audio_data[:max_samples]

        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)

        return audio_data, sr

    except Exception as e:
        return None, f"Error processing audio: {str(e)}"


# Define a wrapper for the feature extraction function that ensures proper shape
def extract_features(audio_data, sample_rate):
    """
    Extract features and ensure they are properly shaped for the model.

    Args:
        audio_data: Audio data array
        sample_rate: Sample rate of the audio

    Returns:
        numpy.ndarray: Features array with shape (1, n_features)
    """
    try:
        # Get features using the source extractor
        features = src_extract_features(audio_data, sample_rate)

        # Ensure features are 2D (add batch dimension if needed)
        if features is not None:
            if features.ndim == 1:
                features = features.reshape(1, -1)

        return features
    except Exception as e:
        raise Exception(f"Error in feature extraction: {str(e)}")
