"""
Audio processing utilities for the server.
"""
import librosa
from pydub import AudioSegment
import io
import torch

from config import USE_GPU, N_MFCC, N_CHROMA, N_MEL_BANDS
from server.config import MIN_AUDIO_DURATION, MAX_AUDIO_DURATION, SAMPLE_RATE
from src.features.extractors import extract_features as src_extract_features

# Determine device for computations
device = torch.device("cuda" if torch.cuda.is_available()
                      and USE_GPU else "cpu")


def process_audio_file(audio_file):
    """
    Process audio file for prediction.

    Args:
        audio_file: File-like object containing audio data

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

        # Check minimum duration
        if duration < MIN_AUDIO_DURATION:
            return None, f"Audio duration ({duration:.1f}s) is less than minimum required ({MIN_AUDIO_DURATION}s)"

        # Handle duration
        if duration > MAX_AUDIO_DURATION:
            # Calculate samples for max duration
            max_samples = int(MAX_AUDIO_DURATION * sr)
            # Take the first MAX_AUDIO_DURATION seconds
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
