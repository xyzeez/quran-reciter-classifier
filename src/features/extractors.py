import librosa
import numpy as np
import torch
import logging
from config import *

logger = logging.getLogger(__name__)

# Pick GPU if available and enabled, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available()
                      and USE_GPU else "cpu")


def extract_features(X: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract audio features for reciter identification.

    This is where the magic happens - we pull out the distinctive characteristics
    that help us identify different reciters. We focus on tonal qualities, rhythm
    patterns, and spectral properties that tend to be unique to each reciter.

    X: Raw audio signal
    sample_rate: Audio sampling rate (typically 22050Hz)

    Returns a flattened vector of concatenated audio features
    """
    try:
        # Move data to GPU if we're using it
        if device.type == "cuda":
            X = torch.from_numpy(X).to(device)

        features = []

        # MFCCs capture the tonal qualities and vocal characteristics
        # These are our primary features for voice identification
        mfccs = np.mean(librosa.feature.mfcc(y=X.cpu().numpy() if device.type == "cuda" else X,
                                             sr=sample_rate,
                                             n_mfcc=N_MFCC).T, axis=0)
        delta_mfccs = np.mean(librosa.feature.delta(mfccs).T, axis=0)
        delta2_mfccs = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)
        features.extend([mfccs, delta_mfccs, delta2_mfccs])

        # Spectral features help identify unique aspects of voice timbre
        stft = np.abs(librosa.stft(X.cpu().numpy()
                      if device.type == "cuda" else X))

        chroma = np.mean(librosa.feature.chroma_stft(S=stft,
                                                     sr=sample_rate,
                                                     n_chroma=N_CHROMA).T, axis=0)

        mel = np.mean(librosa.feature.melspectrogram(y=X.cpu().numpy() if device.type == "cuda" else X,
                                                     sr=sample_rate,
                                                     n_mels=N_MEL_BANDS).T, axis=0)

        contrast = np.mean(librosa.feature.spectral_contrast(
            S=stft, sr=sample_rate).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=X.cpu().numpy() if device.type == "cuda" else X,
                                                           sr=sample_rate).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=X.cpu().numpy() if device.type == "cuda" else X,
                                                             sr=sample_rate).T, axis=0)
        features.extend([chroma, mel, contrast, rolloff, centroid])

        # Additional features for even more discriminative power
        zcr = np.mean(librosa.feature.zero_crossing_rate(
            X.cpu().numpy() if device.type == "cuda" else X).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=X.cpu().numpy()
                      if device.type == "cuda" else X).T, axis=0)

        # Rhythm features - capture the unique pace and style of recitation
        tempogram = np.mean(librosa.feature.tempogram(y=X.cpu().numpy() if device.type == "cuda" else X,
                                                      sr=sample_rate).T, axis=0)

        # Tonal features - especially important for maqam recognition
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X.cpu().numpy() if device.type == "cuda" else X),
                                                  sr=sample_rate).T, axis=0)

        features.extend([zcr, rms, tempogram, tonnetz])

        # Stack everything into one feature vector
        combined_features = np.hstack([f.flatten() for f in features])

        return combined_features
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return None
