import librosa
import numpy as np
from pathlib import Path
import logging
import warnings
from config import *

logger = logging.getLogger(__name__)


def load(file_name: str, skip_seconds: float = 0, end_skip_seconds: float = 0) -> tuple:
    """
    Load an audio file while skipping sections at the start and end.
    Uses librosa for all audio file types.

    Args:
        file_name (str): Path to the audio file.
        skip_seconds (float): Seconds to skip from the start of the audio.
        end_skip_seconds (float): Seconds to skip from the end of the audio.

    Returns:
        Tuple[np.ndarray, int]: Loaded audio data and its sample rate.
    """
    try:
        file_path = Path(file_name).resolve()
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return None, None

        # Use librosa for all audio files
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio_data, sr = librosa.load(
                str(file_path),
                sr=None,  # Keep original sample rate
                offset=skip_seconds,
                duration=None if end_skip_seconds == 0 else -end_skip_seconds,
                res_type='kaiser_fast'
            )

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            return audio_data, sr

    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None, None
