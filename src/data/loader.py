import librosa
import numpy as np
from pathlib import Path
from pydub import AudioSegment
import logging
from config import *

logger = logging.getLogger(__name__)

def load(file_name: str, skip_seconds: float = 0, end_skip_seconds: float = 0) -> tuple:
    """
    Load an audio file while skipping sections at the start and end.

    Args:
        file_name (str): Path to the audio file.
        skip_seconds (float): Seconds to skip from the start of the audio.
        end_skip_seconds (float): Seconds to skip from the end of the audio.

    Returns:
        Tuple[np.ndarray, int]: Loaded audio data and its sample rate.
    """
    try:
        file_path = Path(file_name)
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_name}")
            return None, None

        # Use pydub to load MP3 first
        if file_name.lower().endswith('.mp3'):
            try:
                audio = AudioSegment.from_mp3(file_name)
                samples = np.array(audio.get_array_of_samples())
                # Convert to float32 and normalize
                samples = samples.astype(
                    np.float32) / np.iinfo(samples.dtype).max
                sr = audio.frame_rate

                # Handle stereo to mono conversion if necessary
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)

                # Apply skipping
                start_sample = int(skip_seconds * sr)
                end_sample = int(-end_skip_seconds *
                                 sr) if end_skip_seconds > 0 else None
                samples = samples[start_sample:end_sample]

                return samples, sr
            except Exception as e:
                logger.error(f"Error loading MP3 {file_name}: {str(e)}")
                return None, None
        else:
            # For non-MP3 files, use librosa
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_data, sr = librosa.load(
                    file_name,
                    sr=None,
                    offset=skip_seconds,
                    duration=None if end_skip_seconds == 0 else -end_skip_seconds,
                    res_type='kaiser_fast'
                )
            return audio_data, sr

    except Exception as e:
        logger.error(f"Error loading {file_name}: {str(e)}")
        return None, None