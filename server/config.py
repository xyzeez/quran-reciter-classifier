"""
Server configuration settings.
"""
from pathlib import Path

# Import model-related settings from main config
from config import (
    MODEL_OUTPUT_DIR,
    CONFIDENCE_THRESHOLD,
    SECONDARY_CONFIDENCE_THRESHOLD,
    MAX_CONFIDENCE_DIFF
)

# Server settings
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 5000
DEBUG = False

# Audio processing settings
MIN_AUDIO_DURATION = 5  # Minimum audio duration in seconds
MAX_AUDIO_DURATION = 15  # Maximum audio duration in seconds
SAMPLE_RATE = 22050  # Audio sample rate

# Model settings
MODEL_DIR = Path(MODEL_OUTPUT_DIR)
LATEST_MODEL_SYMLINK = MODEL_DIR / 'latest'
MODEL_ID = '20250309_113120_train'  # Specific model ID to use (e.g., '20240306_152417_train'), if None use symlink/latest

# Server-specific prediction settings
TOP_N_PREDICTIONS = 5  # Number of top predictions to return
