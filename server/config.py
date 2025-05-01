"""
Server configuration settings.
"""
from pathlib import Path

# Import model-related settings from main config
from config import MODEL_OUTPUT_DIR

# Server settings
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 5000

# Debug settings
# SHOW_DEBUG_INFO = args.show_debug # Removed this line - will be handled in app.py

# Audio processing settings for reciter identification
MIN_AUDIO_DURATION = 5  # Minimum audio duration in seconds
MAX_AUDIO_DURATION = 15  # Maximum audio duration in seconds
SAMPLE_RATE = 22050  # Audio sample rate

# Audio processing settings for ayah identification
AYAH_MIN_DURATION = 1  # Minimum audio duration in seconds for ayah identification
AYAH_MAX_DURATION = 10  # Maximum audio duration in seconds for ayah identification

# Model settings
MODEL_DIR = Path(MODEL_OUTPUT_DIR)  # Use the path from main config
LATEST_MODEL_SYMLINK = MODEL_DIR / 'latest'
MODEL_ID = '20250417_023215_train'  # Specific model ID to use (e.g., '20240306_152417_train'), if None use symlink/latest

# Reliability Parameters
CONFIDENCE_THRESHOLD = 0.95  # Primary confidence threshold
SECONDARY_CONFIDENCE_THRESHOLD = 0.10  # Threshold for secondary predictions
MAX_CONFIDENCE_DIFF = 0.80  # Required difference between top predictions

# Server-specific prediction settings
TOP_N_PREDICTIONS = 5  # Number of top predictions to return
