"""
Server configuration settings.
"""
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Quran Reciter Classifier Server')
parser.add_argument('--show-debug', action='store_true', help='Enable debug information in responses')
args = parser.parse_args()

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
DEBUG = False  # Production mode by default
SHOW_UNRELIABLE_PREDICTIONS_IN_DEV = True  # When True, shows unreliable predictions regardless of debug mode

# Debug settings
SHOW_DEBUG_INFO = args.show_debug  # Set by command line argument, independent of dev/prod mode

# Audio processing settings for reciter identification
MIN_AUDIO_DURATION = 5  # Minimum audio duration in seconds
MAX_AUDIO_DURATION = 15  # Maximum audio duration in seconds
SAMPLE_RATE = 22050  # Audio sample rate

# Audio processing settings for ayah identification
AYAH_MIN_DURATION = 1  # Minimum audio duration in seconds for ayah identification
AYAH_MAX_DURATION = 10  # Maximum audio duration in seconds for ayah identification

# Model settings
MODEL_DIR = Path(MODEL_OUTPUT_DIR)
LATEST_MODEL_SYMLINK = MODEL_DIR / 'latest'
MODEL_ID = '20250417_023215_train'  # Specific model ID to use (e.g., '20240306_152417_train'), if None use symlink/latest

# Server-specific prediction settings
TOP_N_PREDICTIONS = 5  # Number of top predictions to return
