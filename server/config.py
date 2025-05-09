"""
Server-specific configuration settings.
"""
from pathlib import Path
from config import (
    MODEL_OUTPUT_DIR,
    DEFAULT_SAMPLE_RATE,
    CONFIDENCE_THRESHOLD as SRC_CONFIDENCE_THRESHOLD,  # Alias to avoid direct name clash if needed locally
    SECONDARY_CONFIDENCE_THRESHOLD as SRC_SECONDARY_CONFIDENCE_THRESHOLD,
    MAX_CONFIDENCE_DIFF as SRC_MAX_CONFIDENCE_DIFF,
    RECITER_MIN_DURATION,
    RECITER_MAX_DURATION
)

# --- Server Network Settings --- 
HOST = '0.0.0.0'  
PORT = 5000         

# --- Audio Processing Constraints --- 
MIN_AUDIO_DURATION = RECITER_MIN_DURATION  # Minimum audio duration in seconds for reciter ID
MAX_AUDIO_DURATION = RECITER_MAX_DURATION # Maximum audio duration in seconds for reciter ID
# Use the sample rate defined in the main config
SAMPLE_RATE = DEFAULT_SAMPLE_RATE     # Target sample rate for reciter ID processing

# --- Reciter Model Loading Settings --- 
MODEL_DIR = Path(MODEL_OUTPUT_DIR)  
LATEST_MODEL_SYMLINK = MODEL_DIR / 'latest' 
MODEL_ID = '20250417_023215_train'  

# --- Ayah Matching Settings --- 
WHISPER_MODEL_ID = "tarteel-ai/whisper-base-ar-quran" # Whisper model for transcription
AYAH_DEFAULT_MAX_MATCHES = 5 # Default number of matches to return
AYAH_DEFAULT_MIN_CONFIDENCE = 0.70 # Default minimum confidence score (0.0 to 1.0)

# --- Logging Settings --- 
# LOG_LEVEL_PRODUCTION = "INFO"  # Log level when debug mode is OFF (e.g., INFO, WARNING, ERROR) # Removed

# --- Prediction & Reliability Parameters --- 
# Use reliability parameters from the main config
CONFIDENCE_THRESHOLD = SRC_CONFIDENCE_THRESHOLD  
SECONDARY_CONFIDENCE_THRESHOLD = SRC_SECONDARY_CONFIDENCE_THRESHOLD  
MAX_CONFIDENCE_DIFF = SRC_MAX_CONFIDENCE_DIFF  

# Settings used by the server's response formatting
TOP_N_PREDICTIONS = 5  
