"""
Server-specific configuration settings.
"""
from pathlib import Path
from config import MODEL_OUTPUT_DIR

# --- Server Network Settings --- 
HOST = '0.0.0.0'  
PORT = 5000         

# --- Audio Processing Constraints --- 
MIN_AUDIO_DURATION = 5  # Minimum audio duration in seconds for reciter ID
MAX_AUDIO_DURATION = 15 # Maximum audio duration in seconds for reciter ID
SAMPLE_RATE = 22050     # Target sample rate for reciter ID processing

# --- Reciter Model Loading Settings --- 
MODEL_DIR = Path(MODEL_OUTPUT_DIR)  
LATEST_MODEL_SYMLINK = MODEL_DIR / 'latest' 
MODEL_ID = '20250417_023215_train'  

# --- Prediction & Reliability Parameters --- 
CONFIDENCE_THRESHOLD = 0.95  
SECONDARY_CONFIDENCE_THRESHOLD = 0.10  
MAX_CONFIDENCE_DIFF = 0.80  

# Settings used by the server's response formatting
TOP_N_PREDICTIONS = 5  
