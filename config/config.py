"""
Configuration parameters for Quran reciter identification project.
"""
import os
from pathlib import Path

# Directory Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_DIR = "data/train"  # Directory with training audio files
TEST_DATA_DIR = "data/test"  # Directory with test audio files
PROCESSED_DIR = "processed"  # Directory for preprocessed data
LOGS_DIR = "logs"  # Directory for logs
MODEL_OUTPUT_DIR = "models"  # Directory for saved models
TEST_RESULTS_DIR = "test_results"  # Directory for test results
QURAN_DATA_PATH = "data/quran.json" # Path to the Quran text data file
RECITERS_DATA_PATH = "data/reciters.json" # Path to the reciters metadata file

# Audio Processing Parameters
DEFAULT_SAMPLE_RATE = 44100  # Hz
AYAH_MIN_DURATION = 1.0  # Minimum duration for an Ayah segment in seconds
AYAH_MAX_DURATION = 10.0  # Maximum duration for an Ayah segment in seconds
RECITER_MIN_DURATION = 5.0  # Minimum audio duration for reciter identification in seconds
RECITER_MAX_DURATION = 30.0  # Maximum audio duration for reciter identification in seconds
AUDIO_FILE_PATTERN = '*.mp3' # Glob pattern for audio files to process

# Feature Extraction Parameters
N_MFCC = 32  # Number of MFCC features
N_CHROMA = 12  # Number of Chroma features
N_MEL_BANDS = 64  # Number of Mel bands
MFCC_DELTA_ORDER = 2  # Order for delta MFCC features (e.g., 2 for delta-delta)
WINDOW_SIZE_MS = 20  # Window size in milliseconds
STEP_SIZE_MS = 8  # Step size in milliseconds

# Data Augmentation Parameters
PITCH_STEPS = [1.0, -1.0]  # Pitch shift steps
TIME_STRETCH_RATES = [0.95, 1.05]  # Time stretch rates
NOISE_FACTOR = 0.003  # Noise addition factor
VOLUME_ADJUST = 0.9  # Volume adjustment factor

# Model Parameters
MODEL_TYPE = 'random_forest'  # Default model type
RANDOM_STATE = 42  # Random seed
TEST_SIZE = 0.2  # Test split size
N_FOLDS = 5  # Number of folds for cross-validation

# Random Forest Parameters
N_ESTIMATORS = 100  # Number of trees
MAX_DEPTH = 10  # Maximum tree depth

# BLSTM Parameters
LSTM_UNITS = 64  # Number of LSTM units
DROPOUT_RATE = 0.3  # Dropout rate
DENSE_UNITS = 128  # Dense layer units
LEARNING_RATE = 0.001  # Learning rate
BATCH_SIZE = 64  # Batch size
EPOCHS = 50  # Number of epochs
EARLY_STOPPING_PATIENCE = 10  # Early stopping patience
BLSTM_MFCC_COUNT = 13  # Number of MFCCs used for BLSTM (as per research paper)

# BLSTM Advanced Parameters (Added to replace hardcoded values)
# Enhanced dropout rate for BLSTM (different from default DROPOUT_RATE)
BLSTM_DROPOUT_RATE = 0.5
# Enhanced learning rate for BLSTM (different from default LEARNING_RATE)
BLSTM_LEARNING_RATE = 0.0005
BLSTM_WEIGHT_DECAY = 0.01  # L2 regularization weight
BLSTM_NOISE_LEVEL = 0.05  # Noise level for data augmentation
BLSTM_FEATURE_DROPOUT_RATE = 0.1  # Feature dropout rate for regularization
BLSTM_SEQUENCE_LENGTH = 16  # Sequence length for BLSTM input
BLSTM_LR_SCHEDULER_FACTOR = 0.5  # Factor by which to reduce learning rate
# Number of epochs with no improvement after which learning rate will be reduced
BLSTM_LR_SCHEDULER_PATIENCE = 5
BLSTM_MIN_LR = 1e-6  # Minimum learning rate
BLSTM_GRADIENT_CLIP_NORM = 1.0  # Maximum norm for gradient clipping

# Reliability Parameters
CONFIDENCE_THRESHOLD = 0.95  # Primary confidence threshold
SECONDARY_CONFIDENCE_THRESHOLD = 0.10  # Threshold for secondary predictions
MAX_CONFIDENCE_DIFF = 0.80  # Required difference between top predictions
INTRA_CLASS_THRESHOLD_PERCENTILE = 95  # Percentile for intra-class distance threshold calculation
MAX_DISTANCE_RATIO_THRESHOLD = 1.0  # Maximum allowed ratio of distance to threshold for reliability

# Visualization Parameters
PLOT_FIGSIZE = (15, 10)
PLOT_PALETTE = 'viridis'
PLOT_BINS = 50
PLOT_CORR_FIGSIZE = (12, 10)
PLOT_CORR_FEATURE_LIMIT = 20 # Max features for correlation matrix plot

# GPU Usage
USE_GPU = True  # Whether to use GPU if available
