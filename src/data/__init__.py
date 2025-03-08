"""
Data handling module for Quran reciter identification project.
"""

# Import key functions to make them available through the data module
from src.data.loader import load
from src.data.preprocessing import preprocess_audio, preprocess_audio_with_logic
from src.data.augmentation import augment_audio
