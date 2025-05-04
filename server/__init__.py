"""
Server package for Quran Reciter Classifier.
"""
from server.app import app
from src.utils.audio_utils import process_audio_file, extract_features
from server.utils.prediction import get_predictions

__all__ = [
    'app',
    'process_audio_file',
    'extract_features',
    'get_predictions'
]

def run_server():
    """Run the Flask server in debug mode."""
    app.run(debug=True)
