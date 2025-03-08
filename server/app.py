"""
Flask server for Quran Reciter Classifier.
"""
from server.config import HOST, PORT, DEBUG
from server.audio_utils import process_audio_file, extract_features
from server.prediction_utils import load_latest_model, get_predictions
from flask import Flask, request, jsonify
import sys
import logging


# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at startup
model, error = load_latest_model()
if error:
    logger.error(f"Failed to load model: {error}")
    sys.exit(1)
logger.info("Model loaded successfully")


@app.route('/getReciter', methods=['POST'])
def get_reciter():
    """
    Endpoint to identify reciter from audio file.

    Expected input: Audio file in form data with key 'audio'
    Returns: JSON with prediction results
    """
    try:
        # Check if file is present in request
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided. Please send the file with key "audio" in form data.'
            }), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({
                'error': 'No audio file selected'
            }), 400

        # Process audio file
        result = process_audio_file(file)
        if result[0] is None:  # If first element is None, second element is error message
            return jsonify({
                'error': result[1]  # Return the error message
            }), 400

        audio_data, sr = result  # Unpack the successful result

        # Extract features
        try:
            features = extract_features(audio_data, sr)
            if features is None:
                return jsonify({
                    'error': 'Feature extraction failed'
                }), 500
        except Exception as e:
            return jsonify({
                'error': f'Error extracting features: {str(e)}'
            }), 500

        # Get predictions
        try:
            result = get_predictions(model, features)
            return jsonify(result), 200

        except Exception as e:
            return jsonify({
                'error': f'Error making prediction: {str(e)}'
            }), 500

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


def run_server():
    """Run the Flask server."""
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    run_server()
