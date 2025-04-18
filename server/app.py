"""
Flask server for Quran Reciter Classifier.
"""
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify

# Add project root to Python path when running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from server.config import HOST, PORT, DEBUG, SHOW_UNRELIABLE_PREDICTIONS_IN_DEV
from server.audio_utils import process_audio_file, extract_features
from server.prediction_utils import load_latest_model, get_predictions


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
    Optional input: show_unreliable_predictions (boolean) - Show predictions even when unreliable
    Returns: JSON with prediction results
    """
    try:
        # Check if file is present in request
        if 'audio' not in request.files:
            error_msg = 'No audio file provided. Please send the file with key "audio" in form data.'
            logger.warning(f"Bad Request (400): {error_msg}")
            return jsonify({
                'error': error_msg
            }), 400

        file = request.files['audio']
        if file.filename == '':
            error_msg = 'No audio file selected'
            logger.warning(f"Bad Request (400): {error_msg}")
            return jsonify({
                'error': error_msg
            }), 400

        # Get show_unreliable_predictions parameter from form data
        show_unreliable = request.form.get('show_unreliable_predictions', '').lower() == 'true'
        
        # Use the global setting if not explicitly set in request
        if not show_unreliable and SHOW_UNRELIABLE_PREDICTIONS_IN_DEV:
            show_unreliable = True

        # Process audio file
        result = process_audio_file(file)
        if result[0] is None:  # If first element is None, second element is error message
            error_msg = result[1] # Get the error message
            logger.warning(f"Bad Request (400) during audio processing: {error_msg}")
            return jsonify({
                'error': error_msg  # Return the error message
            }), 400

        audio_data, sr = result  # Unpack the successful result

        # Extract features
        try:
            features = extract_features(audio_data, sr)
            if features is None:
                error_msg = 'Feature extraction failed'
                logger.error(f"Internal Server Error (500): {error_msg}")
                return jsonify({
                    'error': error_msg
                }), 500
        except Exception as e:
            error_msg = f'Error extracting features: {str(e)}'
            logger.error(f"Internal Server Error (500): {error_msg}", exc_info=True) # Log exception info
            return jsonify({
                'error': error_msg
            }), 500

        # Get predictions
        try:
            result = get_predictions(model, features, show_unreliable_predictions=show_unreliable)
            
            # Ensure all values are JSON serializable
            try:
                return jsonify(result), 200
            except TypeError as e:
                logger.error(f"JSON serialization error: {str(e)}", exc_info=True)
                # Try to fix common serialization issues
                if "is not JSON serializable" in str(e):
                    error_msg = f"Response contains non-JSON serializable data: {str(e)}"
                    logger.error(error_msg)
                    return jsonify({
                        'error': error_msg
                    }), 500
                raise  # Re-raise if it's a different TypeError

        except Exception as e:
            error_msg = f'Error making prediction: {str(e)}'
            logger.error(f"Internal Server Error (500): {error_msg}", exc_info=True) # Log exception info
            return jsonify({
                'error': error_msg
            }), 500

    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        logger.error(f"Internal Server Error (500): {error_msg}", exc_info=True) # Log exception info
        return jsonify({
            'error': error_msg
        }), 500


def run_server():
    """Run the Flask server."""
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    run_server()
