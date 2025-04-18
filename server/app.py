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

from server.config import (
    HOST, PORT, DEBUG, SHOW_UNRELIABLE_PREDICTIONS_IN_DEV,
    SHOW_DEBUG_INFO
)
from server.audio_utils import process_audio_file, extract_features
from server.prediction_utils import load_latest_model, get_predictions
from server.transcription_utils import load_model as load_whisper_model, transcribe_audio
from server.ayah_matcher import load_quran_data, find_matching_ayah


# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and data at startup
try:
    # Load reciter classification model
    model, error = load_latest_model()
    if error:
        logger.error(f"Failed to load reciter model: {error}")
        sys.exit(1)
    logger.info("Reciter model loaded successfully")

    # Load Whisper model
    load_whisper_model()
    logger.info("Whisper model loaded successfully")

    # Load Quran data
    load_quran_data()
    logger.info("Quran data loaded successfully")
except Exception as e:
    logger.error(f"Error loading models or data: {str(e)}")
    sys.exit(1)


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

        # Process audio file with reciter duration constraints
        result = process_audio_file(file, for_ayah=False)
        if result[0] is None:
            error_msg = result[1]
            logger.warning(f"Bad Request (400) during audio processing: {error_msg}")
            return jsonify({
                'error': error_msg
            }), 400

        audio_data, sr = result

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


@app.route('/getAyah', methods=['POST'])
def get_ayah():
    """
    Endpoint to identify Quranic ayah from audio using Whisper model.
    
    Returns:
    - List of potential matching verses, each containing:
        - Ayah text
        - Chapter (surah) number
        - Ayah number
        - Chapter name (in Arabic and English)
        - Confidence score for the match
    - Best match
    - Total number of matches found

    Expected input: 
    - Audio file in form data with key 'audio'
    - Optional: max_matches (int) - Maximum number of matches to return (default: 5)
    - Optional: min_confidence (float) - Minimum confidence threshold (default: 0.70)

    Debug information is included when server is started with --show-debug flag.

    Audio constraints:
    - Minimum duration: 1 second
    - Maximum duration: 10 seconds (longer audio will be truncated)

    Returns: JSON with ayah information and transcription
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

        # Get optional parameters
        try:
            max_matches = int(request.form.get('max_matches', '5'))
            min_confidence = float(request.form.get('min_confidence', '0.70'))
        except ValueError as e:
            error_msg = 'Invalid parameter values. max_matches must be an integer and min_confidence must be a float.'
            logger.warning(f"Bad Request (400): {error_msg}")
            return jsonify({
                'error': error_msg
            }), 400

        # Process audio file with ayah duration constraints
        result = process_audio_file(file, for_ayah=True)
        if result[0] is None:
            error_msg = result[1]
            logger.warning(f"Bad Request (400) during audio processing: {error_msg}")
            return jsonify({
                'error': error_msg
            }), 400

        audio_data, sr = result

        # Transcribe audio and find matches
        try:
            transcription = transcribe_audio(audio_data, sr)
            matches_result = find_matching_ayah(
                transcription,
                min_confidence=min_confidence,
                max_matches=max_matches
            )
            
            # Prepare response
            response = {
                'matches_found': bool(matches_result['matches']),
                'total_matches': matches_result['total_matches']
            }

            if not matches_result['matches']:
                response['message'] = 'No matching verses found with sufficient confidence'
            else:
                response['matches'] = matches_result['matches']
                response['best_match'] = matches_result['best_match']

            # Add debug info when enabled via command line
            if SHOW_DEBUG_INFO:
                # Add transcription at root level for easy access
                response['transcription'] = transcription
                # Add detailed debug info
                response['debug_info'] = matches_result.get('debug_info', {})
                response['debug_info']['transcription'] = transcription
                logger.info("Including debug information in response")
            
            return jsonify(response), 200

        except Exception as e:
            error_msg = f'Error in transcription or matching: {str(e)}'
            logger.error(f"Internal Server Error (500): {error_msg}", exc_info=True)
            return jsonify({
                'error': error_msg
            }), 500

    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        logger.error(f"Internal Server Error (500): {error_msg}", exc_info=True)
        return jsonify({
            'error': error_msg
        }), 500


def run_server():
    """Run the Flask server."""
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    run_server()
