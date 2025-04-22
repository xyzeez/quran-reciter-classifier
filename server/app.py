"""
Flask server for Quran Reciter Classifier.
"""
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify
import os
import json

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

# --- Helper Function for Arabic Numerals ---
def int_to_arabic_numeral(number: int) -> str:
    """Converts an integer to Eastern Arabic numeral string."""
    arabic_digits = { '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩' }
    return "".join(arabic_digits[digit] for digit in str(number))
# --- End Helper Function ---

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

# --- Reciters Data Loading --- 
reciters_data = None
def load_reciters_data():
    global reciters_data
    if reciters_data:
        return reciters_data
    try:
        # Try different paths relative to app.py
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent 
        possible_paths = [
            project_root / 'data' / 'reciters.json',
            script_dir / '..' / 'data' / 'reciters.json' # Relative path
        ]
        
        reciters_file = None
        for path in possible_paths:
            if path.exists():
                reciters_file = path
                break

        if reciters_file is None:
             raise FileNotFoundError(
                f"Could not find reciters.json in expected locations relative to app.py: {[str(p) for p in possible_paths]}"
            )
            
        logger.info(f"Loading reciters data from: {reciters_file}")
        with open(reciters_file, 'r', encoding='utf-8') as f:
            reciters_data = json.load(f)
        return reciters_data
    except Exception as e:
        logger.error(f"Error loading reciters.json: {str(e)}")
        return None # Return None on error

# Ensure reciters data is loaded at startup
load_reciters_data()
# --- End Reciters Data Loading ---

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

        # --- Helper to transform match data --- 
        def transform_match_to_ayah(match_obj):
            if not match_obj:
                return None
            # Get the unicode directly from the match object
            ayah_unicode = match_obj.get('unicode', '') 

            return {
                'surah_number': int_to_arabic_numeral(match_obj.get('surah_number', 0)),
                'surah_number_en': match_obj.get('surah_number', 0),
                'surah_name': match_obj.get('surah_name', ''),
                'surah_name_en': match_obj.get('surah_name_en', ''),
                'ayah_number': int_to_arabic_numeral(match_obj.get('ayah_number', 0)),
                'ayah_number_en': match_obj.get('ayah_number', 0),
                'ayah_text': match_obj.get('ayah_text', ''),
                'unicode': ayah_unicode # Use the actual unicode value
            }
        # --- End Helper --- 

        # Transcribe audio and find matches
        try:
            transcription = transcribe_audio(audio_data, sr)
            matches_result = find_matching_ayah(
                transcription,
                min_confidence=min_confidence,
                max_matches=max_matches
            )

            # Determine reliability based on best_match presence
            reliable = bool(matches_result.get('best_match'))

            # Transform best_match and top_matches
            matchedAyah = transform_match_to_ayah(matches_result.get('best_match'))
            similarAyahs = [transform_match_to_ayah(m) for m in matches_result.get('matches', []) if m]
            
            # Prepare response in the structure expected by the app
            response = {
                'reliable': reliable,
                'matchedAyah': matchedAyah,       # Will be None if not reliable
                'similarAyahs': similarAyahs    # List of transformed matches
            }
            
            # Optional: Include original debug info if needed 
            # (though the app doesn't expect it in this structure)
            if SHOW_DEBUG_INFO:
                 response['debug_info_original'] = matches_result.get('debug_info', {})
                 response['debug_info_original']['transcription'] = transcription
                 logger.info("Including original debug information in response under 'debug_info_original'")

            return jsonify(response), 200

        except Exception as e:
            error_msg = f'Error in transcription or matching: {str(e)}'
            logger.error(f"Internal Server Error (500): {error_msg}", exc_info=True)
            return jsonify({'error': error_msg}), 500

    except Exception as e:
        error_msg = f'Server error: {str(e)}'
        logger.error(f"Internal Server Error (500): {error_msg}", exc_info=True)
        return jsonify({
            'error': error_msg
        }), 500

# --- New Endpoint for Reciters --- 
@app.route('/getAllReciters', methods=['GET'])
def get_all_reciters():
    """Endpoint to get a list of all available reciters."""
    global reciters_data
    if not reciters_data:
        # Attempt to reload if it failed at startup
        load_reciters_data()
        if not reciters_data:
             return jsonify({'error': 'Reciters data could not be loaded.'}), 500

    reciters_list = []
    try:
        for name, data in reciters_data.items():
            servers = data.get('servers', [])
            server_url = ''
            if isinstance(servers, list):
                server_url = servers[0] if servers else ''
            elif isinstance(servers, str):
                server_url = servers
                
            reciter_info = {
                'name': name,
                'nationality': data.get('nationality', 'Unknown'),
                'flagUrl': data.get('flagUrl', ''),
                'imageUrl': data.get('imageUrl', ''),
                'serverUrl': server_url
            }
            reciters_list.append(reciter_info)
            
        return jsonify(reciters_list), 200
    except Exception as e:
        logger.error(f"Error processing reciters data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error processing reciters data.'}), 500
# --- End New Endpoint --- 

def run_server():
    """Run the Flask server."""
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    run_server()
