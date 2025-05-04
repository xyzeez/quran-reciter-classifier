"""
Routes for Reciter identification and listing endpoints.
"""
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

# Import service functions
from server.services.reciter_service import identify_reciter_from_audio, get_all_reciters_list

logger = logging.getLogger(__name__)

# Define blueprint with root prefix
reciter_bp = Blueprint('reciter_bp', __name__, url_prefix='/')

@reciter_bp.route('/getReciter', methods=['POST'])
def handle_get_reciter():
    """Handle POST requests for Reciter identification."""
    debug_save_dir = None
    audio_file = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided. Key must be "audio".'}), 400

        audio_file = request.files['audio']
        if not audio_file or not audio_file.filename:
            return jsonify({'error': 'Invalid or empty audio file provided.'}), 400

        # Setup debug directory if in debug mode
        if current_app.debug:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_save_dir_base = Path("api-responses") / "getReciter"
                debug_save_dir = debug_save_dir_base / timestamp
                os.makedirs(debug_save_dir, exist_ok=True)
                logger.info(f"[ReciterRoute-Debug] Saving request data to {debug_save_dir}")

                # Save original audio
                original_filename = audio_file.filename
                safe_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '-', '_')).rstrip()
                if not safe_filename: safe_filename = 'audio.unknown'
                original_audio_path = debug_save_dir / f"original_{safe_filename}"
                try:
                    # Save a copy for debugging, ensure original stream is seeked back
                    audio_file.save(original_audio_path)
                    audio_file.seek(0) 
                    logger.info(f"[ReciterRoute-Debug] Saved original audio to {original_audio_path}")
                except Exception as save_err:
                    logger.warning(f"[ReciterRoute-Debug] Failed to save original audio: {save_err}")
                    # Don't fail the request, just log the warning

            except Exception as e:
                logger.warning(f"[ReciterRoute-Debug] Error setting up debug save directory: {e}")
                debug_save_dir = None # Ensure it's None if setup fails

        # Extract params
        params = {
            'show_unreliable_predictions': request.form.get('show_unreliable_predictions', '')
        }

        # Call service, passing debug status
        response_data, error_message, status_code = identify_reciter_from_audio(
            audio_file_storage=audio_file,
            params=params,
            debug=current_app.debug # Pass debug status to service
        )

        if error_message:
            logger.error(f"[ReciterRoute] Service error: {error_message} (Status: {status_code})")
            error_response = {'error': error_message}
            # Save error response in debug mode
            if current_app.debug and debug_save_dir:
                try:
                    json_path = debug_save_dir / f"response_error_{status_code}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(error_response, f, ensure_ascii=False, indent=4)
                    logger.info(f"[ReciterRoute-Debug] Saved error response to {json_path}")
                except Exception as debug_save_err:
                    logger.warning(f"[ReciterRoute-Debug] Failed to save error details: {debug_save_err}")
            return jsonify(error_response), status_code

        # Save successful response in debug mode
        if current_app.debug and debug_save_dir:
             try:
                 json_path = debug_save_dir / "response.json"
                 with open(json_path, 'w', encoding='utf-8') as f:
                     json.dump(response_data, f, ensure_ascii=False, indent=4)
                 logger.info(f"[ReciterRoute-Debug] Saved final response to {json_path}")
             except Exception as json_save_err:
                 logger.warning(f"[ReciterRoute-Debug] Failed to save final JSON response: {json_save_err}")

        return jsonify(response_data), status_code

    except Exception as e:
        logger.exception(f"[ReciterRoute] Unexpected error in /getReciter handler: {e}")
        error_response = {'error': f'An unexpected server error occurred in route handler: {type(e).__name__}'}
        # Save debug info if possible
        if current_app.debug and debug_save_dir:
             try:
                 json_path = debug_save_dir / "response_route_error_500.json"
                 with open(json_path, 'w', encoding='utf-8') as f:
                     json.dump(error_response, f, ensure_ascii=False, indent=4)
             except Exception:
                 pass # Ignore errors during error reporting
        return jsonify(error_response), 500

@reciter_bp.route('/getAllReciters', methods=['GET'])
def handle_get_all_reciters():
    """Handle GET requests for listing all reciters."""
    try:
        response_data, error_message, status_code = get_all_reciters_list()
        
        if error_message:
            logger.error(f"[ReciterRoute] Service error: {error_message} (Status: {status_code})")
            return jsonify({'error': error_message}), status_code
            
        return jsonify(response_data), status_code

    except Exception as e:
        logger.exception(f"[ReciterRoute] Unexpected error in /getAllReciters handler: {e}")
        return jsonify({'error': f'An unexpected server error occurred in route handler: {type(e).__name__}'}), 500 