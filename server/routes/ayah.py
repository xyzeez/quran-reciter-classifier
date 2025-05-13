"""
Routes for Ayah identification endpoints.
"""
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from server.services.ayah_service import identify_ayah_from_audio

logger = logging.getLogger(__name__)

# Blueprint definition: url_prefix='/' means routes are directly under the root
ayah_bp = Blueprint('ayah_bp', __name__, url_prefix='/')

@ayah_bp.route('/getAyah', methods=['POST'])
def handle_get_ayah():
    """Handle POST requests for Ayah identification."""
    debug_save_dir = None
    audio_file = None

    try:
        # Access shared components from the app context
        quran_matcher_instance = current_app.quran_matcher
        raw_quran_data = current_app.raw_quran_data
        
        if quran_matcher_instance is None or raw_quran_data is None:
             logger.error("Ayah Service dependencies (matcher/data) not available.")
             return jsonify({'error': 'Ayah identification service is not properly configured.'}), 503

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided. Key must be "audio".'}), 400

        audio_file = request.files['audio']
        if not audio_file or not audio_file.filename:
            return jsonify({'error': 'Invalid or empty audio file provided.'}), 400

        # Setup debug directory if in debug mode
        if current_app.debug:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_save_dir_base = Path("api-responses") / "getAyah"
                debug_save_dir = debug_save_dir_base / timestamp
                os.makedirs(debug_save_dir, exist_ok=True)
                logger.info(f"[AyahRoute-Debug] Saving request data to {debug_save_dir}")

                # Save original audio
                original_filename = audio_file.filename
                safe_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '-', '_')).rstrip()
                if not safe_filename: safe_filename = 'audio.unknown'
                original_audio_path = debug_save_dir / f"original_{safe_filename}"
                try:
                    audio_file.save(original_audio_path)
                    audio_file.seek(0) 
                    logger.info(f"[AyahRoute-Debug] Saved original audio to {original_audio_path}")
                except Exception as save_err:
                    logger.warning(f"[AyahRoute-Debug] Failed to save original audio: {save_err}")

            except Exception as e:
                logger.warning(f"[AyahRoute-Debug] Error setting up debug save directory: {e}")
                debug_save_dir = None 
        
        # Extract parameters for the service
        params = { 
            'max_matches': request.form.get('max_matches'), 
            'min_confidence': request.form.get('min_confidence')
        }
        
        # Call the service layer
        response_data, error_message, status_code = identify_ayah_from_audio(
            audio_file_storage=audio_file, 
            params=params,
            quran_matcher=quran_matcher_instance,
            raw_quran_data=raw_quran_data,
            debug=current_app.debug,
            debug_save_dir=debug_save_dir
        )
        
        # Handle potential errors from the service
        if error_message:
             logger.error(f"[AyahRoute] Service error: {error_message} (Status: {status_code})")
             error_response = {'error': error_message}
             
             # Save error response in debug mode
             if current_app.debug and debug_save_dir:
                 try:
                     json_path = debug_save_dir / f"response_error_{status_code}.json"
                     with open(json_path, 'w', encoding='utf-8') as f:
                         json.dump(error_response, f, ensure_ascii=False, indent=4)
                     logger.info(f"[AyahRoute-Debug] Saved error response to {json_path}")
                     # Try saving original audio again on error
                     original_audio_error_path = debug_save_dir / f"original_audio_on_error.bin"
                     if audio_file and not os.path.exists(original_audio_error_path):
                          try:
                               audio_file.seek(0) 
                               audio_file.save(original_audio_error_path)
                          except Exception as save_err:
                               logger.warning(f"[AyahRoute-Debug] Failed to save original audio on error: {save_err}")
                 except Exception as debug_save_err:
                     logger.warning(f"[AyahRoute-Debug] Failed to save error details: {debug_save_err}")
             
             return jsonify(error_response), status_code
        
        # Save successful response in debug mode
        if current_app.debug and debug_save_dir:
            try:
                json_path = debug_save_dir / "response.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, ensure_ascii=False, indent=4)
                logger.info(f"[AyahRoute-Debug] Saved final response to {json_path}")
            except Exception as json_save_err:
                logger.warning(f"[AyahRoute-Debug] Failed to save final JSON response: {json_save_err}")

        # Return successful response
        return jsonify(response_data), status_code

    except Exception as e:
        # Catch unexpected errors in the route handler
        logger.exception(f"[AyahRoute] Unexpected error in /getAyah handler: {e}")
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