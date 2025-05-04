"""
Service layer for Ayah identification.
"""
import logging
import os
import tempfile
import shutil
from pathlib import Path # Import Path for type hint
from typing import Dict, Optional, Tuple
import subprocess # Needed for CalledProcessError

from server.utils.audio_processor_matcher import AudioProcessor
from server.utils.quran_matcher import QuranMatcher
from server.utils.text_utils import to_arabic_number

logger = logging.getLogger(__name__)

# These would typically be passed via dependency injection in a larger app,
# but for simplicity here, we might access them if initialized globally in the factory
# or pass them explicitly to the service function.
# For now, assume they are passed to the function.

def identify_ayah_from_audio( 
    audio_file_storage, # Type hint could be werkzeug.datastructures.FileStorage
    params: Dict,
    quran_matcher: QuranMatcher, 
    raw_quran_data: list,
    debug: bool = False,
    debug_save_dir: Optional[Path] = None 
) -> Tuple[Optional[Dict], Optional[str], Optional[int]]:
    """Processes audio, identifies Ayah, formats result.

    Args:
        audio_file_storage: The FileStorage object from Flask request.
        params: Dictionary containing request parameters (e.g., 'max_matches', 'min_confidence').
        quran_matcher: The initialized QuranMatcher instance.
        raw_quran_data: The raw Quran data (list of surahs) from quran.json.
        debug: Boolean indicating if debug mode is active.
        debug_save_dir: Path object for saving debug files.

    Returns:
        Tuple containing: (response_data, error_message, status_code).
    """
    temp_dir = None
    temp_input_path = None
    wav_path = None
    temp_dir_created_by_processor = False
    processed_audio_temp_dir = None

    try:
        max_matches = int(params.get('max_matches') or 5) # Handle None from form.get
        min_confidence = float(params.get('min_confidence') or 0.70)
        score_threshold = int(min_confidence * 100)
        logger.info(f"[AyahService] Identifying Ayah: max_matches={max_matches}, min_confidence={min_confidence} (score_threshold={score_threshold})")

        # --- Audio Processing --- 
        temp_dir = tempfile.mkdtemp()
        logger.info(f"[AyahService] Created temp directory: {temp_dir}")
        
        temp_input_filename = "input_audio_" + os.path.basename(audio_file_storage.filename or "audio.bin")
        safe_temp_input_filename = "".join(c for c in temp_input_filename if c.isalnum() or c in ('.', '-', '_')).rstrip()
        if not safe_temp_input_filename: safe_temp_input_filename = "input_audio.bin"
        temp_input_path = os.path.join(temp_dir, safe_temp_input_filename)
        
        try:
             audio_file_storage.seek(0)
             audio_file_storage.save(temp_input_path)
             logger.info(f"[AyahService] Saved uploaded audio to: {temp_input_path}")
        except Exception as save_err:
             logger.error(f"[AyahService] Failed to save uploaded audio to {temp_input_path}: {save_err}")
             return None, f"Failed to save uploaded audio file: {save_err}", 500 
        
        logger.info(f"[AyahService] Converting {temp_input_path} to WAV...")
        wav_path, processed_audio_temp_dir = AudioProcessor.convert_to_wav(temp_input_path, output_dir=temp_dir)
        if processed_audio_temp_dir != temp_dir and tempfile.gettempdir() in str(processed_audio_temp_dir): # Ensure comparison works
             temp_dir_created_by_processor = True
             logger.warning(f"[AyahService] AudioProcessor created its own temp directory: {processed_audio_temp_dir}")
        logger.info(f"[AyahService] Converted WAV path: {wav_path}")

        # --- Debug Saving Processed Audio --- 
        if debug and debug_save_dir and wav_path and os.path.exists(wav_path):
            try:
                processed_audio_debug_path = debug_save_dir / "processed_audio.wav"
                shutil.copyfile(wav_path, processed_audio_debug_path)
                logger.info(f"[AyahService-Debug] Copied processed audio to {processed_audio_debug_path}")
            except Exception as processed_save_err:
                logger.warning(f"[AyahService-Debug] Failed to save processed audio: {processed_save_err}")
        
        # --- Transcription --- 
        logger.info("[AyahService] Transcribing audio...")
        transcription = quran_matcher.transcribe_audio(wav_path)
        logger.info(f"[AyahService] Transcription result: '{transcription}'")

        # --- Matching --- 
        logger.info(f"[AyahService] Finding matches...")
        raw_matches, normalized_transcription, normalized_matches_debug = quran_matcher.find_matches(
            transcription, top_n=max_matches, score_threshold=score_threshold
        )

        # --- Format response --- 
        formatted_matches = []
        if not raw_quran_data:
             logger.error("[AyahService] Cannot format matches: raw_quran_data is empty or None.")
        else:
            for match in raw_matches:
                try:
                    surah_num = int(match['surah_number'])
                    ayah_num = int(match['ayah_number'])
                    confidence_score = match['confidence_score']

                    surah_name = "Unknown Surah"
                    surah_name_en = "Unknown Surah"
                    ayah_text = "Unknown Ayah Text"
                    unicode_char = ""
                    
                    surah_data = next((s for s in raw_quran_data if isinstance(s, dict) and s.get('id') == surah_num), None)
                    
                    if surah_data:
                        surah_name = surah_data.get('name', surah_name)
                        surah_name_en = surah_data.get('transliteration', surah_data.get('translation', surah_name_en))
                        unicode_char = surah_data.get('unicode', unicode_char)
                        
                        ayah_data = next((v for v in surah_data.get('verses', []) if isinstance(v, dict) and v.get('id') == ayah_num), None)
                        if ayah_data:
                            ayah_text = ayah_data.get('text', ayah_text)
                    else:
                        logger.warning(f"[AyahService] Could not find Surah {surah_num} in provided raw_quran_data for formatting.")

                    formatted_match = {
                        'surah_number': to_arabic_number(surah_num),
                        'surah_number_en': surah_num,
                        'surah_name': surah_name,
                        'surah_name_en': surah_name_en,
                        'ayah_number': to_arabic_number(ayah_num),
                        'ayah_number_en': ayah_num,
                        'ayah_text': ayah_text,
                        'confidence_score': confidence_score, 
                        'unicode': unicode_char
                    }
                    formatted_matches.append(formatted_match)
                except (TypeError, KeyError, ValueError) as format_err:
                     logger.error(f"[AyahService] Error formatting match data ({match}): {format_err}", exc_info=True)
                     continue 

        logger.info(f"[AyahService] Formatted {len(formatted_matches)} matches.")

        # --- Construct Final Response Data --- 
        response_data = {
            'matches_found': len(formatted_matches) > 0,
            'total_matches': len(formatted_matches),
            'matches': formatted_matches,
            'best_match': formatted_matches[0] if formatted_matches else None
        }

        # Add debug info if requested
        if debug:
            response_data['transcription'] = transcription
            response_data['debug_info'] = {
                'transcription': transcription,
                'normalized_transcription': normalized_transcription,
                'normalized_matches': normalized_matches_debug
            }
            logger.info("[AyahService-Debug] Added debug info to response data.")

        # Success: Return data, no error message, suggest 200 OK
        return response_data, None, 200

    # --- Error Handling --- 
    except FileNotFoundError as e:
         logger.error(f"[AyahService] File not found error: {e}", exc_info=True)
         return None, f'File operation error: {e}', 500
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else "No stderr output from ffmpeg"
        logger.error(f"[AyahService] FFmpeg execution failed: {stderr_msg}")
        return None, f'Audio processing failed (ffmpeg error). Details: {stderr_msg}', 500
    except RuntimeError as e:
        logger.error(f"[AyahService] Runtime error: {e}", exc_info=True)
        error_msg = str(e)
        status_code = 500
        if "Audio conversion failed" in error_msg: status_code = 400
        elif "Audio transcription failed" in error_msg: status_code = 500
        elif "Quran data" in error_msg: status_code = 503
        return None, f'Processing error: {error_msg}', status_code
    except Exception as e:
        logger.exception(f"[AyahService] Unexpected error: {e}") 
        return None, f'An unexpected server error occurred: {type(e).__name__}', 500

    # --- Cleanup --- 
    finally:
        if temp_dir and os.path.exists(temp_dir):
             try:
                 shutil.rmtree(temp_dir)
                 logger.info(f"[AyahService] Cleaned up temp directory: {temp_dir}")
             except Exception as e:
                 logger.warning(f"[AyahService] Failed to clean up temp directory {temp_dir}: {e}")
        
        if temp_dir_created_by_processor and processed_audio_temp_dir and os.path.exists(processed_audio_temp_dir):
            try:
                 shutil.rmtree(processed_audio_temp_dir)
                 logger.info(f"[AyahService] Cleaned up AudioProcessor temp directory: {processed_audio_temp_dir}")
            except Exception as e:
                 logger.warning(f"[AyahService] Failed to clean up AudioProcessor temp directory {processed_audio_temp_dir}: {e}") 