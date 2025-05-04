"""
Main Flask server for Quran reciter and ayah identification API endpoints.
"""
import logging
from flask import Flask, request, jsonify
import numpy as np
import json
from pathlib import Path
import argparse
import os
import tempfile
import subprocess
import re
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from rapidfuzz import process, fuzz

from src.utils.audio_utils import process_audio_file
from src.features import extract_features
from server.utils.model_loader import initialize_models, get_reciter_model
from src.utils.ayah_matching import load_quran_data
from src.utils.distance_utils import calculate_distances, analyze_prediction_reliability
from server.config import TOP_N_PREDICTIONS, HOST, PORT
from server.utils.text_utils import to_arabic_number

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load Quran data at startup
quran_data_result = load_quran_data()
if not quran_data_result:
    logger.error("Failed to load Quran data")
    raw_quran_data = []
else:
    raw_quran_data = quran_data_result.get('raw_data', [])

reciter_success, ayah_success = initialize_models()
if not (reciter_success and ayah_success):
    logger.error("Failed to initialize one or more models")

# --- Quran Matcher Logic Start ---
# (Copied and adapted from quran-matcher/app)

class AudioProcessor:
    @staticmethod
    def convert_to_wav(input_file, output_dir=None):
        """Convert any audio file to 16kHz mono WAV using ffmpeg"""
        try:
            if output_dir is None:
                output_dir = tempfile.mkdtemp()

            os.makedirs(output_dir, exist_ok=True)
            # Use a unique name to avoid collisions if multiple requests happen concurrently
            output_path = os.path.join(output_dir, f"processed_{os.path.basename(input_file)}.wav")

            # FFmpeg command to convert to mono, 16kHz WAV
            command = [
                "ffmpeg",
                "-y",                   # Overwrite output file if exists
                "-i", input_file,      # Input file
                "-ac", "1",            # Mono
                "-ar", "16000",        # Sample rate 16kHz
                output_path
            ]

            # Use Flask logger
            logger.info(f"Running ffmpeg command: {' '.join(command)}")
            process = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"ffmpeg stdout: {process.stdout}")
            logger.error(f"ffmpeg stderr: {process.stderr}") # Log stderr even on success for debugging
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            raise RuntimeError("Audio conversion failed: unsupported format or ffmpeg error")
        except Exception as e:
            logger.error(f"Error during audio conversion: {e}")
            raise

class QuranMatcher:
    def __init__(self, loaded_quran_data):
        self.model_id = "tarteel-ai/whisper-base-ar-quran"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[QuranMatcher] Initializing model {self.model_id} on {self.device.upper()} device")

        # load Whisper model
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        except Exception as e:
            logger.error(f"[QuranMatcher] Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to load Whisper model {self.model_id}") from e

        # Use the passed-in data
        logger.info("[QuranMatcher] Using pre-loaded Quran data.")
        self.quran_data = loaded_quran_data

        # load and prepare verses from the passed-in data
        try:
            if not self.quran_data:
                 logger.error("[QuranMatcher] Provided Quran data is empty or invalid.")
                 raise ValueError("Provided Quran data is empty.")
                 
            self.all_verses = self._prepare_verse_database()
            # cache normalized strings for matching
            self.normalized_verses = [v["normalized"] for v in self.all_verses]
            logger.info(f"[QuranMatcher] Prepared {len(self.all_verses)} verses from pre-loaded data.")
            except Exception as e:
            logger.error(f"[QuranMatcher] Failed to process pre-loaded Quran data: {e}")
            # If processing fails, ensure the matcher is in a state where it won't crash later
            self.quran_data = []
            self.all_verses = []
            self.normalized_verses = []
            raise RuntimeError("Failed to process pre-loaded Quran data") from e # Re-raise

    def _prepare_verse_database(self):
        """Flatten and normalize all verses"""
        verses = []
        if not self.quran_data:
            logger.error("[QuranMatcher] Cannot prepare verse database: Quran data not loaded.")
            return []
        for chapter in self.quran_data:
            # Check if chapter is a dictionary before accessing keys
            if not isinstance(chapter, dict):
                logger.warning(f"[QuranMatcher] Skipping invalid chapter data item: {type(chapter)}")
                continue
            for verse in chapter.get("verses", []):
                 if not isinstance(verse, dict):
                    logger.warning(f"[QuranMatcher] Skipping invalid verse data item in Surah {chapter.get('id')}: {type(verse)}")
                    continue
                 orig = verse.get("text", "")
                 verses.append({
                    "surah_num": chapter.get("id"),
                    "surah_name": chapter.get("name"), # Assuming name is Arabic
                    # Use transliteration for en name, fallback as before
                    "surah_name_en": chapter.get('transliteration', chapter.get('translation', '')),
                    "ayah_num": verse.get("id"),
                    "original": orig,
                    "normalized": self._normalize_text(orig)
                })
        return verses

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize Arabic text for robust matching"""
        # remove diacritics
        diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]')
        text = diacritics.sub('', text)
        # remove tatweel
        text = text.replace('ـ', '')
        # remove punctuation (keep only Arabic letters and spaces)
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text) # Adjusted range for Arabic letters

        # standardize variants
        variants = {
            'آ': 'ا', 'أ': 'ا', 'إ': 'ا', 'ٱ': 'ا',
            'ى': 'ي', 'ئ': 'ي', 'ؤ': 'و',
            'ة': 'ه', # Keep 'ﷲ' transformation if needed, or remove if base 'الله' preferred
            # 'ﷲ': 'الله' # Uncomment if needed, check if source data uses ligature
        }
        for old, new in variants.items():
            text = text.replace(old, new)

        # collapse whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper model"""
        if not hasattr(self, 'model') or not hasattr(self, 'processor'):
             raise RuntimeError("QuranMatcher not properly initialized. Cannot transcribe.")
        try:
            logger.info(f"[QuranMatcher] Loading audio file: {audio_path}")
            wav, sr = torchaudio.load(audio_path)
            # Ensure mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # Check sample rate (should be 16k after conversion, but double-check)
            if sr != 16000:
                 logger.warning(f"[QuranMatcher] Audio sample rate is {sr}Hz, expected 16000Hz. Resampling...")
                 resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                 wav = resampler(wav)
                 sr = 16000

            logger.info("[QuranMatcher] Extracting features...")
            inputs = self.processor(wav.squeeze(0), sampling_rate=sr, return_tensors='pt') # Use processor directly
            input_feats = inputs.input_features.to(self.device)

            logger.info("[QuranMatcher] Generating transcription...")
            # Specify language and task for potentially better results
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ar", task="transcribe")
            pred_ids = self.model.generate(input_feats, forced_decoder_ids=forced_decoder_ids)
            transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            logger.info(f"[QuranMatcher] Raw transcription: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"[QuranMatcher] Transcription failed for {audio_path}: {e}")
            # Re-raise with the original exception for better debugging if needed
            raise RuntimeError("Audio transcription failed") from e

    def find_matches(self, transcription: str, top_n: int = 5, score_threshold: int = 30) -> list:
        """Find top matching Quran verses via fuzzy matching"""
        if not self.all_verses:
             logger.error("[QuranMatcher] Cannot find matches: Verse database is empty.")
             return [], "" # Return empty list and empty normalized text
        clean = self._normalize_text(transcription)
        if not clean:
            logger.warning("[QuranMatcher] Normalized transcription is empty. Cannot perform matching.")
            return [], clean # Return empty list but the (empty) normalized text
        logger.info(f"[QuranMatcher] Normalized transcription for matching: {clean}")

        # get fuzzy scores against precomputed normalized verses
        try:
            logger.info(f"[QuranMatcher] Running fuzzy matching (top_n={top_n}, threshold={score_threshold})...")
            results = process.extract(clean, self.normalized_verses,
                                      scorer=fuzz.token_set_ratio,
                                      limit=top_n * 3) # Fetch more initially for filtering
        except Exception as e:
            logger.error(f"[QuranMatcher] Fuzzy matching failed: {e}")
            return [], clean # Return empty list but the normalized text

        # sort by score descending
        sorted_res = sorted(results, key=lambda x: x[1], reverse=True)

        matches = []
        seen = set()
        # Store normalized matches for potential debug output
        normalized_matches_debug = []

        logger.info(f"[QuranMatcher] Processing {len(sorted_res)} potential matches...")
        for choice, score, idx in sorted_res:
            if score < score_threshold:
                 logger.info(f"[QuranMatcher] Score {score:.2f} below threshold {score_threshold}. Stopping.")
                 break # Scores are sorted, no need to check further

            verse = self.all_verses[idx]
            key = (verse['surah_num'], verse['ayah_num'])
            if key in seen:
                continue

            seen.add(key)
            # Match response format expected by quran-matcher client (adjust if needed)
            match_data = {
                'surah_number': verse['surah_num'], # Keep as int for internal processing
                'surah_name': verse['surah_name'],
                'surah_name_en': verse.get('surah_name_en', ''), # Add English name if available
                'ayah_number': verse['ayah_num'], # Keep as int for internal processing
                'ayah_text': verse['original'], # Renamed from 'original'
                'confidence_score': float(score), # Ensure float
                # Add placeholder for unicode - will be added in endpoint
                'unicode': ''
            }
            matches.append(match_data)
            # Add normalized match info for debug
            normalized_matches_debug.append({
                'normalized_text': choice, # The matched normalized verse text
                'score': score,
                'surah': verse['surah_num'],
                'ayah': verse['ayah_num']
            })

            logger.info(f"[QuranMatcher] Match found: S{key[0]}:A{key[1]} Score={score:.2f}")

            if len(matches) >= top_n:
                logger.info(f"[QuranMatcher] Reached top_n limit ({top_n}).")
                break
        logger.info(f"[QuranMatcher] Found {len(matches)} final matches.")
        # Return the matches, the normalized input transcription, and normalized match details
        return matches, clean, normalized_matches_debug

# Instantiate the matcher globally, passing the loaded data
try:
    # Ensure raw_quran_data was loaded successfully before passing it
    if raw_quran_data:
        quran_matcher_instance = QuranMatcher(loaded_quran_data=raw_quran_data)
    else:
        logger.error("CRITICAL: Global raw_quran_data is empty. Cannot initialize QuranMatcher.")
        quran_matcher_instance = None
except Exception as init_err:
    logger.error(f"CRITICAL: Failed to initialize QuranMatcher: {init_err}. /getAyah endpoint will likely fail.")
    quran_matcher_instance = None # Set to None to indicate failure

# --- Quran Matcher Logic End ---

@app.route('/getReciter', methods=['POST'])
def identify_reciter():
    """Identify a Quran reciter from an audio recording."""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided. Please send the file with key "audio" in form data.'
            }), 400

        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({
                'error': 'No audio file selected'
            }), 400

        show_unreliable = request.form.get('show_unreliable_predictions', '').lower() == 'true'
        
        result = process_audio_file(audio_file, for_ayah=False)
        if isinstance(result[1], str):
            return jsonify({'error': result[1]}), 400

        audio_data, sample_rate = result
        
        model = get_reciter_model()
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        try:
            features = extract_features(audio_data, sample_rate)
            if features is None:
                return jsonify({'error': 'Feature extraction failed'}), 500
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return jsonify({
                'error': f'Error extracting features: {str(e)}'
            }), 500

        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            distances = calculate_distances(features[0], model.centroids)
            reliability = analyze_prediction_reliability(
                probabilities, distances, model.thresholds, prediction)

            sorted_indices = np.argsort(probabilities)[::-1][:TOP_N_PREDICTIONS]
            predictions = []
            
            reciters_data = {}
            try:
                reciters_file = Path(__file__).resolve().parent.parent / 'data' / 'reciters.json'
                if reciters_file.exists():
                    with open(reciters_file, 'r', encoding='utf-8') as f:
                        reciters_data = json.load(f)
                else:
                    logger.warning(f"Reciters metadata file not found at: {reciters_file}")
            except Exception as e:
                logger.warning(f"Could not load reciters metadata: {str(e)}")
            
            for idx in sorted_indices:
                reciter_name = str(model.classes_[idx])
                confidence = float(probabilities[idx])
                
                reciter_info = reciters_data.get(reciter_name, {})
                servers = reciter_info.get('servers', [])
                server_url = servers[0] if isinstance(servers, list) and servers else ''
                
                predictions.append({
                    'name': reciter_name,
                    'confidence': confidence * 100,
                    'nationality': reciter_info.get('nationality', ''),
                    'serverUrl': server_url,
                    'flagUrl': reciter_info.get('flagUrl', ''),
                    'imageUrl': reciter_info.get('imageUrl', '')
                })

            if not reliability['is_reliable'] and not show_unreliable:
                response = {
                    'reliable': False,
                    'main_prediction': None,
                    'top_predictions': []
                }
            else:
                response = {
                    'reliable': bool(reliability['is_reliable']),
                    'main_prediction': predictions[0] if predictions else None,
                    'top_predictions': predictions
                }

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({
                'error': f'Error making prediction: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/getAllReciters', methods=['GET'])
def get_all_reciters():
    """Get a list of all available reciters."""
    try:
        try:
            reciters_file = Path(__file__).resolve().parent.parent / 'data' / 'reciters.json'
            if not reciters_file.exists():
                return jsonify({'error': 'Reciters data file not found.'}), 500
                
            with open(reciters_file, 'r', encoding='utf-8') as f:
                reciters_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading reciters data: {str(e)}")
            return jsonify({'error': 'Error loading reciters data.'}), 500

        reciters_list = []
        for name, data in reciters_data.items():
            servers = data.get('servers', [])
            server_url = servers[0] if isinstance(servers, list) and servers else ''
            if isinstance(servers, str):
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
        logger.error(f"Server error in getAllReciters: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/getAyah', methods=['POST'])
def get_ayah():
    """
    Identify a Quranic verse from an audio recording using the Tarteel Whisper model
    and fuzzy matching logic. (Replaces old /getAyah)
    """
    temp_dir = None # Track temporary directory for cleanup
    temp_input_path = None # Track temporary input file path
    wav_path = None # Track converted WAV file path

    if quran_matcher_instance is None:
        logger.error("QuranMatcher instance not available for /getAyah")
        return jsonify({'error': 'Quran Matcher service is not initialized.'}), 503 # Service Unavailable

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided. Key must be "audio".'}), 400

        audio_file = request.files['audio']
        if not audio_file or not audio_file.filename:
            return jsonify({'error': 'Invalid or empty audio file provided.'}), 400

        # Create a temporary directory for this request
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temp directory for request: {temp_dir}")

        # Save uploaded file temporarily
        original_filename = audio_file.filename
        safe_filename = "".join(c for c in original_filename if c.isalnum() or c in ('.', '_')).rstrip()
        if not safe_filename: safe_filename = "uploaded_audio"
        temp_input_path = os.path.join(temp_dir, safe_filename)

        logger.info(f"Saving uploaded audio to temporary path: {temp_input_path}")
        audio_file.save(temp_input_path)

        # Convert audio
        logger.info(f"Converting {temp_input_path} to WAV format...")
        wav_path = AudioProcessor.convert_to_wav(temp_input_path, output_dir=temp_dir)
        logger.info(f"Converted WAV file path: {wav_path}")

        # Transcribe
        logger.info("Starting transcription...")
        transcription = quran_matcher_instance.transcribe_audio(wav_path)
        logger.info(f"Transcription result: '{transcription}'")

        # Find matches
        top_n = int(request.form.get('top_n', 5))
        score_threshold = int(request.form.get('score_threshold', 30))
        logger.info(f"Finding matches (top_n={top_n}, threshold={score_threshold})...")
        # Get normalized text and debug matches back from find_matches
        raw_matches, normalized_transcription, normalized_matches_debug = quran_matcher_instance.find_matches(
            transcription, top_n=top_n, score_threshold=score_threshold
        )

        # --- Format response like /getAyah --- 
        formatted_matches = []
        for match in raw_matches: # raw_matches contains results from QuranMatcher
            surah_num = int(match['surah_number'])
            ayah_num = int(match['ayah_number'])
            confidence_score = match['confidence_score'] # Get score from matcher result

            # Look up all descriptive data from the global raw_quran_data
            surah_name = "Unknown Surah"
            surah_name_en = "Unknown Surah"
            ayah_text = "Unknown Ayah Text"
            unicode_char = ""
            
            # Find the Surah in the globally loaded data
            surah_data = next((s for s in raw_quran_data if s.get('id') == surah_num), None)
            
            if surah_data:
                surah_name = surah_data.get('name', surah_name) # Get Arabic name
                # Use 'transliteration' for the English name, fallback to 'translation' or default
                surah_name_en = surah_data.get('transliteration', surah_data.get('translation', surah_name_en))
                unicode_char = surah_data.get('unicode', unicode_char) # Get Unicode
                
                # Find the specific Ayah within the Surah
                ayah_data = next((v for v in surah_data.get('verses', []) if v.get('id') == ayah_num), None)
                if ayah_data:
                    ayah_text = ayah_data.get('text', ayah_text) # Get Ayah text
            else:
                 logger.warning(f"Could not find Surah {surah_num} in global raw_quran_data for formatting.")

            # Construct the match using looked-up data
            formatted_match = {
                'surah_number': to_arabic_number(surah_num),
                'surah_number_en': surah_num,
                'surah_name': surah_name,
                'surah_name_en': surah_name_en,
                'ayah_number': to_arabic_number(ayah_num),
                'ayah_number_en': ayah_num,
                'ayah_text': ayah_text,
                'confidence_score': confidence_score, # Use score from matcher
                'unicode': unicode_char
            }
            formatted_matches.append(formatted_match)

        logger.info(f"Match results: {len(formatted_matches)} matches found.")

        # Construct final response object
        response_data = {
            'matches_found': len(formatted_matches) > 0,
            'total_matches': len(formatted_matches),
            'matches': formatted_matches,
            'best_match': formatted_matches[0] if formatted_matches else None
        }

        # Add debug info if requested
        if app.debug:
            debug_response = {
                'transcription': transcription, # Raw transcription
                'debug_info': {
                    'transcription': transcription,
                    'normalized_transcription': normalized_transcription,
                    'normalized_matches': normalized_matches_debug # Use the detailed list from find_matches
                }
            }
            response_data.update(debug_response)

        return jsonify(response_data)

    except FileNotFoundError as e:
         logger.error(f"File not found error during processing: {e}")
         return jsonify({'error': f'File operation error: {e}. Check file paths and permissions.'}), 500
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg execution failed: {e.stderr}")
        return jsonify({'error': f'Audio processing failed (ffmpeg error). Details: {e.stderr}'}), 500
    except RuntimeError as e:
        logger.error(f"Runtime error in get_ayah: {e}")
        # Provide specific messages for known issues
        if "Audio conversion failed" in str(e):
             return jsonify({'error': 'Failed to convert audio. Ensure ffmpeg is installed and the audio format is supported.'}), 400
        elif "Audio transcription failed" in str(e):
             return jsonify({'error': 'Failed to transcribe audio.'}), 500
        elif "Quran data unavailable" in str(e):
             return jsonify({'error': 'Quran text data is currently unavailable for matching.'}), 503
        else:
             return jsonify({'error': f'An internal processing error occurred: {e}'}), 500
    except Exception as e:
        logger.exception(f"Unexpected error in /getAyah endpoint: {e}") # Log full traceback for unexpected errors
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500

    finally:
        # Cleanup temporary files and directory
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
                logger.info(f"Deleted temporary WAV file: {wav_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary WAV file {wav_path}: {e}")
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
                logger.info(f"Deleted temporary input file: {temp_input_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary input file {temp_input_path}: {e}")
        if temp_dir and os.path.exists(temp_dir):
             try:
                 # Check if dir is empty before removing (optional, rmdir fails otherwise)
                 if not os.listdir(temp_dir):
                     os.rmdir(temp_dir)
                     logger.info(f"Deleted empty temporary directory: {temp_dir}")
                 else:
                     # If files failed to delete, attempt recursive delete (use with caution)
                     import shutil
                     shutil.rmtree(temp_dir)
                     logger.info(f"Recursively deleted temporary directory and contents: {temp_dir}")

             except Exception as e:
                 logger.warning(f"Failed to delete temporary directory {temp_dir}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Quran Reciter API server.")
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode.')
    args = parser.parse_args()

    app.run(host=HOST, port=PORT, debug=args.debug)
