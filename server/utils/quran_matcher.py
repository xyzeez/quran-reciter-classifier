"""
Handles Quran verse identification using Whisper transcription and fuzzy matching.
"""
import logging
import json
import re
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from rapidfuzz import process, fuzz
from pathlib import Path

# Import the utility for converting numbers to Arabic script
# This assumes text_utils.py is in server/utils/
from .text_utils import to_arabic_number

logger = logging.getLogger(__name__)

class QuranMatcher:
    """Identifies Quran verses in audio using Whisper and fuzzy matching."""

    def __init__(self, loaded_quran_data: list):
        """Initializes the matcher with a Whisper model and pre-loaded Quran data.

        Args:
            loaded_quran_data: A list of Surah dictionaries, matching the 
                               structure expected (e.g., from quran.json).
        """
        self.model_id = "tarteel-ai/whisper-base-ar-quran"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[QuranMatcher] Initializing model {self.model_id} on {self.device.upper()} device")

        # Load Whisper model
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        except Exception as e:
            logger.error(f"[QuranMatcher] Failed to load Whisper model {self.model_id}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load Whisper model {self.model_id}") from e

        # Process the passed-in data
        logger.info("[QuranMatcher] Processing pre-loaded Quran data...")
        self.quran_data = loaded_quran_data # Keep reference if needed later?

        try:
            if not self.quran_data or not isinstance(self.quran_data, list):
                 logger.error("[QuranMatcher] Provided Quran data is empty, None, or not a list.")
                 raise ValueError("Invalid or empty Quran data provided.")
                 
            self.all_verses = self._prepare_verse_database(self.quran_data)
            if not self.all_verses:
                logger.error("[QuranMatcher] Verse database preparation resulted in an empty list.")
                raise ValueError("Verse database is empty after preparation.")

            self.normalized_verses = [v["normalized"] for v in self.all_verses]
            logger.info(f"[QuranMatcher] Prepared {len(self.all_verses)} verses internal database.")
        except Exception as e:
            logger.error(f"[QuranMatcher] Failed during internal data preparation: {e}", exc_info=True)
            self.quran_data = []
            self.all_verses = []
            self.normalized_verses = []
            raise RuntimeError("Failed to prepare internal verse database") from e

    def _prepare_verse_database(self, quran_data_list: list) -> list:
        """Flattens and normalizes verses from the loaded Quran data list."""
        verses = []
        # Data validation happens in __init__ before calling this

        for chapter in quran_data_list:
            if not isinstance(chapter, dict):
                logger.warning(f"[QuranMatcher] Skipping invalid chapter data item (expected dict): {type(chapter)}")
                continue
                
            surah_num = chapter.get("id")
            surah_name = chapter.get("name")
            surah_name_en = chapter.get('transliteration', chapter.get('translation', ''))
            
            if surah_num is None or surah_name is None:
                logger.warning(f"[QuranMatcher] Skipping chapter due to missing id or name: {chapter}")
                continue

            for verse in chapter.get("verses", []):
                 if not isinstance(verse, dict):
                    logger.warning(f"[QuranMatcher] Skipping invalid verse data item in Surah {surah_num} (expected dict): {type(verse)}")
                    continue
                    
                 ayah_num = verse.get("id")
                 orig_text = verse.get("text")
                 
                 if ayah_num is None or orig_text is None:
                     logger.warning(f"[QuranMatcher] Skipping verse in Surah {surah_num} due to missing id or text: {verse}")
                     continue
                     
                 verses.append({
                    "surah_num": surah_num,
                    "surah_name": surah_name,
                    "surah_name_en": surah_name_en,
                    "ayah_num": ayah_num,
                    "original": orig_text,
                    "normalized": self._normalize_text(orig_text)
                })
        return verses

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize Arabic text for robust matching"""
        if not isinstance(text, str):
            return "" # Return empty string if input is not a string
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
            'ة': 'ه',
        }
        for old, new in variants.items():
            text = text.replace(old, new)

        # collapse whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper model"""
        if not hasattr(self, 'model') or not hasattr(self, 'processor'):
             raise RuntimeError("QuranMatcher model/processor not properly initialized. Cannot transcribe.")
        try:
            logger.info(f"[QuranMatcher] Loading audio file: {audio_path}")
            wav, sr = torchaudio.load(audio_path)
            # Ensure mono
            if wav.shape[0] > 1:
                logger.info("[QuranMatcher] Converting stereo audio to mono.")
                wav = wav.mean(dim=0, keepdim=True)

            # Check sample rate (should be 16k after conversion, but double-check)
            if sr != 16000:
                 logger.warning(f"[QuranMatcher] Audio sample rate is {sr}Hz, expected 16000Hz. Resampling...")
                 resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                 wav = resampler(wav)
                 sr = 16000

            logger.info("[QuranMatcher] Extracting features...")
            inputs = self.processor(wav.squeeze(0), sampling_rate=sr, return_tensors='pt')
            input_feats = inputs.input_features.to(self.device)

            logger.info("[QuranMatcher] Generating transcription...")
            # Use forced_decoder_ids for language/task specification
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ar", task="transcribe")
            pred_ids = self.model.generate(input_feats, forced_decoder_ids=forced_decoder_ids)
            transcription = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            logger.info(f"[QuranMatcher] Raw transcription: {transcription}")
            return transcription
        except Exception as e:
            logger.error(f"[QuranMatcher] Transcription failed for {audio_path}: {e}", exc_info=True)
            raise RuntimeError("Audio transcription failed") from e

    def find_matches(self, transcription: str, top_n: int = 5, score_threshold: int = 30) -> tuple[list, str, list]:
        """Find top matching Quran verses via fuzzy matching. Returns (matches, normalized_input, normalized_matches_debug)."""
        if not self.all_verses or not self.normalized_verses:
             logger.error("[QuranMatcher] Cannot find matches: Verse database is empty or not prepared.")
             return [], "", [] # Return empty list and empty normalized text/debug list
             
        clean = self._normalize_text(transcription)
        if not clean:
            logger.warning("[QuranMatcher] Normalized transcription is empty. Cannot perform matching.")
            return [], clean, [] # Return empty list but the (empty) normalized text
        logger.info(f"[QuranMatcher] Normalized transcription for matching: '{clean}'")

        # get fuzzy scores against precomputed normalized verses
        try:
            logger.info(f"[QuranMatcher] Running fuzzy matching (top_n={top_n}, threshold={score_threshold})...")
            # Increase limit slightly more to allow filtering duplicates more effectively
            results = process.extract(clean, self.normalized_verses,
                                      scorer=fuzz.token_set_ratio,
                                      limit=top_n * 5) 
        except Exception as e:
            logger.error(f"[QuranMatcher] Fuzzy matching failed: {e}", exc_info=True)
            return [], clean, [] # Return empty list but the normalized text

        # sort by score descending
        sorted_res = sorted(results, key=lambda x: x[1], reverse=True)

        matches = []
        seen = set()
        normalized_matches_debug = []

        logger.info(f"[QuranMatcher] Processing {len(sorted_res)} potential fuzzy matches...")
        for choice, score, idx in sorted_res:
            # Safety check
            if idx >= len(self.all_verses):
                logger.warning(f"[QuranMatcher] Fuzzy match index {idx} out of bounds for all_verses (len={len(self.all_verses)}). Skipping.")
                continue
                
            # Threshold check
            if score < score_threshold:
                 logger.info(f"[QuranMatcher] Score {score:.2f} below threshold {score_threshold}. Stopping match processing.")
                 break 

            verse = self.all_verses[idx]
            key = (verse['surah_num'], verse['ayah_num'])
            
            # Skip duplicates
            if key in seen:
                continue

            # Stop if desired number of unique matches found
            if len(matches) >= top_n:
                logger.info(f"[QuranMatcher] Reached top_n limit ({top_n}) for unique matches.")
                break
                
            seen.add(key)
            # Prepare internal representation for the service layer
            match_data = {
                'surah_number': verse['surah_num'],
                'surah_name': verse['surah_name'],
                'surah_name_en': verse['surah_name_en'],
                'ayah_number': verse['ayah_num'], 
                'ayah_text': verse['original'], 
                'confidence_score': float(score), 
                'unicode': '' # Placeholder added by service
            }
            matches.append(match_data)
            
            # Add debug info
            normalized_matches_debug.append({
                'normalized_text': choice, 
                'score': score,
                'surah': verse['surah_num'],
                'ayah': verse['ayah_num']
            })
            logger.debug(f"[QuranMatcher] Match added: S{key[0]}:A{key[1]} ({verse['surah_name_en']}) Score={score:.2f}") # Debug level

        logger.info(f"[QuranMatcher] Found {len(matches)} final unique matches meeting threshold.")
        return matches, clean, normalized_matches_debug 