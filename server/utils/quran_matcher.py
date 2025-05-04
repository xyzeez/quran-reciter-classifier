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
# Assuming text_utils.py is in the same directory or accessible via PYTHONPATH
# If text_utils is in server.utils, use: from .text_utils import to_arabic_number
# If text_utils is in src.utils, use: from src.utils.text_utils import to_arabic_number 
# Adjust based on actual location. Using relative import for now.
from server.utils.text_utils import to_arabic_number

logger = logging.getLogger(__name__)

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
            logger.error(f"[QuranMatcher] Failed to load Whisper model: {e}", exc_info=True)
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
            if not self.all_verses:
                logger.error("[QuranMatcher] Verse database preparation resulted in an empty list.")
                raise ValueError("Verse database is empty after preparation.")

            # cache normalized strings for matching
            self.normalized_verses = [v["normalized"] for v in self.all_verses]
            logger.info(f"[QuranMatcher] Prepared {len(self.all_verses)} verses from pre-loaded data.")
        except Exception as e:
            logger.error(f"[QuranMatcher] Failed to process pre-loaded Quran data: {e}", exc_info=True)
            self.quran_data = []
            self.all_verses = []
            self.normalized_verses = []
            raise RuntimeError("Failed to process pre-loaded Quran data") from e

    def _prepare_verse_database(self):
        """Flatten and normalize all verses from the loaded data"""
        verses = []
        if not self.quran_data:
            logger.error("[QuranMatcher] Cannot prepare verse database: Quran data is empty.")
            return []
            
        # Check if quran_data is a list of dictionaries (expected format)
        if not isinstance(self.quran_data, list):
             logger.error(f"[QuranMatcher] Expected quran_data to be a list, but got {type(self.quran_data)}")
             return []

        for chapter in self.quran_data:
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
                    "normalized": self._normalize_text(orig_text) # Pass orig_text here
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
            # Ensure idx is within bounds (safety check)
            if idx >= len(self.all_verses):
                logger.warning(f"[QuranMatcher] Fuzzy match index {idx} out of bounds for all_verses (len={len(self.all_verses)}). Skipping.")
                continue
                
            # Apply threshold check early
            if score < score_threshold:
                 # Since list is sorted, we can break early
                 logger.info(f"[QuranMatcher] Score {score:.2f} below threshold {score_threshold}. Stopping match processing.")
                 break 

            verse = self.all_verses[idx]
            key = (verse['surah_num'], verse['ayah_num'])
            
            # Skip if we've already added this exact verse
            if key in seen:
                continue

            # Check if we have reached the desired number of unique matches
            if len(matches) >= top_n:
                logger.info(f"[QuranMatcher] Reached top_n limit ({top_n}) for unique matches.")
                break
                
            seen.add(key)
            # Prepare data structure for the endpoint (internal representation)
            match_data = {
                'surah_number': verse['surah_num'],
                'surah_name': verse['surah_name'], # Use name stored during _prepare_verse_database
                'surah_name_en': verse['surah_name_en'], # Use name stored during _prepare_verse_database
                'ayah_number': verse['ayah_num'], 
                'ayah_text': verse['original'], # Use original text stored during _prepare_verse_database
                'confidence_score': float(score), 
                'unicode': '' # Placeholder - will be added by endpoint using global data
            }
            matches.append(match_data)
            
            # Add normalized match info for debug
            normalized_matches_debug.append({
                'normalized_text': choice, # The matched normalized verse text from rapidfuzz
                'score': score,
                'surah': verse['surah_num'],
                'ayah': verse['ayah_num']
            })
            logger.info(f"[QuranMatcher] Match added: S{key[0]}:A{key[1]} ({verse['surah_name_en']}) Score={score:.2f}")

        logger.info(f"[QuranMatcher] Found {len(matches)} final unique matches meeting threshold.")
        return matches, clean, normalized_matches_debug 