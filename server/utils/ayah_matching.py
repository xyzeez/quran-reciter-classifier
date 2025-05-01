"""
Ayah matching utilities for identifying Quranic verses.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Global cache for Quran data
_quran_data = None
_normalized_verses = None

def normalize_arabic_text(text: str, verbose: bool = False) -> str:
    """
    Normalize Arabic text for comparison.
    
    Args:
        text: Arabic text to normalize
        verbose: Whether to log normalization steps
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
        
    if verbose:
        logger.debug(f"Original text: {text}")
    
    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize alef variations
    text = re.sub(r'[إأٱآا]', 'ا', text)
    
    # Normalize hamza
    text = re.sub(r'[ؤئ]', 'ء', text)
    
    # Remove tatweel (stretching character)
    text = re.sub(r'ـ', '', text)
    
    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')
    
    # Remove any non-Arabic characters
    text = re.sub(r'[^\u0621-\u063A\u0641-\u064A\s]', '', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    if verbose:
        logger.debug(f"Normalized text: {text}")
    
    return text

def load_quran_data(data_dir: Optional[Path] = None) -> Dict:
    """
    Load and prepare Quran data for matching.
    
    Args:
        data_dir: Directory containing quran.json
        
    Returns:
        Dict containing normalized Quran data
    """
    global _quran_data, _normalized_verses
    
    # Return cached data if available
    if _quran_data is not None and _normalized_verses is not None:
        return {
            'verses': _normalized_verses,
            'raw_data': _quran_data
        }
    
    try:
        # Find quran.json
        if data_dir is None:
            possible_paths = [
                Path.cwd() / 'data' / 'quran.json',
                Path.cwd().parent / 'data' / 'quran.json',
                Path(__file__).parent.parent.parent / 'data' / 'quran.json'
            ]
            quran_file = next((p for p in possible_paths if p.exists()), None)
        else:
            quran_file = data_dir / 'quran.json'

        if quran_file is None or not quran_file.exists():
            raise FileNotFoundError(
                f"Could not find quran.json in expected locations: {[str(p) for p in possible_paths]}"
            )

        logger.info(f"Loading Quran data from: {quran_file}")
        with open(quran_file, 'r', encoding='utf-8') as f:
            _quran_data = json.load(f)

        # Convert the data into a list of verses with all required information
        _normalized_verses = []
        for surah in _quran_data:
            surah_number = surah['id']
            surah_name = surah['name']
            surah_name_en = surah['transliteration']
            
            for verse in surah['verses']:
                _normalized_verses.append({
                    'surah_number': surah_number,
                    'ayah_number': verse['id'],
                    'surah_name': surah_name,
                    'surah_name_en': surah_name_en,
                    'ayah_text': verse['text'],
                    'normalized_text': normalize_arabic_text(verse['text'])
                })

        logger.info(f"Loaded and normalized {len(_normalized_verses)} verses successfully")
        return {
            'verses': _normalized_verses,
            'raw_data': _quran_data
        }

    except Exception as e:
        logger.error(f"Error loading Quran data: {str(e)}")
        return None

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using a combination of methods.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Get word sets
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # Calculate word overlap
    common_words = words1.intersection(words2)
    word_similarity = len(common_words) / max(len(words1), len(words2))
    
    # Calculate sequence similarity
    sequence_similarity = SequenceMatcher(None, text1, text2).ratio()
    
    # Combine scores (weighted average)
    return 0.7 * word_similarity + 0.3 * sequence_similarity

def find_matching_ayah(
    transcribed_text: str, 
    min_confidence: float = 0.70, 
    max_matches: int = 5
) -> Dict:
    """
    Find matching Quran verses for the transcribed text.
    
    Args:
        transcribed_text: Transcribed Arabic text
        min_confidence: Minimum confidence threshold
        max_matches: Maximum number of matches to return
        
    Returns:
        Dict containing matches and best match information
    """
    try:
        # Ensure Quran data is loaded
        quran_data = load_quran_data()
        if quran_data is None:
            raise Exception("Failed to load Quran data")
            
        # Get normalized verses
        verses = quran_data['verses']
        
        # Normalize transcribed text
        normalized_transcription = normalize_arabic_text(transcribed_text)
        
        # Find matches using similarity scoring
        matches = []
        for verse in verses:
            # Calculate similarity score
            score = calculate_similarity(
                normalized_transcription,
                verse['normalized_text']
            )
            
            if score > 0:
                match = {
                    'surah_number': verse['surah_number'],
                    'ayah_number': verse['ayah_number'],
                    'surah_name': verse['surah_name'],
                    'surah_name_en': verse['surah_name_en'],
                    'ayah_text': verse['ayah_text'],
                    'confidence_score': score
                }
                matches.append(match)

        # Sort matches by confidence
        matches.sort(key=lambda x: x['confidence_score'], reverse=True)
        matches = matches[:max_matches]

        # Get best match above threshold
        best_match = matches[0] if matches and matches[0]['confidence_score'] >= min_confidence else None

        return {
            'matches': matches,
            'best_match': best_match,
            'total_matches': len(matches)
        }

    except Exception as e:
        logger.error(f"Error finding matching ayah: {str(e)}")
        return {
            'matches': [],
            'best_match': None,
            'total_matches': 0,
            'error': str(e)
        } 