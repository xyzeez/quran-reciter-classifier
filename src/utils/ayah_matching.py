"""
Common ayah matching utilities for the Quran Reciter Classifier system.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for comparison.
    
    This function:
    1. Removes all diacritics (tashkeel)
    2. Normalizes various forms of alef
    3. Normalizes hamza forms
    4. Removes tatweel (stretching character)
    5. Normalizes teh marbuta to heh
    """
    if not text:
        return ""
    
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
    
    return text

def load_quran_data(data_dir: Optional[Path] = None) -> Dict:
    """Load and prepare Quran data for matching.
    
    Args:
        data_dir: Directory containing quran.json, defaults to looking in common locations
    
    Returns:
        Dict containing normalized Quran data
    """
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

        with open(quran_file, 'r', encoding='utf-8') as f:
            surahs = json.load(f)

        # Convert the data into a list of verses with all required information
        verses = []
        for surah in surahs:
            surah_number = surah['id']
            surah_name = surah['name']
            surah_name_en = surah['transliteration']
            
            for verse in surah['verses']:
                verses.append({
                    'surah_number': surah_number,
                    'ayah_number': verse['id'],
                    'surah_name': surah_name,
                    'surah_name_en': surah_name_en,
                    'text': verse['text'],
                    'normalized_text': normalize_arabic_text(verse['text'])
                })

        return {'verses': verses}

    except Exception as e:
        logger.error(f"Error loading Quran data: {str(e)}")
        return None

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using a combination of methods.
    
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

def find_matching_ayah(transcription: str, min_confidence: float = 0.70, max_matches: int = 5) -> Dict:
    """Find matching Quranic verses for the transcribed text.
    
    Args:
        transcription: Transcribed Arabic text
        min_confidence: Minimum confidence threshold for matches
        max_matches: Maximum number of matches to return
    
    Returns:
        Dict containing matches and best match
    """
    try:
        # Normalize input text
        normalized_input = normalize_arabic_text(transcription)

        # Load Quran data if not already loaded
        quran_data = load_quran_data()
        if quran_data is None:
            raise Exception("Failed to load Quran data")

        # Find matches
        matches = []
        for verse in quran_data['verses']:
            # Calculate similarity score
            score = calculate_similarity(normalized_input, verse['normalized_text'])
            if score > 0:
                match = {
                    'surah_number': verse['surah_number'],
                    'ayah_number': verse['ayah_number'],
                    'surah_name': verse['surah_name'],
                    'surah_name_en': verse['surah_name_en'],
                    'ayah_text': verse['text'],
                    'confidence_score': score
                }
                matches.append(match)

        # Sort matches by confidence
        matches.sort(key=lambda x: x['confidence_score'], reverse=True)
        matches = matches[:max_matches]

        # Get best match
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