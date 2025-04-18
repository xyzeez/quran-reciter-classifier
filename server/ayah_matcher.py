"""
Utility module for matching transcribed text to Quran verses.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Initialize cache for Quran data
quran_data = None
normalized_verses = None

def normalize_arabic_text(text: str, verbose: bool = False) -> str:
    """
    Normalize Arabic text by removing diacritics and standardizing characters.
    
    Args:
        text (str): Arabic text to normalize
        verbose (bool): Whether to log normalization details
    
    Returns:
        str: Normalized text
    """
    if verbose:
        logger.info(f"Normalizing text: {text}")
    
    # First, remove all diacritical marks and special characters
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', text)
    
    # Remove small high characters and other special marks
    text = re.sub(r'[\u0615-\u061A\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', text)
    
    # Remove superscript alef
    text = re.sub(r'\u0670', '', text)
    
    # Remove tatweel (elongation character)
    text = re.sub(r'\u0640', '', text)
    
    # Standardize alef variants
    text = re.sub(r'[إأآاٱ]', 'ا', text)
    
    # Standardize hamza variants
    text = re.sub(r'[ؤئءٔ]', 'ء', text)
    
    # Standardize yah variants
    text = re.sub(r'[يىئ]', 'ي', text)
    
    # Standardize tah marbuta
    text = re.sub(r'ة', 'ه', text)
    
    # Keep only basic Arabic letters and spaces
    text = re.sub(r'[^\u0621-\u063A\u0641-\u064A\s]', '', text)
    
    # Remove extra spaces and trim
    result = ' '.join(text.split()).strip()
    
    if verbose:
        logger.info(f"Normalized result: {result}")
    return result

def get_ngrams(text: str, n: int = 3) -> List[str]:
    """Get n-grams from text."""
    return [text[i:i+n] for i in range(len(text)-n+1)]

def calculate_sequence_similarity(text1: str, text2: str) -> float:
    """Calculate character sequence similarity."""
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_word_similarity(text1: str, text2: str) -> float:
    """Calculate word-level similarity."""
    words1 = set(text1.split())
    words2 = set(text2.split())
    common_words = words1.intersection(words2)
    return len(common_words) / max(len(words1), len(words2))

def calculate_ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """Calculate n-gram similarity."""
    ngrams1 = set(get_ngrams(text1, n))
    ngrams2 = set(get_ngrams(text2, n))
    common_ngrams = ngrams1.intersection(ngrams2)
    return len(common_ngrams) / max(len(ngrams1), len(ngrams2)) if max(len(ngrams1), len(ngrams2)) > 0 else 0

def calculate_confidence(transcribed: str, verse: str) -> float:
    """
    Calculate confidence score using multiple similarity metrics.
    
    Args:
        transcribed (str): Normalized transcribed text
        verse (str): Normalized verse text
    
    Returns:
        float: Combined confidence score between 0 and 1
    """
    # Calculate individual scores
    sequence_score = calculate_sequence_similarity(transcribed, verse)
    word_score = calculate_word_similarity(transcribed, verse)
    ngram_score = calculate_ngram_similarity(transcribed, verse)
    
    # Adjust weights to favor word-level matching more heavily
    weights = [0.2, 0.6, 0.2]  # Sequence (0.2), Word (0.6), N-gram (0.2)
    
    logger.debug(f"Similarity scores for comparison:")
    logger.debug(f"Text 1: {transcribed}")
    logger.debug(f"Text 2: {verse}")
    logger.debug(f"Sequence score: {sequence_score:.4f}")
    logger.debug(f"Word score: {word_score:.4f}")
    logger.debug(f"Ngram score: {ngram_score:.4f}")
    
    final_score = (
        sequence_score * weights[0] + 
        word_score * weights[1] + 
        ngram_score * weights[2]
    )
    
    logger.debug(f"Final confidence score: {final_score:.4f}")
    return final_score

def load_quran_data() -> None:
    """
    Load and cache Quran data from JSON file.
    """
    global quran_data, normalized_verses
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Try different possible locations for quran.json
        possible_paths = [
            project_root / 'data' / 'quran.json',
            project_root / 'quran-reciter-classifier' / 'data' / 'quran.json',
            Path.cwd() / 'data' / 'quran.json'
        ]

        quran_file = None
        for path in possible_paths:
            if path.exists():
                quran_file = path
                break

        if quran_file is None:
            raise FileNotFoundError(
                f"Could not find quran.json in any of these locations: {[str(p) for p in possible_paths]}"
            )
        
        logger.info(f"Loading Quran data from: {quran_file}")
        with open(quran_file, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        
        # Pre-compute normalized verses for faster matching
        normalized_verses = []
        verse_count = 0
        for surah in quran_data:
            for verse in surah['verses']:
                normalized_verses.append({
                    'surah_number': surah['id'],
                    'surah_name': surah['name'],
                    'surah_name_en': surah['transliteration'],
                    'ayah_number': verse['id'],
                    'ayah_text': verse['text'],
                    'normalized_text': normalize_arabic_text(verse['text'], verbose=False)
                })
                verse_count += 1
        
        logger.info(f"Loaded and normalized {verse_count} verses successfully")
    except Exception as e:
        logger.error(f"Error loading Quran data: {str(e)}")
        raise

def find_matching_ayah(transcribed_text: str, min_confidence: float = 0.60, max_matches: int = 5) -> Dict:
    """
    Find matching Quran verses for the transcribed text.
    
    Args:
        transcribed_text (str): Transcribed Arabic text
        min_confidence (float): Minimum confidence score to consider a match valid for best_match
        max_matches (int): Maximum number of matches to return in the matches list

    Returns:
        Dict: Contains:
            - matches: Top N matches sorted by confidence (regardless of threshold)
            - best_match: Highest confidence match that meets the threshold
            - total_matches: Number of matches found
            - debug_info: Debug information if enabled
    """
    try:
        # Ensure Quran data is loaded
        if quran_data is None or normalized_verses is None:
            load_quran_data()
        
        # Normalize transcribed text
        normalized_transcription = normalize_arabic_text(transcribed_text, verbose=True)
        
        # Find matches using combined confidence scoring
        all_matches = []  # Store all matches with their scores
        
        for verse in normalized_verses:
            # Calculate confidence score
            score = calculate_confidence(
                normalized_transcription,
                verse['normalized_text']
            )
            
            # Create match info
            match_info = {
                'surah_number': verse['surah_number'],
                'surah_name': verse['surah_name'],
                'surah_name_en': verse['surah_name_en'],
                'ayah_number': verse['ayah_number'],
                'ayah_text': verse['ayah_text'],
                'confidence_score': round(score, 4),
                'normalized_text': verse['normalized_text']  # Store normalized text for debug info
            }
            all_matches.append(match_info)
        
        # Sort all matches by confidence score in descending order
        all_matches.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Get the best match that meets the threshold
        best_match = all_matches[0] if all_matches and all_matches[0]['confidence_score'] >= min_confidence else None
        
        # Get top N matches regardless of confidence
        top_matches = all_matches[:max_matches]
        
        # Log the top matches for debugging
        logger.info("Top matches (including those below threshold):")
        for match in top_matches:
            logger.info(f"Score: {match['confidence_score']:.4f} - Verse: {match['ayah_text']}")
        
        # Prepare response
        result = {
            'matches': [{k: v for k, v in m.items() if k != 'normalized_text'} for m in top_matches],  # Remove normalized_text from main matches
            'best_match': {k: v for k, v in top_matches[0].items() if k != 'normalized_text'} if best_match else None,
            'total_matches': len(top_matches),
            'debug_info': {
                'normalized_transcription': normalized_transcription,
                'normalized_matches': [
                    {
                        'transcription': normalized_transcription,
                        'verse': match['normalized_text'],
                        'score': match['confidence_score']
                    }
                    for match in top_matches
                ]
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error matching ayah: {str(e)}")
        raise 