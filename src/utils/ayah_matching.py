"""
Common ayah matching utilities for the Quran Reciter Classifier system.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import re

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
        Dict containing normalized Quran data and raw data
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
            unicode_char = surah['unicode']
            
            for verse in surah['verses']:
                verses.append({
                    'surah_number': surah_number,
                    'ayah_number': verse['id'],
                    'surah_name': surah_name,
                    'surah_name_en': surah_name_en,
                    'text': verse['text'],
                    'normalized_text': normalize_arabic_text(verse['text']),
                    'unicode': unicode_char
                })

        return {
            'verses': verses,
            'raw_data': surahs
        }

    except Exception as e:
        logger.error(f"Error loading Quran data: {str(e)}")
        return None 