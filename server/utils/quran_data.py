"""
Utilities for loading and handling Quran JSON data.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import re

logger = logging.getLogger(__name__)

# Global cache for Quran data to avoid reloading
_quran_data_cache = None

def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for comparison.
    
    This function:
    1. Removes all diacritics (tashkeel)
    2. Normalizes various forms of alef
    3. Normalizes hamza forms
    4. Removes tatweel (stretching character)
    5. Normalizes teh marbuta to heh
    6. Removes specific Quranic symbols if needed (optional)
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove diacritics (tashkeel) - refined range
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
    
    # Normalize alef variations to plain alef
    text = re.sub(r'[إأٱآ]', 'ا', text)
    
    # Normalize hamza forms (on Waw, Yeh, standalone) to plain Hamza
    # Note: Keeping ء as is, only changing carriers unless specific rules needed
    text = re.sub(r'[ؤئ]', 'ء', text) 
    
    # Remove tatweel (stretching character)
    text = text.replace('ـ', '')
    
    # Normalize teh marbuta to heh
    text = text.replace('ة', 'ه')
    
    # Normalize alef maksura to yeh
    text = text.replace('ى', 'ي')
    
    # Optional: Remove specific symbols often found in Quran text if needed
    # text = re.sub(r'[ٜٝٛٞٚٙ]', '', text) # Example: Small high signs etc.

    # Keep only Arabic letters, Hamza, and spaces. Adjust range if needed.
    text = re.sub(r'[^\u0621-\u063A\u0641-\u064A\u0600-\u0605\s]', '', text) 
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()

def load_quran_data(force_reload: bool = False) -> Optional[Dict]:
    """Load and prepare Quran data (from data/quran.json) for matching.
    Uses a global cache.
    
    Args:
        force_reload: If True, ignores the cache and reloads the file.

    Returns:
        Dict containing processed Quran data (e.g., list of Surahs or verses)
        or None if loading fails.
    """
    global _quran_data_cache
    if _quran_data_cache is not None and not force_reload:
        logger.info("Returning cached Quran data.")
        return _quran_data_cache
        
    try:
        # Determine the path relative to this file's location
        # Assumes this file is in server/utils/, data/ is ../../data/
        # Adjust if structure is different. Using Path.cwd() as a fallback might be needed.
        base_path = Path(__file__).resolve().parent.parent.parent
        quran_file = base_path / 'data' / 'quran.json'

        if not quran_file.exists():
             # Fallback to searching from CWD (useful if run differently)
             logger.warning(f"Quran data not found at {quran_file}, trying relative to CWD.")
             quran_file = Path.cwd() / 'data' / 'quran.json'
             if not quran_file.exists():
                logger.error(f"Could not find quran.json at primary path or relative to CWD ({Path.cwd()}).")
                raise FileNotFoundError(f"Could not find quran.json")

        logger.info(f"Loading Quran data from: {quran_file}")
        with open(quran_file, 'r', encoding='utf-8') as f:
            # Load the raw structure (assuming it's a list of Surah objects)
            raw_data = json.load(f)
            if not isinstance(raw_data, list):
                 logger.error(f"Loaded Quran data is not a list as expected: {type(raw_data)}")
                 return None
                 
            logger.info(f"Successfully loaded raw Quran data ({len(raw_data)} surahs).")
            
            # Store the raw data in the cache
            _quran_data_cache = raw_data 
            return _quran_data_cache # Return the raw list of surahs

    except FileNotFoundError as e:
        logger.error(f"Error loading Quran data: {e}")
        _quran_data_cache = None
        return None
    except json.JSONDecodeError as e:
         logger.error(f"Error decoding Quran JSON data from {quran_file}: {e}")
         _quran_data_cache = None
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading Quran data: {e}", exc_info=True)
        _quran_data_cache = None
        return None

# Example of a function to get the raw data (list of surahs)
def get_raw_quran_data() -> Optional[list]:
    """Returns the cached raw Quran data (list of Surah objects). Loads if not cached."""
    data = load_quran_data() 
    # The cache now stores the raw data directly
    return data

# Remove commented out example function