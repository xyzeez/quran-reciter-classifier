"""
Core ayah identification utilities shared between single and batch processing.
"""
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

from src.models.ayah_model_factory import load_ayah_model
from src.utils.audio_processing import process_audio_file
from src.utils.ayah_matching import find_matching_ayah

logger = logging.getLogger(__name__)

def identify_ayah(
    audio_path: Union[str, Path], 
    model_type: str = "whisper",
    min_confidence: float = 0.70,
    max_matches: int = 5,
    true_surah: Optional[int] = None,
    true_ayah: Optional[int] = None
) -> Dict:
    """
    Core ayah identification function used by both single and batch processing.
    
    Args:
        audio_path: Path to audio file
        model_type: Type of model to use (default: whisper)
        min_confidence: Minimum confidence threshold
        max_matches: Maximum number of matches to return
        true_surah: Optional true surah number for validation
        true_ayah: Optional true ayah number for validation
        
    Returns:
        Dict containing:
            - success: bool indicating if processing was successful
            - error: Error message if not successful
            - transcription: Transcribed text if successful
            - matches: List of potential matches
            - best_match: Best match details
            - processing_time: Time taken to process
            - metrics: Additional metrics if true values provided
    """
    start_time = time.time()
    result = {
        'success': False,
        'error': None,
        'transcription': None,
        'matches': [],
        'best_match': None,
        'processing_time': 0,
        'metrics': {}
    }
    
    try:
        # Process audio file
        audio_result = process_audio_file(audio_path, for_ayah=True)
        if audio_result[0] is None:
            result['error'] = audio_result[1]
            return result
        
        audio_data, sr = audio_result
        
        # Load model and transcribe
        model = load_ayah_model(model_type)
        transcription = model.transcribe(audio_data, sr)
        result['transcription'] = transcription
        
        # Find matching ayah
        matches = find_matching_ayah(
            transcription,
            min_confidence=min_confidence,
            max_matches=max_matches
        )
        
        # Update result with matches
        result['matches'] = matches.get('matches', [])
        result['best_match'] = matches.get('best_match')
        result['success'] = bool(result['best_match'])
        
        # Calculate metrics if true values provided
        if true_surah and true_ayah and result['best_match']:
            best_match = result['best_match']
            result['metrics'] = {
                'correct': (
                    best_match['surah_number'] == true_surah and 
                    best_match['ayah_number'] == true_ayah
                ),
                'confidence': best_match['confidence_score']
            }
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {audio_path}: {str(e)}")
        
    finally:
        result['processing_time'] = time.time() - start_time
        
    return result

def process_batch_file(audio_file: Path, model_type: str = "whisper") -> Dict:
    """
    Process a single file in batch mode, extracting true values from filename.
    
    Args:
        audio_file: Path to audio file
        model_type: Type of model to use
        
    Returns:
        Dict containing processed results
    """
    try:
        # Extract true values from filename
        filename = audio_file.name
        true_surah = int(filename[:3])
        true_ayah = int(filename[3:6])
        
        # Process using shared logic
        result = identify_ayah(
            audio_file,
            model_type=model_type,
            true_surah=true_surah,
            true_ayah=true_ayah,
            max_matches=1  # Batch mode uses single match
        )
        
        # Format result for batch processing
        return {
            'file': filename,
            'true_surah': true_surah,
            'true_ayah': true_ayah,
            'predicted_surah': result['best_match']['surah_number'] if result['best_match'] else None,
            'predicted_ayah': result['best_match']['ayah_number'] if result['best_match'] else None,
            'predicted_surah_name': result['best_match']['surah_name'] if result['best_match'] else None,
            'predicted_surah_name_en': result['best_match']['surah_name_en'] if result['best_match'] else None,
            'predicted_ayah_text': result['best_match']['ayah_text'] if result['best_match'] else None,
            'confidence': result['best_match']['confidence_score'] if result['best_match'] else 0.0,
            'correct': result['metrics'].get('correct', False) if result['success'] else False,
            'error': result['error'],
            'transcription': result['transcription'],
            'processing_time': result['processing_time']
        }
        
    except Exception as e:
        logger.error(f"Error processing {audio_file}: {str(e)}")
        return {
            'file': audio_file.name,
            'true_surah': None,
            'true_ayah': None,
            'predicted_surah': None,
            'predicted_ayah': None,
            'predicted_surah_name': None,
            'predicted_surah_name_en': None,
            'predicted_ayah_text': None,
            'confidence': 0.0,
            'correct': False,
            'error': str(e),
            'transcription': None,
            'processing_time': 0
        } 