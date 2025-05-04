"""
Service layer for Reciter identification and listing.
"""
import logging
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple

# Assuming these are the correct paths after refactoring
from src.utils.audio_utils import process_audio_file 
from src.features import extract_features
from src.utils.distance_utils import calculate_distances, analyze_prediction_reliability
from server.config import TOP_N_PREDICTIONS
# Import the model access function
from server.utils.model_loader import get_reciter_model

logger = logging.getLogger(__name__)

def identify_reciter_from_audio(
    audio_file_storage, # Type hint: werkzeug.datastructures.FileStorage
    params: Dict,
    debug: bool = False # Add debug parameter
) -> Tuple[Optional[Dict], Optional[str], Optional[int]]:
    """Processes audio, identifies Reciter, formats result.

    Args:
        audio_file_storage: The FileStorage object from Flask request.
        params: Dictionary containing request parameters (e.g., 'show_unreliable_predictions').
        debug: Boolean indicating if debug mode is active.

    Returns:
        Tuple containing: (response_data, error_message, status_code).
    """
    try:
        show_unreliable = params.get('show_unreliable_predictions', '').lower() == 'true'
        
        # --- Audio Processing --- 
        logger.info("[ReciterService] Processing audio for reciter identification...")
        result = process_audio_file(audio_file_storage, for_ayah=False)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str):
            error_msg = result[1]
            logger.error(f"[ReciterService] Audio processing failed: {error_msg}")
            return None, error_msg, 400
        elif not isinstance(result, tuple) or len(result) != 2:
             logger.error(f"[ReciterService] Unexpected return format from process_audio_file: {type(result)}")
             return None, "Internal error during audio processing.", 500

        audio_data, sample_rate = result
        logger.info(f"[ReciterService] Audio processed successfully. Sample rate: {sample_rate}, Data shape: {getattr(audio_data, 'shape', 'N/A')}")

        # --- Feature Extraction --- 
        logger.info("[ReciterService] Extracting features...")
        features = extract_features(audio_data, sample_rate)
        if features is None:
            logger.error("[ReciterService] Feature extraction returned None.")
            return None, 'Feature extraction failed', 500
        logger.info(f"[ReciterService] Features extracted. Shape: {features.shape}")

        # --- Prediction --- 
        logger.info("[ReciterService] Performing prediction...")
        model = get_reciter_model() # Get the loaded model instance

        if features.ndim == 1:
            features = features.reshape(1, -1)

        prediction_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        predicted_class_name = str(model.classes_[prediction_idx])
        logger.info(f"[ReciterService] Raw prediction: {predicted_class_name}, Index: {prediction_idx}")

        # --- Reliability Analysis --- 
        distances = calculate_distances(features[0], model.centroids)
        reliability = analyze_prediction_reliability(
            probabilities, distances, model.thresholds, prediction_idx
        )
        logger.info(f"[ReciterService] Prediction reliability: {reliability}")

        # --- Format Response --- 
        sorted_indices = np.argsort(probabilities)[::-1][:TOP_N_PREDICTIONS]
        predictions_list = []
        
        reciters_metadata = _load_reciters_metadata()
        if reciters_metadata is None:
             reciters_metadata = {}
             logger.warning("[ReciterService] Proceeding without reciters metadata.")

        for idx in sorted_indices:
            reciter_name = str(model.classes_[idx])
            confidence = float(probabilities[idx]) * 100 
            
            reciter_info = reciters_metadata.get(reciter_name, {})
            servers = reciter_info.get('servers', [])
            server_url = servers[0] if isinstance(servers, list) and servers else ''
            if isinstance(servers, str):
                 server_url = servers

            predictions_list.append({
                'name': reciter_name,
                'confidence': confidence,
                'nationality': reciter_info.get('nationality', ''),
                'serverUrl': server_url,
                'flagUrl': reciter_info.get('flagUrl', ''),
                'imageUrl': reciter_info.get('imageUrl', '')
            })

        # Construct response based on reliability AND debug mode
        # In debug mode, always include predictions
        # Otherwise, include only if reliable OR show_unreliable is true
        include_predictions = debug or reliability['is_reliable'] or show_unreliable

        if not include_predictions:
            response_data = {
                'reliable': False,
                'main_prediction': None,
                'top_predictions': []
            }
            logger.info("[ReciterService] Unreliable prediction, hiding results (debug={}, show_unreliable={}).".format(debug, show_unreliable))
        else:
            response_data = {
                'reliable': bool(reliability['is_reliable']), # Still report actual reliability
                'main_prediction': predictions_list[0] if predictions_list else None,
                'top_predictions': predictions_list
            }
            if debug and not reliability['is_reliable']:
                 logger.info("[ReciterService] Showing unreliable predictions because debug mode is active.")
            elif reliability['is_reliable']:
                 logger.info(f"[ReciterService] Reliable prediction. Found {len(predictions_list)} predictions.")
            else: # Unreliable but show_unreliable was true
                 logger.info(f"[ReciterService] Unreliable prediction, showing results because show_unreliable=true. Found {len(predictions_list)} predictions.")

        return response_data, None, 200

    except RuntimeError as e:
         logger.error(f"[ReciterService] Runtime error: {e}", exc_info=True)
         return None, f"Server runtime error: {e}", 500
    except Exception as e:
        logger.exception(f"[ReciterService] Unexpected error during reciter identification: {e}")
        return None, f'An unexpected server error occurred: {type(e).__name__}', 500

def get_all_reciters_list() -> Tuple[Optional[List[Dict]], Optional[str], Optional[int]]:
    """Retrieves the list of all known reciters from metadata.

     Returns:
        Tuple containing: (reciters_list, error_message, status_code).
    """
    try:
        reciters_metadata = _load_reciters_metadata()
        if reciters_metadata is None:
            return None, "Reciters data file not found or failed to load.", 500

        reciters_list = []
        for name, data in reciters_metadata.items():
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
            
        logger.info(f"[ReciterService] Retrieved {len(reciters_list)} reciters.")
        return reciters_list, None, 200
        
    except Exception as e:
        logger.exception(f"[ReciterService] Server error in get_all_reciters_list: {e}")
        return None, f'Server error retrieving reciter list: {type(e).__name__}', 500

# Helper function to load reciters.json
def _load_reciters_metadata() -> Optional[Dict]:
    """Loads reciter metadata from data/reciters.json."""
    reciters_file = None # Define for use in exception logging
    try:
        base_path = Path(__file__).resolve().parent.parent.parent
        reciters_file = base_path / 'data' / 'reciters.json'
        
        if not reciters_file.exists():
             reciters_file = Path.cwd() / 'data' / 'reciters.json'
             if not reciters_file.exists():
                  logger.error(f"Reciters metadata file not found at primary path or relative to CWD ({Path.cwd()}).")
                  return None

        with open(reciters_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        path_str = str(reciters_file) if reciters_file else "unknown path"
        logger.error(f"Could not load or parse reciters metadata from {path_str}: {e}", exc_info=True)
        return None 