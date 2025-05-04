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
) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[int], Optional[str], Optional[int]]:
    """Processes audio, identifies Reciter, formats result.

    Args:
        audio_file_storage: The FileStorage object from Flask request.
        params: Dictionary containing request parameters (e.g., 'show_unreliable_predictions').
        debug: Boolean indicating if debug mode is active.

    Returns:
        Tuple containing: (response_data, processed_audio_data, processed_sample_rate, error_message, status_code).
    """
    logging.debug(f"Starting Reciter identification process (debug={debug})")
    processed_audio_data: Optional[np.ndarray] = None
    processed_sample_rate: Optional[int] = None
    try:
        show_unreliable = params.get('show_unreliable_predictions', '').lower() == 'true'
        if debug:
            logging.debug(f"[ReciterService-Debug] Parameters: show_unreliable={show_unreliable}")
        
        # --- Audio Processing --- 
        if debug:
            logging.debug("[ReciterService-Debug] Processing audio for reciter identification...")
        result = process_audio_file(audio_file_storage, for_ayah=False)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], str):
            error_msg = result[1]
            logging.error(f"[ReciterService] Audio processing failed: {error_msg}")
            return None, None, None, error_msg, 400
        elif not isinstance(result, tuple) or len(result) != 2:
             logging.error(f"[ReciterService] Unexpected return format from process_audio_file: {type(result)}")
             return None, None, None, "Internal error during audio processing.", 500

        processed_audio_data, processed_sample_rate = result
        if debug:
            logging.debug(f"[ReciterService-Debug] Audio processed successfully. Sample rate: {processed_sample_rate}, Data shape: {getattr(processed_audio_data, 'shape', 'N/A')}")

        # --- Feature Extraction --- 
        if debug:
            logging.debug("[ReciterService-Debug] Extracting features...")
        features = extract_features(processed_audio_data, processed_sample_rate)
        if features is None:
            logging.error("[ReciterService] Feature extraction returned None.")
            return None, None, None, 'Feature extraction failed', 500
        if debug:
            logging.debug(f"[ReciterService-Debug] Features extracted. Shape: {features.shape}")

        # --- Prediction --- 
        if debug:
            logging.debug("[ReciterService-Debug] Performing prediction...")
        model = get_reciter_model() # Get the loaded model instance

        if features.ndim == 1:
            features = features.reshape(1, -1)

        probabilities = model.predict_proba(features)[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = str(model.classes_[predicted_class_index])
        if debug:
            logging.debug(f"[ReciterService-Debug] Raw prediction: {predicted_class_name}, Index: {predicted_class_index}")

        # --- Reliability Analysis --- 
        if debug:
            logging.debug("[ReciterService-Debug] Analyzing reliability...")
        distances = calculate_distances(features[0], model.centroids)
        reliability = analyze_prediction_reliability(
            probabilities, distances, model.thresholds, predicted_class_name
        )
        if debug:
            logging.debug(f"[ReciterService-Debug] Prediction reliability: {reliability}")

        # --- Format Response --- 
        if debug:
            logging.debug("[ReciterService-Debug] Formatting response...")
        sorted_indices = np.argsort(probabilities)[::-1][:TOP_N_PREDICTIONS]
        predictions_list = []
        
        reciters_metadata = _load_reciters_metadata()
        if reciters_metadata is None:
             reciters_metadata = {}
             logging.warning("[ReciterService] Proceeding without reciters metadata.")

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
        include_predictions = debug or reliability['is_reliable'] or show_unreliable

        if not include_predictions:
            response_data = {
                'reliable': False,
                'main_prediction': None,
                'top_predictions': []
            }
            logging.debug("[ReciterService] Unreliable prediction, hiding results (debug={}, show_unreliable={}).".format(debug, show_unreliable))
        else:
            response_data = {
                'reliable': bool(reliability['is_reliable']), # Still report actual reliability
                'main_prediction': predictions_list[0] if predictions_list else None,
                'top_predictions': predictions_list
            }
            if debug:
                 if not reliability['is_reliable']:
                     logging.debug("[ReciterService-Debug] Showing unreliable predictions because debug mode is active.")
                 else:
                     logging.debug(f"[ReciterService-Debug] Reliable prediction or showing unreliable. Found {len(predictions_list)} predictions.")

        logging.debug("Reciter identification process completed successfully.")
        return response_data, processed_audio_data, processed_sample_rate, None, 200

    except RuntimeError as e:
         logging.error(f"[ReciterService] Runtime error: {e}", exc_info=debug)
         return None, None, None, f"Server runtime error: {e}", 500
    except Exception as e:
        logging.error(f"[ReciterService] Unexpected error during reciter identification: {e}", exc_info=True)
        return None, None, None, f'An unexpected server error occurred: {type(e).__name__}', 500

def get_all_reciters_list() -> Tuple[Optional[List[Dict]], Optional[str], Optional[int]]:
    """Retrieves the list of all known reciters from metadata.

     Returns:
        Tuple containing: (reciters_list, error_message, status_code).
    """
    try:
        reciters_metadata = _load_reciters_metadata()
        if reciters_metadata is None:
            logging.error("[ReciterService] Reciters data file not found or failed to load.")
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
            
        logging.info(f"[ReciterService] Retrieved {len(reciters_list)} reciters.")
        return reciters_list, None, 200
        
    except Exception as e:
        logging.error(f"[ReciterService] Server error in get_all_reciters_list: {e}", exc_info=True)
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
                  logging.error(f"Reciters metadata file not found at primary path or relative to CWD ({Path.cwd()}).")
                  return None

        with open(reciters_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        path_str = str(reciters_file) if reciters_file else "unknown path"
        logging.error(f"Could not load or parse reciters metadata from {path_str}: {e}", exc_info=True)
        return None 