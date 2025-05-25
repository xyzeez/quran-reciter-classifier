from flask import Blueprint, jsonify, current_app
import logging
from pathlib import Path

from server.utils.model_loader import get_reciter_model
from server.config import AYAH_DEFAULT_MAX_MATCHES, AYAH_DEFAULT_MIN_CONFIDENCE # Import Ayah configs
# QuranMatcher is on current_app, raw_quran_data is also on current_app

models_bp = Blueprint('models_bp', __name__, url_prefix='/')
logger = logging.getLogger(__name__)

@models_bp.route('/models', methods=['GET'])
def get_models_info():
    """Provides detailed information about the loaded models."""
    models_info = {}
    reciter_model_details = {"status": "not_loaded", "details": {}}
    ayah_matcher_details = {"status": "not_initialized", "details": {}}

    # --- Reciter Model Information ---
    try:
        model = get_reciter_model()
        if model:
            # Get the details stored by app_factory
            stored_model_info = getattr(current_app, 'reciter_model_details_for_info', None)

            if stored_model_info:
                loader_info = stored_model_info.get('model_info_from_loader', {})
                path_from_loader = stored_model_info.get('model_path_from_loader', "Unknown")
                
                # Determine model_id: Prefer model_id from loader_info, then parent dir of path_from_loader
                current_model_id = loader_info.get('model_id', "N/A")
                if current_model_id == "N/A" and path_from_loader != "Unknown":
                    try:
                        current_model_id = Path(path_from_loader).parent.name
                    except Exception:
                        current_model_id = "N/A" # if path is malformed or parent cannot be determined

                # Extract training parameters from loader_info
                training_params = {}
                # Add 'classes', 'n_classes', 'num_classes' here to prevent them from being duplicated 
                # at the top level of training_params. They should only appear nested if present in training_info.
                explicitly_handled_keys = ['model_type', 'model_id', 'model_path', 'classes', 'n_classes', 'num_classes']
                if isinstance(loader_info, dict):
                    for key, value in loader_info.items():
                        if key not in explicitly_handled_keys:
                            # Specifically check for and handle nested 'history' and 'classes' in 'training_info'
                            if key == 'training_info' and isinstance(value, dict):
                                modified_training_info = value.copy()
                                if 'history' in modified_training_info:
                                    del modified_training_info['history']
                                # We WANT to keep 'classes' and 'n_classes' if they are in training_info
                                training_params[key] = modified_training_info
                            else:
                                training_params[key] = value

                reciter_model_details["status"] = "loaded"
                reciter_model_details["details"] = {
                    "model_type": loader_info.get('model_type', 'N/A'),
                    "model_id": current_model_id,
                    "training_parameters": training_params if training_params else "Not available or not in model info"
                }
            else:
                reciter_model_details["status"] = "loaded_no_factory_info"
                reciter_model_details["details"]["message"] = "Model is loaded, but factory-provided details are missing."
                # If we reach here, model is loaded but loader_info was missing. Provide direct attributes if any.
                # reciter_model_details["details"]["num_classes_direct"] = len(model.classes_) if hasattr(model, 'classes_') else 'N/A'
                # reciter_model_details["details"]["classes_direct"] = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else []
    except RuntimeError as e: # Model not initialized
        reciter_model_details["status"] = "error_not_initialized"
        reciter_model_details["details"]["error_message"] = str(e)
        logger.warning(f"Models info: Reciter model not initialized: {e}")
    except Exception as e:
        reciter_model_details["status"] = "error_accessing"
        reciter_model_details["details"]["error_message"] = str(e)
        logger.error(f"Models info: Error accessing reciter model: {e}", exc_info=True)

    models_info["reciter_model"] = reciter_model_details

    # --- Ayah Matcher (Whisper) Information ---
    try:
        quran_matcher = getattr(current_app, 'quran_matcher', None)
        if quran_matcher:
            ayah_matcher_details["status"] = "initialized"
            ayah_matcher_details["details"] = {
                "matcher_type": "Whisper",
                "whisper_model_id": getattr(quran_matcher, 'model_id', 'N/A'),
                "device": getattr(quran_matcher, 'device', 'N/A'),
                "normalized_verses_count": len(quran_matcher.normalized_verses) if hasattr(quran_matcher, 'normalized_verses') else 'N/A',
                "all_verses_count": len(quran_matcher.all_verses) if hasattr(quran_matcher, 'all_verses') else 'N/A',
                "service_defaults": {
                    "max_matches": AYAH_DEFAULT_MAX_MATCHES,
                    "min_confidence": AYAH_DEFAULT_MIN_CONFIDENCE
                }
            }
            
            # Information about the raw Quran data source for the matcher
            raw_data = getattr(current_app, 'raw_quran_data', None)
            if raw_data and isinstance(raw_data, list):
                 ayah_matcher_details["details"]["quran_data_source_surahs"] = len(raw_data)
            else:
                 ayah_matcher_details["details"]["quran_data_source_surahs"] = "Not available or empty"
        else:
            logger.warning("Models info: Quran matcher not found on current_app.")

    except Exception as e:
        ayah_matcher_details["status"] = "error_accessing"
        ayah_matcher_details["details"]["error_message"] = str(e)
        logger.error(f"Models info: Error accessing quran_matcher: {e}", exc_info=True)
        
    models_info["ayah_matcher"] = ayah_matcher_details

    return jsonify(models_info), 200 