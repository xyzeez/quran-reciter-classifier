from flask import Blueprint, jsonify, current_app
import logging
from server.utils.model_loader import get_reciter_model

health_bp = Blueprint('health_bp', __name__, url_prefix='/')
logger = logging.getLogger(__name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Provides an endpoint to check the health of application services."""
    services_status = {}
    overall_status_code = 200 # Assume OK initially

    # Check Reciter Model
    try:
        reciter_model = get_reciter_model()
        if reciter_model and hasattr(reciter_model, 'get_model_info') and reciter_model.get_model_info():
            services_status["reciter_model"] = "loaded"
        else:
            services_status["reciter_model"] = "error"
            logger.warning("Health check: Reciter model retrieved but seems invalid or lacks info.")
    except RuntimeError as e:
        services_status["reciter_model"] = "error"
        logger.warning(f"Health check: Reciter model not initialized: {e}")
    except Exception as e:
        services_status["reciter_model"] = "error"
        logger.error(f"Health check: Unexpected error accessing reciter model: {e}", exc_info=True)

    # Check Quran Data
    try:
        raw_quran_data = getattr(current_app, 'raw_quran_data', None)
        if raw_quran_data and isinstance(raw_quran_data, list) and len(raw_quran_data) > 0:
            services_status["quran_data"] = "loaded"
        else:
            services_status["quran_data"] = "error"
            logger.warning("Health check: Quran data not loaded or empty.")
    except Exception as e:
        services_status["quran_data"] = "error"
        logger.error(f"Health check: Error accessing Quran data: {e}", exc_info=True)

    # Check Quran Matcher (and its internal Whisper model)
    try:
        quran_matcher = getattr(current_app, 'quran_matcher', None)
        if quran_matcher and hasattr(quran_matcher, 'model') and hasattr(quran_matcher, 'processor'):
            services_status["quran_matcher"] = "initialized"
        else:
            services_status["quran_matcher"] = "error"
            logger.warning("Health check: Quran matcher (or its Whisper model) not initialized.")
    except Exception as e:
        services_status["quran_matcher"] = "error"
        logger.error(f"Health check: Error accessing Quran matcher: {e}", exc_info=True)
    
    # Determine overall status
    if any(status == "error" for status in services_status.values()):
        overall_status = "error"
        overall_status_code = 503 # Service Unavailable
    else:
        overall_status = "ok"

    return jsonify({"status": overall_status, "services": services_status}), overall_status_code 