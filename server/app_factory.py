"""
Application Factory for creating the Flask app instance.
"""
import logging
import os
from flask import Flask
from pathlib import Path

# Import configuration
from server.config import HOST, PORT # Import any needed config values

# Import utilities for initialization
from server.utils.quran_data import get_raw_quran_data
from server.utils.model_loader import initialize_models # Initializes reciter model
from server.utils.quran_matcher import QuranMatcher

# Import Blueprints
from server.routes.ayah import ayah_bp
from server.routes.reciter import reciter_bp

logger = logging.getLogger(__name__)

def create_app(config_object=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)

    # --- Configuration --- 
    # Load default config or from object/file
    # Example: app.config.from_object('server.config.DevelopmentConfig')
    # Or load directly if config.py just holds variables:
    app.config.from_pyfile('config.py', silent=True)
    if config_object:
        app.config.from_object(config_object)
        
    # Set debug status explicitly based on environment or args if needed
    # app.debug = app.config.get('DEBUG', False) 
    
    # --- Logging Configuration --- 
    # Basic config done here, might be overridden by command line args in app.py
    # Default to INFO level to reduce verbosity from libraries like urllib3
    log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    logging.basicConfig(level=log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Flask app created. Initial log level set to: {logging.getLevelName(log_level)}")

    # --- Initialize Shared Resources --- 
    # Load Quran data once
    logger.info("Loading global Quran data...")
    raw_quran_data = get_raw_quran_data() # Uses the function from quran_data util
    if not raw_quran_data:
        logger.error("CRITICAL: Failed to load Quran data. Ayah routes will be impaired.")
        app.raw_quran_data = [] # Attach empty list to avoid None errors later
    else:
        app.raw_quran_data = raw_quran_data
        logger.info("Global Quran data loaded successfully.")

    # Initialize Reciter model
    logger.info("Initializing machine learning models...")
    reciter_success, _ = initialize_models() # Ignore the second bool
    if not reciter_success:
        logger.error("CRITICAL: Failed to initialize Reciter model. Reciter routes will be impaired.")

    # Initialize QuranMatcher instance
    logger.info("Initializing QuranMatcher...")
    try:
        # Use the data attached to the app context
        matcher_quran_data = getattr(app, 'raw_quran_data', None)
        if matcher_quran_data:
            app.quran_matcher = QuranMatcher(loaded_quran_data=matcher_quran_data)
            logger.info("QuranMatcher instance initialized successfully.")
        else:
            logger.error("Cannot initialize QuranMatcher: Quran data is not available.")
            app.quran_matcher = None
    except Exception as init_err:
        logger.error(f"CRITICAL: Failed to initialize QuranMatcher: {init_err}", exc_info=True)
        app.quran_matcher = None 

    # --- Register Blueprints --- 
    # Note: Using url_prefix='/' means the routes defined within blueprints
    # (e.g., /getAyah) will be directly under the root.
    # If the blueprints had url_prefix defined (e.g., /ayah), then the final
    # route would be /ayah/getAyah.
    # Removing url_prefix from blueprint definitions for direct routes.
    app.register_blueprint(ayah_bp, url_prefix='/')
    app.register_blueprint(reciter_bp, url_prefix='/')
    logger.info("Registered API blueprints.")

    # --- Health Check Endpoint --- 
    @app.route("/health")
    def health_check():
        # Basic check, can be expanded
        # Check component status (e.g., if models/data loaded)
        services_status = {
             "reciter_model": "loaded" if reciter_success else "error",
             "quran_data": "loaded" if app.raw_quran_data else "error",
             "quran_matcher": "initialized" if getattr(app, 'quran_matcher', None) else "error"
        }
        overall_status = "ok" if all(s != "error" for s in services_status.values()) else "error"
        return jsonify({"status": overall_status, "services": services_status}), 200 if overall_status == "ok" else 503

    logger.info("Application factory setup complete.")
    return app 