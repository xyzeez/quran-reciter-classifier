"""
Application Factory for creating the Flask app instance.
"""
import logging
import os
import time # Import time module
from flask import Flask, g, jsonify # Import g and jsonify
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

# Import configuration
from server.config import HOST, PORT # Import any needed config values

# Import utilities for initialization
from server.utils.quran_data import get_raw_quran_data
from server.utils.model_loader import initialize_models, get_reciter_model # Import get_reciter_model for panel
from server.utils.quran_matcher import QuranMatcher
from server.utils.logging_config import setup_logging, get_console, get_module_blocker_filter

# Import Blueprints
from server.routes.ayah import ayah_bp
from server.routes.reciter import reciter_bp
from server.routes.health import health_bp # Added import for health blueprint
from server.routes.models import models_bp # Changed import from info to models

# Get root logger - still useful for direct calls if needed
# root_logger = logging.getLogger() # Removed - Handled internally if needed
# _logger_configured = False # Removed - Handled by logging_config

def create_app(config_object=None, debug_mode=False): # Accept debug_mode flag
    """Create and configure an instance of the Flask application."""
    # global _logger_configured # Removed
    # Get console instance from the logging utility
    _console = get_console()
    _module_blocker = get_module_blocker_filter() # Get the filter instance
    
    # --- Initial Logging Configuration --- 
    # Call the centralized setup function FIRST
    log_level = logging.DEBUG if debug_mode else logging.INFO
    setup_logging(debug_mode=debug_mode)

    app = Flask(__name__)

    # --- Request Timing --- 
    @app.before_request
    def before_request_timing():
        g.start_time = time.perf_counter()

    @app.after_request
    def after_request_timing(response):
        if hasattr(g, 'start_time'):
            elapsed_ms = (time.perf_counter() - g.start_time) * 1000
            response.headers["X-Response-Time-MS"] = f"{elapsed_ms:.2f}"
            
            # Try to add to JSON response body if applicable
            if response.content_type == 'application/json':
                try:
                    data = response.get_json()
                    if isinstance(data, dict): # Ensure data is a dict before adding
                        data['response_time_ms'] = float(f"{elapsed_ms:.2f}")
                        response.data = jsonify(data).data # Re-serialize with new field
                except Exception as e:
                    # Log error if modifying JSON fails, but don't break the response
                    logging.getLogger(__name__).warning(f"Failed to add response_time_ms to JSON: {e}", exc_info=False)
        return response

    # --- Flask App Configuration --- (Moved after logging setup)
    app.config.from_pyfile('config.py', silent=True)
    if config_object:
        app.config.from_object(config_object)
    app.config['DEBUG'] = debug_mode

    # --- Initialize Shared Resources with Progress Bar --- (Logging handled by RichHandler setup above)
    _console.print(f"[bold blue]🚀 Initializing Quran Reciter Classifier Server...[/bold blue]{' [bold yellow]🔧 DEBUG MODE ENABLED[/bold yellow]' if debug_mode else ''}")

    # Get references for panel later
    loaded_reciter_model = None
    loaded_quran_matcher = None
    quran_data_status = "Error"
    reciter_model_status = "Error"
    quran_matcher_status = "Error"
    blueprints_status = "Pending"

    # Define modules to silence during progress display
    modules_to_silence = [
        "server.utils.quran_data",
        "server.utils.model_loader",
        "server.utils.quran_matcher",
        "src.models.model_factory",
        "src.models.blstm_model"
    ]
    _module_blocker.set_blocked_prefixes(modules_to_silence)

    try:
        _module_blocker.set_blocking(True) # Activate blocking

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=_console,
            transient=False
        ) as progress:
            # ... (Progress steps: Load Quran Data, Reciter Model, Quran Matcher, Blueprints)
            # --- Task 1: Load Quran Data ---
            task1 = progress.add_task("📦 Loading Quran Data...", total=1)
            try:
                raw_quran_data = get_raw_quran_data()
                if not raw_quran_data:
                    logging.error("CRITICAL: Failed to load Quran data...") # Log normally
                    app.raw_quran_data = []
                    progress.update(task1, completed=1, description="📦 Loading Quran Data... [red]Failed[/red]")
                else:
                    app.raw_quran_data = raw_quran_data
                    quran_data_status = f"OK ({len(raw_quran_data)} surahs)"
                    progress.update(task1, completed=1, description="📦 Loading Quran Data... [green]OK[/green]")
            except Exception as e:
                logging.error(f"CRITICAL: Error loading Quran data: {e}", exc_info=debug_mode)
                app.raw_quran_data = []
                progress.update(task1, completed=1, description="📦 Loading Quran Data... [red]Error[/red]")

            # --- Task 2: Initialize Reciter Model ---
            task2 = progress.add_task("🧠 Loading Reciter Model...", total=1)
            try:
                reciter_success, model_info_details, model_path_loaded = initialize_models()
                app.reciter_model_details_for_info = None # Initialize attribute
                if not reciter_success:
                    logging.error("CRITICAL: Failed to initialize Reciter model...") # Log normally
                    progress.update(task2, completed=1, description="🧠 Loading Reciter Model... [red]Failed[/red]")
                else:
                    loaded_reciter_model = get_reciter_model()
                    # Store details for the /models endpoint
                    app.reciter_model_details_for_info = {
                        "model_info_from_loader": model_info_details if model_info_details else {},
                        "model_path_from_loader": model_path_loaded if model_path_loaded else "Unknown"
                    }

                    # For the startup panel, use the best available info
                    panel_model_type = "Unknown"
                    panel_model_id_for_display = "Unknown ID"
                    if model_info_details and isinstance(model_info_details, dict):
                        panel_model_type = model_info_details.get('model_type', panel_model_type)
                        # Attempt to get model_id from the model_info_details for panel display
                        # This 'model_id' is what the model itself reports (e.g., from its saved config)
                        panel_model_id_for_display = model_info_details.get('model_id', panel_model_id_for_display)
                    elif model_path_loaded:
                        # Fallback: try to infer model_id (training run ID) from the parent directory of model_path_loaded
                        try:
                            panel_model_id_for_display = Path(model_path_loaded).parent.name
                        except Exception:
                            pass # Keep "Unknown ID"
                    
                    reciter_model_status = f"OK ({panel_model_type}, ID: {panel_model_id_for_display}, {len(loaded_reciter_model.classes_)} classes)"
                    progress.update(task2, completed=1, description=f"🧠 Loading Reciter Model... [green]OK[/green] ({panel_model_type}) ")
            except Exception as e:
                logging.error(f"CRITICAL: Error initializing Reciter model: {e}", exc_info=debug_mode)
                progress.update(task2, completed=1, description="🧠 Loading Reciter Model... [red]Error[/red]")

            # --- Task 3: Initialize QuranMatcher ---
            task3 = progress.add_task("🔍 Initializing Quran Matcher...", total=1)
            try:
                matcher_quran_data = getattr(app, 'raw_quran_data', None)
                if matcher_quran_data:
                    app.quran_matcher = QuranMatcher(loaded_quran_data=matcher_quran_data)
                    loaded_quran_matcher = app.quran_matcher
                    quran_matcher_status = f"OK ({loaded_quran_matcher.model_id} on {loaded_quran_matcher.device.upper()})"
                    progress.update(task3, completed=1, description="🔍 Initializing Quran Matcher... [green]OK[/green] (Whisper)")
                else:
                    logging.error("CRITICAL: Cannot initialize QuranMatcher...") # Log normally
                    app.quran_matcher = None
                    progress.update(task3, completed=1, description="🔍 Initializing Quran Matcher... [red]Failed (No Quran Data)[/red]")
            except Exception as init_err:
                logging.error(f"CRITICAL: Failed to initialize QuranMatcher: {init_err}", exc_info=debug_mode)
                app.quran_matcher = None
                progress.update(task3, completed=1, description="🔍 Initializing Quran Matcher... [red]Error[/red]")

            # --- Task 4: Register Blueprints ---
            task4 = progress.add_task("🔌 Registering API Blueprints...", total=1)
            try:
                app.register_blueprint(ayah_bp, url_prefix='/')
                app.register_blueprint(reciter_bp, url_prefix='/')
                app.register_blueprint(health_bp, url_prefix='/')
                app.register_blueprint(models_bp, url_prefix='/') # Register models_bp
                blueprints_status = "OK"
                progress.update(task4, completed=1, description="🔌 Registering API Blueprints... [green]OK[/green]")
            except Exception as e:
                logging.error(f"CRITICAL: Failed to register blueprints: {e}", exc_info=debug_mode)
                progress.update(task4, completed=1, description="🔌 Registering API Blueprints... [red]Error[/red]")

    finally:
        _module_blocker.set_blocking(False) # Deactivate blocking
        logging.debug("Deactivated module log blocker.") # Only visible if root is DEBUG

    # --- Print Summary Panel --- (Panel code remains the same)
    panel_title = "Server Ready" + (" [DEBUG MODE]" if debug_mode else "")
    panel_border_style = "yellow" if debug_mode else "green"
    panel_content = (
        f"Status:        [bold {panel_border_style}]{'Online (Debug)' if debug_mode else 'Online'}[/bold {panel_border_style}]\n"
        f"Quran Data:    {quran_data_status}\n"
        f"Reciter Model: {reciter_model_status}\n"
        f"Ayah Matcher:  {quran_matcher_status}\n"
        f"Blueprints:    {blueprints_status}\n"
        f"Listening on:  [link=http://{app.config.get('HOST', HOST)}:{app.config.get('PORT', PORT)}]http://{app.config.get('HOST', HOST)}:{app.config.get('PORT', PORT)}[/link]"
    )
    _console.print(Panel(panel_content, title=panel_title, border_style=panel_border_style, expand=False))

    return app 