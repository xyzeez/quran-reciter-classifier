"""
Server package for Quran Reciter Classifier.
Imports the application factory.
"""

from server.app_factory import create_app

__all__ = [
    'create_app'
]

# Remove old code below if app is run via a separate run.py or WSGI server

# def run_server():
#     """Runs the Flask development server."""
#     app = create_app()
#     # Fetch host/port from config AFTER app creation
#     host = app.config.get('HOST', '0.0.0.0')
#     port = app.config.get('PORT', 5000)
#     debug_mode = app.config.get('DEBUG', False)
#     logger.info(f"Starting server on {host}:{port} (Debug: {debug_mode})")
#     app.run(host=host, port=port, debug=debug_mode)
