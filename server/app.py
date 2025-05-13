"""
Main entry point for running the Flask server when executed as a module.
Uses the application factory pattern.
"""
import argparse
import logging

# Import the application factory
from server.app_factory import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Quran Reciter Classifier API server.")
    parser.add_argument("--host", type=str, default=None, help="Hostname to listen on (default: from config or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: from config or 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (Flask debug, enhanced logging)")

    args = parser.parse_args()

    # Create the app instance using the factory, passing debug status
    # Logging is configured inside create_app based on args.debug
    app = create_app(debug_mode=args.debug)

    # Override host/port from command line if provided
    host = args.host or app.config.get('HOST', '0.0.0.0')
    port = args.port or app.config.get('PORT', 5000)

    # Explicitly set root logger level to DEBUG if --debug is used
    # This ensures RichHandler shows DEBUG messages from our app code,
    # overriding the initial INFO level set by setup_logging.
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug mode active. Root logger level set to DEBUG.") # Log this change at DEBUG level

    # Run the Flask development server
    # Pass debug=args.debug to enable/disable Werkzeug reloader and debugger
    # The host/port are taken from app config, potentially overridden by args
    app.run(host=host, port=port, debug=args.debug)
