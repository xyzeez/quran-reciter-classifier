"""
Main entry point for running the Flask server when executed as a module.
Uses the application factory pattern.
"""
import argparse
import logging

# Import the application factory
from server.app_factory import create_app
from server.config import HOST, PORT # Get host/port from config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Quran Reciter API server.")
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode.')
    # Add other potential arguments like --host, --port if needed
    # parser.add_argument('--host', default=HOST, help='Host to run the server on.')
    # parser.add_argument('--port', type=int, default=PORT, help='Port to run the server on.')
    args = parser.parse_args()

    # Create the app instance using the factory
    # Pass debug status to potentially influence factory behavior (e.g., logging level)
    # A more robust approach might use environment variables or a config object
    app = create_app()
    
    # Set debug mode based on command line arg *after* app creation
    # if factory doesn't handle it internally based on config.
    app.debug = args.debug 
    
    # Update logger level if debug flag is set after initial basicConfig
    if app.debug:
         logging.getLogger().setLevel(logging.DEBUG)
         for handler in logging.getLogger().handlers:
              handler.setLevel(logging.DEBUG)
         logger.info("Debug mode enabled via command line. Log level set to DEBUG.")

    logger.info(f"Starting server on {HOST}:{PORT}...")
    # Use HOST and PORT from config for running
    app.run(host=HOST, port=PORT, debug=app.debug)
