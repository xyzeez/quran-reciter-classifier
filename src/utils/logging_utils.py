import logging
import os
from datetime import datetime
from pathlib import Path
from config import LOGS_DIR

def setup_logging(log_name, log_to_console=True):
    """
    Set up logging configuration.
    
    Args:
        log_name (str): Base name for the log file
        log_to_console (bool): Whether to also log to console
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{log_name}_{timestamp}.log")
    
    # Configure handlers
    handlers = [logging.FileHandler(log_file)]
    if log_to_console:
        handlers.append(logging.StreamHandler())
    
    # Set up basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    
    return logger, log_file

def format_duration(seconds):
    """
    Format duration in seconds to a readable string.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    from datetime import timedelta
    
    duration = timedelta(seconds=seconds)
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)