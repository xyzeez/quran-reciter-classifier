"""
Model initialization and management for the server.
Handles loading and access to reciter and verse identification models.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

from src.models.model_factory import load_model
from server.config import (
    MODEL_DIR,
    MODEL_ID,
    LATEST_MODEL_SYMLINK
)

logger = logging.getLogger(__name__)

# Global model instances
reciter_model = None
# Ayah model handled by QuranMatcher

def find_model_path() -> Optional[Path]:
    """
    Locate model file using configured paths and fallback strategies.
    Tries specific model ID, latest symlink, and most recent model.
    
    Returns:
        Path to model file or None if not found
    """
    models_dir = Path(MODEL_DIR)
    
    try:
        # Try specific model ID first
        if MODEL_ID:
            model_dir = models_dir / MODEL_ID
            if model_dir.exists():
                model_files = list(model_dir.glob('model_*.joblib'))
                if model_files:
                    return model_files[0]
                logger.warning(f"No model file found in {model_dir}")
            logger.warning(f"Model directory not found at {model_dir}, trying latest")
            
        # Try latest symlink
        latest_link = models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                target_dir = latest_link.resolve()
            else:
                target_dir = latest_link
                
            model_files = list(target_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            logger.warning(f"No model file found in {target_dir}")
            
        # Find most recent model directory
        model_dirs = [d for d in models_dir.iterdir() 
                     if d.is_dir() and d.name.endswith('_train')]
        if model_dirs:
            latest_dir = max(model_dirs, key=lambda d: d.stat().st_mtime)
            model_files = list(latest_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            logger.warning(f"No model file found in {latest_dir}")
            
        logger.error("No valid model file found in any location")
        return None
        
    except Exception as e:
        logger.error(f"Error finding model path: {str(e)}")
        return None

def initialize_models() -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Load the reciter identification model at startup.
    (Ayah model loading is handled by QuranMatcher)
    
    Returns:
        (reciter_success, model_info, model_path_str)
    """
    global reciter_model
    reciter_success = False
    model_info_to_return = None
    model_path_str_to_return = None
    
    try:
        # Initialize reciter model
        logger.info("Initializing reciter identification model...")
        model_path = find_model_path()
        if model_path:
            reciter_model = load_model(model_path)
            if reciter_model:
                model_path_str_to_return = str(model_path.resolve())
                # Ensure get_model_info() is called and its result stored
                if hasattr(reciter_model, 'get_model_info') and callable(reciter_model.get_model_info):
                    model_info_to_return = reciter_model.get_model_info()
                    # If model_info_to_return doesn't include model_path or model_id, try to add them
                    if not model_info_to_return:
                        model_info_to_return = {}
                    if 'model_path' not in model_info_to_return:
                        model_info_to_return['model_path'] = model_path_str_to_return
                    # model_id should ideally be part of the model_info from get_model_info itself,
                    # often derived from the directory name (e.g. training run ID)
                    # For now, we rely on what get_model_info provides for model_id.
                else:
                    # Fallback if get_model_info is not present
                    model_info_to_return = {
                        'model_type': 'Unknown (get_model_info not found)',
                        'model_path': model_path_str_to_return
                    }
                
                logger.info(f"Reciter model loaded successfully from {model_path_str_to_return}")
                logger.info(f"Model type: {model_info_to_return.get('model_type', 'N/A')}")
                logger.info(f"Number of classes: {len(reciter_model.classes_)}")
                reciter_success = True
            else:
                logger.error(f"load_model({model_path}) returned None")
        else:
            logger.error("No valid model path found for reciter model")
            
    except Exception as e:
        logger.error(f"Error initializing reciter model: {str(e)}", exc_info=True)
        
    return reciter_success, model_info_to_return, model_path_str_to_return

def get_reciter_model():
    """
    Get the global reciter identification model.
    
    Returns:
        Loaded reciter model instance
        
    Raises:
        RuntimeError: If model not initialized
    """
    global reciter_model
    if reciter_model is None:
        raise RuntimeError("Reciter model not initialized")
    return reciter_model