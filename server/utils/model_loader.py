"""
Model loading utilities for server initialization.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

from src.models.model_factory import load_model
from src.models.ayah_model_factory import load_ayah_model
from server.config import (
    MODEL_DIR,
    MODEL_ID,
    LATEST_MODEL_SYMLINK
)

logger = logging.getLogger(__name__)

# Global model instances
reciter_model = None
ayah_model = None

def find_model_path() -> Optional[Path]:
    """
    Find the appropriate model file path based on config.
    
    Returns:
        Path: Path to model file or None if not found
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

def initialize_models() -> Tuple[bool, bool]:
    """
    Initialize both reciter and ayah models at server startup.
    
    Returns:
        Tuple[bool, bool]: (reciter_success, ayah_success)
    """
    global reciter_model, ayah_model
    reciter_success = False
    ayah_success = False
    
    try:
        # Initialize reciter model
        logger.info("Initializing reciter identification model...")
        model_path = find_model_path()
        if model_path:
            reciter_model = load_model(model_path)
            model_info = reciter_model.get_model_info()
            logger.info(f"Reciter model loaded successfully")
            logger.info(f"Model type: {model_info['model_type']}")
            logger.info(f"Number of classes: {len(reciter_model.classes_)}")
            reciter_success = True
        else:
            logger.error("No valid model path found for reciter model")
            
        # Initialize ayah model
        logger.info("Initializing ayah identification model...")
        ayah_model = load_ayah_model()
        logger.info("Ayah model loaded successfully")
        ayah_success = True
            
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        
    return reciter_success, ayah_success

def get_reciter_model():
    """Get the loaded reciter model instance."""
    global reciter_model
    if reciter_model is None:
        raise RuntimeError("Reciter model not initialized")
    return reciter_model

def get_ayah_model():
    """Get the loaded ayah model instance."""
    global ayah_model
    if ayah_model is None:
        raise RuntimeError("Ayah model not initialized")
    return ayah_model 