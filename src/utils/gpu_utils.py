"""
Utility functions for GPU detection and management.
"""
import logging

logger = logging.getLogger(__name__)


def is_gpu_available():
    """
    Check if GPU is actually available at runtime.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    # Try PyTorch first
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(
                0) if device_count > 0 else "Unknown"
            logger.info(
                f"PyTorch detected GPU: {device_name} (Total: {device_count})")
        return gpu_available
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")
    except Exception as e:
        logger.debug(f"Error checking PyTorch GPU: {str(e)}")

    # Try TensorFlow if PyTorch isn't available
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_available = len(gpus) > 0
        if gpu_available:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s)")
        return gpu_available
    except ImportError:
        logger.debug("TensorFlow not available for GPU detection")
    except Exception as e:
        logger.debug(f"Error checking TensorFlow GPU: {str(e)}")

    # If neither is available, assume no GPU
    logger.info(
        "No GPU detection method available, assuming GPU is not available")
    return False
