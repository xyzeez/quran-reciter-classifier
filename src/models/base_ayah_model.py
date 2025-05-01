"""
Interface for Quranic verse identification models.
Defines common functionality for audio transcription models.
"""
from abc import ABC, abstractmethod
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseAyahModel(ABC):
    """Base interface for verse identification models."""
    
    def __init__(self):
        """Initialize common model attributes."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load(self, model_path: Path):
        """
        Load model from disk.
        
        Args:
            model_path: Model file/directory path
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio_data, sample_rate: int) -> str:
        """
        Convert audio to text.
        
        Args:
            audio_data: Audio time series
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        pass
    
    @abstractmethod
    def save(self, save_path: Path):
        """
        Save model to disk.
        
        Args:
            save_path: Output path
        """
        pass
    
    def to_device(self, device=None):
        """
        Move model to specified compute device.
        
        Args:
            device: Target device, uses default if None
        """
        if device is not None:
            self.device = device
        if self.model is not None:
            self.model = self.model.to(self.device)
    
    @staticmethod
    def get_model_type() -> str:
        """Get model identifier string."""
        return "base_ayah" 