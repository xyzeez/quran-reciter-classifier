"""
Base class for Ayah identification models.
"""
from abc import ABC, abstractmethod
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseAyahModel(ABC):
    """Abstract base class for Ayah identification models."""
    
    def __init__(self):
        """Initialize the base model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load(self, model_path: Path):
        """
        Load a model from the given path.
        
        Args:
            model_path (Path): Path to the model file/directory
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio_data, sample_rate: int) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data array
            sample_rate (int): Sample rate of the audio
            
        Returns:
            str: Transcribed text
        """
        pass
    
    @abstractmethod
    def save(self, save_path: Path):
        """
        Save the model to the given path.
        
        Args:
            save_path (Path): Path to save the model
        """
        pass
    
    def to_device(self, device=None):
        """
        Move model to specified device.
        
        Args:
            device: torch device to move model to. If None, uses self.device
        """
        if device is not None:
            self.device = device
        if self.model is not None:
            self.model = self.model.to(self.device)
    
    @staticmethod
    def get_model_type() -> str:
        """Get the type of the model."""
        return "base_ayah" 