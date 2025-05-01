"""
Whisper-based model for Ayah identification.
"""
import torch
from pathlib import Path
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from src.models.base_ayah_model import BaseAyahModel

logger = logging.getLogger(__name__)

class WhisperAyahModel(BaseAyahModel):
    """Whisper model implementation for Ayah identification."""
    
    MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"
    SAMPLE_RATE = 16000  # Whisper model's expected sample rate
    
    def __init__(self):
        """Initialize the Whisper model."""
        super().__init__()
        self.model = None
        self.processor = None
    
    def load(self, model_path: Path = None):
        """
        Load the Whisper model and processor.
        
        Args:
            model_path: Optional path to saved model. If None, loads from HuggingFace.
        """
        try:
            logger.info("Loading Whisper model and processor...")
            self.processor = WhisperProcessor.from_pretrained(self.MODEL_ID)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.MODEL_ID)
            self.to_device()  # Move to appropriate device
            logger.info(f"Whisper model loaded successfully (using {self.device})")
            return self
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def transcribe(self, audio_data, sample_rate: int) -> str:
        """
        Transcribe audio using the Whisper model.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate of the audio
            
        Returns:
            str: Transcribed text
        """
        try:
            # Ensure model is loaded
            if self.model is None or self.processor is None:
                self.load()
            
            # Resample audio if needed
            if sample_rate != self.SAMPLE_RATE:
                logger.info(f"Resampling audio from {sample_rate}Hz to {self.SAMPLE_RATE}Hz")
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=self.SAMPLE_RATE
                )
            
            # Process audio
            input_features = self.processor(
                audio_data, 
                sampling_rate=self.SAMPLE_RATE, 
                return_tensors="pt"
            ).input_features
            
            # Move to appropriate device
            input_features = input_features.to(self.device)
            
            # Generate transcription
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            raise
    
    def save(self, save_path: Path):
        """
        Save the model to the given path.
        
        Args:
            save_path: Path to save the model
        """
        try:
            if self.model is not None:
                self.model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
                logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @staticmethod
    def get_model_type() -> str:
        """Get the type of the model."""
        return "whisper_ayah" 