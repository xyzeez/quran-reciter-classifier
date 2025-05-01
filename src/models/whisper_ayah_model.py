"""
Whisper-based Arabic speech recognition for Quranic verse identification.
Uses fine-tuned Whisper model optimized for Quranic Arabic.
"""
import torch
from pathlib import Path
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from src.models.base_ayah_model import BaseAyahModel

logger = logging.getLogger(__name__)

class WhisperAyahModel(BaseAyahModel):
    """
    Specialized Whisper model for Quranic verse transcription.
    Uses tarteel.ai's fine-tuned model for Arabic Quran.
    """
    
    MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"
    SAMPLE_RATE = 16000  # Required sample rate for Whisper
    
    def __init__(self):
        """Initialize model components."""
        super().__init__()
        self.model = None
        self.processor = None
    
    def load(self, model_path: Path = None):
        """
        Load Whisper model and tokenizer.
        
        Args:
            model_path: Local model path, uses HuggingFace if None
            
        Returns:
            self: Loaded model instance
        """
        try:
            logger.info("Loading Whisper model and processor...")
            self.processor = WhisperProcessor.from_pretrained(self.MODEL_ID)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.MODEL_ID)
            self.to_device()
            logger.info(f"Model loaded successfully (using {self.device})")
            return self
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def transcribe(self, audio_data, sample_rate: int) -> str:
        """
        Convert audio to Arabic text.
        
        Args:
            audio_data: Audio time series
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed Arabic text
        """
        try:
            if self.model is None or self.processor is None:
                self.load()
            
            # Resample if needed
            if sample_rate != self.SAMPLE_RATE:
                logger.info(f"Resampling from {sample_rate}Hz to {self.SAMPLE_RATE}Hz")
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=self.SAMPLE_RATE
                )
            
            # Prepare input
            input_features = self.processor(
                audio_data, 
                sampling_rate=self.SAMPLE_RATE, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate text
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def save(self, save_path: Path):
        """
        Save model to disk.
        
        Args:
            save_path: Output directory
        """
        try:
            if self.model is not None:
                self.model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
                logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Save failed: {str(e)}")
            raise
    
    @staticmethod
    def get_model_type() -> str:
        """Get model identifier string."""
        return "whisper_ayah" 