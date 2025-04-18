"""
Audio transcription utilities using the Tarteel Whisper model.
"""
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
from pathlib import Path
import librosa

logger = logging.getLogger(__name__)

# Initialize model and processor
MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"
WHISPER_SAMPLE_RATE = 16000  # Whisper model's expected sample rate
processor = None
model = None

def load_model():
    """
    Load the Whisper model and processor.
    This should be called once when the server starts.
    """
    global processor, model
    try:
        logger.info("Loading Whisper model and processor...")
        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("Model moved to GPU")
        else:
            logger.info("Using CPU for inference")
            
        logger.info("Whisper model and processor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        raise

def transcribe_audio(audio_data, sample_rate):
    """
    Transcribe audio using the Whisper model.

    Args:
        audio_data (numpy.ndarray): Audio data array
        sample_rate (int): Sample rate of the audio

    Returns:
        str: Transcribed text
    """
    try:
        # Ensure model is loaded
        if model is None or processor is None:
            load_model()

        # Resample audio to 16kHz if needed
        if sample_rate != WHISPER_SAMPLE_RATE:
            logger.info(f"Resampling audio from {sample_rate}Hz to {WHISPER_SAMPLE_RATE}Hz")
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=WHISPER_SAMPLE_RATE
            )

        # Process audio
        input_features = processor(
            audio_data, 
            sampling_rate=WHISPER_SAMPLE_RATE, 
            return_tensors="pt"
        ).input_features

        # Move to GPU if available
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        # Generate transcription
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription.strip()

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise 