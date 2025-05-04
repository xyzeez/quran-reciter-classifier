"""
Utility for processing audio files, specifically converting to WAV using ffmpeg.
"""
import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    @staticmethod
    def convert_to_wav(input_file, output_dir=None):
        """Convert any audio file to 16kHz mono WAV using ffmpeg"""
        try:
            if output_dir is None:
                # Create temp dir within the system's temp location
                output_dir = tempfile.mkdtemp() 
            else:
                # Ensure the provided output directory exists
                os.makedirs(output_dir, exist_ok=True)

            # Use a unique name to avoid collisions if multiple requests happen concurrently
            # Use a .wav extension regardless of the input filename
            base_name = os.path.basename(input_file)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_dir, f"processed_{file_name_without_ext}.wav")

            # FFmpeg command to convert to mono, 16kHz WAV
            command = [
                "ffmpeg",
                "-y",                   # Overwrite output file if exists
                "-i", input_file,      # Input file
                "-ac", "1",            # Mono
                "-ar", "16000",        # Sample rate 16kHz
                output_path
            ]

            logger.info(f"Running ffmpeg command: {' '.join(command)}")
            process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore') # Added encoding/errors
            # Log ffmpeg output for debugging, even on success
            if process.stdout:
                logger.info(f"ffmpeg stdout: {process.stdout.strip()}")
            if process.stderr:
                logger.warning(f"ffmpeg stderr: {process.stderr.strip()}") # Use warning for stderr noise
            
            if not os.path.exists(output_path):
                logger.error(f"FFmpeg command completed but output file not found: {output_path}")
                raise RuntimeError("Audio conversion failed: Output file not created.")

            return output_path, output_dir # Return output_dir too for cleanup if created internally

        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else "No stderr output"
            logger.error(f"FFmpeg conversion failed (CalledProcessError):\nSTDERR: {stderr_output}")
            raise RuntimeError(f"Audio conversion failed: FFmpeg error. Details: {stderr_output}")
        except FileNotFoundError:
            logger.error("FFmpeg command failed: 'ffmpeg' executable not found. Ensure ffmpeg is installed and in PATH.")
            raise RuntimeError("Audio conversion failed: ffmpeg not found.")
        except Exception as e:
            logger.error(f"Unexpected error during audio conversion: {e}", exc_info=True)
            raise RuntimeError(f"Audio conversion failed due to an unexpected error: {e}") 