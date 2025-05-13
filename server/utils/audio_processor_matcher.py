"""
Utility for processing audio files, specifically converting to WAV using ffmpeg.
"""
import os
import subprocess
import tempfile
import logging
import shutil

logger = logging.getLogger(__name__)

class AudioProcessor:
    @staticmethod
    def convert_to_wav(input_file, output_dir=None):
        """Convert any audio file to 16kHz mono WAV using ffmpeg.
        
        Returns:
            Tuple[str, str]: Path to the converted WAV file and the directory it was saved in.
        """
        created_temp_dir = False
        try:
            if output_dir is None:
                output_dir = tempfile.mkdtemp() 
                created_temp_dir = True # Mark if we created it
            else:
                os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.basename(input_file)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_dir, f"processed_{file_name_without_ext}.wav")

            command = [
                "ffmpeg",
                "-y",
                "-i", input_file,
                "-ac", "1",
                "-ar", "16000",
                output_path
            ]

            logger.debug(f"Running ffmpeg command: {' '.join(command)}")
            process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if process.stdout:
                logger.debug(f"ffmpeg stdout: {process.stdout.strip()}")
            if process.stderr:
                logger.debug(f"ffmpeg stderr: {process.stderr.strip()}")
            
            if not os.path.exists(output_path):
                logger.error(f"FFmpeg command completed but output file not found: {output_path}")
                raise RuntimeError("Audio conversion failed: Output file not created.")

            # Return output_dir so caller knows where the file is / can clean up if needed
            return output_path, output_dir 

        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip() if e.stderr else "No stderr output"
            logger.error(f"FFmpeg conversion failed (CalledProcessError):\nSTDERR: {stderr_output}")
            # Clean up temp dir if we created it
            if created_temp_dir and os.path.exists(output_dir):
                 shutil.rmtree(output_dir, ignore_errors=True)
            raise RuntimeError(f"Audio conversion failed: FFmpeg error. Details: {stderr_output}")
        except FileNotFoundError:
            logger.error("FFmpeg command failed: 'ffmpeg' executable not found. Ensure ffmpeg is installed and in PATH.")
            if created_temp_dir and os.path.exists(output_dir):
                 shutil.rmtree(output_dir, ignore_errors=True)
            raise RuntimeError("Audio conversion failed: ffmpeg not found.")
        except Exception as e:
            logger.error(f"Unexpected error during audio conversion: {e}", exc_info=True)
            if created_temp_dir and os.path.exists(output_dir):
                 shutil.rmtree(output_dir, ignore_errors=True)
            raise RuntimeError(f"Audio conversion failed due to an unexpected error: {e}") 