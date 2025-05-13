#!/usr/bin/env python3
"""
Command-line tool for Quran reciter identification.
Loads a trained model and predicts the reciter from an audio file.
"""
import sys
import argparse
from pathlib import Path

# Add project root to Python path when running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
from src.pipelines import predict_pipeline
from config import MODEL_OUTPUT_DIR


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace with:
        - model_file_id: Training run ID (optional)
        - audio: Path to audio file
        - true_label: Expected reciter name (optional)
        - list_models: Show available models flag
    """
    parser = argparse.ArgumentParser(
        description="Identify reciter from audio file using Quran Reciter Classifier")

    parser.add_argument(
        "--model-file-id", type=str,
        help="Run ID of the training run to use (e.g., '20240306_152417_train')")

    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to audio file to analyze (required)")

    parser.add_argument(
        "--true-label", type=str,
        help="True reciter name for verification (optional)")

    parser.add_argument(
        "--list-models", action="store_true",
        help="List all available model runs")

    return parser.parse_args()


def find_model_path(model_id=None):
    """
    Locate model file for prediction.
    
    Args:
        model_id: Training run ID (e.g., '20240306_152417_train')
                 Uses latest model if None
                 
    Returns:
        Path to model file or None if not found
    """
    models_dir = Path(MODEL_OUTPUT_DIR)

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return None

    if model_id:
        # Find the specific model requested
        target_dir = models_dir / model_id
        if target_dir.exists():
            # Look for the model file (e.g., model_random_forest.joblib)
            model_files = list(target_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            else:
                print(f"Error: No model file found in {target_dir}")
                return None
        else:
            print(
                f"Error: Training run '{model_id}' not found in {models_dir}")
            return None
    else:
        # First try to use the 'latest' shortcut
        latest_link = models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                target_dir = latest_link.resolve()
            else:
                target_dir = latest_link

            # Get the model file from that directory
            model_files = list(target_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            else:
                print(f"Error: No model file found in {target_dir}")
                return None

        # No 'latest' link? Find the newest model by file timestamp
        model_dirs = [d for d in models_dir.iterdir()
                      if d.is_dir() and d.name.endswith("_train")]
        if not model_dirs:
            print(f"Error: No training runs found in {models_dir}")
            return None

        latest_dir = max(model_dirs, key=lambda d: d.stat().st_mtime)
        model_files = list(latest_dir.glob('model_*.joblib'))
        if model_files:
            return model_files[0]
        else:
            print(f"Error: No model file found in {latest_dir}")
            return None


def list_available_models():
    """
    Display available trained models.
    Shows run ID, model type, and creation timestamp.
    
    Returns:
        0 on success, 1 on error
    """
    models_dir = Path(MODEL_OUTPUT_DIR)
    if not models_dir.exists():
        print(f"Models directory '{MODEL_OUTPUT_DIR}' not found.")
        return 1

    model_dirs = [d for d in models_dir.iterdir()
                  if d.is_dir() and d.name.endswith("_train")]

    if not model_dirs:
        print(f"No training runs found in {models_dir}")
        return 1

    print(f"\nAvailable training runs in {models_dir}:")
    for i, model_dir in enumerate(sorted(model_dirs, key=lambda d: d.stat().st_mtime, reverse=True), 1):
        # Figure out model type from filename
        model_files = list(model_dir.glob('model_*.joblib'))
        model_type = "unknown"
        if model_files:
            model_name = model_files[0].name
            model_type = model_name.split('_')[1] if len(
                model_name.split('_')) > 1 else "unknown"

        # Format the timestamp nicely
        timestamp = model_dir.name.split('_')[0:2]
        timestamp = '_'.join(timestamp)

        # Add indicator for the default model
        latest_mark = ""
        latest_link = models_dir / "latest"
        if latest_link.exists() and latest_link.is_symlink() and latest_link.resolve() == model_dir:
            latest_mark = " (latest)"

        print(
            f"{i}. Run ID: {model_dir.name} - Type: {model_type}, Created: {timestamp}{latest_mark}")

    print("\nUse --model-file-id with the Run ID to use a specific model")
    return 0


def main():
    """
    Run reciter identification on an audio file.
    Handles model selection, audio loading, and prediction.
    
    Returns:
        0 on success, 1 on error
    """
    args = parse_args()

    # Show model list if requested
    if args.list_models:
        return list_available_models()

    # Make sure the audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Audio file not found: {args.audio}")
        return 1

    # Pick the model to use for prediction
    model_path = find_model_path(args.model_file_id)
    if not model_path:
        print("Use --list-models to see available training runs")
        return 1

    print(f"\n{'='*80}")
    print(f"Starting prediction pipeline")
    print(f"Audio file: {audio_path}")
    print(f"Using model: {model_path}")
    if args.model_file_id:
        print(f"From training run: {args.model_file_id}")
    if args.true_label:
        print(f"True label (for verification): {args.true_label}")
    print(f"{'='*80}\n")

    # Run the prediction pipeline
    return predict_pipeline(str(model_path), str(audio_path), args.true_label)


if __name__ == "__main__":
    sys.exit(main())
