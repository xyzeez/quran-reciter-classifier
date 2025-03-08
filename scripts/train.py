#!/usr/bin/env python3
"""
Training script for the Quran Reciter Classifier system.
Handles model building, training and serialization.
"""
import sys
import argparse
from pathlib import Path
from src.pipelines import train_pipeline
from config import PROCESSED_DIR


def parse_args():
    """Parse CLI args for training config."""
    parser = argparse.ArgumentParser(
        description="Train a Quran Reciter Classifier model")

    parser.add_argument(
        "--model-type", type=str, default=None,
        help="Model type to use (random_forest, blstm)")

    parser.add_argument(
        "--preprocess-file-id", type=str,
        help="Run ID of preprocessing to use (e.g., '20240306_143208_preprocess')")

    return parser.parse_args()


def find_preprocess_dir(file_id=None):
    """
    Locate the preprocessed data directory.

    If file_id is provided, looks for that specific run.
    Otherwise tries to find the latest run based on symlink or timestamp.

    Can fail if no preprocessed data exists - you'll need to run preprocess.py first.
    """
    processed_dir = Path(f"{PROCESSED_DIR}/train")

    if not processed_dir.exists():
        print(f"Error: Processed directory not found: {processed_dir}")
        return None

    if file_id:
        # User specified a run ID - try to find it
        target_dir = processed_dir / file_id
        if target_dir.exists():
            return target_dir
        else:
            print(
                f"Error: Preprocessing run '{file_id}' not found in {processed_dir}")
            return None
    else:
        # Try to use the 'latest' symlink if it exists
        latest_link = processed_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                return latest_link.resolve()
            else:
                return latest_link

        # Fall back to finding most recent directory by timestamp
        preprocess_dirs = [d for d in processed_dir.iterdir()
                           if d.is_dir() and d.name.endswith("_preprocess")]
        if not preprocess_dirs:
            print(f"Error: No preprocessing runs found in {processed_dir}")
            return None

        return max(preprocess_dirs, key=lambda d: d.stat().st_mtime)


def main():
    """Run the training process from end to end."""
    args = parse_args()

    # Find the data we'll use for training
    preprocess_dir = find_preprocess_dir(args.preprocess_file_id)
    if not preprocess_dir:
        return 1

    print(f"\n{'='*80}")
    print(f"Starting training pipeline")
    print(
        f"Model type: {args.model_type if args.model_type else 'default from config'}")
    print(f"Using preprocessing run: {preprocess_dir.name}")
    print(f"{'='*80}\n")

    # Do the actual training
    status, model_path = train_pipeline(
        model_type=args.model_type, preprocess_dir=str(preprocess_dir))

    if status == 0 and model_path:
        model_dir = Path(model_path).parent
        print(f"\n{'='*80}")
        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Run ID: {model_dir.name}")
        print(f"{'='*80}\n")
        return 0
    else:
        print(f"\n{'='*80}")
        print(f"Training failed. Check logs for details.")
        print(f"{'='*80}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
