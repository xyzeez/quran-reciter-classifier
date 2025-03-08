#!/usr/bin/env python3
"""
Preprocessing script for the Quran Reciter Classifier project.
Handles audio processing, feature extraction, and data organization to prepare
for model training or testing.
"""
import sys
import argparse
from pathlib import Path
from src.pipelines import preprocess_pipeline


def parse_args():
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio files for Quran Reciter Classifier")

    parser.add_argument(
        "--mode", choices=["train", "test"], default="train",
        help="Mode to run in: 'train' (default) or 'test'")

    parser.add_argument(
        "--no-augment", action="store_true",
        help="Skip data augmentation (saves time but reduces diversity)")

    return parser.parse_args()


def main():
    """Main driver for preprocessing pipeline."""
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"Starting preprocessing pipeline in {args.mode} mode")
    print(f"Data augmentation: {'disabled' if args.no_augment else 'enabled'}")
    print(f"{'='*80}\n")

    status, output_dir = preprocess_pipeline(
        mode=args.mode, augment=not args.no_augment)

    if status == 0 and output_dir:
        print(f"\n{'='*80}")
        print(f"Preprocessing completed successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Run ID: {Path(output_dir).name}")
        print(f"{'='*80}\n")
        return 0
    else:
        print(f"\n{'='*80}")
        print(f"Preprocessing failed. Check logs for details.")
        print(f"{'='*80}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
