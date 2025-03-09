#!/usr/bin/env python3
"""
Test script for Quran Reciter Classifier models.
Evaluates model performance against test data and generates metrics/visualizations.
"""
import sys
import argparse
from pathlib import Path

# Add project root to Python path when running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
from src.pipelines import test_pipeline
from config import MODEL_OUTPUT_DIR, TEST_RESULTS_DIR


def parse_args():
    """Handle command line options for testing."""
    parser = argparse.ArgumentParser(
        description="Test a trained Quran Reciter Classifier model")

    parser.add_argument(
        "--model-file-id", type=str,
        help="Run ID of the training run to use (e.g., '20240306_152417_train')")

    parser.add_argument(
        "--list-models", action="store_true",
        help="List all available model runs")

    parser.add_argument(
        "--list-tests", action="store_true",
        help="List all previous test runs")

    return parser.parse_args()


def find_model_path(model_id=None):
    """
    Find the model file we want to test.

    If no model_id is provided, we'll try to use the latest model.
    The latest model should be pointed to by a symlink, but we'll
    fall back to timestamp-based search if needed.

    Returns None if we can't find a suitable model - you might need
    to train one first!
    """
    models_dir = Path(MODEL_OUTPUT_DIR)

    # No models? No testing.
    if not models_dir.exists():
        print(f"Error: Models directory '{MODEL_OUTPUT_DIR}' not found.")
        print("Please train a model first: python scripts/train.py")
        return None

    # Find dirs that look like training runs
    model_dirs = [d for d in models_dir.iterdir()
                  if d.is_dir() and d.name.endswith("_train")]

    if not model_dirs:
        print(f"Error: No trained models found in {MODEL_OUTPUT_DIR}")
        print("Please train a model first: python scripts/train.py")
        return None

    if model_id:
        # User asked for a specific model
        target_dir = models_dir / model_id
        if target_dir.exists():
            # Look for the serialized model file - naming convention: model_random_forest.joblib
            model_files = list(target_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            else:
                print(f"Error: No model file found in {target_dir}")
                return None
        else:
            print(
                f"Error: Training run '{model_id}' not found in {models_dir}")
            print("Use --list-models to see available training runs")
            return None
    else:
        # Try to find the latest model
        latest_link = models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                target_dir = latest_link.resolve()
            else:
                target_dir = latest_link

            # Now find the actual model file
            model_files = list(target_dir.glob('model_*.joblib'))
            if model_files:
                return model_files[0]
            else:
                print(f"Error: No model file found in {target_dir}")
                return None

        # No 'latest' link? Fall back to timestamps
        latest_dir = max(model_dirs, key=lambda d: d.stat().st_mtime)
        model_files = list(latest_dir.glob('model_*.joblib'))
        if model_files:
            return model_files[0]
        else:
            print(f"Error: No model file found in {latest_dir}")
            return None


def list_available_models():
    """
    Show all the models we have trained so far.

    This is helpful when you want to test a specific model version
    or compare different training runs.
    """
    models_dir = Path(MODEL_OUTPUT_DIR)

    # Can't list what doesn't exist
    if not models_dir.exists():
        print(f"Models directory '{MODEL_OUTPUT_DIR}' not found.")
        print("Please train a model first: python scripts/train.py")
        return 1

    model_dirs = [d for d in models_dir.iterdir()
                  if d.is_dir() and d.name.endswith("_train")]

    if not model_dirs:
        print(f"No training runs found in {models_dir}")
        print("Please train a model first: python scripts/train.py")
        return 1

    print(f"\nAvailable training runs in {models_dir}:")
    for i, model_dir in enumerate(sorted(model_dirs, key=lambda d: d.stat().st_mtime, reverse=True), 1):
        # Figure out what type of model this is
        model_files = list(model_dir.glob('model_*.joblib'))
        model_type = "unknown"
        if model_files:
            model_name = model_files[0].name
            model_type = model_name.split('_')[1] if len(
                model_name.split('_')) > 1 else "unknown"

        # Parse timestamp from directory name format: YYYYMMDD_HHMMSS_train
        timestamp = model_dir.name.split('_')[0:2]  # Get YYYYMMDD_HHMMSS part
        timestamp = '_'.join(timestamp)

        # Mark the model that's currently set as "latest"
        latest_mark = ""
        latest_link = models_dir / "latest"
        if latest_link.exists() and latest_link.is_symlink() and latest_link.resolve() == model_dir:
            latest_mark = " (latest)"

        print(
            f"{i}. Run ID: {model_dir.name} - Type: {model_type}, Created: {timestamp}{latest_mark}")

    print("\nUse --model-file-id with the Run ID to test a specific model")
    return 0


def list_test_runs():
    """
    Show all previous test runs and their results.

    Great for comparing how different models performed
    without having to re-run tests.
    """
    test_results_dir = Path(TEST_RESULTS_DIR)
    if not test_results_dir.exists():
        print(f"Test results directory '{TEST_RESULTS_DIR}' not found.")
        print("No test runs have been performed yet.")
        return 1

    test_dirs = [d for d in test_results_dir.iterdir()
                 if d.is_dir() and d.name.endswith("_test")]

    if not test_dirs:
        print(f"No test runs found in {test_results_dir}")
        print("No tests have been performed yet.")
        return 1

    print(f"\nAvailable test runs in {test_results_dir}:")
    for i, test_dir in enumerate(sorted(test_dirs, key=lambda d: d.stat().st_mtime, reverse=True), 1):
        # Try to extract metrics from the summary report
        summary_file = test_dir / 'summary_report.json'
        model_type = "unknown"
        accuracy = "unknown"

        if summary_file.exists():
            try:
                import json
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                model_type = summary.get('model_type', 'unknown')
                accuracy = f"{summary.get('accuracy_rate', 0) * 100:.2f}%"
            except Exception:
                pass

        timestamp = test_dir.name.split('_')[0:2]
        timestamp = '_'.join(timestamp)

        print(f"{i}. Run ID: {test_dir.name} - Model: {model_type}, Accuracy: {accuracy}, Created: {timestamp}")

    return 0


def check_prerequisites():
    """
    Make sure we have everything needed to run tests.

    We need two things to test:
    1. Preprocessed test data
    2. A trained model to evaluate

    Returns False if either is missing.
    """
    # Do we have test data?
    preprocessed_dir = Path("processed/test")
    if not preprocessed_dir.exists():
        print(
            f"Preprocessed test data directory '{preprocessed_dir}' not found.")
        print("Please run preprocessing first: python scripts/preprocess.py --mode test")
        return False

    # Do we have the extracted features?
    preprocessed_files = list(preprocessed_dir.glob("**/all_features.npy"))
    if not preprocessed_files:
        print("No preprocessed test data found.")
        print("Please run preprocessing first: python scripts/preprocess.py --mode test")
        return False

    # Do we have any models?
    models_dir = Path(MODEL_OUTPUT_DIR)
    if not models_dir.exists():
        print(f"Models directory '{MODEL_OUTPUT_DIR}' not found.")
        print("Please train a model first: python scripts/train.py")
        return False

    # Do we have actual model files?
    model_dirs = [d for d in models_dir.iterdir()
                  if d.is_dir() and d.name.endswith("_train")]
    if not model_dirs:
        print(f"No trained models found in {MODEL_OUTPUT_DIR}")
        print("Please train a model first: python scripts/train.py")
        return False

    return True


def main():
    """Fire up the testing process and handle results."""
    args = parse_args()

    # Make sure we have a place to store test results
    Path(TEST_RESULTS_DIR).mkdir(exist_ok=True)

    # Handle the list commands first
    if args.list_models:
        return list_available_models()

    if args.list_tests:
        return list_test_runs()

    # Make sure we've got what we need
    if not check_prerequisites():
        return 1

    # Figure out which model to test
    model_path = find_model_path(args.model_file_id)
    if not model_path:
        # Error already shown to user
        return 1

    print(f"\n{'='*80}")
    print(f"Starting testing pipeline")
    print(f"Using model: {model_path}")
    if args.model_file_id:
        print(f"From training run: {args.model_file_id}")
    print(f"Results will be saved to the '{TEST_RESULTS_DIR}' directory")
    print(f"{'='*80}\n")

    # Run the test pipeline with the model path
    status = test_pipeline(str(model_path))

    if status == 0:
        print(f"\n{'='*80}")
        print(f"Testing completed successfully!")
        print(f"Results saved to the '{TEST_RESULTS_DIR}' directory")
        print(f"Use --list-tests to see all test runs")
        print(f"{'='*80}\n")

    return status


if __name__ == "__main__":
    sys.exit(main())
