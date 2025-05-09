#!/usr/bin/env python3
"""
Test script for Quran Reciter Classifier models.
Evaluates model performance against test data and generates metrics/visualizations.
"""
import sys
import argparse
from pathlib import Path
import json

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
    Find the model file we want to test and its canonical run ID.

    If no model_id is provided, we'll try to use the latest model.
    The latest model should be pointed to by a symlink, but we'll
    fall back to timestamp-based search if needed.
    It prioritizes run_id from training_metadata.json for reporting.

    Returns:
        tuple: (Path to model.joblib, str run_id_for_reporting) or None
    """
    models_dir = Path(MODEL_OUTPUT_DIR)

    if not models_dir.exists():
        print(f"Error: Models directory '{MODEL_OUTPUT_DIR}' not found.")
        print("Please train a model first: python scripts/train.py")
        return None

    model_dirs_in_root = [d for d in models_dir.iterdir()
                          if d.is_dir() and d.name.endswith("_train")]

    probed_dir_path = None
    specified_model_id = model_id

    if specified_model_id:
        candidate_dir = models_dir / specified_model_id
        if candidate_dir.exists() and candidate_dir.is_dir():
            probed_dir_path = candidate_dir
        else:
            print(
                f"Error: Specified training run '{specified_model_id}' not found in {models_dir}")
            print("Use --list-models to see available training runs")
            return None
    else:
        # Try to find the latest model
        latest_link = models_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                try:
                    probed_dir_path = latest_link.resolve(strict=True)
                    print(f"Info: 'latest' symlink resolved to {probed_dir_path}")
                except FileNotFoundError:
                    print(f"Error: 'latest' symlink points to a non-existent location: {latest_link.readlink()}")
                    return None
                except Exception as e:
                    print(f"Error resolving 'latest' symlink: {e}")
                    return None # Or fallback to directory search? For now, error.
            elif latest_link.is_dir(): # 'latest' is an actual directory
                probed_dir_path = latest_link
                print(f"Info: Using 'latest' as a directory: {probed_dir_path}")
            else: # 'latest' exists but is not a symlink or dir (e.g. a file)
                print(f"Error: 'latest' exists but is not a symlink or a directory at {latest_link}")
                return None
        
        if not probed_dir_path: # Fallback if 'latest' link/dir didn't yield a path
            if not model_dirs_in_root:
                print(f"Error: No trained models found in {MODEL_OUTPUT_DIR}")
                print("Please train a model first: python scripts/train.py")
                return None
            probed_dir_path = max(model_dirs_in_root, key=lambda d: d.stat().st_mtime)
            print(f"Info: No 'latest' link/dir found or resolved. Using most recent model by timestamp: {probed_dir_path}")

    if not probed_dir_path or not probed_dir_path.exists():
        print(f"Error: Could not determine a valid model directory to probe.")
        return None

    # Now we have probed_dir_path, find model file and metadata
    model_files_in_probed_dir = list(probed_dir_path.glob('model_*.joblib'))
    if not model_files_in_probed_dir:
        print(f"Error: No model file (model_*.joblib) found in {probed_dir_path}")
        return None
    
    model_joblib_in_probed_dir = model_files_in_probed_dir[0]
    model_joblib_name = model_joblib_in_probed_dir.name

    # Attempt to get canonical run_id from metadata in probed_dir_path
    run_id_from_meta = None
    training_meta_file = probed_dir_path / "training_metadata.json"
    if training_meta_file.exists():
        try:
            with open(training_meta_file, 'r') as f_meta:
                training_meta = json.load(f_meta)
            if 'run_id' in training_meta and training_meta['run_id']:
                run_id_from_meta = training_meta['run_id']
                print(f"Info: Found run_id '{run_id_from_meta}' in {training_meta_file}")
            else:
                print(f"Warning: 'run_id' not found or empty in {training_meta_file}.")
        except Exception as e:
            print(f"Warning: Could not read or parse {training_meta_file}: {e}.")
    else:
        print(f"Info: {training_meta_file} not found in {probed_dir_path}.")

    # Determine final paths and run ID for reporting
    final_model_joblib_to_load = model_joblib_in_probed_dir
    final_run_id_to_report = probed_dir_path.name # Fallback to directory name

    if run_id_from_meta:
        # If metadata run_id is different from current probed_dir name,
        # or if we want to ensure reported ID is always from metadata if available.
        final_run_id_to_report = run_id_from_meta # Prioritize metadata's run_id for reporting
        
        # Attempt to use canonical path based on metadata's run_id
        candidate_joblib_path = models_dir / run_id_from_meta / model_joblib_name
        if candidate_joblib_path.exists():
            final_model_joblib_to_load = candidate_joblib_path
            print(f"Info: Using model file '{final_model_joblib_to_load}' based on metadata's run_id '{run_id_from_meta}'.")
        else:
            print(f"Warning: Metadata in {probed_dir_path} suggested run_id '{run_id_from_meta}', "
                  f"but model file {candidate_joblib_path} was not found. "
                  f"Using model directly from {model_joblib_in_probed_dir}.")
            # Keep final_model_joblib_to_load as model_joblib_in_probed_dir
            # final_run_id_to_report is already set to run_id_from_meta
            # If probed_dir_path.name was 'latest' and run_id_from_meta was 'actual_id',
            # we report 'actual_id' but load from 'latest' dir if 'actual_id' dir model is missing.
            # This ensures we use the metadata ID for reporting if possible.
            # The joblib path is the one we can actually load.

    # If specified_model_id was given, ensure reported run_id matches it.
    if specified_model_id and final_run_id_to_report != specified_model_id:
        print(f"Warning: Specified model_id was '{specified_model_id}' but effective run_id for reporting is '{final_run_id_to_report}'. This might happen if metadata inside '{specified_model_id}' points elsewhere.")
        # We trust specified_model_id more for reporting if it was directly given.
        # final_run_id_to_report = specified_model_id
        # final_model_joblib_to_load needs to be from specified_model_id dir in this case.
        # The logic above should handle this if probed_dir_path was set to specified_model_id dir and metadata was consistent.
        # If metadata pointed elsewhere and that was used, this warning is appropriate. For now, we'll stick with run_id from meta or probed_dir.name.

    if not final_model_joblib_to_load.exists():
        # This should be rare given previous checks, but as a safeguard:
        print(f"Error: Final model joblib path to load does not exist: {final_model_joblib_to_load}")
        return None
        
    print(f"Selected model file for loading: {final_model_joblib_to_load}")
    print(f"Run ID for reporting: {final_run_id_to_report}")
    return final_model_joblib_to_load, final_run_id_to_report


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
    print("\nFinding model to test...")
    model_info = find_model_path(args.model_file_id)

    if not model_info:
        print("Could not find a suitable model to test.")
        return 1

    model_path_to_test, model_run_id_for_report = model_info
    
    print(f"\nResolved model path for testing: {model_path_to_test}")
    print(f"Resolved model run ID for reporting: {model_run_id_for_report}")

    # Run the test pipeline with the model path and the resolved run ID for reporting
    print(f"\nInitiating test pipeline for model: {model_path_to_test.name} (from run {model_run_id_for_report})")
    status = test_pipeline(model_path=str(model_path_to_test), model_run_id_for_reporting=model_run_id_for_report)

    if status == 0:
        print(f"\n{'='*80}")
        print(f"Testing completed successfully!")
        print(f"Results saved to the '{TEST_RESULTS_DIR}' directory")
        print(f"Use --list-tests to see all test runs")
        print(f"{'='*80}\n")

    return status


if __name__ == "__main__":
    sys.exit(main())
