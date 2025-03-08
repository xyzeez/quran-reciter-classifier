"""
Testing pipeline for Quran reciter identification.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import json
import warnings
import shutil
import platform

from config import *
from src.models import load_model
from src.utils import calculate_distances, analyze_prediction_reliability, setup_logging, format_duration
from src.evaluation import plot_confusion_matrix
from src.utils import is_gpu_available

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def test_pipeline(model_path=None):
    """
    Run the testing pipeline.

    Args:
        model_path (str, optional): Path to model file. If None, uses the latest model.

    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_test"
        logger, log_file = setup_logging(f"test_report_{run_id}")

        # Create the test results directory
        test_results_dir = Path(TEST_RESULTS_DIR)
        test_results_dir.mkdir(exist_ok=True)

        # Create timestamped directory for this test run
        output_dir = test_results_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting testing pipeline")
        logger.info(f"Test run ID: {run_id}")
        logger.info(f"Results will be saved to: {output_dir}")

        # Step 1: Load the model
        if model_path is None:
            # This should not happen as the script should handle this
            logger.error("No model path provided")
            return 1

        try:
            logger.info(f"Loading model from: {model_path}")
            model = load_model(model_path)
            logger.info('Model loaded successfully')

            # Get model info and save path
            model_info = model.get_model_info()
            model_file_path = Path(model_path)
            model_dir = model_file_path.parent

            logger.info(f"Model type: {model_info['model_type']}")
            logger.info(f"Model directory: {model_dir}")
            logger.info(f"Number of classes: {model_info['n_classes']}")

            # For BLSTM model, log additional info
            if model_info['model_type'] == 'BLSTM':
                logger.info(
                    f"Using first {model_info.get('input_size', BLSTM_MFCC_COUNT)} MFCCs as features")

        except Exception as e:
            logger.error(f'Error loading model: {str(e)}')
            return 1

        # Step 2: Find preprocessed test data directory
        processed_base_dir = Path("processed/test")

        if not processed_base_dir.exists():
            logger.error(
                f"Preprocessed test data directory not found: {processed_base_dir}")
            logger.error(
                "Please run the preprocessing pipeline first with 'python scripts/preprocess.py --mode test'")
            return 1

        # First, check for "latest" symbolic link
        latest_link = processed_base_dir / "latest"
        if latest_link.exists():
            if latest_link.is_symlink():
                try:
                    preprocessed_path = latest_link.resolve()
                    logger.info(
                        f"Using latest preprocessing directory (from symlink): {preprocessed_path}")
                except:
                    # If symlink resolution fails, treat as regular directory
                    preprocessed_path = latest_link
                    logger.info(
                        f"Using latest preprocessing directory: {preprocessed_path}")
            else:
                preprocessed_path = latest_link
                logger.info(
                    f"Using latest preprocessing directory: {preprocessed_path}")
        else:
            # Find most recent preprocessing directory
            preprocess_dirs = [d for d in processed_base_dir.iterdir()
                               if d.is_dir() and d.name.endswith("_preprocess")]

            if not preprocess_dirs:
                logger.error(
                    f"No preprocessing directories found in {processed_base_dir}")
                logger.error(
                    "Please run the preprocessing pipeline first with 'python scripts/preprocess.py --mode test'")
                return 1

            preprocessed_path = max(
                preprocess_dirs, key=lambda d: d.stat().st_mtime)
            logger.info(
                f"Using most recent preprocessing directory: {preprocessed_path}")

        # Step 3: Load preprocessed data
        features_file = preprocessed_path / "all_features.npy"
        metadata_file = preprocessed_path / "all_metadata.csv"
        preprocess_metadata_file = preprocessed_path / "preprocessing_metadata.json"

        logger.info(f"Looking for features file at: {features_file}")
        logger.info(f"Looking for metadata file at: {metadata_file}")

        if not features_file.exists():
            logger.error(f"Features file not found: {features_file}")
            return 1

        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return 1

        # Load preprocessing metadata if available
        preprocess_info = {}
        if preprocess_metadata_file.exists():
            try:
                with open(preprocess_metadata_file, 'r') as f:
                    preprocess_info = json.load(f)
                logger.info(
                    f"Loaded preprocessing metadata from {preprocess_metadata_file}")
                logger.info(
                    f"Preprocessing run ID: {preprocess_info.get('run_id', 'unknown')}")
            except Exception as e:
                logger.warning(
                    f"Could not load preprocessing metadata: {str(e)}")

        logger.info(f"Loading preprocessed features from {features_file}")
        features = np.load(features_file)

        logger.info(f"Loading metadata from {metadata_file}")
        metadata = pd.read_csv(metadata_file)

        # Extract necessary information from metadata
        file_names = metadata['file_name'].values
        true_labels = metadata['reciter'].values

        # Get durations if available
        durations = None
        if 'duration' in metadata.columns:
            durations = metadata['duration'].values
            logger.info(
                f"Found duration information for {len(durations)} files")
        else:
            logger.warning("Duration information not found in metadata")

        logger.info(f"Loaded {len(features)} samples for testing")

        # Step 4: Make predictions for all samples
        all_results = []
        predicted_labels = []
        total_samples = len(features)
        test_start_time = datetime.now()

        # Get training reciters from model
        training_reciters = set(model.classes_)
        logger.info(
            f"Training reciters: {', '.join(sorted(training_reciters))}")

        logger.info("Making predictions...")
        for i, (feature_vector, true_label, file_name) in enumerate(zip(features, true_labels, file_names)):
            logger.info(
                f"Processing sample {i + 1}/{total_samples}: {file_name}")
            logger.info(f"  True Reciter: {true_label}")

            # Check if reciter is in training data
            is_training_reciter = true_label in training_reciters
            logger.info(f"  Is Training Reciter: {is_training_reciter}")

            # Get file duration if available (in seconds)
            file_duration = None
            if durations is not None:
                file_duration = durations[i]
                # Format duration for readability if needed
                if file_duration is not None:
                    file_duration = float(file_duration)  # Ensure it's a float

            # Make prediction
            feature_vector = feature_vector.reshape(1, -1)
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]

            # Calculate distances to centroids
            distances = calculate_distances(feature_vector, model.centroids)

            # Analyze reliability
            reliability = analyze_prediction_reliability(
                probabilities, distances, model.thresholds, prediction)

            # Store prediction
            if reliability['is_reliable']:
                final_prediction = prediction
                logger.info(
                    f"  Result: {prediction} (Confidence: {reliability['top_confidence']:.2%})")
                logger.info(
                    f"  Distance ratio: {reliability['distance_ratio']:.2%}")
            else:
                final_prediction = "Unknown"
                logger.info(f"  Result: Unknown (Reliability issues)")
                for reason in reliability['failure_reasons']:
                    logger.info(f"  - {reason}")

            # Determine if prediction is correct
            is_correct_prediction = (
                (is_training_reciter and final_prediction == true_label) or
                (not is_training_reciter and final_prediction == "Unknown")
            )
            logger.info(f"  Is Correct Prediction: {is_correct_prediction}")

            # Add to results
            result_entry = {
                'file_name': file_name,
                'file_duration': file_duration,
                'true_label': true_label,
                'is_training_reciter': is_training_reciter,
                'predicted_label': prediction,
                'final_prediction': final_prediction,
                'is_correct_prediction': is_correct_prediction,
                'confidence': float(reliability['top_confidence']),
                'is_reliable': bool(reliability['is_reliable']),
                'distance_ratio': float(reliability['distance_ratio']),
                'failure_reasons': reliability['failure_reasons']
            }

            all_results.append(result_entry)
            predicted_labels.append(final_prediction)

        # Calculate total testing time
        test_end_time = datetime.now()
        test_duration = (test_end_time - test_start_time).total_seconds()

        # Step 5: Generate test report
        # Save detailed results
        results_df = pd.DataFrame(all_results)
        results_file = output_dir / 'detailed_results.csv'
        results_df.to_csv(results_file, index=False)
        logger.info(f"Detailed results saved to {results_file}")

        # Generate and save confusion matrix
        confusion_matrix_file = output_dir / 'test_confusion_matrix.png'
        plot_confusion_matrix(
            true_labels,
            predicted_labels,
            classes=sorted(set(true_labels) | set(predicted_labels)),
            output_path=str(confusion_matrix_file)
        )
        logger.info(f"Confusion matrix saved to {confusion_matrix_file}")

        # Calculate statistics
        total_files = len(all_results)
        reliable_predictions = sum(r['is_reliable'] for r in all_results)
        correct_predictions = sum(r['is_correct_prediction']
                                  for r in all_results)
        training_reciters_processed = sum(
            r['is_training_reciter'] for r in all_results)
        non_training_reciters_processed = total_files - training_reciters_processed

        # Calculate new accuracy metrics
        prediction_accuracy = float(
            correct_predictions / total_files if total_files > 0 else 0)
        reliable_correct_predictions = sum(
            1 for r in all_results
            if r['is_reliable'] and r['is_correct_prediction']
        )
        reliable_prediction_accuracy = float(
            reliable_correct_predictions / reliable_predictions if reliable_predictions > 0 else 0)

        # Per-reciter statistics
        reciter_stats = {}
        for reciter in set(true_labels):
            reciter_results = [
                r for r in all_results if r['true_label'] == reciter]
            reciter_correct = sum(r['is_correct_prediction']
                                  for r in reciter_results)
            reciter_reliable = sum(r['is_reliable'] for r in reciter_results)
            is_training = reciter in training_reciters

            reciter_stats[reciter] = {
                'total_files': len(reciter_results),
                'correct_predictions': reciter_correct,
                'reliable_predictions': reciter_reliable,
                'is_training_reciter': is_training,
                'accuracy_rate': float(reciter_correct / len(reciter_results)) if len(reciter_results) > 0 else 0,
                'reliability_rate': float(reciter_reliable / len(reciter_results)) if len(reciter_results) > 0 else 0,
                'reliable_correct_predictions': sum(1 for r in reciter_results if r['is_reliable'] and r['is_correct_prediction']),
                'reliable_prediction_accuracy': float(
                    sum(1 for r in reciter_results if r['is_reliable'] and r['is_correct_prediction']) /
                    sum(1 for r in reciter_results if r['is_reliable'])
                ) if sum(1 for r in reciter_results if r['is_reliable']) > 0 else 0
            }

        # Generate summary report
        summary = {
            'test_run_id': run_id,
            'timestamp': timestamp,
            'model_path': str(model_path),
            'model_type': model_info['model_type'],
            'model_directory': str(model_dir),
            'preprocessing_path': str(preprocessed_path),
            'preprocessing_run_id': preprocess_info.get('run_id', 'unknown'),
            'total_files': int(total_files),
            'training_reciters_processed': int(training_reciters_processed),
            'non_training_reciters_processed': int(non_training_reciters_processed),
            'reliable_predictions': int(reliable_predictions),
            'correct_predictions': int(correct_predictions),
            'reliable_correct_predictions': int(reliable_correct_predictions),
            'reliability_rate': float(reliable_predictions / total_files if total_files > 0 else 0),
            'accuracy_rate': float(correct_predictions / total_files if total_files > 0 else 0),
            'prediction_accuracy': prediction_accuracy,
            'reliable_prediction_accuracy': reliable_prediction_accuracy,
            'test_duration_seconds': test_duration,
            'per_reciter_statistics': reciter_stats
        }

        # Save summary report
        summary_file = output_dir / 'summary_report.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Summary report saved to {summary_file}")

        # Create human-readable summary
        summary_text = [
            f"Test Run Summary ({run_id})",
            f"=====================================",
            f"Model: {Path(model_path).name}",
            f"Model Type: {model_info['model_type']}",
            f"Preprocessing: {preprocessed_path.name}",
            f"",
            f"Overall Statistics:",
            f"  Total files processed: {total_files}",
            f"  Training reciters processed: {training_reciters_processed}",
            f"  Non-training reciters processed: {non_training_reciters_processed}",
            f"  Reliable predictions: {reliable_predictions}",
            f"  Correct predictions: {correct_predictions}",
            f"  Reliable correct predictions: {reliable_correct_predictions}",
            f"  Reliability rate: {summary['reliability_rate']:.2%}",
            f"  Accuracy rate: {summary['accuracy_rate']:.2%}",
            f"  Prediction accuracy: {prediction_accuracy:.2%}",
            f"  Reliable prediction accuracy: {reliable_prediction_accuracy:.2%}",
            f"  Test duration: {format_duration(test_duration)}",
            f"",
            f"Per-Reciter Statistics:"
        ]

        for reciter, stats in reciter_stats.items():
            summary_text.extend([
                f"  {reciter}:",
                f"    Training Reciter: {'Yes' if stats['is_training_reciter'] else 'No'}",
                f"    Files: {stats['total_files']}",
                f"    Correct: {stats['correct_predictions']} ({stats['accuracy_rate']:.2%})",
                f"    Reliable: {stats['reliable_predictions']} ({stats['reliability_rate']:.2%})",
                f"    Reliable Correct: {stats['reliable_correct_predictions']} ({stats['reliable_prediction_accuracy']:.2%})"
            ])

        # Save text summary
        text_summary_file = output_dir / 'test_summary.txt'
        with open(text_summary_file, 'w') as f:
            f.write('\n'.join(summary_text))
        logger.info(f"Text summary saved to {text_summary_file}")

        # Check for actual GPU availability
        gpu_available = is_gpu_available()
        logger.info(f"GPU availability: {'Yes' if gpu_available else 'No'}")

        # Create test metadata file with links to model and preprocessing
        test_metadata = {
            'test_run_id': run_id,
            'timestamp': timestamp,
            'model': {
                'path': str(model_path),
                'directory': str(model_dir),
                'type': model_info['model_type'],
                'n_classes': model_info['n_classes'],
                'classes': list(model.classes_)
            },
            'preprocessing': {
                'path': str(preprocessed_path),
                'run_id': preprocess_info.get('run_id', 'unknown')
            },
            'test_config': {
                'output_directory': str(output_dir),
                'log_file': str(log_file)
            },
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'use_gpu_config': USE_GPU,
                'gpu_available': gpu_available
            },
            'test_duration_seconds': test_duration
        }

        metadata_file = output_dir / 'test_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f, indent=4)
        logger.info(f"Test metadata saved to {metadata_file}")

        # Log final summary
        logger.info("\nFinal Test Summary:")
        logger.info(f"Model: {Path(model_path).name}")
        logger.info(f"Model Type: {model_info['model_type']}")
        logger.info(f"Preprocessing: {preprocessed_path}")
        logger.info(f"Total files processed: {total_files}")
        logger.info(
            f"Training reciters processed: {training_reciters_processed}")
        logger.info(
            f"Non-training reciters processed: {non_training_reciters_processed}")
        logger.info(f"Reliable predictions: {reliable_predictions}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(
            f"Reliable correct predictions: {reliable_correct_predictions}")
        logger.info(f"Reliability rate: {summary['reliability_rate']:.2%}")
        logger.info(f"Accuracy rate: {summary['accuracy_rate']:.2%}")
        logger.info(f"Prediction accuracy: {prediction_accuracy:.2%}")
        logger.info(
            f"Reliable prediction accuracy: {reliable_prediction_accuracy:.2%}")
        logger.info(f"Test duration: {format_duration(test_duration)}")
        logger.info(f"\nDetailed results saved to: {output_dir}")

        return 0

    except KeyboardInterrupt:
        logger.info("\nTesting process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Stack trace:")
        return 1
