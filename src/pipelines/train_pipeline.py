"""
Enhanced training pipeline for Quran reciter identification.
"""
import os
import numpy as np
import pandas as pd
import time
import json
import platform
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging
import warnings
import shutil
from config import *
from src.models import create_model
from src.evaluation import plot_confusion_matrix, plot_feature_importance
from src.utils import setup_logging, format_duration
from src.utils import is_gpu_available

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def train_pipeline(model_type=None, preprocess_dir=None):
    """
    Run the enhanced training pipeline with improved organization and reporting.

    Args:
        model_type (str, optional): Type of model to train ('random_forest', 'blstm')
                                  If None, uses the model type from config.py
        preprocess_dir (str, optional): Path to preprocessed data directory
                                       If None, uses the latest preprocessed data

    Returns:
        int: 0 for success, 1 for failure
        str: Path to the saved model
    """
    try:
        total_start_time = time.time()

        # Generate timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_train"

        # Setup logging
        logger, log_file = setup_logging(f"training_{run_id}")
        logger.info("Starting enhanced training pipeline")
        logger.info(f"Run ID: {run_id}")

        if model_type:
            logger.info(f"Using model type: {model_type}")
        else:
            logger.info(f"Using default model type from config: {MODEL_TYPE}")

        # Create necessary directories
        for directory in [LOGS_DIR, MODEL_OUTPUT_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        # Determine preprocessing directory
        processed_base_dir = Path("processed/train")

        if preprocess_dir:
            # Use the specified preprocessing directory
            preprocessed_path = Path(preprocess_dir)
            logger.info(
                f"Using specified preprocessing directory: {preprocessed_path}")
        else:
            # Use the latest preprocessing directory or symbolic link
            latest_link = processed_base_dir / "latest"
            if latest_link.exists():
                if latest_link.is_symlink():
                    target = latest_link.resolve()
                    preprocessed_path = processed_base_dir / target
                else:
                    preprocessed_path = latest_link
                logger.info(
                    f"Using latest preprocessing directory: {preprocessed_path}")
            else:
                # Find the most recent directory
                preprocess_dirs = [d for d in processed_base_dir.iterdir(
                ) if d.is_dir() and d.name.endswith("_preprocess")]
                if not preprocess_dirs:
                    logger.error(
                        f"No preprocessing directories found in {processed_base_dir}")
                    logger.error("Please run the preprocessing pipeline first")
                    return 1, None

                preprocessed_path = max(
                    preprocess_dirs, key=lambda d: d.stat().st_mtime)
                logger.info(
                    f"Using most recent preprocessing directory: {preprocessed_path}")

        # Check if preprocessed data exists in the directory
        features_file = preprocessed_path / "all_features.npy"
        metadata_file = preprocessed_path / "all_metadata.csv"
        preprocess_metadata_file = preprocessed_path / "preprocessing_metadata.json"

        if not features_file.exists() or not metadata_file.exists():
            logger.error(
                f"Preprocessed data files not found in {preprocessed_path}")
            logger.error("Please run the preprocessing pipeline first")
            return 1, None

        # Load preprocessing metadata to reference in training
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

        # Load preprocessed data
        logger.info(f"Loading preprocessed features from {features_file}")
        features = np.load(features_file)

        logger.info(f"Loading metadata from {metadata_file}")
        metadata = pd.read_csv(metadata_file)

        # Extract labels from metadata
        labels = metadata['reciter'].values

        logger.info(
            f"Loaded {len(features)} samples with {features.shape[1]} features")
        logger.info(f"Found {len(np.unique(labels))} unique reciters")

        # Check for actual GPU availability
        gpu_available = is_gpu_available()
        logger.info(f"GPU availability: {'Yes' if gpu_available else 'No'}")

        # Create training metadata
        training_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type if model_type else MODEL_TYPE,
            "preprocessing_info": {
                "directory": str(preprocessed_path),
                "run_id": preprocess_info.get("run_id", "unknown"),
                "timestamp": preprocess_info.get("timestamp", "unknown"),
                "sample_count": len(features)
            },
            "config": {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "model_specific": {}
            },
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "use_gpu_config": USE_GPU,
                "gpu_available": gpu_available
            }
        }

        # Split data
        X = features
        y = labels

        if len(np.unique(y)) < 2:
            logger.error("Not enough classes for training")
            return 1, None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # For BLSTM model, log the approach we're using
        if model_type and model_type.lower() == 'blstm':
            logger.info("Using BLSTM model:")
            logger.info(f"- Using first {BLSTM_MFCC_COUNT} MFCCs as features")
            logger.info(
                f"- Window size: {WINDOW_SIZE_MS}ms, Step size: {STEP_SIZE_MS}ms")
            logger.info(
                f"- LSTM units: {LSTM_UNITS}, Dense units: {DENSE_UNITS}")
            logger.info(
                f"- Dropout rate: {DROPOUT_RATE}, Learning rate: {LEARNING_RATE}")

            # Add BLSTM-specific parameters to metadata
            training_metadata["config"]["model_specific"] = {
                "blstm_mfcc_count": BLSTM_MFCC_COUNT,
                "lstm_units": LSTM_UNITS,
                "dropout_rate": DROPOUT_RATE,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS
            }
        elif not model_type or model_type.lower() == 'random_forest':
            # Add Random Forest-specific parameters to metadata
            training_metadata["config"]["model_specific"] = {
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
                "n_folds": N_FOLDS
            }

        # Model training start time
        model_train_start = time.time()

        # Create and train model
        model = create_model(model_type)
        model.train(X_train, y_train)

        # Calculate model training time
        model_train_time = time.time() - model_train_start

        # Evaluate model
        predictions = model.predict(X_test)
        metrics = model.evaluate(X_test, y_test)
        logger.info("\nModel Performance Report:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info("\n" + str(metrics['classification_report']))

        # Update training metadata with performance info
        training_metadata["performance"] = {
            "accuracy": float(metrics['accuracy']),
            "training_time": float(model_train_time),
            # This might need conversion for JSON
            "class_report": metrics['classification_report']
        }

        # Create timestamped output directories
        model_dir = Path(MODEL_OUTPUT_DIR) / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        viz_dir = model_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Generate and save confusion matrix
        confusion_matrix_path = viz_dir / 'confusion_matrix.png'
        plot_confusion_matrix(
            y_test,
            predictions,
            classes=model.classes_,
            output_path=str(confusion_matrix_path)
        )
        logger.info(f"Confusion matrix saved to {confusion_matrix_path}")

        # Plot feature importance if available (only for Random Forest model)
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            try:
                feature_importance_path = viz_dir / 'feature_importance.png'
                plot_feature_importance(
                    model,
                    [f"Feature_{i}" for i in range(X.shape[1])],
                    str(feature_importance_path)
                )
                logger.info(
                    f"Feature importance plot saved to {feature_importance_path}")
            except Exception as e:
                logger.warning(f"Could not plot feature importance: {str(e)}")

        # Save model
        model_type_str = model_type if model_type else MODEL_TYPE
        model_filename = model_dir / f'model_{model_type_str}.joblib'
        model.save(model_filename)
        logger.info(f'Saved model to {model_filename}')

        # Calculate total training time
        total_training_time = time.time() - total_start_time
        training_metadata["total_time"] = float(total_training_time)

        # Save training metadata
        metadata_path = model_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            # Convert any non-serializable objects
            json.dump(convert_to_json_serializable(
                training_metadata), f, indent=4)
        logger.info(f"Training metadata saved to {metadata_path}")

        # Add training summary
        training_summary = [
            "\nTraining Summary:",
            f"Run ID: {run_id}",
            f"Model Type: {model_type_str}",
            f"Preprocessing Directory: {preprocessed_path}",
            f"Preprocessing Run ID: {preprocess_info.get('run_id', 'unknown')}",
            f"Total Samples: {len(features)}",
            f"Training Samples: {len(X_train)}",
            f"Testing Samples: {len(X_test)}",
            f"Number of Features: {X.shape[1]}",
            f"Number of Classes: {len(np.unique(y))}",
            f"Model Training Time: {format_duration(model_train_time)}",
            f"Total Processing Time: {format_duration(total_training_time)}",
            f"Test Accuracy: {metrics['accuracy']:.4f}"
        ]

        # For BLSTM, add specific information to the summary
        if model_type_str.lower() == 'blstm':
            blstm_summary = [
                "\nBLSTM Model Details:",
                f"Feature Count: First {BLSTM_MFCC_COUNT} MFCCs",
                f"LSTM Units: {LSTM_UNITS}",
                f"Dropout Rate: {DROPOUT_RATE}",
                f"Learning Rate: {LEARNING_RATE}"
            ]
            training_summary.extend(blstm_summary)

        # Save training summary
        summary_path = model_dir / 'training_summary.txt'
        with open(summary_path, "w") as summary_file:
            summary_file.write("\n".join(training_summary))
        logger.info(f"Training summary saved to {summary_path}")

        # Create a symbolic link or copy to 'latest' for convenience
        latest_dir = Path(MODEL_OUTPUT_DIR) / "latest"
        if latest_dir.exists():
            if latest_dir.is_symlink():
                latest_dir.unlink()
            else:
                shutil.rmtree(latest_dir)

        try:
            # Try to create symbolic link first
            latest_dir.symlink_to(model_dir.name)
            logger.info(
                f"Created symbolic link to latest training run: {latest_dir}")
        except (OSError, NotImplementedError):
            # If symlink fails (e.g., on Windows), copy the directory
            shutil.copytree(model_dir, latest_dir)
            logger.info(f"Created copy of latest training run: {latest_dir}")

        # Log duration summary to console
        logger.info("\nTraining Complete!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {model_dir}")
        logger.info(
            f"Model Training Time: {format_duration(model_train_time)}")
        logger.info(
            f"Total Processing Time: {format_duration(total_training_time)}")

        return 0, str(model_filename)

    except KeyboardInterrupt:
        logger.info("\nTraining process interrupted by user")
        return 1, None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Stack trace:")
        return 1, None


def convert_to_json_serializable(obj):
    """
    Convert non-serializable objects to JSON serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj
