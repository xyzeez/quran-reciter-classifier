"""
Prediction pipeline for Quran reciter identification.
"""
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment
import logging
import warnings
import json

from config import *
from src.data import preprocess_audio_with_logic, preprocess_audio
from src.features import extract_features
from src.models import load_model
from src.utils import calculate_distances, analyze_prediction_reliability, setup_logging
from src.evaluation import plot_prediction_analysis

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def predict_pipeline(model_path=None, audio_path=None, true_label=None):
    """
    Run the prediction pipeline.

    Args:
        model_path (str, optional): Path to model file. If None, uses the latest model.
        audio_path (str, optional): Path to audio file. Required.
        true_label (str, optional): True reciter name for verification. Optional.

    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Setup logging
        logger, log_file = setup_logging("prediction")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Starting prediction pipeline")

        # Check if model_path is provided
        if model_path is None:
            logger.error('No model specified. Please provide a model path.')
            return 1

        # Check if audio_path is provided
        if audio_path is None:
            logger.error('No audio file specified. Please provide an audio file path.')
            return 1

        # Step 1: Load the model
        try:
            logger.info(f'Loading model from: {model_path}')
            model = load_model(model_path)
            logger.info('Model loaded successfully')

            # Log model info
            model_info = model.get_model_info()
            logger.info(f"Model type: {model_info['model_type']}")

            # For BLSTM implementation, log additional info
            if model_info['model_type'] == 'BLSTM':
                logger.info(f"Using first {model_info.get('input_size', BLSTM_MFCC_COUNT)} MFCCs as features")
        except Exception as e:
            logger.error(f'Error loading model: {str(e)}')
            return 1

        # Step 2: Verify input file
        input_path = Path(audio_path)
        if not input_path.is_file():
            logger.error(f'Invalid audio file: {audio_path}')
            return 1
        logger.info(f'Processing audio file: {input_path}')

        # Step 3: Process true label (if provided)
        is_correct = None
        if true_label:
            logger.info(f'Provided true label: {true_label}')
            unknown_variants = ['unknown', 'unknown reciter', 'unk', 'unknown reader']
            if true_label.lower() in unknown_variants:
                true_label = "Unknown"  # Standardize unknown label
            elif true_label not in model.classes_:
                logger.warning(f"Warning: Provided reciter name '{true_label}' is not in model's classes")
                logger.info("Known classes: " + ", ".join(model.classes_))

        # Step 4: Process audio
        audio_data = None
        sr = None

        try:
            audio_data, sr = preprocess_audio_with_logic(str(input_path))
        except Exception as e:
            logger.warning(f"Direct loading failed: {str(e)}")
            try:
                logger.info('Converting MP3 to WAV...')
                audio = AudioSegment.from_mp3(input_path)
                wav_path = input_path.with_suffix('.wav')
                audio.export(wav_path, format='wav')
                logger.info(f'Converted to {wav_path.name}')

                audio_data, sr = preprocess_audio_with_logic(str(wav_path))
                wav_path.unlink()
            except Exception as e:
                logger.error(f"Error during MP3 conversion: {str(e)}")
                return 1

        if audio_data is None:
            logger.error("Error during preprocessing")
            return 1

        # Continue preprocessing
        audio_data, sr = preprocess_audio(audio_data, sr)
        if audio_data is None:
            logger.error("Error during additional preprocessing")
            return 1

        # Step 5: Extract features
        logger.info('Extracting features...')
        features = extract_features(audio_data, sr)
        if features is None:
            logger.error("Error during feature extraction")
            return 1
        features = np.array(features).reshape(1, -1)

        # Step 6: Make predictions and analyze
        logger.info('Predicting...')
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Get model package components
        model_info = model.get_model_info()
        centroids = model.centroids
        thresholds = model.thresholds

        # Calculate distances
        distances = calculate_distances(features, centroids)

        # Analyze reliability
        reliability = analyze_prediction_reliability(
            probabilities, distances, thresholds, prediction)

        # Display results
        logger.info("\nPrediction Results:")
        if not reliability['is_reliable']:
            predicted_reciter = "Unknown"
            logger.info("Predicted Reciter: Unknown")
            logger.info("Reasons for uncertainty:")
            for reason in reliability['failure_reasons']:
                logger.info(f"- {reason}")
        else:
            predicted_reciter = prediction
            logger.info(f"Predicted Reciter: {prediction}")
            logger.info(
                f"Confidence Score: {reliability['top_confidence']:.2%}")
            logger.info(f"Distance Ratio: {reliability['distance_ratio']:.2%}")

        # Show prediction verification only if true label was provided
        if true_label:
            # Handle case where both prediction and truth are "Unknown"
            if true_label == "Unknown" and predicted_reciter == "Unknown":
                is_correct = True
            else:
                is_correct = predicted_reciter == true_label
            logger.info("\nPrediction Verification:")
            logger.info(f"Entered Reciter: {true_label}")
            logger.info(f"Correct Prediction: {'Yes' if is_correct else 'No'}")

        # Create results directory
        results_dir = Path(LOGS_DIR) / f'prediction_results_{timestamp}'
        results_dir.mkdir(exist_ok=True, parents=True)

        # Generate visualization
        plot_path = results_dir / 'prediction_analysis.png'
        plot_prediction_analysis(
            probabilities,
            distances,
            thresholds,
            np.array(model.classes_),
            true_label,
            prediction if reliability['is_reliable'] else None,
            str(plot_path)
        )
        logger.info(f"\nAnalysis plot saved to {plot_path}")

        # Display detailed analysis
        logger.info("\nDetailed Analysis:")
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            reciter = model.classes_[idx]
            prob = probabilities[idx]
            dist_ratio = distances[reciter][0] / thresholds[reciter]

            label = []
            if true_label and reciter == true_label:
                label.append("(True Label)")
            if reliability['is_reliable'] and reciter == prediction:
                label.append("(Predicted)")

            label_str = " ".join(label) if label else ""

            logger.info(
                f"{reciter}: Confidence={prob:.2%}, Distance Ratio={dist_ratio:.2f} {label_str}")

        # Create a summary of results
        # Convert NumPy types to Python native types to ensure JSON serializability
        result_summary = {
            "timestamp": timestamp,
            "audio_file": str(input_path),
            "model": str(model_path),
            "model_type": model_info['model_type'],
            "predicted_reciter": predicted_reciter,
            "is_reliable": bool(reliability['is_reliable']),  # Convert to Python bool
            "confidence": float(reliability['top_confidence']),  # Convert to Python float
            "distance_ratio": float(reliability['distance_ratio']),  # Convert to Python float
            "true_label": true_label,
            "is_correct": bool(is_correct) if is_correct is not None else None,  # Convert to Python bool or None
            "analysis_plot": str(plot_path)
        }

        # Save result summary to a JSON file
        summary_path = results_dir / 'prediction_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(result_summary, f, indent=4)
            logger.info(f"\nPrediction summary saved to {summary_path}")
        except TypeError as e:
            logger.error(f"Error serializing prediction summary: {str(e)}")
            logger.debug(f"Result summary content: {result_summary}")

        logger.info(f"\nAll prediction results saved to: {results_dir}")
        return 0

    except KeyboardInterrupt:
        logger.info("\nPrediction process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Stack trace:")
        return 1