"""
Random Forest classifier for Quran reciter identification.
Uses calibrated probabilities and distance-based verification.
"""
import joblib
import os
from datetime import datetime
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report
from src.models.base_model import BaseModel
from src.utils.distance_utils import calculate_centroids, calculate_intra_class_thresholds
from config import *

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier with probability calibration.
    Uses feature space distances for prediction verification.
    """

    def __init__(self):
        """Initialize model with configuration parameters."""
        self.rf_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        self.calibrated_model = None
        self.centroids = None
        self.thresholds = None
        self.classes_ = None
        self.feature_importances_ = None
        self.training_info = {}

    def train(self, X_train, y_train):
        """
        Train and calibrate the model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            self: Trained model instance
        """
        logger.info("Training Random Forest model...")
        start_time = datetime.now()

        # Validate model performance
        logger.info("Performing cross-validation...")
        kfold = KFold(n_splits=N_FOLDS, shuffle=True,
                      random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            self.rf_model, X_train, y_train, cv=kfold, n_jobs=-1)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(
            f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train base model
        self.rf_model.fit(X_train, y_train)

        # Calibrate prediction probabilities
        logger.info("Calibrating probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.rf_model, method='sigmoid', cv='prefit')
        self.calibrated_model.fit(X_train, y_train)

        # Store model attributes
        self.classes_ = self.calibrated_model.classes_
        self.feature_importances_ = self.rf_model.feature_importances_

        # Calculate verification metrics
        logger.info("Calculating centroids and distance thresholds...")
        self.centroids = calculate_centroids(X_train, y_train)
        self.thresholds = calculate_intra_class_thresholds(
            X_train, y_train, self.centroids)

        end_time = datetime.now()

        # Record training metadata
        self.training_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_classes': len(self.classes_),
            'classes': list(self.classes_),
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_duration': (end_time - start_time).total_seconds(),
            'cross_validation_mean': float(cv_scores.mean()),
            'cross_validation_std': float(cv_scores.std()),
            'model_parameters': {
                'n_estimators': N_ESTIMATORS,
                'max_depth': MAX_DEPTH,
                'random_state': RANDOM_STATE
            }
        }

        logger.info(
            f"Training completed in {(end_time - start_time).total_seconds():.2f} seconds")
        return self

    def predict(self, X):
        """
        Predict reciter labels for new samples.

        Args:
            X: Audio features matrix

        Returns:
            Predicted reciter labels
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.calibrated_model.predict(X)

    def predict_proba(self, X):
        """
        Get calibrated prediction probabilities.

        Args:
            X: Audio features matrix

        Returns:
            Probability distribution over reciters
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.calibrated_model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test features matrix
            y_test: True reciter labels

        Returns:
            Dict of performance metrics
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet.")

        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            'accuracy': report['accuracy'],
            'classification_report': report,
            'test_samples': len(X_test)
        }

    def save(self, filepath):
        """
        Save model to disk.

        Args:
            filepath: Output path
        """
        if self.calibrated_model is None:
            raise ValueError("Model has not been trained yet.")

        # Package model data
        model_package = {
            'model': self.calibrated_model,
            'centroids': self.centroids,
            'thresholds': self.thresholds,
            'training_info': self.training_info,
            'feature_importances': self.feature_importances_
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model package
        model_package = joblib.load(filepath)

        # Create new instance
        instance = cls()
        instance.calibrated_model = model_package['model']
        instance.centroids = model_package['centroids']
        instance.thresholds = model_package['thresholds']
        instance.training_info = model_package['training_info']
        instance.classes_ = instance.calibrated_model.classes_

        # Load feature importances if available
        if 'feature_importances' in model_package:
            instance.feature_importances_ = model_package['feature_importances']
        else:
            # Try to get from the base classifier if available
            try:
                instance.feature_importances_ = instance.calibrated_model.base_estimator.feature_importances_
            except AttributeError:
                instance.feature_importances_ = None

        logger.info(f"Model loaded from {filepath}")
        return instance

    def get_model_info(self):
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'RandomForest',
            'training_info': self.training_info,
            'n_classes': len(self.classes_) if self.classes_ is not None else 0,
            'classes': list(self.classes_) if self.classes_ is not None else [],
            'has_centroids': self.centroids is not None,
            'has_thresholds': self.thresholds is not None,
        }
