"""
Base interface that all Quran reciter classifier models must implement.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class defining the common interface for all reciter models.
    Ensures consistent behavior across different model implementations.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train model on audio features.

        Args:
            X_train: Audio features matrix
            y_train: Reciter labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict reciter for new audio samples.

        Args:
            X: Audio features matrix
            
        Returns:
            Predicted reciter labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Get confidence scores for each reciter.

        Args:
            X: Audio features matrix
            
        Returns:
            Probability distribution over all reciters
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.

        Args:
            X_test: Test features matrix
            y_test: True reciter labels
            
        Returns:
            Dict of performance metrics (accuracy, F1, etc.)
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save model to disk.

        Args:
            filepath: Path to save model file
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath):
        """
        Load model from disk.

        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded model instance
        """
        pass

    @abstractmethod
    def get_model_info(self):
        """
        Get model configuration summary.
        
        Returns:
            Dict of model parameters and settings
        """
        pass
