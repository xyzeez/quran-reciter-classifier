"""
Base model interface for Quran Reciter Classifier.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class that all our reciter models should inherit from.
    This enforces a common interface for different model implementations.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model on the provided dataset.

        X_train: Features extracted from audio samples
        y_train: Corresponding reciter labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Run inference on new audio samples.

        X: Extracted features from new audio
        Returns the most likely reciter label for each sample
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Get confidence scores for each possible reciter.

        X: Features from audio sample(s)
        Returns probability distribution across all reciters
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Check how well the model performs on test data.

        X_test: Test set features
        y_test: Ground truth labels

        Returns various metrics like accuracy, F1, etc.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save the trained model to disk.

        filepath: Where to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath):
        """
        Restore a model from a saved file.

        filepath: Location of the saved model

        Returns a ready-to-use model instance
        """
        pass

    @abstractmethod
    def get_model_info(self):
        """
        Return a summary of this model's configuration.

        Useful for logging and comparing different models.
        """
        pass
