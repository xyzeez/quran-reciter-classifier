"""
Improved Bidirectional LSTM model implementation for Quran reciter identification.
This version includes fixes for overfitting and better generalization.
"""
import os
import numpy as np
import logging
import joblib
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from src.models.base_model import BaseModel
from src.utils.distance_utils import calculate_centroids, calculate_intra_class_thresholds
from config import *

logger = logging.getLogger(__name__)


class SimpleAttentionLayer(nn.Module):
    """
    Simpler attention mechanism to focus on important parts of sequences.
    """

    def __init__(self, hidden_size):
        super(SimpleAttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size*2)
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1), dim=1
        )  # (batch, seq_len)

        # Apply attention weights to LSTM output
        context = torch.bmm(
            attention_weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_size*2)

        return context, attention_weights


class BLSTMNetwork(nn.Module):
    """
    Bidirectional LSTM network with attention mechanism.
    """

    def __init__(self, input_size, hidden_size, num_classes, dropout=None):
        """
        Initialize the BLSTM network.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_classes: Number of output classes
            dropout: Dropout rate (if None, uses BLSTM_DROPOUT_RATE from config)
        """
        super(BLSTMNetwork, self).__init__()

        # Use config dropout if not provided
        if dropout is None:
            dropout = BLSTM_DROPOUT_RATE

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        self.dropout = nn.Dropout(dropout)
        self.attention = SimpleAttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # LSTM layer
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Apply attention
        context, _ = self.attention(lstm_out)

        # Apply dropout again
        context = self.dropout(context)

        # Final fully connected layer
        x = self.fc(context)

        return x


class BLSTMModel(BaseModel):
    """
    Improved Bidirectional LSTM model for reciter identification.

    This implementation focuses on preventing overfitting and
    improving generalization to unseen data.
    """

    def __init__(self):
        """Initialize the BLSTM model with default parameters."""
        self.model = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
        self.label_encoder = LabelEncoder()
        self.centroids = None
        self.thresholds = None
        self.classes_ = None
        self.training_info = {}
        self.feature_importances_ = None  # Kept for compatibility

        # BLSTM specific parameters (using config values)
        self.lstm_units = LSTM_UNITS
        # Using enhanced dropout rate from config
        self.dropout_rate = BLSTM_DROPOUT_RATE
        # Using enhanced learning rate from config
        self.learning_rate = BLSTM_LEARNING_RATE
        self.weight_decay = BLSTM_WEIGHT_DECAY  # Using weight decay from config
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.patience = EARLY_STOPPING_PATIENCE

        # Data augmentation parameters
        self.use_augmentation = True
        self.noise_level = BLSTM_NOISE_LEVEL  # Using noise level from config
        # Using feature dropout rate from config
        self.feature_dropout_rate = BLSTM_FEATURE_DROPOUT_RATE

        # Sequence parameters
        self.sequence_length = BLSTM_SEQUENCE_LENGTH  # Using sequence length from config
        self.num_features = BLSTM_MFCC_COUNT

    def _extract_mfcc_features(self, X):
        """
        Extract only the first N MFCCs from the feature matrix.

        Args:
            X: Input features array or DataFrame

        Returns:
            Numpy array with only the first N features (MFCCs)
        """
        # Convert DataFrame to numpy array if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        # Extract just the first N features (assuming these are the MFCCs)
        # Note: This assumes your feature extraction puts MFCCs first in the feature vector
        X_mfccs = X_array[:, :self.num_features]

        return X_mfccs

    def _create_sequences(self, X, sequence_length=None):
        """
        Create fixed-length sequences from the input features.

        Args:
            X: Input features with shape (samples, features)
            sequence_length: Length of sequences to create

        Returns:
            Numpy array with shape (samples, sequence_length, features)
        """
        if sequence_length is None:
            sequence_length = self.sequence_length

        # Get dimensions
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Create placeholder for sequences
        X_sequences = np.zeros((n_samples, sequence_length, n_features))

        # Fill sequences
        for i in range(n_samples):
            # For simplicity, just repeat the feature vector
            # This approach treats each sample as a standalone representation
            # rather than trying to create artificial time sequences
            for j in range(sequence_length):
                X_sequences[i, j, :] = X[i, :]

        return X_sequences

    def _augment_data(self, X, y):
        """
        Augment training data to improve generalization.

        Args:
            X: Features array with shape (samples, features)
            y: Labels array

        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        if not self.use_augmentation:
            return X, y

        augmented_features = []
        augmented_labels = []

        for i in range(len(X)):
            # Original sample
            augmented_features.append(X[i])
            augmented_labels.append(y[i])

            # Add noise
            noisy_sample = X[i] + \
                np.random.normal(0, self.noise_level, X[i].shape)
            augmented_features.append(noisy_sample)
            augmented_labels.append(y[i])

            # Feature dropout (randomly zero out some features)
            dropout_mask = np.random.binomial(
                1, 1 - self.feature_dropout_rate, X[i].shape)
            dropout_sample = X[i] * dropout_mask
            augmented_features.append(dropout_sample)
            augmented_labels.append(y[i])

        return np.array(augmented_features), np.array(augmented_labels)

    def train(self, X_train, y_train):
        """
        Train the BLSTM model with improved generalization.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            self: Trained model
        """
        logger.info(f"Training BLSTM model on {self.device}")
        train_start_time = time.time()

        # Record training start time
        start_time = datetime.now()

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        num_classes = len(self.classes_)

        # Extract MFCC features
        X_mfccs = self._extract_mfcc_features(X_train)

        # Augment training data
        if self.use_augmentation:
            X_augmented, y_augmented = self._augment_data(X_mfccs, y_encoded)
            logger.info(
                f"Augmented data: {X_augmented.shape[0]} samples (original: {X_mfccs.shape[0]})")
        else:
            X_augmented, y_augmented = X_mfccs, y_encoded

        # Split into training and validation sets
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_augmented, y_augmented, test_size=TEST_SIZE, stratify=y_augmented, random_state=RANDOM_STATE
        )

        logger.info(
            f"Training on {len(X_train_split)} samples, validating on {len(X_val)} samples")

        # Create sequences
        X_train_seq = self._create_sequences(X_train_split)
        X_val_seq = self._create_sequences(X_val)

        # Calculate centroids and thresholds on original features
        # We'll still use this for compatibility with the existing reliability metrics
        logger.info("Calculating centroids and distance thresholds...")
        self.centroids = calculate_centroids(X_train, y_train)
        self.thresholds = calculate_intra_class_thresholds(
            X_train, y_train, self.centroids)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.LongTensor(y_train_split)
        X_val_tensor = torch.FloatTensor(X_val_seq)
        y_val_tensor = torch.LongTensor(y_val)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, len(val_dataset)),
            shuffle=False,
            drop_last=False
        )

        # Create model with simplified BLSTM architecture
        self.model = BLSTMNetwork(
            input_size=self.num_features,
            hidden_size=self.lstm_units,
            num_classes=num_classes,
            dropout=self.dropout_rate
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total")

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=BLSTM_LR_SCHEDULER_FACTOR,
            patience=BLSTM_LR_SCHEDULER_PATIENCE,
            verbose=True,
            min_lr=BLSTM_MIN_LR
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': [],
                            'train_acc': [], 'val_acc': [], 'lr': []}

        logger.info(f"Starting training for up to {self.epochs} epochs...")

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=BLSTM_GRADIENT_CLIP_NORM)
                optimizer.step()

                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # Current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Calculate epoch time
            epoch_time = time.time() - epoch_start

            # Save metrics
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            training_history['lr'].append(current_lr)

            logger.info(f'Epoch {epoch + 1}/{self.epochs} [{epoch_time:.1f}s] | '
                        f'Train Loss: {train_loss:.4f} | '
                        f'Train Acc: {train_acc:.4f} | '
                        f'Val Loss: {val_loss:.4f} | '
                        f'Val Acc: {val_acc:.4f} | '
                        f'LR: {current_lr:.6f}')

            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict()
                logger.info(
                    f"Saved new best model with val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch + 1}')
                    break

        # Load the best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model from training")

        # Record training end time
        end_time = datetime.now()
        total_train_time = time.time() - train_start_time

        # Store training information
        self.training_info = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_classes': len(self.classes_),
            'classes': list(self.classes_),
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'training_duration': total_train_time,
            'epochs_completed': epoch + 1,
            'best_val_accuracy': float(best_val_acc),
            'final_train_accuracy': training_history['train_acc'][-1],
            'final_val_accuracy': training_history['val_acc'][-1],
            'best_val_loss': float(best_val_loss),
            'history': {
                'train_loss': [float(x) for x in training_history['train_loss']],
                'val_loss': [float(x) for x in training_history['val_loss']],
                'train_acc': [float(x) for x in training_history['train_acc']],
                'val_acc': [float(x) for x in training_history['val_acc']],
                'lr': [float(x) for x in training_history['lr']]
            },
            'model_parameters': {
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'sequence_length': self.sequence_length,
                'num_features': self.num_features,
                'use_augmentation': self.use_augmentation
            }
        }

        logger.info(f"Training completed in {total_train_time:.2f} seconds")
        return self

    def predict(self, X):
        """
        Make predictions using the trained BLSTM model.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Extract MFCC features
        X_mfccs = self._extract_mfcc_features(X)

        # Create sequences
        X_sequences = self._create_sequences(X_mfccs)

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, y_pred_indices = torch.max(outputs, 1)

        # Convert indices back to original labels
        y_pred = self.label_encoder.inverse_transform(
            y_pred_indices.cpu().numpy())

        return y_pred

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X: Input features

        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Extract MFCC features
        X_mfccs = self._extract_mfcc_features(X)

        # Create sequences
        X_sequences = self._create_sequences(X_mfccs)

        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)

        # Get probabilities
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()

        # Ensure the probabilities match the expected order of self.classes_
        # This is crucial for the reliability analysis to work correctly
        reordered_proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, label in enumerate(self.classes_):
            idx = np.where(self.label_encoder.classes_ == label)[0][0]
            reordered_proba[:, i] = probabilities[:, idx]

        return reordered_proba

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Get predictions
        y_pred = self.predict(X_test)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()

        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'test_samples': len(X_test)
        }

        return metrics

    def save(self, filepath):
        """
        Save the model.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Create a directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the PyTorch model
        torch_model_path = f"{filepath}_torch_model.pt"
        torch.save(self.model.state_dict(), torch_model_path)

        # Create model package
        model_package = {
            'torch_model_path': torch_model_path,
            'model_architecture': {
                'input_size': self.num_features,
                'hidden_size': self.lstm_units,
                'num_classes': len(self.classes_),
                'dropout': self.dropout_rate
            },
            'centroids': self.centroids,
            'thresholds': self.thresholds,
            'training_info': self.training_info,
            'label_encoder_classes': self.label_encoder.classes_,
            'sequence_length': self.sequence_length,
            'num_features': self.num_features
        }

        # Save model package using joblib
        joblib.dump(model_package, filepath)
        logger.info(f"BLSTM model saved to {filepath}")
        logger.info(f"PyTorch model saved to {torch_model_path}")

    @classmethod
    def load(cls, filepath):
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            # Load model package
            model_package = joblib.load(filepath)

            # Extract components
            torch_model_path = model_package['torch_model_path']
            model_architecture = model_package['model_architecture']
            centroids = model_package['centroids']
            thresholds = model_package['thresholds']
            training_info = model_package['training_info']
            label_encoder_classes = model_package['label_encoder_classes']
            sequence_length = model_package.get('sequence_length', 32)
            num_features = model_package.get('num_features', BLSTM_MFCC_COUNT)

            # Create new instance
            instance = cls()
            instance.device = torch.device(
                "cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

            # Set sequence parameters
            instance.sequence_length = sequence_length
            instance.num_features = num_features

            # Create model with same architecture
            instance.model = BLSTMNetwork(
                input_size=model_architecture['input_size'],
                hidden_size=model_architecture['hidden_size'],
                num_classes=model_architecture['num_classes'],
                dropout=model_architecture['dropout']
            ).to(instance.device)

            # Load model state
            state_dict = torch.load(
                torch_model_path,
                map_location=instance.device
            )
            instance.model.load_state_dict(state_dict)
            instance.model.eval()

            # Set model attributes
            instance.centroids = centroids
            instance.thresholds = thresholds
            instance.training_info = training_info
            instance.classes_ = label_encoder_classes

            # Set up label encoder
            instance.label_encoder = LabelEncoder()
            instance.label_encoder.classes_ = label_encoder_classes

            logger.info(f"BLSTM model loaded from {filepath}")
            return instance

        except Exception as e:
            logger.error(f"Error loading BLSTM model: {str(e)}")
            raise

    def get_model_info(self):
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'BLSTM',
            'training_info': self.training_info,
            'n_classes': len(self.classes_) if self.classes_ is not None else 0,
            'classes': list(self.classes_) if self.classes_ is not None else [],
            'has_centroids': self.centroids is not None,
            'has_thresholds': self.thresholds is not None,
            'sequence_length': self.sequence_length,
            'num_features': self.num_features
        }
