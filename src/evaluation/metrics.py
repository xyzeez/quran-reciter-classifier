import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Dictionary containing classification metrics
    """
    try:
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate additional metrics
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'correct_predictions': correct,
            'total_predictions': total
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return None


def save_metrics(metrics, output_path):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary of metrics
        output_path (str): Path to save the metrics
    """
    try:
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")


def log_metrics_summary(metrics):
    """
    Log a summary of metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
    """
    try:
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['total_predictions']}")
        
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            logger.info("\nClassification Report:")
            for label, values in report.items():
                if label in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                if isinstance(values, dict):
                    precision = values.get('precision', 0)
                    recall = values.get('recall', 0)
                    f1 = values.get('f1-score', 0)
                    support = values.get('support', 0)
                    logger.info(f"  {label}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
    except Exception as e:
        logger.error(f"Error logging metrics summary: {str(e)}")