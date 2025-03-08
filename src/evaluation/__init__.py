"""
Evaluation module for Quran reciter identification project.
"""

from src.evaluation.metrics import calculate_metrics, save_metrics, log_metrics_summary
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_prediction_analysis
)
