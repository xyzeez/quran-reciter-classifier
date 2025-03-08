import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path
import logging
from config import *

logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (array-like): Class labels
        output_path (str): Path to save the plot
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")


def plot_feature_importance(model, feature_names, output_path):
    """
    Plot and save feature importance graph.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Names of features
        output_path (str): Path to save the plot
    """
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(20), importances[indices[:20]])
        plt.xticks(range(20), [feature_names[i]
                   for i in indices[:20]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Feature importance plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")


def plot_prediction_analysis(probabilities, distances, thresholds, classes, true_label=None, prediction=None, output_path=None):
    """
    Plot confidence scores and distance ratios.
    
    Args:
        probabilities (array-like): Prediction probabilities
        distances (dict): Distances to centroids
        thresholds (dict): Distance thresholds
        classes (array-like): Class labels
        true_label (str, optional): True label if known
        prediction (str, optional): Predicted label
        output_path (str, optional): Path to save the plot
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Sort probabilities and classes
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probs = probabilities[sorted_indices]
        sorted_classes = classes[sorted_indices]

        # Plot 1: Confidence Scores
        confidence_colors = ['darkred' if p >= CONFIDENCE_THRESHOLD else 'lightgray'
                            for p in sorted_probs]

        # Update colors for true label and prediction
        if true_label is not None:
            for i, class_name in enumerate(sorted_classes):
                if class_name == true_label:
                    confidence_colors[i] = 'green'
                elif prediction and class_name == prediction and prediction != true_label:
                    confidence_colors[i] = 'orange'

        bars1 = ax1.bar(range(len(sorted_probs)),
                        sorted_probs, color=confidence_colors)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom')

        ax1.axhline(y=CONFIDENCE_THRESHOLD, color='r', linestyle='--', alpha=0.5)
        ax1.text(len(sorted_probs)-1, CONFIDENCE_THRESHOLD,
                f'Confidence Threshold ({CONFIDENCE_THRESHOLD:.2%})',
                ha='right', va='bottom', color='r')

        ax1.set_title('Confidence Scores by Reciter')
        ax1.set_xticks(range(len(sorted_classes)))
        ax1.set_xticklabels(sorted_classes, rotation=45, ha='right')
        ax1.set_ylabel('Confidence Score')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot 2: Distance Ratios
        distance_ratios = {k: distances[k][0] /
                        thresholds[k] for k in sorted_classes}
        ratio_values = [distance_ratios[k] for k in sorted_classes]
        distance_colors = ['darkred' if r <=
                        1 else 'lightgray' for r in ratio_values]

        bars2 = ax2.bar(range(len(ratio_values)),
                        ratio_values, color=distance_colors)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax2.text(len(ratio_values)-1, 1.0, 'Distance Threshold Ratio (1.0)',
                ha='right', va='bottom', color='r')

        ax2.set_title('Distance/Threshold Ratios by Reciter')
        ax2.set_xticks(range(len(sorted_classes)))
        ax2.set_xticklabels(sorted_classes, rotation=45, ha='right')
        ax2.set_ylabel('Distance/Threshold Ratio')
        ax2.set_ylim(0, max(ratio_values) * 1.1)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Add legend
        if true_label is not None:
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='green', label='True Label'),
                plt.Rectangle((0, 0), 1, 1, facecolor='darkred',
                            label='Within Threshold'),
                plt.Rectangle((0, 0), 1, 1, facecolor='lightgray',
                            label='Outside Threshold')
            ]
            if prediction and prediction != true_label:
                legend_elements.append(plt.Rectangle(
                    (0, 0), 1, 1, facecolor='orange', label='Prediction'))
            ax1.legend(handles=legend_elements)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Prediction analysis plot saved to {output_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting prediction analysis: {str(e)}")