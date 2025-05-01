#!/usr/bin/env python3
"""
Batch testing script for Ayah identification.
Tests multiple audio files and generates metrics and visualizations.
"""
import sys
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import logging

# Add project root to Python path when running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.utils.ayah_identification import process_batch_file

# Initialize Rich console for pretty output
console = Console()
logger = logging.getLogger(__name__)

def parse_args():
    """Process command line options for batch testing."""
    parser = argparse.ArgumentParser(
        description="Batch test Ayah identification on a directory of audio files")

    parser.add_argument(
        "--test-dir", type=str, default="data/test-ayah",
        help="Directory containing test audio files (default: data/test-ayah)")

    parser.add_argument(
        "--model-type", type=str, default="whisper",
        help="Type of model to use (default: whisper)")

    parser.add_argument(
        "--output-dir", type=str, default="test_results/ayah",
        help="Directory for test results (default: test_results/ayah)")

    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for processing (default: 32)")

    parser.add_argument(
        "--report-format", type=str, choices=['json', 'csv'], default='json',
        help="Output format for detailed results (default: json)")

    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualizations")

    return parser.parse_args()

def create_test_directory(base_dir):
    """Create timestamped test directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'metrics').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    (output_dir / 'errors').mkdir(exist_ok=True)
    
    return output_dir

def generate_visualizations(results_df, output_dir):
    """Generate visualizations from test results."""
    viz_dir = output_dir / 'visualizations'
    
    # Set style
    plt.style.use('seaborn')
    
    # 1. Accuracy by Surah
    plt.figure(figsize=(15, 6))
    surah_accuracy = results_df[results_df['error'].isna()].groupby('true_surah')['correct'].mean()
    surah_accuracy.plot(kind='bar')
    plt.title('Accuracy by Surah')
    plt.xlabel('Surah Number')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(viz_dir / 'accuracy_by_surah.png')
    plt.close()
    
    # 2. Confidence Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df[results_df['error'].isna()], x='confidence', hue='correct', bins=20)
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(viz_dir / 'confidence_distribution.png')
    plt.close()
    
    # 3. Processing Time Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df[results_df['error'].isna()], x='processing_time', bins=20)
    plt.title('Processing Time Distribution')
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(viz_dir / 'processing_time_distribution.png')
    plt.close()
    
    # 4. Error Patterns (if any errors exist)
    error_df = results_df[~results_df['error'].isna()]
    if not error_df.empty:
        plt.figure(figsize=(12, 6))
        error_counts = error_df['error'].value_counts()
        error_counts.plot(kind='bar')
        plt.title('Error Types Distribution')
        plt.xlabel('Error Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'error_distribution.png')
        plt.close()

def calculate_metrics(results: list) -> dict:
    """Calculate accuracy and other metrics from results."""
    total = len(results)
    successful = len([r for r in results if not r['error']])
    correct = len([r for r in results if not r['error'] and r['correct']])
    
    metrics = {
        'total_files': total,
        'successful': successful,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'success_rate': successful / total if total > 0 else 0,
        'avg_confidence': sum(r.get('confidence', 0) for r in results if not r['error']) / successful if successful > 0 else 0,
        'avg_time': sum(r.get('processing_time', 0) for r in results if not r['error']) / successful if successful > 0 else 0
    }
    
    return metrics

def save_results(results: list, metrics: dict, output_dir: Path):
    """Save results and metrics to files."""
    # Create output directories
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = metrics_dir / 'detailed_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved detailed results to {results_file}")

    # Save summary metrics
    metrics_file = metrics_dir / 'summary_metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved summary metrics to {metrics_file}")

    # Save results as CSV for easy analysis
    df = pd.DataFrame(results)
    csv_file = metrics_dir / 'detailed_results.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8')
    logger.info(f"Saved CSV results to {csv_file}")

def main():
    """Run batch testing of ayah identification."""
    args = parse_args()

    # Validate test directory
    test_dir = Path(args.test_dir)
    if not test_dir.exists() or not test_dir.is_dir():
        console.print(f"[red]Error:[/red] Test directory not found: {args.test_dir}")
        return 1

    # Create output directory structure
    output_dir = create_test_directory(args.output_dir)
    console.print(f"[green]Created test output directory:[/green] {output_dir}")

    # Get test files
    test_files = list(test_dir.glob("*.mp3"))
    if not test_files:
        console.print(f"[red]Error:[/red] No MP3 files found in {test_dir}")
        return 1

    # Initialize results storage
    results = []
    error_patterns = defaultdict(int)

    # Process files with progress bar
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Testing files...", total=len(test_files))

        for audio_file in test_files:
            # Process file using shared logic
            result = process_batch_file(audio_file, model_type=args.model_type)
            
            # Track error patterns if prediction was made but incorrect
            if not result.get('error') and not result['correct']:
                error_patterns[f"{result['true_surah']}->{result['predicted_surah']}"] += 1
            
            results.append(result)
            progress.advance(task)

            # Log warnings for errors
            if result.get('error'):
                progress.console.print(
                    f"[yellow]Warning:[/yellow] {audio_file.name}: {result['error']}"
                )

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Save results
    save_results(results, metrics, output_dir)
    
    # Generate visualizations if requested
    if args.visualize:
        console.print("\n[bold]Generating visualizations...[/bold]")
        generate_visualizations(results_df, output_dir)
    
    # Print summary
    console.print("\n[bold green]Test Results Summary[/bold green]")
    console.print(f"Total files processed: {metrics['total_files']}")
    console.print(f"Successful predictions: {metrics['successful']}")
    console.print(f"Correct predictions: {metrics['correct']}")
    console.print(f"Accuracy: {metrics['accuracy']:.1%}")
    console.print(f"Success rate: {metrics['success_rate']:.1%}")
    console.print(f"Average confidence: {metrics['avg_confidence']:.1%}")
    console.print(f"Average processing time: {metrics['avg_time']:.2f}s")
    
    # Print error patterns if any incorrect predictions
    if error_patterns:
        console.print("\n[bold yellow]Common Error Patterns[/bold yellow]")
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            console.print(f"{pattern}: {count} occurrences")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 