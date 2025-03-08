"""
Enhanced preprocessing pipeline for Quran reciter identification.
This pipeline processes audio files and saves extracted features for later use
with improved organization and comprehensive reporting.
"""
import os
import numpy as np
import pandas as pd
import time
import json
import platform
from pathlib import Path
from datetime import datetime
import logging
import warnings
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from src.data import preprocess_audio_with_logic, preprocess_audio, augment_audio
from src.features import extract_features
from src.utils import setup_logging, format_duration
from src.utils import is_gpu_available

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def preprocess_pipeline(mode="train", augment=True):
    """
    Run the enhanced preprocessing pipeline with improved reporting and organization.

    Args:
        mode (str): Mode to run in ('train' or 'test')
        augment (bool): Whether to apply data augmentation (only for train mode)

    Returns:
        int: 0 for success, 1 for failure
        str: Path to the preprocessing output directory
    """
    try:
        total_start_time = time.time()

        # Generate timestamp for this preprocessing run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_preprocess"
        
        # Setup logging
        logger, log_file = setup_logging(f"preprocessing_{mode}_{run_id}")
        logger.info(f"Starting enhanced preprocessing pipeline in {mode} mode")
        logger.info(f"Data augmentation: {'enabled' if augment else 'disabled'}")
        logger.info(f"Run ID: {run_id}")

        # Set input and output directories based on mode
        if mode == "train":
            input_dir = TRAIN_DATA_DIR
        elif mode == "test":
            input_dir = TEST_DATA_DIR  # Test directory
            # Disable augmentation for test mode
            augment = False
        else:
            logger.error(f"Invalid mode: {mode}. Must be 'train' or 'test'.")
            return 1, None

        # Create timestamped processed directory structure
        processed_dir = Path("processed")
        mode_dir = processed_dir / mode
        output_dir = mode_dir / run_id
        
        # Create output directory and subdirectories
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Find all reciter folders
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1, None

        sub_folders = [f for f in input_path.iterdir() if f.is_dir()]
        reciter_count = len(sub_folders)

        if reciter_count == 0:
            logger.error("No reciter folders found in input directory")
            return 1, None

        logger.info(f'Found {reciter_count} reciters')

        # Initialize data structures for tracking
        features_list = []
        metadata_list = []
        summary_log = []
        reciter_durations = {}
        total_audio_duration = 0
        file_inventory = []
        reciter_profiles = {}
        file_feature_map = []
        feature_index = 0

        # Check for actual GPU availability
        gpu_available = is_gpu_available()
        logger.info(f"GPU availability: {'Yes' if gpu_available else 'No'}")

        # Create preprocessing metadata
        preprocessing_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "augmentation_enabled": augment,
            "config": {
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "min_duration": MINIMUM_DURATION,
                "skip_start": SKIP_START,
                "skip_end": SKIP_END,
                "n_mfcc": N_MFCC,
                "n_mel_bands": N_MEL_BANDS,
                "n_chroma": N_CHROMA
            },
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "use_gpu_config": USE_GPU,
                "gpu_available": gpu_available
            },
            "input_directory": str(input_path),
            "output_directory": str(output_dir),
            "reciters": [f.name for f in sub_folders]
        }

        # Process each reciter's folder
        for index, sub_folder in enumerate(sub_folders, 1):
            reciter_start_time = time.time()
            logger.info(f'Processing {sub_folder.name} ({index}/{reciter_count})')

            # Create reciter directory in output
            reciter_output_dir = output_dir / sub_folder.name
            reciter_output_dir.mkdir(exist_ok=True)

            # Initialize reciter profile
            reciter_profiles[sub_folder.name] = {
                "total_files": 0,
                "processed_files": 0,
                "skipped_files": 0,
                "original_duration": 0,
                "processed_duration": 0,
                "augmented_samples": 0
            }

            files_mp3 = list(sub_folder.glob('*.mp3'))
            reciter_profiles[sub_folder.name]["total_files"] = len(files_mp3)
            
            if not files_mp3:
                logger.warning(f"No MP3 files found in {sub_folder.name}")
                continue

            processed_count = 0
            skipped_count = 0
            skipped_reasons = []
            reciter_audio_duration = 0
            reciter_features = []
            reciter_metadata = []

            for file_mp3 in files_mp3:
                logger.info(f'Processing {file_mp3.name}')
                file_size = file_mp3.stat().st_size
                processing_start = time.time()
                
                # Create inventory entry for this file
                inventory_entry = {
                    "file_path": str(file_mp3),
                    "file_name": file_mp3.name,
                    "reciter": sub_folder.name,
                    "file_size_bytes": file_size,
                    "processing_status": "initialized",
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_duration": None,
                    "processed_duration": None,
                    "sample_rate": None,
                    "reason": None
                }

                try:
                    # Preprocess audio
                    audio_data, sr = preprocess_audio_with_logic(str(file_mp3))
                    
                    if audio_data is None or sr is None:
                        inventory_entry["processing_status"] = "skipped"
                        inventory_entry["reason"] = "Initial preprocessing failed"
                        file_inventory.append(inventory_entry)
                        skipped_count += 1
                        skipped_reasons.append(f"{file_mp3.name}: Initial preprocessing failed")
                        continue
                    
                    # Calculate original audio duration
                    original_duration = len(audio_data) / sr
                    inventory_entry["original_duration"] = original_duration
                    inventory_entry["sample_rate"] = sr
                    reciter_profiles[sub_folder.name]["original_duration"] += original_duration

                    # Additional preprocessing
                    audio_data, sr = preprocess_audio(audio_data, sr)
                    if audio_data is None or sr is None:
                        inventory_entry["processing_status"] = "skipped"
                        inventory_entry["reason"] = "Secondary preprocessing failed"
                        file_inventory.append(inventory_entry)
                        skipped_count += 1
                        skipped_reasons.append(f"{file_mp3.name}: Secondary preprocessing failed")
                        continue
                    
                    # Update audio duration after preprocessing
                    processed_duration = len(audio_data) / sr
                    inventory_entry["processed_duration"] = processed_duration
                    reciter_audio_duration += processed_duration
                    total_audio_duration += processed_duration
                    reciter_profiles[sub_folder.name]["processed_duration"] += processed_duration

                    # Process the original audio
                    features = extract_features(audio_data, sr)
                    if features is None:
                        inventory_entry["processing_status"] = "skipped"
                        inventory_entry["reason"] = "Feature extraction failed"
                        file_inventory.append(inventory_entry)
                        skipped_count += 1
                        skipped_reasons.append(f"{file_mp3.name}: Feature extraction failed")
                        continue

                    # Create metadata for original audio
                    file_metadata = {
                        "file_name": file_mp3.name,
                        "reciter": sub_folder.name,
                        "augmented": False,
                        "duration": processed_duration,
                        "sample_rate": sr,
                        "run_id": run_id,
                        "feature_index": feature_index
                    }

                    # Update processing status
                    inventory_entry["processing_status"] = "processed"
                    inventory_entry["processing_time"] = time.time() - processing_start
                    file_inventory.append(inventory_entry)
                    
                    # Update file-to-feature mapping
                    file_feature_map.append({
                        "file_path": str(file_mp3),
                        "file_name": file_mp3.name,
                        "reciter": sub_folder.name,
                        "feature_index": feature_index,
                        "augmented": False,
                        "feature_length": len(features)
                    })
                    
                    # Add to lists
                    reciter_features.append(features)
                    reciter_metadata.append(file_metadata)
                    features_list.append(features)
                    metadata_list.append(file_metadata)
                    
                    processed_count += 1
                    feature_index += 1
                    reciter_profiles[sub_folder.name]["processed_files"] += 1

                    # Data augmentation (only for training mode)
                    if augment and mode == "train":
                        try:
                            augmented_audios = augment_audio(audio_data, sr)
                            # Skip the first element as it's the original audio
                            augmented_audios = augmented_audios[1:]
                            
                            for i, aug_audio in enumerate(augmented_audios):
                                aug_features = extract_features(aug_audio, sr)
                                if aug_features is None:
                                    continue

                                aug_duration = len(aug_audio) / sr
                                
                                # Create metadata for augmented audio
                                aug_metadata = {
                                    "file_name": f"{file_mp3.stem}_aug{i+1}{file_mp3.suffix}",
                                    "reciter": sub_folder.name,
                                    "augmented": True,
                                    "augmentation_type": i + 1,  # 1: pitch shift, 2: time stretch, etc.
                                    "original_file": file_mp3.name,
                                    "duration": aug_duration,
                                    "sample_rate": sr,
                                    "run_id": run_id,
                                    "feature_index": feature_index
                                }

                                # Update file-to-feature mapping
                                file_feature_map.append({
                                    "file_path": str(file_mp3),
                                    "file_name": f"{file_mp3.stem}_aug{i+1}{file_mp3.suffix}",
                                    "reciter": sub_folder.name,
                                    "feature_index": feature_index,
                                    "augmented": True,
                                    "augmentation_type": i + 1,
                                    "feature_length": len(aug_features)
                                })

                                # Add to lists
                                reciter_features.append(aug_features)
                                reciter_metadata.append(aug_metadata)
                                features_list.append(aug_features)
                                metadata_list.append(aug_metadata)
                                processed_count += 1
                                feature_index += 1
                                reciter_profiles[sub_folder.name]["augmented_samples"] += 1
                        except Exception as e:
                            logger.warning(f"Augmentation failed for {file_mp3.name}: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error processing {file_mp3.name}: {str(e)}")
                    inventory_entry["processing_status"] = "failed"
                    inventory_entry["reason"] = str(e)
                    file_inventory.append(inventory_entry)
                    skipped_count += 1
                    skipped_reasons.append(f"{file_mp3.name}: {str(e)}")

            # Save features and metadata for this reciter
            if reciter_features:
                # Convert to numpy array and save
                reciter_features_array = np.array(reciter_features)
                features_file = reciter_output_dir / "features.npy"
                np.save(features_file, reciter_features_array)
                
                # Save metadata as JSON
                metadata_file = reciter_output_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(reciter_metadata, f, indent=4)
                
                logger.info(f"Saved {len(reciter_features)} feature sets to {features_file}")

            # Calculate and store reciter processing duration
            reciter_duration = time.time() - reciter_start_time
            reciter_durations[sub_folder.name] = {
                'processing_time': reciter_duration,
                'audio_duration': reciter_audio_duration,
                'processed_files': processed_count,
                'skipped_files': skipped_count
            }
            
            # Update reciter profile with processing stats
            reciter_profiles[sub_folder.name]["skipped_files"] = skipped_count
            reciter_profiles[sub_folder.name]["processing_time"] = reciter_duration

            # Log summary
            summary_log.extend([
                f"Reciter: {sub_folder.name}",
                f"  Total Files: {len(files_mp3)}",
                f"  Processed Files: {processed_count}",
                f"  Skipped Files: {skipped_count}",
                f"  Original Audio Duration: {format_duration(reciter_profiles[sub_folder.name]['original_duration'])}",
                f"  Processed Audio Duration: {format_duration(reciter_audio_duration)}",
                f"  Augmented Samples: {reciter_profiles[sub_folder.name]['augmented_samples']}",
                f"  Processing Time: {format_duration(reciter_duration)}"
            ])
            for reason in skipped_reasons:
                summary_log.append(f"    - {reason}")
            summary_log.append("")

        # Check if we have enough data
        if len(features_list) == 0:
            logger.error("No features were extracted successfully")
            return 1, None

        # Save global features and metadata
        all_features = np.array(features_list)
        features_file = output_dir / "all_features.npy"
        np.save(features_file, all_features)
        
        # Convert metadata to DataFrame and save as CSV
        metadata_df = pd.DataFrame(metadata_list)
        metadata_file = output_dir / "all_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        logger.info(f"Total number of processed samples: {len(features_list)}")
        logger.info(f"Saved all features to {features_file}")
        logger.info(f"Saved all metadata to {metadata_file}")

        # Calculate total preprocessing time
        total_preprocessing_time = time.time() - total_start_time

        # Add timing information to summary
        summary_log.extend([
            "\nPreprocessing Duration Summary:",
            f"Total Audio Duration: {format_duration(total_audio_duration)}",
            f"Total Processing Time: {format_duration(total_preprocessing_time)}",
            "\nPer-Reciter Processing Times:"
        ])

        for reciter, times in reciter_durations.items():
            summary_log.extend([
                f"{reciter}:",
                f"  Audio Duration: {format_duration(times['audio_duration'])}",
                f"  Processing Time: {format_duration(times['processing_time'])}",
                f"  Processed Files: {times['processed_files']}",
                f"  Skipped Files: {times['skipped_files']}"
            ])
        
        # Update preprocessing metadata with final stats
        preprocessing_metadata.update({
            "total_reciters": reciter_count,
            "total_files_processed": len(metadata_list) - sum([p["augmented_samples"] for p in reciter_profiles.values()]),
            "total_files_skipped": sum([p["skipped_files"] for p in reciter_profiles.values()]),
            "total_augmented_samples": sum([p["augmented_samples"] for p in reciter_profiles.values()]),
            "total_features": len(features_list),
            "total_audio_duration": total_audio_duration,
            "total_processing_time": total_preprocessing_time
        })

        # Save file inventory as CSV
        inventory_df = pd.DataFrame(file_inventory)
        inventory_file = output_dir / "file_inventory.csv"
        inventory_df.to_csv(inventory_file, index=False)
        logger.info(f"File inventory saved to {inventory_file}")
        
        # Save file-to-feature mapping
        feature_map_df = pd.DataFrame(file_feature_map)
        feature_map_file = output_dir / "file_feature_map.csv"
        feature_map_df.to_csv(feature_map_file, index=False)
        logger.info(f"File-to-feature mapping saved to {feature_map_file}")
        
        # Save reciter profiles
        reciter_profiles_file = output_dir / "reciter_profiles.json"
        with open(reciter_profiles_file, 'w') as f:
            json.dump(reciter_profiles, f, indent=4)
        logger.info(f"Reciter profiles saved to {reciter_profiles_file}")
        
        # Save preprocessing metadata
        metadata_file = output_dir / "preprocessing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(preprocessing_metadata, f, indent=4)
        logger.info(f"Preprocessing metadata saved to {metadata_file}")

        # Save preprocessing summary
        summary_path = output_dir / 'preprocessing_summary.txt'
        with open(summary_path, "w") as summary_file:
            summary_file.write("\n".join(summary_log))
        logger.info(f"Preprocessing summary saved to {summary_path}")
        
        # Calculate and save feature statistics
        try:
            feature_stats = calculate_feature_statistics(all_features, metadata_df)
            feature_stats_file = output_dir / "feature_statistics.json"
            with open(feature_stats_file, 'w') as f:
                # Convert numpy types to Python native types for JSON serialization
                json_compatible_stats = convert_to_json_serializable(feature_stats)
                json.dump(json_compatible_stats, f, indent=4)
            logger.info(f"Feature statistics saved to {feature_stats_file}")
            
            # Generate visualizations
            generate_visualizations(all_features, metadata_df, viz_dir, run_id)
            logger.info(f"Visualizations saved to {viz_dir}")
        except Exception as e:
            logger.error(f"Error generating feature statistics or visualizations: {str(e)}")
        
        # Log duration summary to console
        logger.info("\nPreprocessing Complete!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Total Audio Duration: {format_duration(total_audio_duration)}")
        logger.info(f"Total Processing Time: {format_duration(total_preprocessing_time)}")
        logger.info(f"Total Features: {len(features_list)}")

        # Create a symbolic link or copy to 'latest' for convenience
        latest_dir = mode_dir / "latest"
        if latest_dir.exists():
            if latest_dir.is_symlink():
                latest_dir.unlink()
            else:
                shutil.rmtree(latest_dir)
        
        try:
            # Try to create symbolic link first
            latest_dir.symlink_to(output_dir.name)
            logger.info(f"Created symbolic link to latest preprocessing run: {latest_dir}")
        except (OSError, NotImplementedError):
            # If symlink fails (e.g., on Windows), copy the directory
            shutil.copytree(output_dir, latest_dir)
            logger.info(f"Created copy of latest preprocessing run: {latest_dir}")

        return 0, str(output_dir)

    except KeyboardInterrupt:
        logger.info("\nPreprocessing interrupted by user")
        return 1, None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.exception("Stack trace:")
        return 1, None


def calculate_feature_statistics(features, metadata_df):
    """
    Calculate comprehensive statistics about the extracted features.
    
    Args:
        features (np.ndarray): All extracted features
        metadata_df (pd.DataFrame): Metadata for all processed files
        
    Returns:
        dict: Statistics about the features
    """
    stats = {
        "global": {
            "min": np.min(features, axis=0).tolist(),
            "max": np.max(features, axis=0).tolist(),
            "mean": np.mean(features, axis=0).tolist(),
            "std": np.std(features, axis=0).tolist(),
        },
        "per_reciter": {},
        "feature_correlations": {}
    }
    
    # Calculate per-reciter statistics
    for reciter in metadata_df['reciter'].unique():
        reciter_indices = metadata_df[metadata_df['reciter'] == reciter].index.tolist()
        reciter_features = features[reciter_indices]
        
        stats["per_reciter"][reciter] = {
            "min": np.min(reciter_features, axis=0).tolist(),
            "max": np.max(reciter_features, axis=0).tolist(),
            "mean": np.mean(reciter_features, axis=0).tolist(),
            "std": np.std(reciter_features, axis=0).tolist(),
            "sample_count": len(reciter_features)
        }
    
    # Calculate correlation matrix for the first 20 features (to keep size manageable)
    if features.shape[1] > 1:
        corr_matrix = np.corrcoef(features[:, :min(20, features.shape[1])], rowvar=False)
        # Convert to list of lists for JSON serialization
        stats["feature_correlations"]["first_20_features"] = corr_matrix.tolist()
    
    return stats


def generate_visualizations(features, metadata_df, viz_dir, run_id):
    """
    Generate visualizations for the preprocessing run.
    
    Args:
        features (np.ndarray): All extracted features
        metadata_df (pd.DataFrame): Metadata for all processed files
        viz_dir (Path): Directory to save visualizations
        run_id (str): Run identifier
    """
    # 1. Sample counts per reciter
    plt.figure(figsize=(12, 6))
    reciter_counts = metadata_df['reciter'].value_counts()
    sns.barplot(x=reciter_counts.index, y=reciter_counts.values)
    plt.title('Number of Samples per Reciter')
    plt.ylabel('Sample Count')
    plt.xlabel('Reciter')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(viz_dir / 'reciter_sample_counts.png', dpi=300)
    plt.close()
    
    # 2. Original vs. Augmented samples
    plt.figure(figsize=(8, 6))
    aug_counts = metadata_df['augmented'].value_counts()
    labels = ['Original', 'Augmented']
    values = [aug_counts.get(False, 0), aug_counts.get(True, 0)]
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#99ff99'])
    plt.title('Original vs. Augmented Samples')
    plt.savefig(viz_dir / 'augmentation_distribution.png', dpi=300)
    plt.close()
    
    # 3. Duration distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(metadata_df['duration'], bins=30, kde=True)
    plt.title('Distribution of Audio File Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.savefig(viz_dir / 'duration_distribution.png', dpi=300)
    plt.close()
    
    # 4. Feature correlation heatmap (first 20 features only for readability)
    if features.shape[1] > 1:
        plt.figure(figsize=(12, 10))
        corr_matrix = np.corrcoef(features[:, :min(20, features.shape[1])], rowvar=False)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=.5, annot=False)
        plt.title('Feature Correlation Matrix (First 20 Features)')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_correlation.png', dpi=300)
        plt.close()
    
    # 5. Feature distributions (first 8 features)
    n_features = min(8, features.shape[1])
    if n_features > 0:
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 2*n_features), sharex=True)
        if n_features == 1:
            axes = [axes]  # Make it iterable when there's only one feature
        
        for i in range(n_features):
            sns.histplot(features[:, i], kde=True, ax=axes[i])
            axes[i].set_title(f'Feature {i+1} Distribution')
            axes[i].set_xlabel('')
        
        axes[-1].set_xlabel('Feature Value')
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_distributions.png', dpi=300)
        plt.close()


def convert_to_json_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess audio files for Quran reciter identification")
    parser.add_argument("--mode", choices=["train", "test"], default="train", 
                        help="Mode to run in: 'train' or 'test'")
    parser.add_argument("--no-augment", action="store_true", 
                        help="Disable data augmentation (train mode only)")
    
    args = parser.parse_args()
    
    status, output_dir = preprocess_pipeline(mode=args.mode, augment=not args.no_augment)
    sys.exit(status)