import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import os
import argparse
from collections import Counter

# Path configurations
DATA_DIR = Path("data")
dataset_DIR = Path("dataset")  # Updated path to dataset directory
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CONFIG_FILE = DATA_DIR / "dataset_splits.json"
RECITER_ALL_JSON_PATH = DATA_DIR / "recitersAll.json"
RECITER_JSON_PATH = DATA_DIR / "reciters.json"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare Quran recitation dataset')
    parser.add_argument('--use-all-versions', action='store_true', 
                        help='Use all available versions of recitations (default: False)')
    return parser.parse_args()

def parse_range(range_str):
    """Parse range string in format 'start-end' to tuple of integers"""
    start, end = map(int, range_str.split('-'))
    return start, end

def get_ayah_files(source_dir, surah_num):
    """Get all ayah files for a surah, including multiple versions"""
    # Get base files (without suffix)
    base_files = list(source_dir.glob(f"{surah_num:03d}*.mp3"))
    
    # Group files by ayah number
    ayah_groups = {}
    for file in base_files:
        # Extract ayah number from filename (e.g., "001001.mp3" -> "001")
        ayah_num = file.stem[3:6]
        if ayah_num not in ayah_groups:
            ayah_groups[ayah_num] = []
        ayah_groups[ayah_num].append(file)
    
    return ayah_groups

def get_version_identifier(file_path):
    """Extract version identifier from filename"""
    # The assumption is that versions are differentiated by some suffix after the ayah number
    # For example: 001001A.mp3, 001001B.mp3, etc.
    # If the naming convention is different, this function should be adjusted
    
    # Extract the part after the basic surah-ayah pattern
    base_name = file_path.stem  # Gets filename without extension
    if len(base_name) > 6:  # If there's something after the 6-digit surah-ayah code
        return base_name[6:]
    return "default"  # If no version identifier found

def select_versions_evenly(ayah_groups):
    """
    Select versions evenly across all ayahs
    
    For each ayah, select a version in a way that ensures even distribution
    of version types across all ayahs.
    """
    all_versions = []
    
    # First, identify all available version types
    for ayah_files in ayah_groups.values():
        for file in ayah_files:
            version = get_version_identifier(file)
            all_versions.append(version)
    
    # Count how many of each version we have
    version_counter = Counter(all_versions)
    
    # Create a selection dictionary for each ayah
    selected_files = {}
    
    # Track how many of each version we've already selected
    selected_version_counts = Counter()
    
    # For each ayah, select the version that will best balance our distribution
    for ayah_num, ayah_files in ayah_groups.items():
        if len(ayah_files) == 1:
            # If only one version, use it
            selected_files[ayah_num] = ayah_files[0]
        else:
            # Get all version identifiers for this ayah
            versions = [get_version_identifier(f) for f in ayah_files]
            
            # Calculate selection ratios for each version
            # (How many we've selected so far / How many total we should select)
            selection_ratios = {}
            for version in versions:
                if version_counter[version] == 0:
                    selection_ratios[version] = float('inf')
                else:
                    selection_ratios[version] = selected_version_counts[version] / version_counter[version]
            
            # Select the version with the lowest selection ratio
            best_version = min(versions, key=lambda v: selection_ratios[v])
            
            # Find the file with this version
            for file in ayah_files:
                if get_version_identifier(file) == best_version:
                    selected_files[ayah_num] = file
                    selected_version_counts[best_version] += 1
                    break
    
    return selected_files

def load_split_config():
    """Load and validate the split configuration"""
    print("\n📚 Loading split configuration...")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Get lists directly from JSON
    training_reciters = config.get("training", [])
    testing_reciters_from_json = config.get("testing", []) 
    remaining_reciters_from_json = config.get("remaining", [])

    # --- Validation and Cleaning ---
    training_set = set(training_reciters)
    
    # 1. Validate testing_reciters: 
    #    They must come from the union of reciters in the training list 
    #    and reciters in the remaining list (as defined in the JSON).
    #    "potential_sources_for_testing_from_remaining" are those in json's remaining list
    #    that are NOT already in the training list (to avoid double counting sources).
    potential_sources_for_testing_from_remaining = set(r for r in remaining_reciters_from_json if r not in training_set)
    valid_sources_for_testing = training_set | potential_sources_for_testing_from_remaining
    
    validated_testing_reciters = []
    invalid_testing_entries = []
    for reciter in testing_reciters_from_json:
        if reciter in valid_sources_for_testing:
            validated_testing_reciters.append(reciter)
        else:
            invalid_testing_entries.append(reciter)
    
    if invalid_testing_entries:
        print(f"⚠️ Warning: The following reciters in the 'testing' list are not found in 'training' or 'remaining' lists from JSON and will be removed:")
        for entry in invalid_testing_entries:
            print(f"  - {entry}")
            
    testing_set = set(validated_testing_reciters)

    # 2. Clean remaining_reciters: 
    #    Remove any reciters that are in the training list or in the (validated) testing list.
    #    Start from the raw remaining list from JSON.
    cleaned_remaining_reciters = []
    removed_from_remaining_count = 0
    for reciter in remaining_reciters_from_json:
        if reciter not in training_set and reciter not in testing_set:
            cleaned_remaining_reciters.append(reciter)
        elif reciter in training_set or reciter in testing_set:
            removed_from_remaining_count +=1
            
    if removed_from_remaining_count > 0:
        print(f"ℹ️ Info: {removed_from_remaining_count} reciter(s) were removed from the 'remaining' list because they are present in 'training' or 'testing'.")

    # Update config with validated and cleaned lists
    config["training"] = training_reciters # training list is taken as is
    config["testing"] = validated_testing_reciters
    config["remaining"] = cleaned_remaining_reciters
    
    # Update statistics based on the validated and cleaned lists
    config["n_training_reciters"] = len(config["training"])
    config["n_testing_reciters"] = len(config["testing"])
    config["n_remaining_reciters"] = len(config["remaining"])
    
    # Calculate total_reciters as the count of unique reciters across all three final lists
    all_defined_reciters = set(config["training"]) | set(config["testing"]) | set(config["remaining"])
    config["total_reciters"] = len(all_defined_reciters)
    
    print(f"✓ Loaded {config['n_training_reciters']} training reciters (as per JSON).")
    if invalid_testing_entries:
        print(f"✓ Loaded {config['n_testing_reciters']} testing reciters after validation (removed {len(invalid_testing_entries)} invalid entries).")
    else:
        print(f"✓ Loaded {config['n_testing_reciters']} testing reciters (as per JSON, all valid).")
    print(f"✓ Effective {config['n_remaining_reciters']} remaining reciters after cleaning.")
    print(f"✓ Total unique reciters to be used in data preparation: {config['total_reciters']}.")
    
    # Clean up any existing directories (should happen AFTER config is finalized)
    cleanup_directories(config)
    
    # Save updated configuration (should also happen AFTER config is finalized)
    save_config(config)
    
    return config

def cleanup_directories(config):
    """Remove and recreate data directories for a fresh start"""
    print("\n🧹 Cleaning up directories...")
    
    # Remove and recreate training directory
    if TRAIN_DIR.exists():
        print("  Removing training directory for fresh start")
        shutil.rmtree(TRAIN_DIR)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Remove and recreate testing directory
    if TEST_DIR.exists():
        print("  Removing testing directory for fresh start")
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories cleaned and recreated")

def save_config(config):
    """Save the updated configuration"""
    print("\n💾 Saving configuration...")
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("✓ Configuration saved")

def prepare_training_data(config, use_all_versions=False):
    """Prepare training data by copying ayah files"""
    print("\n🔄 Preparing training data...")
    
    start_surah, end_surah = parse_range(config["train_data_range"])
    training_surahs = list(range(start_surah, end_surah + 1))
    
    for reciter in tqdm(config["training"], desc="Processing training reciters"):
        reciter_dir = TRAIN_DIR / reciter
        reciter_dir.mkdir(exist_ok=True)
        
        source_dir = dataset_DIR / reciter
        
        for surah_num in training_surahs:
            # Get all ayah files for this surah, grouped by ayah number
            ayah_groups = get_ayah_files(source_dir, surah_num)
            
            if use_all_versions:
                # Copy all versions of each ayah
                for ayah_files in ayah_groups.values():
                    for ayah_file in ayah_files:
                        shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
            else:
                # Select versions evenly and copy only selected versions
                selected_files = select_versions_evenly(ayah_groups)
                
                # Copy selected files
                for ayah_file in selected_files.values():
                    shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
    
    version_mode = "all versions" if use_all_versions else "evenly distributed versions"
    print(f"✓ Prepared surahs {start_surah}-{end_surah} for {len(config['training'])} reciters using {version_mode}")
    return config

def prepare_testing_data(config, use_all_versions=False):
    """Prepare testing data by copying ayah files"""
    print("\n🔄 Preparing testing data...")
    
    start_surah, end_surah = parse_range(config["test_data_range"])
    testing_surahs = list(range(start_surah, end_surah + 1))
    
    for reciter in tqdm(config["testing"], desc="Processing testing reciters"):
        reciter_dir = TEST_DIR / reciter
        reciter_dir.mkdir(exist_ok=True)
        
        source_dir = dataset_DIR / reciter
        
        for surah_num in testing_surahs:
            # Get all ayah files for this surah, grouped by ayah number
            ayah_groups = get_ayah_files(source_dir, surah_num)
            
            if use_all_versions:
                # Copy all versions of each ayah
                for ayah_files in ayah_groups.values():
                    for ayah_file in ayah_files:
                        shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
            else:
                # Select versions evenly and copy only selected versions
                selected_files = select_versions_evenly(ayah_groups)
                
                # Copy selected files
                for ayah_file in selected_files.values():
                    shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
    
    version_mode = "all versions" if use_all_versions else "evenly distributed versions"
    print(f"✓ Prepared surahs {start_surah}-{end_surah} for {len(config['testing'])} reciters using {version_mode}")
    return config

def update_training_reciters_json(config):
    """Update reciters.json with data for selected training reciters."""
    print("\n🔄 Updating reciters.json with training reciters...")

    try:
        with open(RECITER_ALL_JSON_PATH, 'r', encoding='utf-8') as f:
            all_reciters_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {RECITER_ALL_JSON_PATH} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {RECITER_ALL_JSON_PATH}.")
        return

    training_reciter_names = config.get("training", [])
    selected_training_reciters_data = {}

    for reciter_name in training_reciter_names:
        if reciter_name in all_reciters_data:
            selected_training_reciters_data[reciter_name] = all_reciters_data[reciter_name]
        else:
            print(f"Warning: Training reciter '{reciter_name}' not found in {RECITER_ALL_JSON_PATH}.")

    try:
        with open(RECITER_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(selected_training_reciters_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Successfully updated {RECITER_JSON_PATH} with {len(selected_training_reciters_data)} training reciters.")
    except IOError:
        print(f"Error: Could not write to {RECITER_JSON_PATH}.")

def main():
    print("\n🚀 Starting data preparation process...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Load and prepare configuration (includes cleanup and initial save)
    config = load_split_config()
    
    # Update reciters.json with the selected training reciters
    update_training_reciters_json(config)
    
    # Process training data
    config = prepare_training_data(config, use_all_versions=args.use_all_versions)
    
    # Process testing data
    config = prepare_testing_data(config, use_all_versions=args.use_all_versions)
    
    version_mode = "all versions" if args.use_all_versions else "evenly distributed versions"
    print(f"\n✨ Data preparation completed successfully using {version_mode}!")
    print(f"📊 Final statistics:")
    print(f"  - Training reciters: {config['n_training_reciters']}")
    print(f"  - Testing reciters: {config['n_testing_reciters']}")
    print(f"  - Remaining reciters: {config['n_remaining_reciters']}")
    print(f"  - Total reciters: {config['total_reciters']}")
    print(f"  - Training range: {config['train_data_range']}")
    print(f"  - Testing range: {config['test_data_range']}")

if __name__ == "__main__":
    main() 