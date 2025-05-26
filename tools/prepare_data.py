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
    return parser.parse_args()

def parse_range(range_str):
    """Parse range string in format 'start-end' to tuple of integers"""
    start, end = map(int, range_str.split('-'))
    return start, end

def get_ayah_files(source_dir, surah_num):
    """Get all ayah files for a surah (single version per ayah)"""
    return list(source_dir.glob(f"{surah_num:03d}*.mp3"))

def load_split_config():
    """Load and validate the split configuration based on recitersAll.json"""
    print("\n📚 Loading split configuration...")

    # 1. Load All Available Reciters from recitersAll.json
    try:
        with open(RECITER_ALL_JSON_PATH, 'r', encoding='utf-8') as f:
            all_reciters_data = json.load(f)
        all_available_reciters_set = set(all_reciters_data.keys())
        print(f"✓ Found {len(all_available_reciters_set)} total available reciters in {RECITER_ALL_JSON_PATH.name}.")
    except FileNotFoundError:
        print(f"🛑 Error: {RECITER_ALL_JSON_PATH} not found. Cannot proceed.")
        exit(1) # Exit if the master list is missing
    except json.JSONDecodeError:
        print(f"🛑 Error: Could not decode JSON from {RECITER_ALL_JSON_PATH}. Cannot proceed.")
        exit(1) # Exit if the master list is malformed

    # 2. Load user-defined splits from dataset_splits.json
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
    except FileNotFoundError:
        print(f"🛑 Error: {CONFIG_FILE} not found. Please ensure it exists with 'training' and 'testing' lists.")
        exit(1)
    except json.JSONDecodeError:
        print(f"🛑 Error: Could not decode JSON from {CONFIG_FILE}. Please check its format.")
        exit(1)

    # Get user-defined training and testing lists
    training_reciters_from_json = user_config.get("training", [])
    testing_reciters_from_json = user_config.get("testing", [])
    
    print(f"ℹ️ User-defined training reciters (from {CONFIG_FILE.name}): {len(training_reciters_from_json)}")
    print(f"ℹ️ User-defined testing reciters (from {CONFIG_FILE.name}): {len(testing_reciters_from_json)}")

    # --- Validation and Final List Determination ---

    # 3. Training Reciters (taken as is from user config, but we can validate if they are in all_available_reciters_set)
    validated_training_reciters = []
    invalid_training_entries = []
    for reciter in training_reciters_from_json:
        if reciter in all_available_reciters_set:
            validated_training_reciters.append(reciter)
        else:
            invalid_training_entries.append(reciter)
    
    if invalid_training_entries:
        print(f"⚠️ Warning: The following reciters in your 'training' list (from {CONFIG_FILE.name}) are NOT found in {RECITER_ALL_JSON_PATH.name} and will be EXCLUDED:")
        for entry in invalid_training_entries:
            print(f"  - {entry}")
    training_set = set(validated_training_reciters)
    final_training_reciters = validated_training_reciters # Use the validated list

    # 4. Validate Testing Reciters
    validated_testing_reciters = []
    invalid_testing_entries_in_test = []
    for reciter in testing_reciters_from_json:
        if reciter in all_available_reciters_set:
            if reciter not in validated_testing_reciters: # Avoid duplicates in testing list
                 validated_testing_reciters.append(reciter)
        else:
            invalid_testing_entries_in_test.append(reciter)
            
    if invalid_testing_entries_in_test:
        print(f"⚠️ Warning: The following reciters in your 'testing' list (from {CONFIG_FILE.name}) are NOT found in {RECITER_ALL_JSON_PATH.name} and will be EXCLUDED:")
        for entry in invalid_testing_entries_in_test:
            print(f"  - {entry}")
    # final_testing_reciters will be validated_testing_reciters.
    # No need for a separate testing_set for calculation if we just use the list length.

    # 5. Determine Remaining Reciters
    # Remaining = All_Available - Training_Set
    remaining_reciters_set = all_available_reciters_set - training_set
    final_remaining_reciters = sorted(list(remaining_reciters_set))
    
    # --- Construct the final config object to be used and saved ---
    # Preserve other keys from user_config if they exist (e.g., ranges)
    final_config = user_config.copy() 
    
    final_config["training"] = final_training_reciters
    final_config["testing"] = validated_testing_reciters # Use the validated list for testing
    final_config["remaining"] = final_remaining_reciters
    
    # Update statistics
    final_config["n_training_reciters"] = len(final_config["training"])
    final_config["n_testing_reciters"] = len(final_config["testing"])
    final_config["n_remaining_reciters"] = len(final_config["remaining"])
    final_config["total_available_reciters"] = len(all_available_reciters_set) # New stat: total from recitersAll.json
    
    # This 'total_reciters' might be misleading if it's meant to be unique across train/test/remaining.
    # The sum of n_training, n_testing, n_remaining can be > total_available if there's overlap (e.g. test can be in remaining)
    # Let's define it as the total unique reciters effectively being *considered* by the split config.
    # For this new logic, it should be total_available_reciters.
    final_config["total_reciters_in_split_config"] = len(all_available_reciters_set)

    print(f"✓ Final training reciters: {final_config['n_training_reciters']} (after validation against {RECITER_ALL_JSON_PATH.name})")
    print(f"✓ Final testing reciters: {final_config['n_testing_reciters']} (after validation against {RECITER_ALL_JSON_PATH.name})")
    print(f"✓ Final remaining reciters: {final_config['n_remaining_reciters']} (calculated as All - Training)")
    print(f"✓ Total reciters available in {RECITER_ALL_JSON_PATH.name}: {final_config['total_available_reciters']}")

    # Clean up directories based on the new config understanding
    cleanup_directories(final_config) # Pass the fully formed final_config
    
    # Save the final, validated, and augmented configuration
    save_config(final_config) # Pass the fully formed final_config
    
    return final_config

def cleanup_directories(config_to_use):
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

def save_config(config_to_save):
    """Save the updated configuration"""
    print("\n💾 Saving configuration...")
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)
    print("✓ Configuration saved")

def prepare_training_data(config):
    """Prepare training data by copying ayah files"""
    print("\n🔄 Preparing training data...")
    start_surah, end_surah = parse_range(config["train_data_range"])
    training_surahs = list(range(start_surah, end_surah + 1))
    for reciter in tqdm(config["training"], desc="Processing training reciters"):
        reciter_dir = TRAIN_DIR / reciter
        reciter_dir.mkdir(exist_ok=True)
        source_dir = dataset_DIR / reciter
        for surah_num in training_surahs:
            ayah_files = get_ayah_files(source_dir, surah_num)
            for ayah_file in ayah_files:
                shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
    print(f"✓ Prepared surahs {start_surah}-{end_surah} for {len(config['training'])} reciters.")
    return config

def prepare_testing_data(config):
    """Prepare testing data by copying ayah files"""
    print("\n🔄 Preparing testing data...")
    start_surah, end_surah = parse_range(config["test_data_range"])
    testing_surahs = list(range(start_surah, end_surah + 1))
    for reciter in tqdm(config["testing"], desc="Processing testing reciters"):
        reciter_dir = TEST_DIR / reciter
        reciter_dir.mkdir(exist_ok=True)
        source_dir = dataset_DIR / reciter
        for surah_num in testing_surahs:
            ayah_files = get_ayah_files(source_dir, surah_num)
            for ayah_file in ayah_files:
                shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
    print(f"✓ Prepared surahs {start_surah}-{end_surah} for {len(config['testing'])} reciters.")
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
    args = parse_args()
    config = load_split_config()
    update_training_reciters_json(config)
    config = prepare_training_data(config)
    config = prepare_testing_data(config)
    print(f"\n✨ Data preparation completed successfully!")
    print(f"📊 Final statistics:")
    print(f"  - Training reciters: {config['n_training_reciters']}")
    print(f"  - Testing reciters: {config['n_testing_reciters']}")
    print(f"  - Remaining reciters: {config['n_remaining_reciters']}")
    print(f"  - Total reciters: {config['total_reciters_in_split_config']}")
    print(f"  - Training range: {config['train_data_range']}")
    print(f"  - Testing range: {config['test_data_range']}")

if __name__ == "__main__":
    main() 