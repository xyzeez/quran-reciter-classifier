import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import os

# Path configurations
DATA_DIR = Path("data")
dataset_DIR = Path("dataset")  # Updated path to dataset directory
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
CONFIG_FILE = DATA_DIR / "dataset_splits.json"

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

def load_split_config():
    """Load and validate the split configuration"""
    print("\n📚 Loading split configuration...")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Store original training reciters
    training_reciters = config["training"]
    
    # Clean remaining array
    config["remaining"] = [r for r in config["remaining"] if r not in training_reciters]
    
    # Empty testing array
    config["testing"] = []
    
    # Update statistics
    config["total_reciters"] = len(training_reciters) + len(config['remaining'])
    config["n_training_reciters"] = len(training_reciters)
    config["n_testing_reciters"] = 0  # Will be updated after selection
    config["n_remaining_reciters"] = len(config['remaining'])
    
    print(f"✓ Found {len(training_reciters)} training reciters")
    print(f"✓ Found {len(config['remaining'])} remaining reciters")
    
    # Clean up any existing directories
    cleanup_directories(config)
    
    # Save initial configuration
    save_config(config)
    
    return config

def select_testing_reciters(config):
    """Select reciters for testing set"""
    print("\n🎯 Selecting testing reciters...")
    
    # Calculate number of additional reciters needed
    n_training = len(config["training"])
    n_additional_needed = n_training // 2
    n_available = len(config["remaining"])
    
    # Determine how many we can actually select
    n_to_select = min(n_additional_needed, n_available)
    
    print(f"✓ Will select {n_to_select} additional reciters from remaining pool")
    
    # Select random reciters from remaining
    selected_additional = random.sample(config["remaining"], n_to_select)
    
    # Build testing array
    config["testing"] = config["training"] + selected_additional
    
    print(f"✓ Testing set contains {len(config['testing'])} reciters")
    print(f"  - {len(config['training'])} from training set")
    print(f"  - {len(selected_additional)} additional reciters")
    
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
            # Get all ayah files for this surah, grouped by ayah number
            ayah_groups = get_ayah_files(source_dir, surah_num)
            
            # Copy all versions of each ayah
            for ayah_files in ayah_groups.values():
                for ayah_file in ayah_files:
                    shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
    
    print(f"✓ Prepared surahs {start_surah}-{end_surah} for {len(config['training'])} reciters")
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
            # Get all ayah files for this surah, grouped by ayah number
            ayah_groups = get_ayah_files(source_dir, surah_num)
            
            # Copy all versions of each ayah
            for ayah_files in ayah_groups.values():
                for ayah_file in ayah_files:
                    shutil.copy2(ayah_file, reciter_dir / ayah_file.name)
    
    print(f"✓ Prepared surahs {start_surah}-{end_surah} for {len(config['testing'])} reciters")
    return config

def main():
    print("\n🚀 Starting data preparation process...")
    
    # Load and prepare configuration (includes cleanup and initial save)
    config = load_split_config()
    
    # Select testing reciters
    config = select_testing_reciters(config)
    
    # Update statistics after testing selection
    config["n_testing_reciters"] = len(config["testing"])
    save_config(config)
    
    # Process training data
    config = prepare_training_data(config)
    
    # Process testing data
    config = prepare_testing_data(config)
    
    print("\n✨ Data preparation completed successfully!")
    print(f"📊 Final statistics:")
    print(f"  - Training reciters: {config['n_training_reciters']}")
    print(f"  - Testing reciters: {config['n_testing_reciters']}")
    print(f"  - Remaining reciters: {config['n_remaining_reciters']}")
    print(f"  - Total reciters: {config['total_reciters']}")
    print(f"  - Training range: {config['train_data_range']}")
    print(f"  - Testing range: {config['test_data_range']}")

if __name__ == "__main__":
    main() 