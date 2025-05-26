import os
import json
import requests
import logging
import warnings
from pathlib import Path
from tqdm import tqdm
import time
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL verification warnings
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format to work with our custom messages
)
logger = logging.getLogger(__name__)

def log_info(message):
    """Print formatted info message"""
    print(f"\nℹ️ {message}")

def log_error(message):
    """Print formatted error message"""
    print(f"\n❌ Error: {message}")

def log_success(message):
    """Print formatted success message"""
    print(f"\n✅ {message}")

def load_reciters():
    """Load reciters information from reciters.json"""
    try:
        with open('data/recitersAll.json', 'r', encoding='utf-8') as f:
            reciters_data = json.load(f)
            
            # Convert the new format to the format expected by the rest of the code
            reciters = {}
            for reciter_name, reciter_info in reciters_data.items():
                # Extract the servers field which contains the URLs
                if isinstance(reciter_info, dict) and "servers" in reciter_info:
                    reciters[reciter_name] = reciter_info["servers"]
                else:
                    # Handle legacy format or unexpected structure
                    reciters[reciter_name] = reciter_info
            
            log_success(f"Loaded {len(reciters)} reciters from configuration")
            return reciters
    except FileNotFoundError:
        log_error("data/reciters.json file not found!")
        raise
    except json.JSONDecodeError:
        log_error("Invalid JSON format in reciters.json!")
        raise

def load_surahs():
    """Load surah information from surahs.json"""
    try:
        with open('data/surahs.json', 'r', encoding='utf-8') as f:
            surahs = json.load(f)
            log_success(f"Loaded {len(surahs)} surahs from configuration")
            return surahs
    except FileNotFoundError:
        log_error("surahs.json file not found!")
        raise
    except json.JSONDecodeError:
        log_error("Invalid JSON format in surahs.json!")
        raise

def create_directory_structure():
    """Create the main directory and subdirectories for each reciter"""
    base_dir = Path('dataset')
    try:
        base_dir.mkdir(exist_ok=True)
        log_info(f"Created/Verified base directory: {base_dir}")
        
        for reciter in RECITERS:
            reciter_dir = base_dir / reciter
            reciter_dir.mkdir(exist_ok=True)
        
        log_success("Directory structure created successfully")
        return base_dir
    except Exception as e:
        log_error(f"Error creating directory structure: {e}")
        raise

def download_ayah(base_url, surah_num, ayah_num, output_path):
    """Download a single ayah and save it to the specified path"""
    file_num = f"{surah_num:03d}{ayah_num:03d}"
    url = f"{base_url}{file_num}.mp3"
    try:
        # Skip if file already exists and is not empty
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        log_error(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error while downloading {url}: {e}")
        return False

def download_reciter_files(reciter_name, url, surahs):
    """Download all ayahs for a reciter"""
    reciter_dir = Path('dataset') / reciter_name
    total_ayahs = sum(int(surahs[str(surah_num)]) for surah_num in [1] + list(range(78, 115)))
    log_info(f"Downloading files for {reciter_name}")
    with tqdm(total=total_ayahs, desc=f"📥 {reciter_name}", unit="file") as pbar:
        # Process Surah 1 first
        surah_num = "1"
        ayah_count = surahs[surah_num]
        for ayah_num in range(1, ayah_count + 1):
            output_path = reciter_dir / f"{int(surah_num):03d}{ayah_num:03d}.mp3"
            success = download_ayah(url, int(surah_num), ayah_num, output_path)
            pbar.update(1)
            if success:
                time.sleep(0.1)
        # Then process Surahs 78-114
        for surah_num in range(78, 115):
            surah_num_str = str(surah_num)
            ayah_count = surahs[surah_num_str]
            for ayah_num in range(1, ayah_count + 1):
                output_path = reciter_dir / f"{surah_num:03d}{ayah_num:03d}.mp3"
                success = download_ayah(url, surah_num, ayah_num, output_path)
                pbar.update(1)
                if success:
                    time.sleep(0.1)
    log_success(f"Completed downloading files for {reciter_name}")

def main():
    log_info("🚀 Starting Quran audio download process...")
    try:
        global RECITERS
        RECITERS = load_reciters()
        create_directory_structure()
        surahs = load_surahs()
        total_reciters = len(RECITERS)
        for i, (reciter_name, url) in enumerate(RECITERS.items(), 1):
            log_info(f"Processing reciter {i}/{total_reciters}: {reciter_name}")
            download_reciter_files(reciter_name, url, surahs)
        log_success(f"✨ Download completed successfully! Processed {total_reciters} reciters")
    except Exception as e:
        log_error(f"An error occurred during the download process: {e}")
        raise

if __name__ == "__main__":
    main()