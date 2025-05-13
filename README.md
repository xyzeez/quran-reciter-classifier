# Quran Reciter Classifier

A machine learning system for identifying Quran reciters from audio recordings using advanced audio processing and classification techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Preprocessing Data](#preprocessing-data)
  - [Training Models](#training-models)
  - [Testing Models](#testing-models)
  - [Making Predictions](#making-predictions)
  - [Using the API](#using-the-api)
- [Models](#models)
- [Audio Features](#audio-features)
- [Configuration](#configuration)

## 🔍 Overview

The Quran Reciter Classifier system uses machine learning to identify Quran reciters from audio recordings. The system extracts various audio features and uses classification models to determine the most likely reciter. It supports both traditional machine learning (Random Forest) and deep learning (BLSTM) approaches.

## ✨ Features

- **Audio Processing**: Handles various audio formats, performs noise reduction, and extracts meaningful segments
- **Feature Extraction**: Extracts comprehensive audio features including MFCCs, spectral features, and rhythm features
- **Multiple Models**: Supports Random Forest and BLSTM (Bidirectional Long Short-Term Memory) neural networks
- **Data Augmentation**: Implements pitch shifting, time stretching, and noise addition for robust training
- **Model Evaluation**: Provides detailed performance metrics and confusion matrices
- **REST API**: Offers a Flask-based API for easy integration with other applications
- **GPU Support**: Utilizes GPU acceleration for faster processing when available

## 📁 Project Structure

```
.
├── api-responses/          # API response cache and examples
│   ├── getReciter/        # Contains timestamped folders with debug data (audio, JSON) when server runs in debug mode
│   └── getAyah/           # Ayah identification responses
├── config/                # Configuration settings
├── data/                  # Training and test data
│   ├── train/             # Training audio files
│   ├── test/              # Test audio files
├── server/                # Flask server for API
│   ├── __init__.py        # Server package initializer
│   ├── app.py             # Main server application entry point
│   ├── app_factory.py     # Creates Flask app, initializes resources
│   ├── config.py          # Server-specific configuration
│   ├── requirements.txt   # Server-specific Python dependencies
│   ├── routes/            # API endpoint Blueprints (ayah.py, reciter.py)
│   ├── services/          # Business logic (ayah_service.py, reciter_service.py)
│   └── utils/             # Server utility functions and classes (e.g., audio_utils.py, quran_matcher.py, model_loader.py)
├── src/                   # Source code for core ML pipeline
│   ├── models/            # ML models implementation
│   ├── features/          # Feature extraction
│   ├── utils/             # Utility functions (e.g., distance_utils.py, gpu_utils.py, logging_utils.py)
│   ├── pipelines/         # Processing pipelines
│   ├── evaluation/        # Model evaluation
│   └── data/              # Data handling
├── scripts/               # Scripts for training, testing, etc.
├── processed/             # Preprocessed data (generated)
│   ├── train/             # Processed training data
│   └── test/              # Processed test data
├── models/                # Saved models (generated)
├── test_results/          # Test results (generated)
├── logs/                  # Log files (generated)
├── tools/                 # Project management and automation tools
├── requirements.txt       # Main project Python dependencies
└── README.md              # This file
```

## 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/xyzeez/quran-reciter-classifier.git
   cd quran-reciter-classifier
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare your data:
   - Place training audio files in `data/train/`
   - Place test audio files in `data/test/`

## 🚀 Usage

### Preprocessing Data

```bash
python scripts/preprocess.py [--mode {train,test}] [--no-augment]
```

Options:

- `--mode`: Mode to run in (`train` or `test`), default is `train`
- `--no-augment`: Skip data augmentation (saves time but reduces diversity)

Output:

- Creates a timestamped directory (e.g., `20240306_143208_preprocess`)
- Saves processed data to `processed/{mode}/TIMESTAMP_preprocess/`
- Creates a `latest` symlink to the most recent run

### Training Models

```bash
python scripts/train.py [--model-type MODEL_TYPE] [--preprocess-file-id PREPROCESS_FILE_ID]
```

Options:

- `--model-type`: Model type to use (`random_forest` or `blstm`). If not specified, uses default from config
- `--preprocess-file-id`: Specific preprocessing run ID to use (e.g., `20240306_143208_preprocess`). If not specified, uses the latest

Output:

- Creates a timestamped directory in `models/` (e.g., `20240306_152417_train`)
- Saves model file as `model_{type}.joblib`
- Creates a `latest` symlink to the most recent model

### Testing Models

```bash
python scripts/test.py [--model-file-id MODEL_FILE_ID] [--list-models] [--list-tests]
```

Options:

- `--model-file-id`: Training run ID to use (e.g., `20240306_152417_train`). If not specified, uses the latest
- `--list-models`: List all available model runs with their types and creation timestamps
- `--list-tests`: List all previous test runs with their accuracies

Output:

- Creates a timestamped directory in `test_results/` (e.g., `20240306_153012_test`)
- Generates performance metrics and visualizations
- Creates a `summary_report.json` with test results

### Making Predictions

```bash
python scripts/predict.py [--audio AUDIO_FILE] [--model-file-id MODEL_FILE_ID] [--true-label RECITER_NAME] [--list-models]
```

Options:

- `--audio`: Path to audio file to analyze (required)
- `--model-file-id`: Training run ID to use, uses latest if not specified
- `--true-label`: True reciter name for verification (optional)
- `--list-models`: List all available model runs

### Using the API

First, ensure you have installed the main project dependencies from `requirements.txt`. Then, install the API-specific dependencies:

```bash
pip install -r server/requirements.txt
```

Start the Flask server:

```bash
python -m server.app [--debug]
```

The server will start on `http://localhost:5000` by default.

#### API Endpoints

**POST** `/getReciter`

Identifies the Quran reciter from an audio file.

##### Request Format

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio`: Audio file (required)
    - Format: MP3 or WAV
    - Duration: 5-15 seconds
    - Sample Rate: 22050 Hz (will be converted if different)
  - `show_unreliable_predictions`: Boolean (optional)
    - Default: false
    - When true, returns predictions even if reliability criteria aren't met

##### Response Format

```json
{
  "reliable": true,
  "main_prediction": {
    "name": "Reciter Name",
    "confidence": 95.5,
    "nationality": "Country",
    "serverUrl": "https://example.com",
    "flagUrl": "https://example.com/flag.png",
    "imageUrl": "https://example.com/image.jpg"
  },
  "top_predictions": [
    {
      "name": "Reciter Name 1",
      "confidence": 95.5,
      "nationality": "Country 1",
      "serverUrl": "https://example.com/1",
      "flagUrl": "https://example.com/flag1.png",
      "imageUrl": "https://example.com/image1.jpg"
    }
    // ... more predictions
  ]
}
```

**POST** `/getAyah`

Identifies the Quranic verse from an audio recording.

##### Request Format

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio`: Audio file (required)
    - Format: MP3 or WAV (or other ffmpeg-supported formats)
    - Duration: 1-10 seconds (Recommended; Note: Limits exist in config but are not currently enforced by the endpoint)
    - Sample Rate: Any (will be automatically converted to 16000 Hz for processing)
  - `max_matches`: Maximum number of matches to return (optional, default: 5)
  - `min_confidence`: Minimum confidence threshold (0.0 to 1.0) for fuzzy matching score (optional, default: 0.70)

##### Response Format

```json
{
  "matches_found": true,
  "total_matches": 5,
  "matches": [
    {
      "surah_number": 105,
      "surah_name": "الفيل",
      "surah_name_en": "Al-Fil",
      "ayah_number": 5,
      "ayah_text": "فَجَعَلَهُمۡ كَعَصۡفٖ مَّأۡكُولِۭ",
      "confidence_score": 0.7749
    }
    // ... more matches sorted by confidence
  ],
  "best_match": {
    // highest confidence match that meets threshold
  }
}
```

Debug Mode Fields (when server started with --debug):

```json
{
  "transcription": "فَجَعَلَهُمْ كَعَصْفٍ مَأْكُوهُ",
  "debug_info": {
    "transcription": "فَجَعَلَهُمْ كَعَصْفٍ مَأْكُوهُ",
    "normalized_transcription": "فجعلهم كعصف ماكوه",
    "normalized_matches": [
      {
        "transcription": "فجعلهم كعصف ماكوه",
        "verse": "فجعلهم كعصف ماكول",
        "score": 0.7749
      }
      // ... more matches with normalization details
    ]
  }
}
```

## 🧠 Models

### Random Forest

A traditional machine learning approach that uses an ensemble of decision trees. Configured with:

- Number of estimators: 100
- Maximum depth: 10

### BLSTM (Bidirectional Long Short-Term Memory)

A deep learning approach that captures temporal patterns in audio features. Configured with:

- LSTM units: 64
- Dropout rate: 0.5 (enhanced from default 0.3)
- Dense units: 128
- Learning rate: 0.0005 (reduced from default 0.001)
- Weight decay: 0.01
- Batch size: 32
- Epochs: 50
- Early stopping patience: 10

## 🎵 Audio Features

The system extracts the following features from audio:

- **MFCCs**: Mel-Frequency Cepstral Coefficients (40 features)
- **Spectral Features**:
  - Chroma (12 features)
  - Mel spectrograms (128 bands)
  - Spectral contrast
  - Spectral rolloff
  - Spectral centroid
- **Rhythm Features**: Tempogram
- **Tonal Features**: Tonnetz
- **Additional Features**:
  - Zero crossing rate
  - RMS energy

## ⚙️ Configuration

The system uses multiple configuration files:

- `config/config.py`: Core system configuration (Used by `scripts/` and `src/`)
- `server/config.py`: API server configuration

### Server Configuration (`server/config.py`)

Key settings include:

- `HOST`, `PORT`, `DEBUG`: Network settings for the server.
- `MODEL_DIR`, `LATEST_MODEL_SYMLINK`, `MODEL_ID`: Paths and identifiers for loading the reciter model.
- `TOP_N_PREDICTIONS`: Number of predictions to return in the `/getReciter` response (default: 5).
- `CONFIDENCE_THRESHOLD`, `SECONDARY_CONFIDENCE_THRESHOLD`, `MAX_CONFIDENCE_DIFF`: Parameters related to the _original_ training and threshold calculations stored within the model. The actual reliability check in the service uses thresholds loaded _with the model_, not these specific config values directly at runtime for the check itself.
- `MIN_AUDIO_DURATION`, `MAX_AUDIO_DURATION`: Duration limits for `/getReciter` audio.
- `AYAH_MIN_DURATION`, `AYAH_MAX_DURATION`: Duration limits for `/getAyah` audio.

### Audio Processing Configuration

- `SAMPLE_RATE`: 22050 Hz
- `MIN_DURATION`: 5 seconds (for reciter identification)
- `MAX_DURATION`: 15 seconds (for reciter identification)
- `MIN_AYAH_DURATION`: 1 second
- `MAX_AYAH_DURATION`: 10 seconds

### Feature Extraction Configuration

- `N_MFCC`: 40 (number of MFCC features)
- `N_CHROMA`: 12 (number of chroma features)
- `N_MEL_BANDS`: 128 (number of mel spectrogram bands)

### Data Augmentation Configuration

- `PITCH_STEPS`: [1.5, -1.5]
- `TIME_STRETCH_RATES`: [0.9, 1.1]
- `NOISE_FACTOR`: 0.005
- `VOLUME_ADJUST`: 0.8

## 🚀 Usage

### Preprocessing Data

```bash
python scripts/preprocess.py [--mode {train,test}] [--no-augment]
```

Options:

- `--mode`: Mode to run in (`train` or `test`), default is `train`
- `--no-augment`: Skip data augmentation (saves time but reduces diversity)

Output:

- Creates a timestamped directory (e.g., `20240306_143208_preprocess`)
- Saves processed data to `processed/{mode}/TIMESTAMP_preprocess/`
- Creates a `latest` symlink to the most recent run

### Training Models

```bash
python scripts/train.py [--model-type MODEL_TYPE] [--preprocess-file-id PREPROCESS_FILE_ID]
```

Options:

- `--model-type`: Model type to use (`random_forest` or `blstm`). If not specified, uses default from config
- `--preprocess-file-id`: Specific preprocessing run ID to use (e.g., `20240306_143208_preprocess`). If not specified, uses the latest

Output:

- Creates a timestamped directory in `models/` (e.g., `20240306_152417_train`)
- Saves model file as `model_{type}.joblib`
- Creates a `latest` symlink to the most recent model

### Testing Models

```bash
python scripts/test.py [--model-file-id MODEL_FILE_ID] [--list-models] [--list-tests]
```

Options:

- `--model-file-id`: Training run ID to use (e.g., `20240306_152417_train`). If not specified, uses the latest
- `--list-models`: List all available model runs with their types and creation timestamps
- `--list-tests`: List all previous test runs with their accuracies

Output:

- Creates a timestamped directory in `test_results/` (e.g., `20240306_153012_test`)
- Generates performance metrics and visualizations
- Creates a `summary_report.json` with test results

### Making Predictions

```bash
python scripts/predict.py [--audio AUDIO_FILE] [--model-file-id MODEL_FILE_ID] [--true-label RECITER_NAME] [--list-models]
```

Options:

- `--audio`: Path to audio file to analyze (required)
- `--model-file-id`: Training run ID to use, uses latest if not specified
- `--true-label`: True reciter name for verification (optional)
- `--list-models`: List all available model runs

### Using the API

First, ensure you have installed the main project dependencies from `requirements.txt`. Then, install the API-specific dependencies:

```bash
pip install -r server/requirements.txt
```

Start the Flask server:

```bash
python -m server.app [--debug]
```

The server will start on `http://localhost:5000` by default.

#### API Endpoints

**POST** `/getReciter`

Identifies the Quran reciter from an audio file.

##### Request Format

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio`: Audio file (required)
    - Format: MP3 or WAV
    - Duration: 5-15 seconds
    - Sample Rate: 22050 Hz (will be converted if different)
  - `show_unreliable_predictions`: Boolean (optional)
    - Default: false
    - When true, returns predictions even if reliability criteria aren't met

##### Response Format

```json
{
  "reliable": true,
  "main_prediction": {
    "name": "Reciter Name",
    "confidence": 95.5,
    "nationality": "Country",
    "serverUrl": "https://example.com",
    "flagUrl": "https://example.com/flag.png",
    "imageUrl": "https://example.com/image.jpg"
  },
  "top_predictions": [
    {
      "name": "Reciter Name 1",
      "confidence": 95.5,
      "nationality": "Country 1",
      "serverUrl": "https://example.com/1",
      "flagUrl": "https://example.com/flag1.png",
      "imageUrl": "https://example.com/image1.jpg"
    }
    // ... more predictions
  ]
}
```

**POST** `/getAyah`

Identifies the Quranic verse from an audio recording.

##### Request Format

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio`: Audio file (required)
    - Format: MP3 or WAV (or other ffmpeg-supported formats)
    - Duration: 1-10 seconds (Recommended; Note: Limits exist in config but are not currently enforced by the endpoint)
    - Sample Rate: Any (will be automatically converted to 16000 Hz for processing)
  - `max_matches`: Maximum number of matches to return (optional, default: 5)
  - `min_confidence`: Minimum confidence threshold (0.0 to 1.0) for fuzzy matching score (optional, default: 0.70)

##### Response Format

```json
{
  "matches_found": true,
  "total_matches": 5,
  "matches": [
    {
      "surah_number": 105,
      "surah_name": "الفيل",
      "surah_name_en": "Al-Fil",
      "ayah_number": 5,
      "ayah_text": "فَجَعَلَهُمۡ كَعَصۡفٖ مَّأۡكُولِۭ",
      "confidence_score": 0.7749
    }
    // ... more matches sorted by confidence
  ],
  "best_match": {
    // highest confidence match that meets threshold
  }
}
```

Debug Mode Fields (when server started with --debug):

```json
{
  "transcription": "فَجَعَلَهُمْ كَعَصْفٍ مَأْكُوهُ",
  "debug_info": {
    "transcription": "فَجَعَلَهُمْ كَعَصْفٍ مَأْكُوهُ",
    "normalized_transcription": "فجعلهم كعصف ماكوه",
    "normalized_matches": [
      {
        "transcription": "فجعلهم كعصف ماكوه",
        "verse": "فجعلهم كعصف ماكول",
        "score": 0.7749
      }
      // ... more matches with normalization details
    ]
  }
}
```
