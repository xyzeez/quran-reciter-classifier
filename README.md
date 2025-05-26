# Quran Reciter Classifier

A comprehensive machine learning system for identifying Quran reciters from audio recordings and identifying specific Quranic verses (Ayahs) using advanced audio processing, feature extraction, and classification techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Training Pipeline](#training-pipeline)
  - [Testing Pipeline](#testing-pipeline)
  - [Prediction Pipeline](#prediction-pipeline)
  - [Server/API Pipeline](#serverapi-pipeline)
- [Models](#models)
- [Audio Features](#audio-features)
- [API Documentation](#api-documentation)
- [Development Tools](#development-tools)
- [Data Management](#data-management)

## 🔍 Overview

The Quran Reciter Classifier is a dual-purpose machine learning system that:

1. **Reciter Identification**: Identifies Quran reciters from audio recordings using traditional ML (Random Forest) and deep learning (BLSTM) approaches
2. **Ayah Identification**: Identifies specific Quranic verses from audio using Whisper ASR and fuzzy text matching

The system supports both traditional machine learning and deep learning approaches, with comprehensive preprocessing, training, testing, and deployment pipelines.

## 🏗️ System Architecture

The system consists of five main pipelines:

1. **Preprocessing Pipeline**: Audio processing and feature extraction
2. **Training Pipeline**: Model training with Random Forest or BLSTM
3. **Testing Pipeline**: Model evaluation and performance analysis
4. **Prediction Pipeline**: Single audio file prediction
5. **Server Pipeline**: REST API for real-time inference

## ✨ Features

- **Multi-Model Support**: Random Forest and Bidirectional LSTM models
- **Comprehensive Audio Processing**: Noise reduction, normalization, and feature extraction
- **Advanced Feature Engineering**: MFCCs, spectral features, rhythm features, and tonal features
- **Data Augmentation**: Pitch shifting, time stretching, noise addition, and volume adjustment
- **Reliability Analysis**: Distance-based prediction verification and confidence scoring
- **REST API**: Flask-based server with endpoints for both reciter and Ayah identification
- **Extensive Evaluation**: Confusion matrices, performance metrics, and visualization tools
- **GPU Support**: CUDA acceleration for faster processing
- **Robust Logging**: Comprehensive logging and progress tracking
- **Modular Design**: Extensible architecture with clear separation of concerns

## 📁 Project Structure

```
.
├── api-responses/              # API response cache and debug data
│   ├── getReciter/            # Reciter identification debug data
│   └── getAyah/               # Ayah identification debug data
├── config/                    # Configuration settings
│   ├── __init__.py           # Configuration package initializer
│   └── config.py             # Main configuration parameters
├── data/                      # Input data directories
│   ├── train/                # Training audio files (organized by reciter)
│   ├── test/                 # Test audio files (organized by reciter)
│   ├── quran.json            # Quran text data for Ayah identification
│   ├── reciters.json         # Reciter metadata for training models
│   ├── recitersAll.json      # Complete reciter database
│   ├── surahs.json          # Surah information
│   └── dataset_splits.json   # Training/testing split configuration
├── processed/                 # Preprocessed data (generated)
│   ├── train/                # Training data processing outputs
│   │   ├── latest/           # Symlink to most recent preprocessing
│   │   └── YYYYMMDD_HHMMSS_preprocess/  # Timestamped preprocessing runs
│   └── test/                 # Test data processing outputs
├── models/                    # Trained models (generated)
│   ├── latest/               # Symlink to most recent model
│   └── YYYYMMDD_HHMMSS_train/  # Timestamped training runs
│       ├── model_random_forest.joblib  # Model file
│       ├── training_metadata.json      # Training configuration
│       ├── training_summary.txt        # Human-readable summary
│       └── visualizations/             # Training visualizations
├── test_results/              # Test results (generated)
│   └── YYYYMMDD_HHMMSS_test/  # Timestamped test runs
│       ├── detailed_results.csv        # Per-file test results
│       ├── summary_report.json         # Test metrics
│       ├── test_summary.txt           # Human-readable summary
│       └── confusion_matrix_*.png     # Confusion matrices
├── logs/                      # Log files (generated)
├── server/                    # Flask API server
│   ├── __init__.py           # Server package initializer
│   ├── app.py                # Main server entry point
│   ├── app_factory.py        # Flask application factory
│   ├── config.py             # Server-specific configuration
│   ├── requirements.txt      # Server dependencies
│   ├── routes/               # API endpoint blueprints
│   │   ├── ayah.py          # Ayah identification endpoints
│   │   ├── reciter.py       # Reciter identification endpoints
│   │   ├── health.py        # Health check endpoints
│   │   └── models.py        # Model information endpoints
│   ├── services/             # Business logic layer
│   │   ├── ayah_service.py  # Ayah identification service
│   │   └── reciter_service.py  # Reciter identification service
│   └── utils/                # Server utility functions
│       ├── audio_processor_matcher.py  # Audio processing utilities
│       ├── logging_config.py           # Logging configuration
│       ├── model_loader.py             # Model loading utilities
│       ├── quran_data.py              # Quran data management
│       ├── quran_matcher.py           # Whisper-based Ayah matching
│       └── text_utils.py              # Text processing utilities
├── src/                       # Core ML pipeline source code
│   ├── data/                 # Data handling modules
│   │   ├── __init__.py      # Data package exports
│   │   ├── augmentation.py  # Audio data augmentation
│   │   ├── loader.py        # Audio file loading
│   │   └── preprocessing.py # Audio preprocessing
│   ├── features/             # Feature extraction
│   │   ├── __init__.py      # Features package exports
│   │   └── extractors.py    # Audio feature extraction
│   ├── models/               # ML model implementations
│   │   ├── __init__.py      # Models package exports
│   │   ├── base_model.py    # Abstract base model class
│   │   ├── random_forest.py # Random Forest implementation
│   │   ├── blstm_model.py   # BLSTM implementation
│   │   └── model_factory.py # Model creation and loading
│   ├── pipelines/            # ML pipeline implementations
│   │   ├── __init__.py      # Pipelines package exports
│   │   ├── preprocess_pipeline.py  # Preprocessing pipeline
│   │   ├── train_pipeline.py       # Training pipeline
│   │   ├── test_pipeline.py        # Testing pipeline
│   │   └── predict_pipeline.py     # Prediction pipeline
│   ├── evaluation/           # Model evaluation tools
│   │   ├── __init__.py      # Evaluation package exports
│   │   ├── metrics.py       # Performance metrics
│   │   └── visualization.py # Evaluation visualizations
│   └── utils/                # Utility functions
│       ├── __init__.py      # Utils package exports
│       ├── distance_utils.py    # Distance calculations and reliability
│       ├── gpu_utils.py         # GPU detection utilities
│       └── logging_utils.py     # Logging utilities
├── scripts/                   # Entry point scripts
│   ├── __init__.py           # Scripts package initializer
│   ├── preprocess.py         # Preprocessing script entry point
│   ├── train.py              # Training script entry point
│   ├── test.py               # Testing script entry point
│   └── predict.py            # Prediction script entry point
├── tools/                     # Project management tools
│   ├── download_dataset.py   # Dataset download utility
│   ├── prepare_data.py       # Data preparation utility
│   └── test_case_runner.py   # Comprehensive API testing tool
├── requirements.txt           # Main project dependencies
└── README.md                 # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- GPU with CUDA support (optional, for acceleration)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd quran-reciter-classifier
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Main project dependencies
pip install -r requirements.txt

# Server-specific dependencies (if using API)
pip install -r server/requirements.txt
```

### Step 4: Prepare Data Structure

```bash
# Create necessary directories
mkdir -p data/train data/test

# Place audio files in appropriate directories:
# data/train/{reciter_name}/*.mp3
# data/test/{reciter_name}/*.mp3
```

## ⚙️ Configuration

The system uses multiple configuration files:

### Main Configuration (`config/config.py`)

Key parameters include:

- **Directories**: Data paths, output locations
- **Audio Processing**: Sample rates, duration limits
- **Feature Extraction**: MFCC count, spectral features
- **Model Parameters**: Random Forest and BLSTM settings
- **Reliability Thresholds**: Confidence and distance thresholds

### Server Configuration (`server/config.py`)

Server-specific settings:

- **Network**: Host, port, debug mode
- **Model Loading**: Model paths and identifiers
- **API Limits**: Audio duration constraints
- **Whisper Settings**: ASR model configuration

## 🚀 Usage

### Data Preparation

#### Option 1: Download Dataset (Automated)

```bash
# Download audio files from configured sources
python tools/download_dataset.py

# Prepare training/testing splits
python tools/prepare_data.py
```

#### Option 2: Manual Data Organization

Place audio files in the following structure:

```
data/
├── train/
│   ├── reciter1/
│   │   ├── audio1.mp3
│   │   └── audio2.mp3
│   └── reciter2/
│       ├── audio1.mp3
│       └── audio2.mp3
└── test/
    ├── reciter1/
    │   └── test_audio.mp3
    └── unknown_reciter/
        └── test_audio.mp3
```

### Preprocessing Pipeline

**Entry Point**: `scripts/preprocess.py`  
**Core Implementation**: `src/pipelines/preprocess_pipeline.py`

#### Command Syntax

```bash
python scripts/preprocess.py [--mode {train,test}] [--no-augment]
```

#### Arguments

- `--mode`: Processing mode (`train` or `test`) - Default: `train`
- `--no-augment`: Skip data augmentation (faster processing, reduced diversity)

#### Examples

```bash
# Preprocess training data with augmentation
python scripts/preprocess.py --mode train

# Preprocess test data (no augmentation)
python scripts/preprocess.py --mode test

# Preprocess training data without augmentation
python scripts/preprocess.py --mode train --no-augment
```

#### Process Flow

1. **Audio Loading**: Load MP3/WAV files using librosa
2. **Noise Reduction**: Apply spectral noise reduction
3. **Normalization**: Amplitude normalization and silence trimming
4. **Resampling**: Convert to 22050 Hz sample rate
5. **Feature Extraction**: Extract comprehensive audio features
6. **Data Augmentation** (train mode only):
   - Pitch shifting (±1.0, ±1.5 semitones)
   - Time stretching (0.9x, 1.1x rates)
   - Noise addition (0.003 factor)
   - Volume adjustment (0.9x scaling)

#### Outputs

**Location**: `processed/{mode}/YYYYMMDD_HHMMSS_preprocess/`

- `all_features.npy`: Extracted features array
- `all_metadata.csv`: File metadata and labels
- `preprocessing_metadata.json`: Run configuration and statistics
- `file_inventory.csv`: Processing status per file
- `file_feature_map.csv`: Feature index mapping
- `reciter_profiles.json`: Per-reciter processing statistics
- `preprocessing_summary.txt`: Human-readable summary
- `feature_statistics.json`: Feature distribution statistics
- `visualizations/`: Feature distribution and correlation plots

**Symlink**: `processed/{mode}/latest` → most recent run

### Training Pipeline

**Entry Point**: `scripts/train.py`  
**Core Implementation**: `src/pipelines/train_pipeline.py`

#### Command Syntax

```bash
python scripts/train.py [--model-type {random_forest,blstm}] [--preprocess-file-id PREPROCESS_FILE_ID]
```

#### Arguments

- `--model-type`: Model architecture (`random_forest` or `blstm`) - Default: from config
- `--preprocess-file-id`: Specific preprocessing run ID (e.g., `20240306_143208_preprocess`) - Default: latest

#### Examples

```bash
# Train Random Forest with latest preprocessing
python scripts/train.py --model-type random_forest

# Train BLSTM with specific preprocessing run
python scripts/train.py --model-type blstm --preprocess-file-id 20240306_143208_preprocess

# Train with default model type from config
python scripts/train.py
```

#### Process Flow

1. **Data Loading**: Load preprocessed features and metadata
2. **Train/Validation Split**: 80/20 split with stratification
3. **Model Training**:
   - **Random Forest**: 100 estimators, max depth 10
   - **BLSTM**: Bidirectional LSTM with attention mechanism
4. **Reliability Metrics**: Calculate centroids and distance thresholds
5. **Model Evaluation**: Performance metrics and validation
6. **Visualization**: Confusion matrices and feature importance

#### Model Architectures

**Random Forest**:

- 100 estimators with max depth 10
- Calibrated probabilities using sigmoid method
- Feature importance analysis
- Cross-validation with 5 folds

**BLSTM (Bidirectional LSTM)**:

- 64 LSTM units with bidirectional processing
- Attention mechanism for temporal focus
- 0.5 dropout rate for regularization
- 128 dense units in classification layer
- Learning rate: 0.0005 with weight decay: 0.01
- Early stopping with patience: 10 epochs

#### Outputs

**Location**: `models/YYYYMMDD_HHMMSS_train/`

- `model_{type}.joblib`: Trained model file
- `training_metadata.json`: Complete training configuration
- `training_summary.txt`: Human-readable training summary
- `visualizations/confusion_matrix.png`: Training confusion matrix
- `visualizations/feature_importance.png`: Feature importance (Random Forest only)

**Symlink**: `models/latest` → most recent training run

### Testing Pipeline

**Entry Point**: `scripts/test.py`  
**Core Implementation**: `src/pipelines/test_pipeline.py`

#### Command Syntax

```bash
python scripts/test.py [--model-file-id MODEL_FILE_ID] [--list-models] [--list-tests]
```

#### Arguments

- `--model-file-id`: Training run ID (e.g., `20240306_152417_train`) - Default: latest
- `--list-models`: List all available trained models
- `--list-tests`: List all previous test runs

#### Examples

```bash
# Test latest model
python scripts/test.py

# Test specific model
python scripts/test.py --model-file-id 20240306_152417_train

# List available models
python scripts/test.py --list-models

# List previous test results
python scripts/test.py --list-tests
```

#### Process Flow

1. **Model Loading**: Load specified or latest trained model
2. **Test Data Loading**: Load preprocessed test features
3. **Prediction Generation**: Generate predictions for all test samples
4. **Reliability Analysis**: Analyze prediction confidence and distance metrics
5. **Performance Evaluation**: Calculate accuracy, precision, recall, F1-score
6. **Confusion Matrices**: Separate matrices for training vs non-training reciters
7. **Statistical Analysis**: Per-reciter performance breakdown

#### Evaluation Metrics

- **Overall Accuracy**: Correct predictions / Total predictions
- **Reliability Rate**: Reliable predictions / Total predictions
- **False Positive Rate**: Incorrect reliable predictions for unknown reciters
- **Per-Reciter Statistics**: Individual reciter performance analysis

#### Outputs

**Location**: `test_results/YYYYMMDD_HHMMSS_test/`

- `detailed_results.csv`: Per-file prediction results
- `summary_report.json`: Complete test metrics
- `test_summary.txt`: Human-readable test summary
- `test_metadata.json`: Test configuration and system info
- `confusion_matrix_training_reciters.png`: Confusion matrix for known reciters
- `confusion_matrix_nontraining_reciters.png`: Confusion matrix for unknown reciters

### Prediction Pipeline

**Entry Point**: `scripts/predict.py`  
**Core Implementation**: `src/pipelines/predict_pipeline.py`

#### Command Syntax

```bash
python scripts/predict.py --audio AUDIO_FILE [--model-file-id MODEL_FILE_ID] [--true-label RECITER_NAME] [--list-models]
```

#### Arguments

- `--audio`: Path to audio file (required)
- `--model-file-id`: Training run ID - Default: latest
- `--true-label`: True reciter name for verification (optional)
- `--list-models`: List available models

#### Examples

```bash
# Predict reciter for audio file
python scripts/predict.py --audio path/to/audio.mp3

# Predict with verification
python scripts/predict.py --audio path/to/audio.mp3 --true-label "Mishary Alafasy"

# Use specific model
python scripts/predict.py --audio path/to/audio.mp3 --model-file-id 20240306_152417_train

# List available models
python scripts/predict.py --list-models
```

#### Process Flow

1. **Audio Loading**: Load and validate audio file
2. **Preprocessing**: Apply same preprocessing as training
3. **Feature Extraction**: Extract audio features
4. **Prediction**: Generate reciter prediction with confidence
5. **Reliability Analysis**: Analyze prediction reliability
6. **Visualization**: Generate prediction analysis plots

#### Outputs

**Location**: `logs/prediction_results_YYYYMMDD_HHMMSS/`

- `prediction_analysis.png`: Confidence and distance analysis visualization
- `prediction_summary.json`: Prediction results and metadata

### Server/API Pipeline

**Entry Point**: `server/app.py`  
**Core Implementation**: `server/app_factory.py`

#### Command Syntax

```bash
python -m server.app [--host HOST] [--port PORT] [--debug]
```

#### Arguments

- `--host`: Hostname to bind (default: 0.0.0.0)
- `--port`: Port to bind (default: 5000)
- `--debug`: Enable debug mode with enhanced logging

#### Examples

```bash
# Start server with defaults
python -m server.app

# Start with custom host and port
python -m server.app --host localhost --port 8080

# Start in debug mode
python -m server.app --debug
```

#### Initialization Process

1. **Logging Setup**: Configure Rich logging with appropriate levels
2. **Quran Data Loading**: Load Quran text data for Ayah identification
3. **Model Loading**: Initialize reciter identification model
4. **Whisper Initialization**: Setup QuranMatcher with Whisper ASR
5. **Blueprint Registration**: Register API endpoints

#### Debug Mode Features

- Enhanced logging with stack traces
- Debug data saving for requests/responses
- Detailed error reporting
- Request timing information

## 🧠 Models

### Random Forest

A traditional ensemble learning approach:

- **Architecture**: 100 decision trees with max depth 10
- **Calibration**: Sigmoid probability calibration
- **Features**: Full feature set with importance analysis
- **Validation**: 5-fold cross-validation
- **Advantages**: Fast training, interpretable, robust to overfitting

### BLSTM (Bidirectional Long Short-Term Memory)

A deep learning approach for temporal pattern recognition:

- **Architecture**: Bidirectional LSTM with attention mechanism
- **Input**: First 13 MFCC coefficients arranged in sequences
- **LSTM Units**: 64 units per direction (128 total)
- **Attention**: Simple attention layer for temporal focus
- **Regularization**: 0.5 dropout rate and L2 weight decay
- **Training**: Adam optimizer with learning rate scheduling
- **Advantages**: Captures temporal patterns, attention mechanism, state-of-the-art performance

### Reliability Analysis

Both models use distance-based reliability verification:

- **Centroids**: Calculate class centroids in feature space
- **Thresholds**: 95th percentile of intra-class distances
- **Verification**: Multiple criteria for prediction reliability:
  - Confidence ≥ 95%
  - Secondary confidence < 10%
  - Confidence difference ≥ 80%
  - Distance ratio ≤ 1.0

## 🎵 Audio Features

The system extracts comprehensive audio features optimized for voice identification:

### Core Features (32 MFCC Coefficients)

- **MFCCs**: Mel-Frequency Cepstral Coefficients capturing vocal tract characteristics
- **Delta MFCCs**: First-order derivatives capturing temporal changes
- **Delta-Delta MFCCs**: Second-order derivatives capturing acceleration

### Spectral Features

- **Chroma (12 features)**: Pitch class profiles for tonal analysis
- **Mel Spectrograms (64 bands)**: Perceptually-motivated frequency representation
- **Spectral Contrast**: Difference between peaks and valleys in spectrum
- **Spectral Rolloff**: Frequency below which 85% of energy is concentrated
- **Spectral Centroid**: Center of mass of the spectrum

### Temporal Features

- **Zero Crossing Rate**: Rate of sign changes in the signal
- **RMS Energy**: Root mean square energy measure
- **Tempogram**: Rhythm and tempo analysis

### Tonal Features

- **Tonnetz**: Tonal centroid features for harmonic analysis

### Feature Processing

- **Normalization**: All features are normalized to unit variance
- **Aggregation**: Mean values computed across time frames
- **Total Dimensions**: ~350+ features per audio sample

## 📡 API Documentation

### Base URL

```
http://localhost:5000
```

### Endpoints

#### POST /getReciter

Identify Quran reciter from audio file.

**Request**:

```bash
curl -X POST http://localhost:5000/getAyah \
  -F "audio=@ayah_audio.mp3" \
  -F "max_matches=5" \
  -F "min_confidence=0.70"
```

**Parameters**:

- `audio`: Audio file (MP3/WAV, 1-10 seconds recommended)
- `max_matches`: Maximum number of matches to return (default: 5)
- `min_confidence`: Minimum confidence threshold 0.0-1.0 (default: 0.70)

**Response**:

```json
{
  "matches_found": true,
  "total_matches": 3,
  "matches": [
    {
      "surah_number": "١٠٥",
      "surah_number_en": 105,
      "surah_name": "الفيل",
      "surah_name_en": "Al-Fil",
      "ayah_number": "٥",
      "ayah_number_en": 5,
      "ayah_text": "فَجَعَلَهُمۡ كَعَصۡفٖ مَّأۡكُولِۭ",
      "confidence_score": 0.7749,
      "unicode": "🐘"
    }
  ],
  "best_match": {
    "surah_number": "١٠٥",
    "surah_number_en": 105,
    "surah_name": "الفيل",
    "surah_name_en": "Al-Fil",
    "ayah_number": "٥",
    "ayah_number_en": 5,
    "ayah_text": "فَجَعَلَهُمۡ كَعَصۡفٖ مَّأۡكُولِۭ",
    "confidence_score": 0.7749,
    "unicode": "🐘"
  },
  "response_time_ms": 2150.3
}
```

**Debug Mode Response** (when server started with `--debug`):

```json
{
  "matches_found": true,
  "total_matches": 3,
  "matches": [...],
  "best_match": {...},
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
    ]
  }
}
```

#### GET /getAllReciters

Get list of all known reciters from the system.

**Request**:

```bash
curl -X GET http://localhost:5000/getAllReciters
```

**Response**:

```json
{
  "reciters": [
    {
      "name": "Mishary Alafasy",
      "nationality": "Kuwait",
      "flagUrl": "https://flags.example.com/kw.png",
      "imageUrl": "https://images.example.com/mishary.jpg",
      "serverUrl": "https://server.example.com"
    }
  ]
}
```

#### GET /health

Check the health status of all system components.

**Request**:

```bash
curl -X GET http://localhost:5000/health
```

**Response**:

```json
{
  "status": "ok",
  "services": {
    "reciter_model": "loaded",
    "quran_data": "loaded",
    "quran_matcher": "initialized"
  }
}
```

#### GET /models

Get detailed information about loaded models and their configurations.

**Request**:

```bash
curl -X GET http://localhost:5000/models
```

**Response**:

```json
{
  "reciter_model": {
    "status": "loaded",
    "details": {
      "model_type": "RandomForest",
      "model_id": "20240306_152417_train",
      "training_parameters": {
        "n_samples": 5000,
        "n_features": 350,
        "n_classes": 10,
        "classes": ["Mishary Alafasy", "Abdul Rahman Al-Sudais", ...],
        "cross_validation_mean": 0.94,
        "model_parameters": {
          "n_estimators": 100,
          "max_depth": 10
        }
      }
    }
  },
  "ayah_matcher": {
    "status": "initialized",
    "details": {
      "matcher_type": "Whisper",
      "whisper_model_id": "tarteel-ai/whisper-base-ar-quran",
      "device": "CUDA",
      "normalized_verses_count": 6236,
      "all_verses_count": 6236,
      "quran_data_source_surahs": 114,
      "service_defaults": {
        "max_matches": 5,
        "min_confidence": 0.70
      }
    }
  }
}
```

### Error Responses

All endpoints return appropriate HTTP status codes and error messages:

```json
{
  "error": "No audio file provided. Key must be 'audio'."
}
```

Common status codes:

- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## 🛠️ Development Tools

### Test Case Runner

Comprehensive API testing tool for validating system performance:

```bash
python tools/test_case_runner.py [--test-case TC_ID] [--endpoint URL]
```

**Test Cases**:

- `tc-001`: Reciter ID with clean audio
- `tc-002`: Non-training reciter handling (clean)
- `tc-003`: Ayah ID with clean audio
- `tc-004`: Reciter ID with noisy audio
- `tc-005`: Non-training reciter handling (noisy)
- `tc-006`: Ayah ID with noisy audio
- `tc-007`: Reciter ID with very short clips
- `tc-008`: Reciter ID with silence/non-speech
- `tc-009`: Ayah ID with very short clips
- `tc-010`: Ayah ID with silence/non-speech

**Example**:

```bash
# Run all test cases
python tools/test_case_runner.py --endpoint http://localhost:5000

# Run specific test case
python tools/test_case_runner.py --test-case tc-001 --endpoint http://localhost:5000
```

**Output**: `test-cases-report/YYYYMMDD_HHMMSS_test/`

- CSV reports with detailed results
- JSON data for programmatic analysis
- Text summaries with statistics
- Model information metadata

### Dataset Management

#### Download Dataset

```bash
python tools/download_dataset.py
```

Downloads audio files from configured sources based on `data/recitersAll.json` and `data/surahs.json`.

#### Prepare Data Splits

```bash
python tools/prepare_data.py
```

Organizes downloaded data into training and testing splits based on `data/dataset_splits.json`.

## 💾 Data Management

### Configuration Files

#### `data/quran.json`

Complete Quran text data structured as:

```json
[
  {
    "id": 1,
    "name": "الفاتحة",
    "transliteration": "Al-Fatihah",
    "translation": "The Opening",
    "unicode": "🕌",
    "verses": [
      {
        "id": 1,
        "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
      }
    ]
  }
]
```

#### `data/reciters.json`

Reciter metadata for training models:

```json
{
  "Mishary Alafasy": {
    "nationality": "Kuwait",
    "flagUrl": "https://flags.example.com/kw.png",
    "imageUrl": "https://images.example.com/mishary.jpg",
    "servers": ["https://server1.example.com", "https://server2.example.com"]
  }
}
```

#### `data/recitersAll.json`

Complete reciter database with all available reciters.

#### `data/dataset_splits.json`

Training/testing split configuration:

```json
{
  "training": ["Mishary Alafasy", "Abdul Rahman Al-Sudais"],
  "testing": ["Maher Al Mueaqly"],
  "train_data_range": "1-1",
  "test_data_range": "78-114",
  "n_training_reciters": 2,
  "n_testing_reciters": 1
}
```

### File Organization Patterns

#### Timestamped Directories

All processing runs create timestamped directories:

- Format: `YYYYMMDD_HHMMSS_operation`
- Example: `20240306_143208_preprocess`

#### Symlinks for Latest

Each operation maintains a `latest` symlink pointing to the most recent run:

- `processed/train/latest` → most recent preprocessing
- `models/latest` → most recent training
- Enables easy reference without knowing specific timestamps

#### Output Structure

Each operation generates comprehensive outputs:

- **Primary Data**: Main results (features, models, predictions)
- **Metadata**: Configuration and run information
- **Summaries**: Human-readable reports
- **Visualizations**: Charts and analysis plots
- **Debug Data**: Detailed processing information

### GPU Support

The system automatically detects and utilizes GPU acceleration:

```python
# GPU detection in config
USE_GPU = True  # Enable GPU usage if available

# Automatic device selection
device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
```

**GPU Benefits**:

- **Preprocessing**: Faster audio processing with PyTorch tensors
- **BLSTM Training**: Significant speedup for neural network training
- **Whisper ASR**: Accelerated speech recognition for Ayah identification

### Logging and Monitoring

#### Structured Logging

- **Rich Console Output**: Enhanced terminal output with progress bars
- **File Logging**: Detailed logs saved to `logs/` directory
- **Timestamped Logs**: Each operation generates timestamped log files
- **Debug Levels**: Configurable logging levels (DEBUG, INFO, WARNING, ERROR)

#### Progress Tracking

- **tqdm Progress Bars**: Real-time progress indication
- **Rich Progress**: Enhanced progress visualization
- **ETA Calculations**: Estimated time to completion
- **File-by-File Tracking**: Individual file processing status

#### Performance Monitoring

- **Response Time Tracking**: API response time measurement
- **Memory Usage**: GPU and CPU memory monitoring
- **Processing Statistics**: Detailed timing information
- **Error Tracking**: Comprehensive error logging and reporting

### Reliability and Robustness

#### Error Handling

- **Graceful Degradation**: System continues operation on partial failures
- **Comprehensive Error Messages**: Detailed error reporting
- **Automatic Recovery**: Retry mechanisms for transient failures
- **Validation**: Input validation at all pipeline stages

#### Quality Assurance

- **Cross-Validation**: Statistical validation of model performance
- **Reliability Metrics**: Multiple reliability indicators
- **Confidence Scoring**: Prediction confidence assessment
- **Unknown Handling**: Proper handling of unknown/out-of-domain inputs

#### Data Integrity

- **Checksum Validation**: File integrity verification
- **Metadata Tracking**: Complete processing history
- **Version Control**: Configuration and data versioning
- **Reproducibility**: Deterministic processing with random seeds

## 🔍 Troubleshooting

### Common Issues

#### 1. FFmpeg Not Found

```bash
# Install FFmpeg
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### 2. GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA toolkit if needed
# Follow instructions at https://pytorch.org/get-started/locally/
```

#### 3. Model Loading Issues

```bash
# Check model directory
ls -la models/latest/

# Retrain if model is corrupted
python scripts/train.py --model-type random_forest
```

#### 4. Memory Issues

```bash
# Reduce batch size in config/config.py
BATCH_SIZE = 16  # Reduce from default 32

# Process smaller chunks
python scripts/preprocess.py --no-augment
```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
# Server debug mode
python -m server.app --debug

# Enable debug logging in scripts
export PYTHONPATH=.
python scripts/train.py  # Uses debug logging automatically
```

## 📊 Performance Benchmarks

### Typical Performance Metrics

#### Preprocessing

- **Speed**: ~10-50 files/minute (depends on audio length and augmentation)
- **Augmentation**: 5x data expansion (1 original + 4 augmented per file)
- **Feature Extraction**: ~350 features per audio sample

#### Training

- **Random Forest**: 5-30 minutes for 10,000 samples
- **BLSTM**: 30-120 minutes for 10,000 samples (GPU recommended)
- **Memory Usage**: 2-8 GB RAM depending on dataset size

#### Inference

- **API Response Time**: 1-3 seconds per request
- **Batch Processing**: 100-500 files/hour
- **Model Loading**: 5-15 seconds at startup

#### Accuracy Benchmarks

- **Known Reciters**: 85-95% accuracy (depends on audio quality and training data)
- **Unknown Reciters**: 90-98% correct rejection rate
- **Ayah Identification**: 70-90% accuracy (depends on audio clarity and verse length)

## 🙏 Acknowledgments

- **Audio Processing**: librosa, soundfile, pydub
- **Machine Learning**: scikit-learn, PyTorch, TensorFlow
- **ASR**: OpenAI Whisper, Tarteel Whisper models
- **Web Framework**: Flask with Rich logging
- **Text Processing**: rapidfuzz for fuzzy matching

## 📞 Support

For troubleshooting and system maintenance:

1. **Debug Mode**: Enable debug logging for detailed troubleshooting
2. **Test Cases**: Use the test case runner to validate system functionality
3. **Log Files**: Check detailed logs in the `logs/` directory
4. **Health Endpoint**: Monitor system status via `/health` API endpoint

---

This comprehensive system provides a complete solution for Quran reciter and verse identification with robust machine learning pipelines and production-ready API endpoints.
curl -X POST http://localhost:5000/getReciter \
 -F "audio=@audio_file.mp3" \
 -F "show_unreliable_predictions=false"

````

**Parameters**:
- `audio`: Audio file (MP3/WAV, 5-15 seconds, 22050 Hz)
- `show_unreliable_predictions`: Show results even if unreliable (default: false)

**Response**:
```json
{
  "reliable": true,
  "main_prediction": {
    "name": "Mishary Alafasy",
    "confidence": 95.5,
    "nationality": "Kuwait",
    "serverUrl": "https://server.example.com",
    "flagUrl": "https://flags.example.com/kw.png",
    "imageUrl": "https://images.example.com/mishary.jpg"
  },
  "top_predictions": [
    {
      "name": "Mishary Alafasy",
      "confidence": 95.5,
      "nationality": "Kuwait",
      "serverUrl": "https://server.example.com",
      "flagUrl": "https://flags.example.com/kw.png",
      "imageUrl": "https://images.example.com/mishary.jpg"
    }
  ],
  "response_time_ms": 1250.5
}
````

#### POST /getAyah

Identify Quranic verse from audio recording.

**Request**:

```bash

```
