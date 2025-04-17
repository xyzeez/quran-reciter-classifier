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
├── config/                 # Configuration settings
├── data/                   # Training and test data
│   ├── train/              # Training audio files
│   └── test/               # Test audio files
├── server/                 # Flask server for API
├── src/                    # Source code
│   ├── models/             # ML models implementation
│   ├── features/           # Feature extraction
│   ├── utils/              # Utility functions
│   ├── pipelines/          # Processing pipelines
│   ├── evaluation/         # Model evaluation
│   └── data/               # Data handling
├── scripts/                # Scripts for training, testing, etc.
├── processed/              # Preprocessed data (generated)
├── models/                 # Saved models (generated)
├── test_results/           # Test results (generated)
├── logs/                   # Log files (generated)
├── tools/                  # Utility scripts for project management and automation
├── requirements.txt        # Python dependencies
└── README.md               # This file
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

Preprocess your audio data before training:

```bash
python scripts/preprocess.py --mode train
```

Options:

- `--mode`: Mode to run in (`train` or `test`), default is `train`
- `--no-augment`: Disable data augmentation (only applies to train mode)

This will:

- Load audio files from the data directory
- Apply noise reduction
- Extract meaningful segments
- Save processed data to the `processed/` directory
- Create a timestamped directory (e.g., `20240306_143208_preprocess`)

### Training Models

Train a model using the preprocessed data:

```bash
python scripts/train.py --model-type random_forest
```

Options:

- `--model-type`: Model type to use (`random_forest` or `blstm`), if not specified, uses the default from config
- `--preprocess-file-id`: Specific preprocessing run ID to use (e.g., `20240306_143208_preprocess`), if not specified, uses the latest

The training process will:

- Load preprocessed data from the specified directory
- Train the selected model type
- Save the model to the `models/` directory in a timestamped folder
- Create a symlink to the latest model

### Testing Models

Evaluate model performance on test data:

```bash
python scripts/test.py --model-file-id 20240306_152417_train
```

Options:

- `--model-file-id`: Training run ID to use (e.g., `20240306_152417_train`), if not specified, uses the latest
- `--list-models`: List all available model runs
- `--list-tests`: List all previous test runs

The testing process will:

- Load the specified model
- Evaluate it on the test data
- Generate performance metrics and visualizations
- Save results to the `test_results/` directory

### Making Predictions

Make predictions on new audio files:

```bash
python scripts/predict.py --audio path/to/audio.mp3
```

Options:

- `--audio`: Path to audio file to analyze (required)
- `--model-file-id`: Training run ID to use (e.g., `20240306_152417_train`), if not specified, uses the latest
- `--true-label`: True reciter name for verification (optional)
- `--list-models`: List all available model runs

The prediction will:

- Load the specified model
- Process the audio file
- Extract features
- Identify the most likely reciter
- Display confidence scores and secondary predictions

### Using the API

First, install the API-specific dependencies:

```bash
pip install -r server/requirements.txt
```

Start the Flask server:

```bash
python -m server.app
```

Send a POST request to the `/getReciter` endpoint with an audio file:

```bash
curl -X POST -F "audio=@path/to/audio.mp3" http://localhost:5000/getReciter
```

Response format:

```json
{
  "primary_prediction": "Sheikh Mishary Rashid Alafasy",
  "confidence": 0.92,
  "secondary_predictions": [
    { "reciter": "Abdul Rahman Al-Sudais", "confidence": 0.05 },
    { "reciter": "Saud Al-Shuraim", "confidence": 0.03 }
  ],
  "processing_time": 1.25
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

Configuration settings are stored in `config/config.py`. Key parameters include:

- **Audio Processing**:

  - Sample rate: 22050 Hz
  - Minimum duration: 20 seconds
  - Skip start/end: 7/5 seconds
  - Minimum usable duration: 5 seconds

- **Feature Extraction**:

  - Number of MFCCs: 40
  - Number of Chroma features: 12
  - Number of Mel bands: 128

- **Data Augmentation**:

  - Pitch steps: [1.5, -1.5]
  - Time stretch rates: [0.9, 1.1]
  - Noise factor: 0.005
  - Volume adjust: 0.8

- **Model Parameters**:
  - Random Forest: 100 estimators, max depth 10
  - BLSTM: 64 units, 0.5 dropout rate, 0.0005 learning rate
