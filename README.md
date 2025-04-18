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

The server will start on `http://localhost:5000` by default.

#### API Endpoint

**POST** `/getReciter`

Identifies the Quran reciter from an audio file.

##### Request Format

- **Content-Type**: `multipart/form-data`
- **Parameter**:
  - `audio`: Audio file (MP3 or WAV format)
- **Audio Requirements**:
  - Duration: 5-15 seconds
  - Format: MP3 or WAV
  - Sample Rate: 22050 Hz (will be converted if different)

Example using curl:

```bash
curl -X POST -F "audio=@path/to/audio.mp3" http://localhost:5000/getReciter
```

Example using Python requests:

```python
import requests

url = 'http://localhost:5000/getReciter'
files = {'audio': open('path/to/audio.mp3', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

##### Response Formats

1. **Successful Response (200 OK)**

For reliable predictions:

```json
{
  "reliable": true,
  "main_prediction": {
    "name": "Reciter Name",
    "confidence": 95.5, // float between 0 and 100
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
    },
    {
      "name": "Reciter Name 2",
      "confidence": 3.5,
      "nationality": "Country 2",
      "serverUrl": "https://example.com/2",
      "flagUrl": "https://example.com/flag2.png",
      "imageUrl": "https://example.com/image2.jpg"
    }
    // ... more predictions
  ]
}
```

For unreliable predictions in production mode:

```json
{
  "reliable": false
}
```

For unreliable predictions in development mode (when SHOW_UNRELIABLE_PREDICTIONS_IN_DEV is true):

```json
{
  "reliable": false,
  "top_predictions": [
    {
      "name": "Reciter Name 1",
      "confidence": 45.5,
      "nationality": "Country 1",
      "serverUrl": "https://example.com/1",
      "flagUrl": "https://example.com/flag1.png",
      "imageUrl": "https://example.com/image1.jpg"
    }
    // ... more predictions
  ]
}
```

2. **Bad Request Errors (400)**

```json
{
  "error": "No audio file provided. Please send the file with key 'audio' in form data."
}
```

or

```json
{
  "error": "No audio file selected"
}
```

or

```json
{
  "error": "Audio file is too short. Minimum duration is 5 seconds."
}
```

or

```json
{
  "error": "Audio file is too long. Maximum duration is 15 seconds."
}
```

or

```json
{
  "error": "Invalid audio file format. Supported formats: MP3, WAV"
}
```

3. **Internal Server Errors (500)**

```json
{
  "error": "Feature extraction failed"
}
```

or

```json
{
  "error": "Error extracting features: [specific error message]"
}
```

or

```json
{
  "error": "Error making prediction: [specific error message]"
}
```

or

```json
{
  "error": "Server error: [specific error message]"
}
```

##### Response Fields

- **predicted_reciter**: The identified reciter's name, or "Unknown" if the prediction is not reliable
- **is_reliable**: Boolean indicating if the prediction meets reliability criteria
- **top_predictions**: Array of top N predictions with confidence scores
- **confidence**: (Optional) The confidence score of the top prediction
- **distance_ratio**: (Optional) Ratio of distances used in reliability analysis
- **failure_reasons**: Array of strings explaining why a prediction might not be reliable

##### Error Handling

The API uses standard HTTP status codes:

- **200**: Successful request
- **400**: Bad request (client error)
- **500**: Internal server error

All error responses include an `error` field with a descriptive message.

##### Reliability Criteria

A prediction is considered reliable when:

1. The top confidence score exceeds the threshold (default: 0.95)
2. The distance ratio meets the minimum requirement
3. No failure reasons are present

If any of these criteria are not met, `is_reliable` will be false and `predicted_reciter` will be "Unknown".

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
