# SpeechDETECT: Acoustic Feature Extraction for Speech Analysis

## Overview

SpeechDETECT is a comprehensive acoustic feature extraction tool for speech analysis. It extracts a wide range of acoustic parameters across multiple domains:

- **Vocal Traits**
  - Frequency Parameters (pitch, jitter, formants)
  - Spectral Features (MFCC, LPC, spectral envelope)
  - Voice Quality (shimmer, HNR, APQ)
  - Loudness and Intensity
  - Speech Signal Complexity

- **Temporal Aspects**
  - Speech Fluency
  - Rhythmic Structure
  - Speech Production Dynamics

## Installation

There are two ways to install SpeechDETECT:

### Option 1: Install from GitHub

```bash
pip install git+https://github.com/SpeechCARE/SpeechDETECT-toolkit.git
```

### Option 2: Manual Installation

```bash
git clone https://github.com/SpeechCARE/SpeechDETECT-toolkit.git
cd SpeechDETECT-toolkit
pip install -e .
```

## Usage Examples

### Basic Feature Extraction

```python
from speechdetect import AcousticFeatureExtractor

# Initialize the feature extractor
extractor = AcousticFeatureExtractor(sampling_rate=16000)

# Extract features with detailed options
features = extractor.extract_features(
    "path/to/audio.wav", 
    features_to_calculate=[
        "spectral",     # Cepstral coefficients and spectral features (MFCC, LPC, etc.)
        "complexity",   # Signal complexity metrics (entropy, fractal dimension)
        "frequency",    # Pitch and formant features (F0, jitter, formants)
        "intensity",    # Loudness and amplitude features (RMS, SPL, etc.)
        "voice_quality" # Voice quality metrics (shimmer, HNR, etc.)
        # Other options: "rhythmic", "fluency", "transcription", "all", "raw"
    ],
    separate_groups=False  # Set to True to organize output by feature categories
)

# Extract all available features and organize by feature groups
grouped_features = extractor.extract_features(
    "path/to/audio.wav", 
    features_to_calculate=["all"],
    separate_groups=True  # Results will be organized by feature categories
)

# Accessing features when separate_groups=True
mel_features = grouped_features["Spectral/Mel"]
print(mel_features)

# Extract features and save to CSV files automatically
extractor.extract_features(
    "path/to/audio.wav",
    features_to_calculate=["all"],
    output_dir="path/to/output_folder"  # Each file will be saved as a CSV in this directory
)
```

### Batch Processing

```python
# Process multiple files
audio_files = [
    "path/to/file1.wav",
    "path/to/file2.wav",
    "path/to/file3.wav"
] # or a pytorch dataloader with paths of audio files

# Extract selected feature types for all files
results = extractor.extract_features(
    audio_files,
    features_to_calculate=["frequency", "voice_quality", "intensity"],
    separate_groups=False  # Default: flat dictionary of features
)

# Access results for a specific file
file1_features = results["path/to/file1.wav"]

# Process files with grouped features
grouped_results = extractor.extract_features(
    audio_files,
    features_to_calculate=["frequency", "voice_quality", "intensity"],
    separate_groups=True  # Features organized by categories
)

# Access grouped results for a specific file
file1_grouped_features = grouped_results["path/to/file1.wav"]
voice_quality_features = file1_grouped_features.get("Voice Quality/Perturbation", {})

# Process multiple files and save each to its own CSV file
extractor.extract_features(
    audio_files,
    features_to_calculate=["frequency", "voice_quality", "intensity"],
    output_dir="path/to/output_folder"  # Each file will get its own CSV with the same filename
)
```

## Available Feature Types

The extractor supports the following feature types:

- `spectral`: Cepstral coefficients and spectral features
- `complexity`: Speech signal complexity features
- `frequency`: Frequency and pitch-related parameters
- `intensity`: Loudness and intensity features
- `rhythmic`: Rhythmic structure parameters
- `fluency`: Speech fluency features
- `voice_quality`: Voice quality metrics
- `all`: All available features
- `raw`: Raw acoustic features
- `transcription`: Features requiring VAD and transcription models


### Advanced Usage with Voice Activity Detection

```python
# For transcription-based features, you need to set up models first
# This example assumes you have VAD and transcription models
import torch
import whisperx

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
transcription_model = whisperx.load_model("large-v3", device, compute_type="int8")

# Set models in the extractor
extractor.set_models(
    vad_model=vad_model,
    vad_utils=vad_utils,
    transcription_model=transcription_model
)

# Extract transcription-based features
transcription_features = extractor.extract_features(
    "path/to/audio.wav",
    features_to_calculate=["transcription"]
)
```

## Logging

SpeechDETECT uses Python's logging framework. You can configure the logging level:

```python
import logging

# Configure logging at the application level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or set the level specifically for the extractor
extractor = AcousticFeatureExtractor(log_level=logging.DEBUG)
```

## Saving and Loading Features

```python
import pandas as pd

# Extract features
features = extractor.extract_all_features("path/to/audio.wav")

# Method 1: Use the built-in CSV export
extractor.extract_features(
    "path/to/audio.wav",
    features_to_calculate=["all"],
    output_dir="path/to/output_folder"
)

# Method 2: Manual saving to CSV
df = pd.DataFrame([features])
df.to_csv("features.csv", index=False)

# Load features from CSV
loaded_features = pd.read_csv("features.csv").iloc[0].to_dict()
```
