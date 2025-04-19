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

# Extract all available features from an audio file
features = extractor.extract_all_features("path/to/audio.wav")
print(f"Extracted {len(features)} features")

# Extract specific feature types
spectral_features = extractor.extract_features(
    "path/to/audio.wav", 
    features_to_calculate=["spectral", "complexity"]
)
```

### Batch Processing

```python
# Process multiple files
audio_files = [
    "path/to/file1.wav",
    "path/to/file2.wav",
    "path/to/file3.wav"
]

# Extract selected feature types for all files
results = extractor.extract_features(
    audio_files,
    features_to_calculate=["frequency", "voice_quality", "intensity"]
)

# Access results for a specific file
file1_features = results["path/to/file1.wav"]
```

### Visualizing Features

```python
import os

# Extract features
features = extractor.extract_all_features("path/to/audio.wav")

# Create a directory for plots
os.makedirs("feature_plots", exist_ok=True)

# Plot all features and save to directory
plot_paths = extractor.plot_features(features, output_dir="feature_plots")

# Plot specific features
selected_features = [
    "F0_sma_amean",         # Mean pitch
    "INTENSITY_sma_max",    # Maximum intensity
    "HNR_sma_amean",        # Mean harmonics-to-noise ratio
    "SHIMMER_sma_amean"     # Mean shimmer
]
extractor.plot_features(
    features, 
    feature_names=selected_features, 
    output_dir="feature_plots/selected"
)
```

### Advanced Usage with Voice Activity Detection

```python
# For transcription-based features, you need to set up models first
# This example assumes you have VAD and transcription models
from your_vad_library import VADModel, VADUtils
from your_transcription_library import TranscriptionModel

# Initialize models
vad_model = VADModel()
vad_utils = VADUtils()
transcription_model = TranscriptionModel()

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

## Saving and Loading Features

```python
import pandas as pd

# Extract features
features = extractor.extract_all_features("path/to/audio.wav")

# Save features to CSV
df = pd.DataFrame([features])
df.to_csv("features.csv", index=False)

# Load features from CSV
loaded_features = pd.read_csv("features.csv").iloc[0].to_dict()
```

## License

[License information]