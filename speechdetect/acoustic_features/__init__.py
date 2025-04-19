"""
This module provides access to all acoustic feature extraction functions.
It imports and exposes functions from all subdirectories for ease of use.
"""

# Import functions from cepstral_coefficients_and_spectral_features.py
from .cepstral_coefficients_and_spectral_features import *

# Import functions from rhythmic_structure.py
from .rhythmic_structure import *

# Import functions from frequency_parameters.py
from .frequency_parameters import *

# Import functions from complexity.py
from .complexity import *

# Import functions from loudness_and_intensity.py
from .loudness_and_intensity import *

# Import functions from voice_quality.py
from .voice_quality import *

# Import functions from statistical_functions.py
from .statistical_functions import *

# Import functions from speech_fluency_and_speech_production_dynamics.py
from .speech_fluency_and_speech_production_dynamics import *

# Import functions from utils.py
from .utils import *

# Define __all__ to control what gets imported with "from acoustic_features import *"
__all__ = []

# Automatically populate __all__ with all imported functions
# This will be executed at import time
import sys
for name, obj in list(sys.modules[__name__].__dict__.items()):
    # Only add function objects that don't start with underscore (not private)
    if callable(obj) and not name.startswith('_'):
        __all__.append(name)
