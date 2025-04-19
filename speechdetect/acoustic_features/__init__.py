"""
Acoustic Features Package - Collection of speech and audio analysis functions.
"""

# Import all public functions and classes from voice_quality module
from .voice_quality import (
    calculate_APQ_from_peaks,
    calculate_frame_based_APQ,
    shimmer,
    analyze_audio_shimmer,
    calculate_frame_level_hnr,
    get_voice_quality_metrics,
    amplitude_range
)

# Import all public functions from statistical_functions module
from .statistical_functions import (
    sma, de, max, min, span, maxPos, minPos, amean, 
    linregc1, linregc2, linregerrA, linregerrQ, 
    stddev, skewness, kurtosis,
    quartile1, quartile2, quartile3, 
    iqr1_2, iqr2_3, iqr1_3,
    percentile1, percentile99, pctlrange0_1,
    upleveltime75, upleveltime90
)

# Import core classes and functions from speech_fluency_and_speech_production_dynamics
from .speech_fluency_and_speech_production_dynamics import (
    SpeechBehavior,
    calculate_duration_ms,
    remove_subranges,
    read_wav_segment_new,
    read_wav_segment,
    extract_syllables,
    calculate_statistics,
    calculate_raw_pvi,
    calculate_normalized_pvi,
    calculate_silence_segments,
    calculate_alternating_durations
)

# Import PauseBehavior class from rhythmic_structure
from .rhythmic_structure import PauseBehavior

# Import all public functions from loudness_and_intensity module
from .loudness_and_intensity import (
    rms_amplitude,
    spl_per,
    peak_amplitude,
    ste_amplitude,
    intensity
)

# Import all public functions from frequency_parameters module
from .frequency_parameters import (
    get_pitch,
    calculate_time_varying_jitter,
    get_formants_frame_based,
    calculate_jitter_shimmer,
    formant_trajectory_info
)

# Import all public functions from cepstral_coefficients_and_spectral_features
from .cepstral_coefficients_and_spectral_features import (
    compute_msc,
    spectral_centriod,
    ltas,
    alpha_ratio,
    log_mel_spectrogram,
    mfcc,
    lpc,
    lpcc,
    spectral_envelope,
    calculate_cpp,
    hammIndex,
    plp_features,
    harmonicity,
    calculate_lsp_freqs_for_frames,
    calculate_frame_wise_zcr
)

# Import all public functions from complexity module
from .complexity import (
    calculate_hfd_per_frame,
    calculate_frequency_entropy,
    calculate_amplitude_entropy
)

# Define what should be imported with "from acoustic_features import *"
__all__ = [
    # voice_quality
    'calculate_APQ_from_peaks', 'calculate_frame_based_APQ', 'shimmer',
    'analyze_audio_shimmer', 'calculate_frame_level_hnr', 'get_voice_quality_metrics',
    'amplitude_range',
    
    # statistical_functions
    'sma', 'de', 'max', 'min', 'span', 'maxPos', 'minPos', 'amean',
    'linregc1', 'linregc2', 'linregerrA', 'linregerrQ',
    'stddev', 'skewness', 'kurtosis',
    'quartile1', 'quartile2', 'quartile3',
    'iqr1_2', 'iqr2_3', 'iqr1_3',
    'percentile1', 'percentile99', 'pctlrange0_1',
    'upleveltime75', 'upleveltime90',
    
    # speech_fluency_and_speech_production_dynamics
    'SpeechBehavior', 'calculate_duration_ms', 'remove_subranges',
    'read_wav_segment_new', 'read_wav_segment', 'extract_syllables',
    'calculate_statistics', 'calculate_raw_pvi', 'calculate_normalized_pvi',
    'calculate_silence_segments', 'calculate_alternating_durations',
    
    # rhythmic_structure
    'PauseBehavior',
    
    # loudness_and_intensity
    'rms_amplitude', 'spl_per', 'peak_amplitude', 'ste_amplitude', 'intensity',
    
    # frequency_parameters
    'get_pitch', 'calculate_time_varying_jitter', 'get_formants_frame_based',
    'calculate_jitter_shimmer', 'formant_trajectory_info',
    
    # cepstral_coefficients_and_spectral_features
    'compute_msc', 'spectral_centriod', 'ltas', 'alpha_ratio',
    'log_mel_spectrogram', 'mfcc', 'lpc', 'lpcc', 'spectral_envelope',
    'calculate_cpp', 'hammIndex', 'plp_features', 'harmonicity',
    'calculate_lsp_freqs_for_frames', 'calculate_frame_wise_zcr',
    
    # complexity
    'calculate_hfd_per_frame', 'calculate_frequency_entropy', 'calculate_amplitude_entropy'
]
