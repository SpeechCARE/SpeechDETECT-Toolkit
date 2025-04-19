import warnings
import librosa
import numpy as np
import parselmouth
from functools import lru_cache
from scipy.signal import find_peaks
from typing import Tuple, Optional, Union, List


def calculate_APQ_from_peaks(
    data: np.ndarray, 
    fs: int, 
    window_length_ms: int = 40, 
    window_step_ms: int = 20, 
    num_cycles: int = 3,
    min_peak_height: Optional[float] = None,
    min_peak_distance: Optional[int] = None
) -> np.ndarray:
    """
    Calculate the frame-based Amplitude Perturbation Quotient (APQ) using peak amplitudes.
    
    The APQ is a measure of the short-term (cycle-to-cycle) amplitude variations in voice.
    Higher values indicate more variation in amplitude, which may be associated with
    dysphonia or other voice disorders.

    Args:
        data: Time series audio signal
        fs: Sampling rate in Hz
        window_length_ms: Length of each frame in milliseconds
        window_step_ms: Step size between consecutive frames in milliseconds
        num_cycles: Number of cycles over which to average the APQ
        min_peak_height: Minimum height for a peak to be considered (default: None)
        min_peak_distance: Minimum distance between peaks in samples (default: None)

    Returns:
        Frame-wise APQ values as percentages
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    
    if data.size == 0:
        return np.array([])
    
    # Convert ms to samples
    frame_length = int(window_length_ms * fs / 1000)
    hop_length = int(window_step_ms * fs / 1000)

    # Ensure frame_length is not larger than signal
    if frame_length > len(data):
        frame_length = len(data)
        warnings.warn(f"Window length ({window_length_ms}ms) exceeds signal duration. Using full signal.")

    # Frame the signal
    frames = librosa.util.frame(data, frame_length=frame_length, hop_length=hop_length)
    apq_values = np.zeros(frames.shape[1])

    for i, frame in enumerate(frames.T):
        # Find peaks in the frame
        peaks, _ = find_peaks(frame, height=min_peak_height, distance=min_peak_distance)
        
        if len(peaks) < num_cycles + 1:
            # Not enough peaks, assign default value
            continue
            
        peak_amplitudes = frame[peaks]
        
        # Calculate the absolute differences in amplitude between consecutive peaks
        amplitude_differences = np.abs(np.diff(peak_amplitudes))
        
        # Use vectorized operations for efficiency
        frame_apq_values = []
        for j in range(len(amplitude_differences) - num_cycles + 1):
            cycle_amplitudes = peak_amplitudes[j:j + num_cycles]
            cycle_differences = amplitude_differences[j:j + num_cycles]
            
            avg_diff = np.mean(cycle_differences)
            avg_amp = np.mean(np.abs(cycle_amplitudes))
            
            # Avoid division by zero
            if avg_amp > 1e-10:
                apq = avg_diff / avg_amp
                frame_apq_values.append(apq)
        
        # Calculate mean APQ for the frame
        if frame_apq_values:
            apq_values[i] = np.mean(frame_apq_values) * 100  # Convert to percentage
    
    return apq_values


def calculate_frame_based_APQ(
    data: np.ndarray, 
    fs: int, 
    window_length_ms: int = 40, 
    window_step_ms: int = 20, 
    num_samples: int = 3
) -> np.ndarray:
    """
    Calculate the frame-based Amplitude Perturbation Quotient (APQ) based on sample differences.
    
    This method calculates APQ using direct sample-to-sample amplitude differences rather than
    focusing on peaks, which makes it suitable for more general analysis.

    Args:
        data: Time series audio signal
        fs: Sampling rate in Hz
        window_length_ms: Length of each frame in milliseconds
        window_step_ms: Step size between consecutive frames in milliseconds
        num_samples: Number of samples over which to average the APQ

    Returns:
        Frame-wise APQ values as percentages
        
    Raises:
        ValueError: If frame length is too small for the specified number of samples
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    
    if data.size == 0:
        return np.array([])
    
    # Convert ms to samples
    frame_length = int(window_length_ms * fs / 1000)
    hop_length = int(window_step_ms * fs / 1000)
    
    # Ensure we can calculate APQ with given parameters
    if frame_length < num_samples + 1:
        raise ValueError(f"Frame length ({frame_length} samples) too small for APQ calculation over {num_samples} samples.")
    
    # Frame the signal
    frames = librosa.util.frame(data, frame_length=frame_length, hop_length=hop_length)
    apq_values = np.zeros(frames.shape[1])
    
    for i, frame in enumerate(frames.T):
        # Calculate amplitude differences
        amplitude_differences = np.abs(np.diff(frame))
        
        # Use vectorized approach for efficiency
        frame_apq_values = []
        for j in range(len(amplitude_differences) - num_samples + 1):
            local_diffs = amplitude_differences[j:j+num_samples]
            local_amps = np.abs(frame[j:j+num_samples])
            
            local_diff_avg = np.mean(local_diffs)
            local_amp_avg = np.mean(local_amps)
            
            # Avoid division by zero
            if local_amp_avg > 1e-10:
                apq = local_diff_avg / local_amp_avg
                frame_apq_values.append(apq)
        
        # Calculate mean APQ for the frame
        if frame_apq_values:
            apq_values[i] = np.mean(frame_apq_values) * 100  # Convert to percentage
    
    return apq_values


def shimmer(frame: np.ndarray) -> float:
    """
    Calculate shimmer for a single frame of audio.
    
    Shimmer measures the cycle-to-cycle variations of amplitude in the voice,
    expressed as a percentage or relative value.
    
    Args:
        frame: A single frame of audio data
        
    Returns:
        Shimmer value (0.0 to 1.0, where higher values indicate more amplitude variation)
    """
    if len(frame) < 2:
        return 0.0
    
    # Calculate the amplitude differences between consecutive samples
    amplitude_differences = np.abs(np.diff(frame))
    frame_amplitudes = np.abs(frame[:-1])
    
    # Avoid division by zero
    mean_amplitude = np.mean(frame_amplitudes)
    if mean_amplitude < 1e-10:
        return 0.0
    
    # Compute shimmer
    shimmer_value = np.mean(amplitude_differences) / mean_amplitude
    
    return float(shimmer_value)


def analyze_audio_shimmer(
    data: np.ndarray, 
    fs: int, 
    frame_length_ms: int = 40, 
    hop_length_ms: int = 20
) -> np.ndarray:
    """
    Calculate shimmer values for framed audio data.
    
    Args:
        data: Time series audio signal
        fs: Sampling rate in Hz
        frame_length_ms: Length of each frame in milliseconds
        hop_length_ms: Step size between consecutive frames in milliseconds
        
    Returns:
        Frame-wise shimmer values
    """
    # Validate inputs
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    
    if data.size == 0:
        return np.array([])
    
    # Convert ms to samples
    frame_length = int(frame_length_ms * fs / 1000)
    hop_length = int(hop_length_ms * fs / 1000)
    
    # Ensure frame_length is not larger than signal
    if frame_length > len(data):
        frame_length = len(data)
        warnings.warn(f"Window length ({frame_length_ms}ms) exceeds signal duration. Using full signal.")
    
    # Frame the signal
    frames = librosa.util.frame(data, frame_length=frame_length, hop_length=hop_length)
    
    # Calculate shimmer for each frame
    shimmer_values = np.zeros(frames.shape[1])
    for i, frame in enumerate(frames.T):
        shimmer_values[i] = shimmer(frame)
    
    return shimmer_values


@lru_cache(maxsize=32)
def _cached_hamming_window(length: int) -> np.ndarray:
    """Create and cache a Hamming window of specified length."""
    return np.hamming(length)


def calculate_frame_level_hnr(
    signal: np.ndarray, 
    sample_rate: int, 
    frame_length_ms: int = 40, 
    frame_step_ms: int = 20, 
    windowing_function: str = "hamming"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate frame-level Harmonics-to-Noise Ratio (HNR) and Noise-to-Harmonics Ratio (NHR).
    
    HNR measures the ratio of harmonic energy to noise energy in voiced speech.
    Higher HNR indicates a clearer voice with less noise.
    
    Args:
        signal: Time series audio signal
        sample_rate: Sampling rate in Hz
        frame_length_ms: Length of each frame in milliseconds
        frame_step_ms: Step size between consecutive frames in milliseconds
        windowing_function: Window function to apply to each frame ('hamming' or None)
        
    Returns:
        Tuple containing:
            - Frame-wise HNR values
            - Frame-wise NHR values (reciprocal of HNR where HNR > 0)
    """
    # Validate inputs
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal, dtype=float)
    
    if signal.size == 0:
        return np.array([]), np.array([])
    
    # Convert ms to samples
    frame_length = int(frame_length_ms * sample_rate / 1000)
    frame_step = int(frame_step_ms * sample_rate / 1000)
    
    # Calculate number of frames and prepare for padding
    num_frames = max(1, int(np.ceil(float(len(signal) - frame_length) / frame_step + 1)))
    
    # Pad the signal to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.zeros(pad_signal_length)
    pad_signal[:len(signal)] = signal
    
    # Get window function
    if windowing_function == "hamming":
        window = _cached_hamming_window(frame_length)
    else:
        window = np.ones(frame_length)
    
    hnr_values = np.full(num_frames, np.nan)
    
    for i in range(num_frames):
        start_idx = i * frame_step
        end_idx = start_idx + frame_length
        
        if end_idx > len(pad_signal):
            break
            
        frame = pad_signal[start_idx:end_idx] * window
        
        # Skip processing if frame is mostly zeros or very low energy
        if np.sum(np.abs(frame)) < 1e-10:
            continue
        
        try:
            # Create Praat Sound object and calculate harmonicity
            snd_frame = parselmouth.Sound(frame, sampling_frequency=sample_rate)
            hnr = snd_frame.to_harmonicity()
            
            # Get valid HNR values (non-NaN)
            valid_hnr = hnr.values[~np.isnan(hnr.values)]
            
            if len(valid_hnr) > 0:
                hnr_values[i] = np.mean(valid_hnr)
        except Exception as e:
            warnings.warn(f"Error calculating HNR for frame {i}: {str(e)}")
            continue
    
    # Calculate NHR (reciprocal of HNR where HNR > 0)
    # Replace NaN with 0 for computation, then restore NaN
    hnr_for_calc = np.copy(hnr_values)
    mask_nan = np.isnan(hnr_for_calc)
    hnr_for_calc[mask_nan] = 0
    
    # Calculate NHR with safeguards against division by zero
    nhr_values = np.zeros_like(hnr_for_calc)
    mask_valid = ~mask_nan & (np.abs(hnr_for_calc) > 1e-10)
    nhr_values[mask_valid] = 1.0 / hnr_for_calc[mask_valid]
    nhr_values[mask_nan] = np.nan
    
    return hnr_values, nhr_values


def get_voice_quality_metrics(
    signal: np.ndarray, 
    sample_rate: int,
    frame_length_ms: int = 40, 
    frame_step_ms: int = 20
) -> dict:
    """
    Calculate multiple voice quality metrics for the given audio signal.
    
    This function serves as a convenience wrapper to calculate multiple 
    voice quality metrics in a single call.
    
    Args:
        signal: Time series audio signal
        sample_rate: Sampling rate in Hz
        frame_length_ms: Length of each frame in milliseconds
        frame_step_ms: Step size between consecutive frames in milliseconds
        
    Returns:
        Dictionary containing the following metrics:
            - 'shimmer': Frame-wise shimmer values
            - 'apq': Frame-wise APQ values
            - 'apq_peaks': Frame-wise APQ values calculated from peaks
            - 'hnr': Frame-wise HNR values
            - 'nhr': Frame-wise NHR values
    """
    # Validate signal
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal, dtype=float)
    
    # Calculate all metrics
    shimmer_values = analyze_audio_shimmer(
        signal, sample_rate, frame_length_ms, frame_step_ms
    )
    
    apq_values = calculate_frame_based_APQ(
        signal, sample_rate, frame_length_ms, frame_step_ms
    )
    
    apq_peaks_values = calculate_APQ_from_peaks(
        signal, sample_rate, frame_length_ms, frame_step_ms
    )
    
    hnr_values, nhr_values = calculate_frame_level_hnr(
        signal, sample_rate, frame_length_ms, frame_step_ms
    )
    
    # Return all metrics in a dictionary
    return {
        'shimmer': shimmer_values,
        'apq': apq_values,
        'apq_peaks': apq_peaks_values,
        'hnr': hnr_values,
        'nhr': nhr_values
    }
