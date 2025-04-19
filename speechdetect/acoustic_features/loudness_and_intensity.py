import numpy as np
from scipy.signal import get_window
from utils import frame_signal


# Constants for acoustics
REF_PRESSURE = 2e-5  # Reference pressure in air (20 µPa)
REF_INTENSITY = 10e-12  # Reference intensity threshold (10 pW/m²)

def mean_energy_concentration(data):
    """
    Calculate the mean energy concentration of a signal.
    
    Args:
        data (numpy.ndarray): Input audio signal
        
    Returns:
        float: Mean energy concentration
    """
    if len(data) == 0:
        return 0.0
    return np.sum(np.square(data)) / len(data)

def process_frames(signal, sample_rate, frame_length_ms, frame_step_ms, window_type="hamming", 
                  normalize=False):
    """
    Process a signal into windowed frames for further analysis.
    
    Args:
        signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sample rate of the signal in Hz
        frame_length_ms (float): Frame length in milliseconds
        frame_step_ms (float): Frame step in milliseconds
        window_type (str): Type of windowing function to apply
        normalize (bool): Whether to normalize the signal
        
    Returns:
        numpy.ndarray: Array of windowed frames with shape (num_frames, frame_length)
    """
    # Convert to mono if needed
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    # Normalize if requested
    if normalize and np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    # Use the frame_signal utility if available
    try:
        frames = frame_signal(signal, sample_rate, frame_length_ms, frame_step_ms, window_type)
    except:
        # Fallback implementation if frame_signal is not available
        frame_length_samples = int(frame_length_ms * sample_rate / 1000)
        frame_step_samples = int(frame_step_ms * sample_rate / 1000)

        # Get window function
        if isinstance(window_type, str):
            window = get_window(window_type, frame_length_samples)
        elif isinstance(window_type, np.ndarray):
            window = window_type
        else:
            window = np.ones(frame_length_samples)
        
        # Calculate number of frames
        num_frames = max(1, int(1 + np.floor((len(signal) - frame_length_samples) / frame_step_samples)))
        
        # Allocate output array
        frames = np.zeros((num_frames, frame_length_samples))
        
        # Extract frames
        for i in range(num_frames):
            start = i * frame_step_samples
            end = min(start + frame_length_samples, len(signal))
            
            # Zero-padding if needed
            if end - start < frame_length_samples:
                frames[i, :(end-start)] = signal[start:end] * window[:(end-start)]
            else:
                frames[i, :] = signal[start:end] * window
    
    return frames

def rms_amplitude(signal, sample_rate, frame_length_ms, frame_step_ms, window_type="hamming"):
    """
    Calculate the root mean square (RMS) amplitude for each frame of a signal.
    
    Args:
        signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sample rate of the signal in Hz
        frame_length_ms (float): Frame length in milliseconds
        frame_step_ms (float): Frame step in milliseconds
        window_type (str): Type of windowing function to apply
        
    Returns:
        numpy.ndarray: Array of RMS values for each frame
    """
    frames = process_frames(signal, sample_rate, frame_length_ms, frame_step_ms, window_type)
    
    # Calculate RMS for each frame (vectorized)
    rms_values = np.sqrt(np.mean(frames**2, axis=1))
    
    return rms_values

def spl_per_frame(signal, sample_rate, frame_length_ms, frame_step_ms, window_type="hamming", 
                 ref_pressure=REF_PRESSURE):
    """
    Calculate the sound pressure level (SPL) in dB for each frame of a signal.
    
    Args:
        signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sample rate of the signal in Hz
        frame_length_ms (float): Frame length in milliseconds
        frame_step_ms (float): Frame step in milliseconds
        window_type (str): Type of windowing function to apply
        ref_pressure (float): Reference pressure level (default: 20 µPa)
        
    Returns:
        numpy.ndarray: Array of SPL values for each frame in dB
    """
    # Convert to float if necessary
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
    
    # Calculate RMS for each frame
    rms_values = rms_amplitude(signal, sample_rate, frame_length_ms, frame_step_ms, window_type)
    
    # Calculate SPL (20 log10(rms/ref)) with proper handling of zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        spl_values = 20 * np.log10(rms_values / ref_pressure)
    
    # Replace -inf with NaN for silent frames
    spl_values[rms_values == 0] = np.nan
    
    return spl_values

def peak_amplitude(signal, sample_rate, frame_length_ms, frame_step_ms, window_type="hamming"):
    """
    Calculate the peak amplitude for each frame of a signal.
    
    Args:
        signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sample rate of the signal in Hz
        frame_length_ms (float): Frame length in milliseconds
        frame_step_ms (float): Frame step in milliseconds
        window_type (str): Type of windowing function to apply
        
    Returns:
        numpy.ndarray: Array of peak amplitude values for each frame
    """
    frames = process_frames(signal, sample_rate, frame_length_ms, frame_step_ms, window_type)
    
    # Calculate peak amplitude for each frame (vectorized)
    peak_values = np.max(np.abs(frames), axis=1)
    
    return peak_values

def short_time_energy(signal, sample_rate, frame_length_ms, frame_step_ms, window_type="hamming"):
    """
    Calculate the short-time energy (STE) for each frame of a signal.
    
    Args:
        signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sample rate of the signal in Hz
        frame_length_ms (float): Frame length in milliseconds
        frame_step_ms (float): Frame step in milliseconds
        window_type (str): Type of windowing function to apply
        
    Returns:
        numpy.ndarray: Array of STE values for each frame
    """
    frames = process_frames(signal, sample_rate, frame_length_ms, frame_step_ms, window_type)
    
    # Calculate STE for each frame (vectorized)
    ste_values = np.sum(frames**2, axis=1)
    
    return ste_values

def intensity(signal, sample_rate, frame_length_ms, frame_step_ms, window_type="hamming", 
             loudness=False, ref_intensity=REF_INTENSITY):
    """
    Calculate the intensity or loudness for each frame of a signal.
    
    Args:
        signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sample rate of the signal in Hz
        frame_length_ms (float): Frame length in milliseconds
        frame_step_ms (float): Frame step in milliseconds
        window_type (str): Type of windowing function to apply
        loudness (bool): Whether to convert intensity to loudness (Stevens' power law)
        ref_intensity (float): Reference intensity level (default: 10 pW/m²)
        
    Returns:
        numpy.ndarray: Array of intensity or loudness values for each frame
    """
    frames = process_frames(signal, sample_rate, frame_length_ms, frame_step_ms, window_type, normalize=True)
    
    # Calculate mean squared amplitude (intensity) for each frame
    intensity_values = np.mean(frames**2, axis=1)
    
    # Convert to loudness using Stevens' power law if requested
    if loudness:
        # Ensure we don't have negative values before applying power
        valid_mask = intensity_values > 0
        result = np.zeros_like(intensity_values)
        result[valid_mask] = (intensity_values[valid_mask] / ref_intensity) ** 0.3
        return result
    
    return intensity_values

def loudness_contour(intensity_values, frequencies, equal_loudness_contour='A'):
    """
    Apply frequency weighting based on equal loudness contours (A, B, C, or D weighting).
    
    Args:
        intensity_values (numpy.ndarray): Intensity values in frequency domain
        frequencies (numpy.ndarray): Corresponding frequencies in Hz
        equal_loudness_contour (str): Type of weighting ('A', 'B', 'C', or 'D')
        
    Returns:
        numpy.ndarray: Weighted intensity values
    """
    # Make sure intensities and frequencies are numpy arrays
    intensity_values = np.asarray(intensity_values)
    frequencies = np.asarray(frequencies)
    
    # Initialize weights array
    weights = np.ones_like(frequencies, dtype=float)
    
    # A-weighting (IEC 61672:2003)
    if equal_loudness_contour.upper() == 'A':
        f_sq = frequencies**2
        weights = 12200**2 * f_sq**2 / ((f_sq + 20.6**2) * 
                                        np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * 
                                        (f_sq + 12200**2))
    
    # C-weighting (IEC 61672:2003)
    elif equal_loudness_contour.upper() == 'C':
        f_sq = frequencies**2
        weights = 12200**2 * f_sq / ((f_sq + 20.6**2) * (f_sq + 12200**2))
    
    # Apply weighting
    weighted_intensity = intensity_values * weights
    
    return weighted_intensity

def speech_activity_detection(signal, sample_rate, frame_length_ms=25, frame_step_ms=10, 
                             energy_threshold=0.1, zero_crossing_threshold=0.2):
    """
    Detect speech activity in an audio signal using energy and zero-crossing rate.
    
    Args:
        signal (numpy.ndarray): Input audio signal
        sample_rate (int): Sample rate of the signal in Hz
        frame_length_ms (float): Frame length in milliseconds
        frame_step_ms (float): Frame step in milliseconds
        energy_threshold (float): Threshold for normalized energy (0-1)
        zero_crossing_threshold (float): Threshold for zero-crossing rate
        
    Returns:
        numpy.ndarray: Boolean array indicating speech activity for each frame
    """
    # Calculate energy
    energy = short_time_energy(signal, sample_rate, frame_length_ms, frame_step_ms)
    
    # Normalize energy to 0-1 range
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    # Calculate zero-crossing rate
    frames = process_frames(signal, sample_rate, frame_length_ms, frame_step_ms)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    
    # Detect speech frames
    speech_frames = (energy > energy_threshold) & (zcr < zero_crossing_threshold)
    
    return speech_frames