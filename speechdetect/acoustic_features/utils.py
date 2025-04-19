import numpy as np
import scipy
from typing import Union, Tuple, Optional, Literal

def frame_signal(signal: np.ndarray, sample_rate: int, frame_length_ms: float, 
                frame_step_ms: float, windowing_function: Union[str, np.ndarray] = "hamming") -> np.ndarray:
    """
    Convert a signal into a sequence of overlapping frames.
    
    Args:
        signal: The audio signal to frame
        sample_rate: The sample rate of the signal
        frame_length_ms: Frame length in milliseconds
        frame_step_ms: Frame step in milliseconds
        windowing_function: Either "hamming" or a custom window function array
    
    Returns:
        Framed signal as a 2D array of shape (num_frames, frame_length_samples)
    """
    # Convert frame length and step from milliseconds to samples
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)
    
    # Create window function only if needed
    if isinstance(windowing_function, str) and windowing_function == "hamming":
        window = np.hamming(frame_length_samples)
    else:
        window = windowing_function

    # Calculate total number of frames
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length_samples)) / frame_step_samples))
    
    # Pre-allocate frames array for better performance
    frames = np.zeros((num_frames, frame_length_samples))
    
    # Calculate zero padding once
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    
    # Only create the exact padding needed
    if pad_signal_length > signal_length:
        pad_signal = np.zeros(pad_signal_length)
        pad_signal[:signal_length] = signal
    else:
        pad_signal = signal[:pad_signal_length]
    
    # Vectorized frame extraction
    for i in range(num_frames):
        start = i * frame_step_samples
        end = start + frame_length_samples
        frames[i] = pad_signal[start:end] * window

    return frames

def freq2mel(f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequency to mel scale."""
    return 2595 * np.log10(1 + (f / 700))

def mel2freq(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert mel scale to frequency."""
    return 700 * (10**(m / 2595) - 1)

def hz_to_bark(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert frequency in Hz to Bark scale."""
    return 6 * np.arcsinh(hz / 600)

def bark_to_hz(bark: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Bark scale to frequency in Hz."""
    return 600 * np.sinh(bark / 6)

def stft(data: np.ndarray, fs: int, window_length_ms: float = 30, 
         window_step_ms: float = 20, windowing_function: Union[str, np.ndarray] = "hamming") -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform of a signal.
    
    Args:
        data: Input signal
        fs: Sample rate in Hz
        window_length_ms: Window length in milliseconds
        window_step_ms: Window step in milliseconds
        windowing_function: Either "hamming" or a custom window function array
    
    Returns:
        Spectrogram as a 2D array
    """
    window_length = int(window_length_ms * fs / 1000)
    window_step = int(window_step_ms * fs / 1000)
    
    if isinstance(windowing_function, str) and windowing_function == "hamming":
        window = np.hanning(window_length)
    else:
        window = windowing_function

    total_length = len(data)
    window_count = int((total_length - window_length) / window_step) + 1
    spectrum_length = int(window_length / 2) + 1
    
    # Pre-allocate the array for better performance
    spectrogram = np.zeros((window_count, spectrum_length))
    
    # Vectorize the windowing operation
    for k in range(window_count):
        starting_position = k * window_step
        data_vector = data[starting_position:starting_position + window_length]
        
        # Apply window and compute FFT in one step
        window_spectrum = np.abs(scipy.fft.rfft(data_vector * window, n=window_length))
        spectrogram[k, :] = window_spectrum

    return spectrogram

def lifter_ceps(ceps: np.ndarray, lift: int = 3) -> np.ndarray:
    """
    Apply a cepstral lifter to the matrix of cepstra.
    
    This increases the magnitude of high frequency DCT coefficients.
    
    Args:
        ceps: Matrix of mel-cepstra, shape (numframes, numcep)
        lift: Liftering coefficient (default: 3)
            - Positive values use exponential liftering
            - Negative values use sine curve liftering
    
    Returns:
        Liftered cepstra
    """
    if lift == 0:
        return ceps
        
    num_ceps = ceps.shape[1]
    
    if lift > 0:
        # Exponential liftering - more efficient implementation
        lift_vec = np.power(np.arange(num_ceps), lift)
        lift_vec[0] = 1  # First coefficient doesn't get liftered
        return ceps * lift_vec
    else:
        # Sine curve liftering
        lift = -lift  # Convert to positive
        lift_vec = 1 + (lift / 2.0) * np.sin(np.pi * np.arange(1, num_ceps + 1) / lift)
        return ceps * lift_vec

def preprocess_audio(fs: int, x: np.ndarray, min_duration_sec: float = 5.0, 
                    min_sample_rate: int = 12000) -> np.ndarray:
    """
    Preprocess audio signal for feature extraction.
    
    Args:
        fs: Sample rate in Hz
        x: Input audio signal
        min_duration_sec: Minimum required duration in seconds
        min_sample_rate: Minimum required sample rate in Hz
    
    Returns:
        Preprocessed audio signal
    
    Raises:
        ValueError: If signal duration or sample rate is insufficient
    """
    # Ensure only one channel
    if x.ndim > 1:
        mono_signal = np.mean(x[:, :2], axis=1)  # Average first two channels if multi-channel
    else:
        mono_signal = x

    # Keep original signal without normalization for better numerical precision
    norm_signal = mono_signal

    # Check minimum signal duration
    t = len(norm_signal) / fs  # Signal duration in seconds
    if t < min_duration_sec:
        raise ValueError(f'Signal duration must be greater than {min_duration_sec}s')

    # Check sampling frequency
    if fs < min_sample_rate:
        raise ValueError(f'Sampling Rate fs must be greater than {min_sample_rate}Hz')

    return norm_signal

def poly2lsf(a: np.ndarray) -> np.ndarray:
    """
    Convert prediction polynomial to line spectral frequencies.
    
    Args:
        a: Prediction polynomial coefficients
    
    Returns:
        Line spectral frequencies
    """
    # Line spectral frequencies are not defined for complex polynomials
    a = np.asarray(a)
    
    if not np.any(a):
        return np.zeros(len(a) - 1)

    # Normalize the polynomial
    if a[0] != 1 and a[0] != 0:
        a = a / a[0]

    # Check stability of the polynomial
    if max(np.abs(np.roots(a))) >= 1.0:
        return np.zeros(len(a) - 1)

    # Form the sum and difference filters
    p = len(a) - 1  # The leading one in the polynomial is not used
    a1 = np.concatenate((a, [0]))
    a2 = a1[::-1]
    P1 = a1 - a2  # Difference filter
    Q1 = a1 + a2  # Sum Filter

    # If order is even, remove the known root at z = 1 for P1 and z = -1 for Q1
    # If odd, remove both the roots from P1
    if p % 2:  # Odd order
        P, _ = scipy.signal.deconvolve(P1, [1, 0, -1])
        Q = Q1
    else:      # Even order
        P, _ = scipy.signal.deconvolve(P1, [1, -1])
        Q, _ = scipy.signal.deconvolve(Q1, [1, 1])

    # Find roots and extract angles
    rP = np.roots(P)
    rQ = np.roots(Q)
    
    # Only use every other root as they come in conjugate pairs
    aP = np.angle(rP[1::2])
    aQ = np.angle(rQ[1::2])

    # Combine and sort the angles
    lsf = np.sort(np.concatenate((-aP, -aQ)))

    return lsf