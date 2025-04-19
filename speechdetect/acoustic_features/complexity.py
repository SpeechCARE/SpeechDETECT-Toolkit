import numpy as np
from scipy.signal import welch
from itertools import permutations
from utils import frame_signal

# Fractal Dimension
def hfd(audio_frame, k_max):
    """Higuchi Fractal Dimension implementation with vectorized operations."""
    x = np.asarray(audio_frame)
    N = len(x)
    L = np.zeros(k_max - 1)
    
    for k in range(1, k_max):
        Lk = 0
        for m in range(k):
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=np.int32)
            if len(idxs) > 0:  # Prevent empty sequences
                Lmk = np.sum(np.abs(x[m + idxs * k] - x[m + (idxs - 1) * k])) / len(idxs) / k
                Lk += Lmk / k
        L[k-1] = np.log(Lk / (m + 1) + 1e-20)  # Add small epsilon to prevent log(0)
    
    # Use vectorized least squares solution
    X = np.vstack([np.log(np.arange(1, k_max)), np.ones(k_max - 1)]).T
    p = np.linalg.lstsq(X, L, rcond=None)[0]
    return p[0]

def calculate_hfd_per_frame(data, fs, window_length_ms, window_step_ms, k_max, windowing_function="hamming"):
    """Calculate Higuchi Fractal Dimension for each frame using frame_signal."""
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    
    # Use the imported frame_signal function
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)
    
    # Calculate HFD for each frame
    hfd_values = np.array([hfd(frame, k_max) for frame in frames])
    return hfd_values


# Shannon Entropy
def calculate_frequency_entropy(signal, fs, frame_length_ms, frame_step_ms, windowing_function="hamming"):
    """Calculate Shannon entropy in frequency domain using vectorized operations."""
    frames = frame_signal(signal, fs, frame_length_ms, frame_step_ms, windowing_function)
    entropy_values = np.zeros(len(frames))
    
    for i, frame in enumerate(frames):
        # Calculate power spectral density
        frequencies, power_spectrum = welch(frame, fs=fs)
        
        # Convert to probability distribution and calculate entropy
        power_sum = np.sum(power_spectrum) + 1e-10  # Prevent division by zero
        power_spectrum_prob = power_spectrum / power_sum
        entropy = -np.sum(power_spectrum_prob * np.log2(power_spectrum_prob + 1e-10))
        entropy_values[i] = entropy
        
    return entropy_values

# Shannon Entropy
def calculate_amplitude_entropy(signal, fs, frame_length_ms, frame_step_ms, windowing_function="hamming", nbins=256):
    """Calculate Shannon entropy in amplitude domain with improved histogram method."""
    frames = frame_signal(signal, fs, frame_length_ms, frame_step_ms, windowing_function)
    entropy_values = np.zeros(len(frames))
    
    for i, frame in enumerate(frames):
        # Only compute histogram if frame has sufficient variation
        if np.ptp(frame) > 1e-10:
            histogram, _ = np.histogram(frame, bins=nbins, density=True)
            prob_dist = histogram / np.sum(histogram)
            
            # Remove zero entries for log calculation
            prob_dist = prob_dist[prob_dist > 0]
            
            # Calculate Shannon entropy
            entropy_values[i] = -np.sum(prob_dist * np.log2(prob_dist))
        else:
            entropy_values[i] = 0  # Zero entropy for constant signal
            
    return entropy_values


# Multi Sclae permutation entropy
def permutation_entropy(time_series, m, tau):
    """Calculate permutation entropy with better memory efficiency."""
    n = len(time_series)
    # Pre-calculate all possible permutations
    all_permutations = list(permutations(range(m)))
    perm_dict = {p: 0 for p in all_permutations}
    
    # Count occurrences of each permutation pattern
    for i in range(0, n - tau * (m - 1), 1):  # Step by 1 for complete coverage
        pattern = time_series[i:i + tau * m:tau]
        # Only process complete patterns
        if len(pattern) == m:
            sorted_idx = tuple(np.argsort(pattern))
            perm_dict[sorted_idx] += 1
    
    # Convert counts to probabilities
    counts = np.array(list(perm_dict.values()))
    total = np.sum(counts)
    
    # Return 0 if no permutations were found
    if total == 0:
        return 0
        
    probabilities = counts / total
    # Calculate Shannon entropy (only for non-zero probabilities)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


# Function to coarse-grain the time series
def coarse_grain(time_series, scale):
    """Coarse-grain a time series with vectorized operations."""
    n = len(time_series)
    num_elements = n // scale
    
    # Handle empty result case
    if num_elements == 0:
        return np.array([np.mean(time_series)])
    
    # Reshape for vectorized mean calculation
    reshaped = time_series[:num_elements * scale].reshape((num_elements, scale))
    return np.mean(reshaped, axis=1)


# Function to calculate multiscale permutation entropy
def multiscale_permutation_entropy_helper(time_series, m, tau, scales):
    """Helper function for multiscale permutation entropy calculation."""
    mpe = np.zeros(len(scales))
    
    for i, scale in enumerate(scales):
        # Skip scales that would produce too short sequences
        min_length = tau * (m - 1) + 1
        if len(time_series) // scale < min_length:
            mpe[i] = np.nan
            continue
            
        # Coarse-grain the time series
        cg_series = coarse_grain(time_series, scale)
        
        # Calculate permutation entropy
        mpe[i] = permutation_entropy(cg_series, m, tau)
        
    return mpe


def multiscale_permutation_entropy(data, fs, window_length_ms, window_step_ms, m, tau, scales, windowing_function="hamming"):
    """Calculate multiscale permutation entropy for frames with proper error handling."""
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)
    
    # Initialize output array
    entropies = np.zeros((len(scales), frames.shape[0]))
    
    # Calculate multiscale permutation entropy for each frame
    for i, frame in enumerate(frames):
        entropies[:, i] = multiscale_permutation_entropy_helper(frame, m, tau, scales)
    
    return entropies


