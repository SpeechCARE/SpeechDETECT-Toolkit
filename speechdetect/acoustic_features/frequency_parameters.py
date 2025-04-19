import numpy as np
import parselmouth
import librosa
from scipy.signal import get_window
from .utils import frame_signal


def get_pitch(data, fs, frame_duration_ms=40, step_size_ms=10, pitch_floor=75, pitch_ceiling=600,
             method="ac", voicing_threshold=0.45):
    """
    Extract pitch (F0) for each frame using Parselmouth.
    
    Args:
        data: Audio signal
        fs: Sample rate
        frame_duration_ms: Frame duration in milliseconds
        step_size_ms: Step size in milliseconds
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
        method: Pitch extraction method ('ac' for autocorrelation, 'cc' for cross-correlation)
        voicing_threshold: Voicing threshold (0-1)
        
    Returns:
        numpy.ndarray: Array of pitch values (NaN for unvoiced frames)
    """
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Normalize data to prevent numerical issues
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))
    
    # Try to use Praat directly on the whole signal first (more efficient)
    try:
        sound = parselmouth.Sound(data, fs)
        pitch_obj = sound.to_pitch(
            time_step=step_size_ms/1000,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            method=method,
            voicing_threshold=voicing_threshold
        )
        
        # Extract pitch values at regular intervals
        times = np.arange(0, sound.duration, step_size_ms/1000)
        pitch_values = np.array([pitch_obj.get_value_at_time(t) or np.nan for t in times])
        return pitch_values
        
    except:
        # Fall back to frame-by-frame analysis if whole-signal approach fails
        frames = frame_signal(data, fs, frame_duration_ms, step_size_ms)
        pitch_values = np.full(len(frames), np.nan)
        
        for i, frame in enumerate(frames):
            if np.sum(np.abs(frame)) < 1e-10:  # Skip silent frames
                continue
                
            try:
                snd = parselmouth.Sound(frame, fs)
                pitch = snd.to_pitch(
                    time_step=frame_duration_ms/1000, 
                    pitch_floor=pitch_floor, 
                    pitch_ceiling=pitch_ceiling,
                    method=method,
                    voicing_threshold=voicing_threshold
                )
                f0 = pitch.selected_array['frequency']
                voiced_frames = f0[f0 > 0]
                
                if len(voiced_frames) > 0:
                    pitch_values[i] = np.mean(voiced_frames)
            except:
                pass
                
        return pitch_values


def f0_estimation(signal, sample_rate, frame_length_ms, frame_step_ms, fmin=librosa.note_to_hz('C2'), 
                 fmax=librosa.note_to_hz('C7'), win_length=None, center=True):
    """
    Estimate fundamental frequency (F0) using librosa's pYIN algorithm.
    
    Args:
        signal: Audio signal
        sample_rate: Sample rate
        frame_length_ms: Frame length in milliseconds
        frame_step_ms: Frame step in milliseconds
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        win_length: Window length in samples (defaults to frame_length)
        center: Whether to center the windows
        
    Returns:
        tuple: (F0 estimates, voiced flag, voiced probabilities)
    """
    # Convert to mono if stereo
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)
    
    # Set win_length to frame_length if not specified
    if win_length is None:
        win_length = frame_length_samples
    
    # Normalize signal to prevent numerical issues
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))
    
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            signal, 
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length_samples, 
            hop_length=frame_step_samples,
            win_length=win_length,
            center=center
        )
        return f0, voiced_flag, voiced_probs
    except Exception as e:
        # Return arrays of NaNs if pYIN fails
        num_frames = 1 + (len(signal) - frame_length_samples) // frame_step_samples
        return (np.full(num_frames, np.nan), 
                np.zeros(num_frames, dtype=bool), 
                np.zeros(num_frames))


def calculate_time_varying_jitter(f0, fs, pyin_hop_length_ms=25, jitter_frame_length_ms=250, 
                                 jitter_hop_length_ms=125, smoothing_window=None):
    """
    Calculate time-varying jitter from F0 contour.
    
    Args:
        f0: Array of F0 values
        fs: Sample rate
        pyin_hop_length_ms: Hop length used for F0 extraction in milliseconds
        jitter_frame_length_ms: Frame length for jitter calculation in milliseconds
        jitter_hop_length_ms: Hop length for jitter calculation in milliseconds
        smoothing_window: Optional window length for smoothing jitter values
        
    Returns:
        numpy.ndarray: Array of jitter values
    """
    # Make sure f0 is a numpy array
    f0 = np.asarray(f0)
    
    # Calculate hop and frame lengths in F0 frames
    pyin_hop_length = int(pyin_hop_length_ms * fs / 1000)
    jitter_frame_length = int(jitter_frame_length_ms * fs / 1000)
    jitter_hop_length = int(jitter_hop_length_ms * fs / 1000)
    
    f0_frames_per_jitter_frame = max(1, jitter_frame_length // pyin_hop_length)
    f0_frames_per_jitter_hop = max(1, jitter_hop_length // pyin_hop_length)
    
    # Calculate number of jitter frames
    num_jitter_frames = max(0, (len(f0) - f0_frames_per_jitter_frame) // f0_frames_per_jitter_hop + 1)
    jitter_values = np.full(num_jitter_frames, np.nan)
    
    # Vectorized approach for jitter calculation
    for i in range(num_jitter_frames):
        start_idx = i * f0_frames_per_jitter_hop
        end_idx = min(start_idx + f0_frames_per_jitter_frame, len(f0))
        
        # Get F0 values for current jitter frame
        frame_f0 = f0[start_idx:end_idx]
        
        # Keep only valid F0 values (non-zero, non-NaN)
        valid_mask = (frame_f0 > 0) & ~np.isnan(frame_f0)
        valid_f0 = frame_f0[valid_mask]
        
        if len(valid_f0) > 1:
            # Calculate jitter (normalized by mean F0)
            f0_diffs = np.abs(np.diff(valid_f0))
            mean_f0 = np.mean(valid_f0)
            
            if mean_f0 > 0:
                jitter_values[i] = np.mean(f0_diffs) / mean_f0
    
    # Apply optional smoothing
    if smoothing_window is not None and smoothing_window > 0:
        # Create a simple moving average window
        window = np.ones(smoothing_window) / smoothing_window
        # Use valid mode to avoid edge effects
        valid_jitter = jitter_values[~np.isnan(jitter_values)]
        if len(valid_jitter) >= smoothing_window:
            smoothed = np.convolve(valid_jitter, window, mode='valid')
            # Place smoothed values back in the original array
            valid_indices = np.where(~np.isnan(jitter_values))[0]
            if len(valid_indices) >= len(smoothed) + smoothing_window - 1:
                jitter_values[valid_indices[smoothing_window//2:-(smoothing_window//2)]] = smoothed
    
    return jitter_values


def get_formants(data, fs, max_formants=5, window_length=0.025, pre_emphasis=50, 
                time_step=None, max_formant_freq=5500):
    """
    Extract formants for the entire signal using Parselmouth.
    
    Args:
        data: Audio signal
        fs: Sample rate
        max_formants: Maximum number of formants to extract
        window_length: Window length for formant analysis in seconds
        pre_emphasis: Pre-emphasis frequency in Hz
        time_step: Time step between formant measurements (default: 0.01 seconds)
        max_formant_freq: Maximum formant frequency in Hz
        
    Returns:
        dict: Dictionary containing formant data and timestamps
    """
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Set default time step if not provided
    if time_step is None:
        time_step = 0.01  # 10 ms default
    
    try:
        sound = parselmouth.Sound(data, fs)
        formants = sound.to_formant_burg(
            time_step=time_step,
            max_number_of_formants=max_formants,
            maximum_formant=max_formant_freq,
            window_length=window_length,
            pre_emphasis_from=pre_emphasis
        )
        
        # Get number of frames
        num_frames = formants.get_number_of_frames()
        times = np.array([formants.get_time_from_frame_number(i+1) for i in range(num_frames)])
        
        # Extract all formants for all frames
        formant_data = {}
        for formant_num in range(1, max_formants + 1):
            frequencies = np.array([
                formants.get_value_at_time(formant_num, t) for t in times
            ])
            bandwidths = np.array([
                formants.get_bandwidth_at_time(formant_num, t) for t in times
            ])
            
            formant_data[f'F{formant_num}'] = frequencies
            formant_data[f'F{formant_num}_bandwidth'] = bandwidths
        
        formant_data['times'] = times
        return formant_data
        
    except Exception as e:
        # Return empty dict if formant extraction fails
        return {'error': str(e)}


def get_formants_frame_based(data, fs, frame_duration_ms=40, step_size_ms=10, formant_number=None, 
                           max_formants=5, window_length=0.025, pre_emphasis=50, max_formant_freq=5500):
    """
    Extract formant frequencies for each frame using Parselmouth.
    
    Args:
        data: Audio signal
        fs: Sample rate
        frame_duration_ms: Frame duration in milliseconds
        step_size_ms: Step size in milliseconds
        formant_number: List of formant numbers to extract (1-based indexing, e.g., [1,2,3] for F1,F2,F3)
                       If None, extracts all formants up to max_formants
        max_formants: Maximum number of formants to extract
        window_length: Window length for formant analysis in seconds
        pre_emphasis: Pre-emphasis frequency in Hz
        max_formant_freq: Maximum formant frequency in Hz
        
    Returns:
        numpy.ndarray: Array of formant frequencies with shape (num_formants, num_frames)
    """
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    frames = frame_signal(data, fs, frame_duration_ms, step_size_ms)
    
    # Determine which formants to extract
    if formant_number is None:
        formant_number = list(range(1, max_formants+1))
    
    num_formants = len(formant_number)
    formant_values = np.full((num_formants, len(frames)), np.nan)
    bandwidth_values = np.full((num_formants, len(frames)), np.nan)
    
    # Try whole-signal approach first (more efficient)
    try:
        sound = parselmouth.Sound(data, fs)
        formants = sound.to_formant_burg(
            time_step=step_size_ms/1000,
            max_number_of_formants=max_formants,
            maximum_formant=max_formant_freq,
            window_length=window_length,
            pre_emphasis_from=pre_emphasis
        )
        
        # Calculate timestamps for each frame
        frame_times = np.arange(frame_duration_ms/2000, sound.duration, step_size_ms/1000)
        frame_times = frame_times[:len(frames)]  # Ensure we don't exceed number of frames
        
        # Extract formants for each frame at the calculated times
        for j, n in enumerate(formant_number):
            if 1 <= n <= max_formants:
                for i, t in enumerate(frame_times):
                    if i < len(frames):
                        formant_values[j, i] = formants.get_value_at_time(n, t) or np.nan
                        bandwidth_values[j, i] = formants.get_bandwidth_at_time(n, t) or np.nan
        
        return formant_values, bandwidth_values
    
    except:
        # Fall back to frame-by-frame analysis if whole-signal approach fails
        for i, frame in enumerate(frames):
            if np.sum(np.abs(frame)) < 1e-10:  # Skip silent frames
                continue
                
            try:
                snd = parselmouth.Sound(frame, fs)
                formants = snd.to_formant_burg(
                    time_step=frame_duration_ms/1000,
                    max_number_of_formants=max_formants,
                    maximum_formant=max_formant_freq,
                    window_length=window_length,
                    pre_emphasis_from=pre_emphasis
                )
                
                # Extract each requested formant at the midpoint
                for j, n in enumerate(formant_number):
                    if 1 <= n <= max_formants:
                        formant_values[j, i] = formants.get_value_at_time(n, snd.duration/2) or np.nan
                        bandwidth_values[j, i] = formants.get_bandwidth_at_time(n, snd.duration/2) or np.nan
            except:
                pass
        
        return formant_values, bandwidth_values


def calculate_frequency_variability(f0, frame_step_ms, window_size_ms=250):
    """
    Calculate frequency variability metrics from an F0 contour.
    
    Args:
        f0: Array of F0 values
        frame_step_ms: Step size of F0 frames in milliseconds
        window_size_ms: Window size for variability calculation in milliseconds
        
    Returns:
        dict: Dictionary with variability metrics
    """
    # Make sure f0 is a numpy array
    f0 = np.asarray(f0)
    
    # Remove NaN and zero values
    valid_mask = ~np.isnan(f0) & (f0 > 0)
    valid_f0 = f0[valid_mask]
    
    if len(valid_f0) < 2:
        return {
            'f0_mean': np.nan,
            'f0_median': np.nan,
            'f0_std': np.nan,
            'f0_range': np.nan,
            'f0_slope': np.nan,
            'f0_contour_variability': np.nan,
            'f0_coefficient_of_variation': np.nan
        }
    
    # Basic statistics
    f0_mean = np.mean(valid_f0)
    f0_median = np.median(valid_f0)
    f0_std = np.std(valid_f0)
    f0_range = np.max(valid_f0) - np.min(valid_f0)
    f0_coefficient_of_variation = f0_std / f0_mean if f0_mean > 0 else np.nan
    
    # Calculate average slope
    time_points = np.arange(len(valid_f0)) * frame_step_ms / 1000
    if len(time_points) > 1:
        coeffs = np.polyfit(time_points, valid_f0, 1)
        f0_slope = coeffs[0]  # Hz/second
    else:
        f0_slope = 0
    
    # Calculate contour variability (mean absolute difference)
    f0_contour_variability = np.mean(np.abs(np.diff(valid_f0)))
    
    return {
        'f0_mean': f0_mean,
        'f0_median': f0_median,
        'f0_std': f0_std,
        'f0_range': f0_range,
        'f0_slope': f0_slope,
        'f0_contour_variability': f0_contour_variability,
        'f0_coefficient_of_variation': f0_coefficient_of_variation
    }


def calculate_shimmer(signal, fs, f0=None, frame_duration_ms=40, step_size_ms=10):
    """
    Calculate shimmer (amplitude variation) in a speech signal.
    
    Args:
        signal: Audio signal
        fs: Sample rate
        f0: Optional pre-calculated F0 values
        frame_duration_ms: Frame duration in milliseconds
        step_size_ms: Step size in milliseconds
        
    Returns:
        numpy.ndarray: Array of shimmer values
    """
    # Get pitch if not provided
    if f0 is None:
        f0 = get_pitch(signal, fs, frame_duration_ms, step_size_ms)
    
    frames = frame_signal(signal, fs, frame_duration_ms, step_size_ms)
    shimmer_values = np.full(len(frames), np.nan)
    
    for i, frame in enumerate(frames):
        # Skip unvoiced frames
        if np.isnan(f0[i]) or f0[i] <= 0:
            continue
            
        # Estimate period in samples
        period_samples = int(fs / f0[i])
        
        if period_samples < 2:
            continue
            
        # Find peaks in the frame (one per period)
        try:
            peaks = librosa.util.peak_pick(
                np.abs(frame), 
                pre_max=period_samples//2,
                post_max=period_samples//2,
                pre_avg=period_samples//2,
                post_avg=period_samples//2,
                delta=0.01,
                wait=period_samples//2
            )
            
            # Extract peak amplitudes
            peak_amps = np.abs(frame[peaks])
            
            # Calculate shimmer (normalized amplitude difference)
            if len(peak_amps) > 1:
                amp_diffs = np.abs(np.diff(peak_amps))
                shimmer_values[i] = np.mean(amp_diffs) / np.mean(peak_amps)
        except:
            pass
    
    return shimmer_values