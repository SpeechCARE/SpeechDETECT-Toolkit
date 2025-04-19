import numpy as np
from scipy import stats
from functools import wraps
from typing import Callable, Any, Union, List, Tuple, Optional, TypeVar, cast


# Type aliases
NumericArray = Union[np.ndarray, List[float]]
T = TypeVar('T')

def handle_nan(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to handle NaN values in input arrays.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with NaN handling
    """
    @wraps(func)
    def wrapper(frame: NumericArray, *args, **kwargs) -> T:
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=float)
        
        # Skip NaN handling for certain functions
        if func.__name__ in ['sma', 'de']:
            return func(frame, *args, **kwargs)
            
        # Handle empty arrays
        if frame.size == 0:
            if func.__name__ in ['max', 'skewness', 'kurtosis']:
                return np.nan
            elif func.__name__ in ['min', 'stddev']:
                return np.nan
            elif func.__name__ in ['span', 'iqr1_2', 'iqr2_3', 'iqr1_3', 'pctlrange0_1']:
                return 0.0
            elif func.__name__ in ['maxPos', 'minPos']:
                return 0
            elif func.__name__ in ['amean', 'quartile1', 'quartile2', 'quartile3',
                                  'percentile1', 'percentile99', 'linregc1', 'linregc2']:
                return 0.0
            return 0.0
            
        # Remove NaN values
        clean_frame = frame[~np.isnan(frame)]
        
        # Handle arrays with only NaN values
        if clean_frame.size == 0:
            if func.__name__ in ['max', 'min', 'skewness', 'kurtosis']:
                return np.nan
            elif func.__name__ in ['stddev']:
                return np.nan
            elif func.__name__ in ['span', 'iqr1_2', 'iqr2_3', 'iqr1_3', 'pctlrange0_1']:
                return 0.0
            elif func.__name__ in ['maxPos', 'minPos']:
                return 0
            return 0.0
            
        return func(clean_frame, *args, **kwargs)
    return wrapper

@handle_nan
def sma(signal: NumericArray, window_size: int = 3) -> np.ndarray:
    """Apply a simple moving average (SMA) smoothing filter.
    
    Args:
        signal: Input signal to smooth
        window_size: Size of the smoothing window
        
    Returns:
        Smoothed signal
    """
    # Ensure window size is valid
    if window_size < 1:
        window_size = 1
    
    # Remove NaN values before convolution
    signal = signal[~np.isnan(signal)]
    
    if len(signal) == 0:
        return np.array([])
    elif len(signal) < window_size:
        return signal
    
    # Apply convolution for moving average
    return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

@handle_nan
def de(signal: NumericArray) -> np.ndarray:
    """Calculate the delta (first order derivative) of a signal.
    
    Args:
        signal: Input signal
        
    Returns:
        First derivative of the signal
    """
    # Handle empty arrays
    if len(signal) <= 1:
        return np.array([])
    
    # Remove NaN values
    signal = signal[~np.isnan(signal)]
    
    # Calculate derivative
    return np.diff(signal, n=1)

@handle_nan
def max(frame: NumericArray) -> float:
    """Calculate the maximum value of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Maximum value
    """
    return float(np.max(frame))

@handle_nan
def min(frame: NumericArray) -> float:
    """Calculate the minimum value of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Minimum value
    """
    return float(np.min(frame))

@handle_nan
def span(frame: NumericArray) -> float:
    """Calculate the range (max - min) of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Range of values
    """
    if len(frame) < 2:
        return 0.0
    return float(np.max(frame) - np.min(frame))

@handle_nan
def maxPos(frame: NumericArray) -> int:
    """Find the position of the maximum value in the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Index of maximum value
    """
    return int(np.argmax(frame))

@handle_nan
def minPos(frame: NumericArray) -> int:
    """Find the position of the minimum value in the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Index of minimum value
    """
    return int(np.argmin(frame))

@handle_nan
def amean(frame: NumericArray) -> float:
    """Calculate the arithmetic mean of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Arithmetic mean
    """
    return float(np.mean(frame))

@handle_nan
def linregc1(frame: NumericArray) -> float:
    """Calculate the slope (first coefficient) of a linear regression.
    
    Args:
        frame: Input array
        
    Returns:
        Slope coefficient of linear regression
    """
    if len(frame) < 2:
        return 0.0
    
    x = np.arange(len(frame))
    try:
        m, _ = np.polyfit(x, frame, 1)
        return float(m)
    except (ValueError, np.linalg.LinAlgError):
        return 0.0

@handle_nan
def linregc2(frame: NumericArray) -> float:
    """Calculate the intercept (second coefficient) of a linear regression.
    
    Args:
        frame: Input array
        
    Returns:
        Intercept coefficient of linear regression
    """
    if len(frame) < 2:
        return 0.0
    
    x = np.arange(len(frame))
    try:
        _, t = np.polyfit(x, frame, 1)
        return float(t)
    except (ValueError, np.linalg.LinAlgError):
        return 0.0

@handle_nan
def linregerrA(frame: NumericArray) -> float:
    """Calculate the absolute error of linear regression fit.
    
    Args:
        frame: Input array
        
    Returns:
        Sum of absolute differences between original values and linear fit
    """
    if len(frame) < 2:
        return 0.0
    
    x = np.arange(len(frame))
    try:
        m, t = np.polyfit(x, frame, 1)
        linear_fit = m * x + t
        return float(np.sum(np.abs(linear_fit - frame)))
    except (ValueError, np.linalg.LinAlgError):
        return 0.0

@handle_nan
def linregerrQ(frame: NumericArray) -> float:
    """Calculate the squared error of linear regression fit.
    
    Args:
        frame: Input array
        
    Returns:
        Sum of squared differences between original values and linear fit
    """
    if len(frame) < 2:
        return 0.0
    
    x = np.arange(len(frame))
    try:
        m, t = np.polyfit(x, frame, 1)
        linear_fit = m * x + t
        return float(np.sum((linear_fit - frame) ** 2))
    except (ValueError, np.linalg.LinAlgError):
        return 0.0

@handle_nan
def stddev(frame: NumericArray) -> float:
    """Calculate the standard deviation of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Standard deviation
    """
    if len(frame) < 2:
        return 0.0
    return float(np.std(frame, ddof=1))  # Using ddof=1 for sample standard deviation

@handle_nan
def skewness(frame: NumericArray) -> float:
    """Calculate the skewness of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Skewness value
    """
    if len(frame) < 3:
        return 0.0
    return float(stats.skew(frame))

@handle_nan
def kurtosis(frame: NumericArray) -> float:
    """Calculate the kurtosis of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Kurtosis value
    """
    if len(frame) < 4:
        return 0.0
    return float(stats.kurtosis(frame))

@handle_nan
def quartile1(frame: NumericArray) -> float:
    """Calculate the first quartile (25th percentile) of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        First quartile value
    """
    return float(np.percentile(frame, 25))

@handle_nan
def quartile2(frame: NumericArray) -> float:
    """Calculate the second quartile (median, 50th percentile) of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Median value
    """
    return float(np.percentile(frame, 50))

@handle_nan
def quartile3(frame: NumericArray) -> float:
    """Calculate the third quartile (75th percentile) of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        Third quartile value
    """
    return float(np.percentile(frame, 75))

@handle_nan
def iqr1_2(frame: NumericArray) -> float:
    """Calculate the interquartile range between Q1 and Q2.
    
    Args:
        frame: Input array
        
    Returns:
        Interquartile range Q2-Q1
    """
    q1 = quartile1(frame)
    q2 = quartile2(frame)
    return float(q2 - q1)

@handle_nan
def iqr2_3(frame: NumericArray) -> float:
    """Calculate the interquartile range between Q2 and Q3.
    
    Args:
        frame: Input array
        
    Returns:
        Interquartile range Q3-Q2
    """
    q2 = quartile2(frame)
    q3 = quartile3(frame)
    return float(q3 - q2)

@handle_nan
def iqr1_3(frame: NumericArray) -> float:
    """Calculate the interquartile range between Q1 and Q3.
    
    Args:
        frame: Input array
        
    Returns:
        Interquartile range Q3-Q1
    """
    q1 = quartile1(frame)
    q3 = quartile3(frame)
    return float(q3 - q1)

@handle_nan
def percentile1(frame: NumericArray) -> float:
    """Calculate the 1st percentile of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        1st percentile value
    """
    return float(np.percentile(frame, 1))

@handle_nan
def percentile99(frame: NumericArray) -> float:
    """Calculate the 99th percentile of the frame.
    
    Args:
        frame: Input array
        
    Returns:
        99th percentile value
    """
    return float(np.percentile(frame, 99))

@handle_nan
def pctlrange0_1(frame: NumericArray) -> float:
    """Calculate the range between 1st and 99th percentiles.
    
    Args:
        frame: Input array
        
    Returns:
        Range between 1st and 99th percentiles
    """
    p1 = percentile1(frame)
    p99 = percentile99(frame)
    return float(p99 - p1)

@handle_nan
def upleveltime75(frame: NumericArray) -> float:
    """Calculate the percentage of time the signal is above the 3rd quartile.
    
    Args:
        frame: Input array
        
    Returns:
        Percentage of values above Q3
    """
    threshold = quartile3(frame)
    return float(np.mean(frame > threshold) * 100)

@handle_nan
def upleveltime90(frame: NumericArray) -> float:
    """Calculate the percentage of time the signal is above 90% of its range.
    
    Args:
        frame: Input array
        
    Returns:
        Percentage of values above the 90% threshold
    """
    if len(frame) < 2:
        return 0.0
    
    min_val, max_val = np.min(frame), np.max(frame)
    threshold = min_val + 0.9 * (max_val - min_val)
    return float(np.mean(frame > threshold) * 100)

# List of function names for external reference
function_names = [
    "max",
    "min",
    "span",
    "maxPos",
    "minPos",
    "amean",
    "linregc1",
    "linregc2",
    "linregerrA",
    "linregerrQ",
    "stddev",
    "skewness",
    "kurtosis",
    "quartile1",
    "quartile2",
    "quartile3",
    "iqr1_2",
    "iqr2_3",
    "iqr1_3",
    "percentile1",
    "percentile99",
    "pctlrange0_1",
    "upleveltime75",
    "upleveltime90"
]
