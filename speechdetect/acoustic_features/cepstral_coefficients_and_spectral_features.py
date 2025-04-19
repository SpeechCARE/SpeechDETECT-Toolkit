import numpy as np
import scipy
import librosa
import parselmouth
from scipy.fftpack import dct
from scipy.signal import find_peaks, periodogram
from scipy.io import wavfile
from scipy.linalg import solve_toeplitz
from spafe.fbanks import mel_fbanks, bark_fbanks
from utils import frame_signal, freq2mel, mel2freq, lifter_ceps, Preprocessing, poly2lsf, stft


def amplitude_range(data, fs, window_length_ms, window_step_ms, windowing_function="hamming"):
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)
    ranges = np.ptp(frames, axis=1)  # Peak-to-peak calculation
    std = np.std(frames, axis=1)
    return ranges, std

def compute_msc(signal, sample_rate, nfft=512, window_length_ms=90, window_step_ms=25, num_msc=13):
    frames = frame_signal(signal, sample_rate, window_length_ms, window_step_ms)
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))
    modulation_spectra = np.abs(np.fft.rfft(pow_frames, nfft, axis=0))
    msc = dct(modulation_spectra, type=2, axis=0, norm='ortho')[:num_msc]
    return msc

def calculate_centriod(x, fs):
    magnitudes = np.abs(np.fft.rfft(x))
    freqs = np.abs(np.fft.fftfreq(len(x), 1.0/fs)[:len(x)//2+1])
    return np.nan if np.sum(magnitudes) == 0 else np.sum(magnitudes*freqs) / np.sum(magnitudes)
    
def spectral_centriod(data, fs, window_length_ms, window_step_ms, windowing_function="hamming"):
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)
    return [calculate_centriod(frame, fs) for frame in frames]

def ltas(x, fs, window_length_ms, window_step_ms, units='db', graph=False):
    win = int(window_length_ms / 1000 * fs)
    hop = int(window_step_ms / 1000 * fs)
    
    f, t, S = scipy.signal.stft(x, fs, window='hann', nperseg=win, noverlap=win-hop)
    PSD = np.abs(S)**2
    ltas_result = np.mean(PSD, axis=1)
    
    if units == 'db':
        ltas_result = 10 * np.log10(ltas_result + 1e-12)
        
    if graph:
        import matplotlib.pyplot as plt
        plt.plot(f, ltas_result)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (dB)')
        plt.title('Long-term average spectrum')
        plt.show()
        
    return ltas_result, f

def harmonic_difference(signal, sample_rate, window_length_ms, window_step_ms, n_first_formant, n_second_fromant):
    frames = frame_signal(signal, sample_rate, window_length_ms, window_step_ms)
    differences = np.zeros(len(frames))
    
    for i, frame in enumerate(frames):
        snd = parselmouth.Sound(frame, sample_rate)
        pitch = snd.to_pitch()
        f0 = pitch.selected_array['frequency']
        f0 = f0[f0 > 0]
        
        if len(f0) == 0:
            differences[i] = np.nan
            continue
            
        estimated_f0 = np.mean(f0)
        
        formants = snd.to_formant_burg(max_number_of_formants=5)
        fn_freq = formants.get_value_at_time(n_second_fromant, snd.duration / 2)
        
        frequencies, spectrum = periodogram(signal, sample_rate)
        harmonic_freqs = estimated_f0 * np.arange(1, int(sample_rate / (2 * estimated_f0)) + 1)
        harmonic_amplitudes = np.interp(harmonic_freqs, frequencies, spectrum)
        
        harmonic_fn_range = harmonic_amplitudes[(harmonic_freqs >= fn_freq - estimated_f0) & 
                                               (harmonic_freqs <= fn_freq + estimated_f0)]
        An_amplitude = np.max(harmonic_fn_range) if len(harmonic_fn_range) > 0 else 1e-12
        H1_amplitude = harmonic_amplitudes[n_first_formant-1] if n_first_formant <= len(harmonic_amplitudes) else 1e-12
        
        H1_An_dB = 20 * np.log10(max(H1_amplitude, 1e-12)) - 20 * np.log10(max(An_amplitude, 1e-12))
        differences[i] = H1_An_dB
        
    return differences

def alpha_ratio(data, fs, window_length_ms, window_step_ms, lower_band, higher_band):
    window_length_samples = int(window_length_ms*fs/1000)
    window_step_samples = int(window_step_ms*fs/1000)
    
    f, lt = scipy.signal.welch(data, fs, nperseg=window_length_samples, noverlap=window_step_samples)
    
    low_mask = (f > lower_band[0]) & (f < lower_band[1])
    high_mask = (f > higher_band[0]) & (f <= higher_band[1])
    
    low_frequency_energy = np.sum(lt[low_mask])
    high_frequency_energy = np.sum(lt[high_mask])
    
    return high_frequency_energy / (low_frequency_energy + 1e-12)

def log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands=120, fmin=20, fmax=8000, window="hamming"):
    spectrogram = stft(data, fs, window_length_ms=window_length_ms, window_step_ms=window_step_ms, windowing_function=window)
    
    nfft = 2048
    melfilterbank, _ = mel_fbanks.mel_filter_banks(
        nfilts=melbands,
        nfft=spectrogram.shape[1] * 2 - 1,
        fs=fs,
        low_freq=fmin,
        high_freq=fmax
    )
    
    logmelspectrogram = np.matmul(np.abs(spectrogram)**2, melfilterbank.T) + 1e-12
    return logmelspectrogram.T

def mfcc(data, fs, window_length_ms, window_step_ms, melbands=120, fmin=20, fmax=8000, lifter=0, window="hamming"):
    logmelspectrogram = log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands, fmin, fmax, window)
    logmelspectrogram = np.log(logmelspectrogram)
    mfcc_result = scipy.fft.dct(logmelspectrogram.T).T
    
    if lifter > 0:
        mfcc_result = lifter_ceps(mfcc_result.T, lifter).T
        
    return mfcc_result

def lpc(data, fs, window_length_ms, window_step_ms, lpc_length=0, windowing_function="hamming"):
    if lpc_length == 0:
        lpc_length = int(1.25*fs/1000)
        
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)
    lpc_coeffs = np.zeros((lpc_length + 1, frames.shape[0]))
    
    for k, frame in enumerate(frames):
        if np.max(np.abs(frame)) == 0:
            continue
            
        frame = frame / np.max(np.abs(frame))
        
        try:
            X = scipy.fft.fft(frame)
            autocovariance = np.real(scipy.fft.ifft(np.abs(X)**2))
            b = np.zeros(lpc_length + 1)
            b[0] = 1.
            a = solve_toeplitz(autocovariance[:lpc_length + 1], b)
            lpc_coeffs[:, k] = a / a[0]
        except:
            pass
            
    return lpc_coeffs

def lpcc(data, fs, window_length_ms, window_step_ms, lpc_length=0, windowing_function="hamming"):
    if lpc_length == 0:
        lpc_length = int(1.25*fs/1000)
        
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)
    lpccs = np.zeros((lpc_length + 1, frames.shape[0]))
    
    for k, frame in enumerate(frames):
        if np.max(np.abs(frame)) == 0:
            continue
            
        frame = frame / np.max(np.abs(frame))
        
        X = scipy.fft.fft(frame)
        autocovariance = np.real(scipy.fft.ifft(np.abs(X)**2))
        b = np.zeros(lpc_length + 1)
        b[0] = 1.
        
        try:
            a = solve_toeplitz(autocovariance[:lpc_length + 1], b)
            a = a / a[0]
            
            if np.sum(a) == 0:
                lpcc = np.zeros(len(a))
            else:
                powerspectrum = np.abs(np.fft.fft(a))**2
                lpcc = np.fft.ifft(np.log(powerspectrum + 1e-12))
                
            lpccs[:, k] = np.abs(lpcc)
        except:
            pass
            
    return lpccs

def spectral_envelope(data, fs, window_length_ms, window_step_ms, spectrum_length=512, windowing_function="hamming"):
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)
    nyquist_frequency = fs // 2
    
    len_spectrum = spectrum_length // 2 + 1
    frequency_step_Hz = 500
    frequency_step = int(len_spectrum * frequency_step_Hz / nyquist_frequency)
    frequency_bins = int(len_spectrum / frequency_step + 0.5)
    
    envelopes = np.zeros((frequency_bins, frames.shape[0]))
    
    # Create filterbank once
    slope = np.arange(0.5, frequency_step + 0.5, 1) / (frequency_step + 1)
    backslope = np.flipud(slope)
    filterbank = np.zeros((len_spectrum, frequency_bins))
    filterbank[0:frequency_step, 0] = 1
    filterbank[-frequency_step:, -1] = 1
    
    for k in range(frequency_bins - 1):
        idx = int((k + 0.25) * frequency_step) + np.arange(0, frequency_step)
        idx = idx[idx < len_spectrum]  # Ensure indices are valid
        filterbank[idx, k + 1] = slope[:len(idx)]
        filterbank[idx, k] = backslope[:len(idx)]
    
    for i, window in enumerate(frames):
        spectrum = scipy.fft.rfft(window, n=spectrum_length)
        spectrum_smoothed = np.matmul(filterbank.T, np.abs(spectrum)**2)
        logspectrum_smoothed = 10 * np.log10(spectrum_smoothed + np.finfo(float).eps)
        envelopes[:, i] = logspectrum_smoothed
        
    return envelopes

def calculate_cpp(data, fs, window_length_ms, window_step_ms, window="hamming"):
    frame_length = int(window_length_ms * fs / 1000)
    frame_step = int(window_step_ms * fs / 1000)
    
    frames = librosa.util.frame(data, frame_length=frame_length, hop_length=frame_step)
    
    if window == "hamming":
        window = np.hamming(frame_length)
    
    windowed_frames = frames * window.reshape(-1, 1)
    
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(windowed_frames.T, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    log_pow_frames = np.log(pow_frames + np.finfo(float).eps)
    real_cepstrum = np.fft.irfft(log_pow_frames, NFFT)
    
    peak = np.max(real_cepstrum, axis=1)
    baseline = np.mean(real_cepstrum, axis=1)
    cpp = peak - baseline
    
    return cpp

def hammIndex(x, fs):
    f1, f2 = 2000, 5000
    
    # Assuming Preprocessing is imported from helper
    norm_signal = Preprocessing(fs, x)
    
    fq, Pxx = scipy.signal.welch(norm_signal, fs, nperseg=2048)
    Pxx_dB = 10 * np.log10(Pxx + 1e-12)
    
    n1 = np.abs(fq - f1).argmin()
    n2 = np.abs(fq - f2).argmin()
    
    seg1 = Pxx_dB[0:n1]
    seg2 = Pxx_dB[n1:n2]
    
    SPL02 = np.max(seg1)
    SPL25 = np.max(seg2)
    HammIndex = SPL02 - SPL25
    
    return HammIndex, fq, Pxx_dB

def plp_features(signal, sample_rate, num_filters=20, N_fft=512, fmin=20, fmax=8000):
    frames = librosa.util.frame(signal, frame_length=N_fft, hop_length=N_fft // 2)
    windowed_frames = frames * np.hamming(N_fft)[:, None]
    
    fft_frames = np.fft.rfft(windowed_frames.T, N_fft)
    power_frames = np.abs(fft_frames) ** 2
    power_frames_db = 10 * np.log10(power_frames + np.finfo(float).eps)
    
    filter_bank, _ = bark_fbanks.bark_filter_banks(
        nfilts=num_filters,
        nfft=N_fft,
        fs=sample_rate,
        low_freq=fmin,
        high_freq=fmax
    )
    
    filter_bank_energy = np.dot(power_frames_db, filter_bank.T)
    
    # Pre-emphasis function
    def E(w):
        return ((w**2 + 56.8e6) * w**4) / ((w**2 + 6.3e6) * (w**2 + 0.38e9) * (w**6 + 9.58e26))
    
    # Apply pre-emphasis (non-vectorized approach)
    filter_bank_energy_emphasized = np.zeros_like(filter_bank_energy)
    for i, frame in enumerate(filter_bank_energy):
        filter_bank_energy_emphasized[i] = np.array([E(w) for w in frame])
    
    # Non-linear transformation (cubic root)
    filter_bank_energy_nonlinear = np.power(np.abs(filter_bank_energy), 1/3)
    
    lpc_length = 8
    order = lpc_length + 1
    lpccs = np.zeros((filter_bank_energy_nonlinear.shape[0], order))
    
    for k, frame in enumerate(filter_bank_energy_nonlinear):
        X = scipy.fft.fft(frame)
        autocovariance = np.real(scipy.fft.ifft(np.abs(X)**2))
        b = np.zeros(lpc_length + 1)
        b[0] = 1.
        
        try:
            a = solve_toeplitz(autocovariance[:lpc_length + 1], b)
            a = a / a[0]
            
            if np.sum(a) == 0:
                lpcc = np.zeros(len(a))
            else:
                powerspectrum = np.abs(np.fft.fft(a)) ** 2
                lpcc = np.fft.ifft(np.log(powerspectrum + 1e-12))
                
            lpccs[k, :] = np.abs(lpcc)
        except:
            pass
            
    return lpccs.T

def harmonicity(data, fs):
    snd = parselmouth.Sound(data, fs)
    hnr = snd.to_harmonicity()
    hnr_values = hnr.values
    hnr_values = hnr_values[hnr_values > 0]
    return np.nanmean(hnr_values)

def calculate_lsp_freqs_for_frames(signal, fs, window_length_ms, window_step_ms, order, window="hamming"):
    c = lpc(signal, fs, window_length_ms, window_step_ms, lpc_length=order, windowing_function=window)
    lsp_freqs = np.zeros((order, c.shape[1]))
    
    for i, frame in enumerate(c.T):
        try:
            lsp_freqs[:, i] = poly2lsf(frame)
        except:
            pass
            
    return lsp_freqs

def calculate_frame_wise_zcr(signal, fs, window_length_ms, window_step_ms):
    window_length = int(window_length_ms * fs / 1000)
    window_step = int(window_step_ms * fs / 1000)
    frames = librosa.util.frame(signal, frame_length=window_length, hop_length=window_step)
    zcr = np.array([np.sum(np.abs(np.diff(np.signbit(frame)))) / (2 * len(frame)) for frame in frames.T])
    return zcr
