"""
AcousticFeatureExtractor - Main interface for extracting acoustic features from speech recordings
"""
import torch
import os
import numpy as np
import inspect
import librosa
import logging
import csv
from typing import List, Dict, Union, Any, Optional
from .acoustic_features import (\

    get_pitch, calculate_time_varying_jitter, get_formants_frame_based,
    compute_msc, spectral_centriod, ltas, alpha_ratio, log_mel_spectrogram,
    mfcc, lpc, lpcc, spectral_envelope, calculate_cpp, hammIndex,
    plp_features, harmonicity, calculate_lsp_freqs_for_frames, calculate_frame_wise_zcr,
    analyze_audio_shimmer, calculate_frame_level_hnr, amplitude_range,
    rms_amplitude, spl_per_frame, peak_amplitude, short_time_energy, intensity,
    calculate_hfd_per_frame, calculate_frequency_entropy, calculate_amplitude_entropy,
    
    # Classes needed for process_file_model
    PauseBehavior, SpeechBehavior,
    
    # Statistical functions and names
    sma, de, function_names,
)

class AcousticFeatureExtractor:
    """
    A unified interface for extracting various acoustic features from speech recordings.
    """
    
    def __init__(self, sampling_rate=16000, log_level=logging.INFO):
        """
        Initialize the feature extractor
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of the audio files to process
        log_level : int
            Logging level (default: logging.INFO)
        """
        self.sampling_rate = sampling_rate
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        
        # Available feature types mapped to their extraction methods
        self.available_features = {
            'spectral': self.extract_spectral_features,
            'complexity': self.extract_complexity_features,
            'frequency': self.extract_frequency_features,
            'intensity': self.extract_intensity_features,
            'rhythmic': self.extract_rhythmic_features,
            'fluency': self.extract_fluency_features,
            'voice_quality': self.extract_voice_quality_features,
            'all': self.extract_all_features,
            'raw': self.process_file,
            'transcription': self.process_file_model
        }
                
        # VAD model and transcription model (to be set later if needed)
        self.vad_model = None
        self.vad_utils = None
        self.transcription_model = None
        
        self.logger.info("AcousticFeatureExtractor initialized with sampling rate %d Hz", sampling_rate)
    
    def set_models(self, vad_model=None, vad_utils=None, transcription_model=None):
        """
        Set the VAD and transcription models for advanced feature extraction
        
        Parameters:
        -----------
        vad_model : object
            Voice Activity Detection model
        vad_utils : object
            Utility functions for the VAD model
        transcription_model : object
            Transcription model for speech-to-text
        """
        self.vad_model = vad_model
        self.vad_utils = vad_utils
        self.transcription_model = transcription_model
        
        models_set = []
        if vad_model is not None:
            models_set.append("VAD model")
        if vad_utils is not None:
            models_set.append("VAD utils")
        if transcription_model is not None:
            models_set.append("transcription model")
            
        self.logger.info("Models configured: %s", ", ".join(models_set))
    
    def extract_all_features(self, audio_path):
        """
        Extract all available acoustic features from an audio file
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
            
        Returns:
        --------
        dict
            Dictionary containing all extracted features
        """
        self.logger.info("Extracting all features from: %s", audio_path)
        features = {}
        
        # Extract all available feature sets except 'all' to avoid infinite recursion
        for feature_type, feature_method in self.available_features.items():
            # Skip 'all' to prevent infinite recursion
            if feature_type == 'all':
                continue
                
            try:
                feature_result = feature_method(audio_path)
                features.update(feature_result)
            except Exception as e:
                self.logger.error("Error extracting %s features: %s", feature_type, str(e))
        
        self.logger.info("Extracted %d features in total from 'all' option", len(features))
        return features
    
    def extract_spectral_features(self, audio_path):
        """Extract spectral features from audio"""
        self.logger.debug("Extracting spectral features from: %s", audio_path)
        try:
            # Load audio file
            data, fs = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Define window parameters
            window_length_ms = 50
            window_step_ms = 25
            
            features = {}
            
            # Spectral features extraction
            try:
                # Modulation Spectrum Coefficients
                msc = compute_msc(data, fs, nfft=512, window_length_ms=window_length_ms, window_step_ms=window_step_ms, num_msc=13)
                features.update(self.process_matrix(msc, 'MSC'))
                
                # Spectral centroids
                centroids = spectral_centriod(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(np.array(centroids), 'CENTRIODS'))
                
                # Long Term Average Spectrum
                LTAS, _ = ltas(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(LTAS, 'LTAS'))
                
                # Alpha ratio
                ALPHA_RATIO = alpha_ratio(data, fs, window_length_ms, window_step_ms, (0, 1000), (1000, 5000))
                features['ALPHA_RATIO'] = ALPHA_RATIO
                
                # Log Mel Spectrogram
                LOG_MEL_SPECTROGRAM = log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands=8, fmin=20, fmax=6500)
                features.update(self.process_matrix(LOG_MEL_SPECTROGRAM, 'LOG_MEL_SPECTROGRAM'))
                
                # MFCCs
                MFCC = mfcc(data, fs, window_length_ms, window_step_ms, melbands=26, lifter=20)[:15]
                features.update(self.process_matrix(MFCC, 'MFCC'))
                
                # Linear Prediction Coefficients
                LPC = lpc(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_matrix(LPC, 'LPC'))
                
                # Linear Prediction Cepstral Coefficients
                LPCC = lpcc(data, fs, window_length_ms, window_step_ms, lpc_length=8)
                features.update(self.process_matrix(LPCC, 'LPCC'))
                
                # Spectral Envelope
                ENVELOPE = spectral_envelope(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_matrix(ENVELOPE, 'ENVELOPE'))
                
                # Cepstral Peak Prominence
                CPP = calculate_cpp(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(CPP, 'CPP'))
                
                # Hammarberg Index
                HAMM_INDEX, _, _ = hammIndex(data, fs)
                features.update(self.process_row(HAMM_INDEX, 'HAMMARBERG_INDEX'))
                
                # Perceptual Linear Prediction
                PLP = plp_features(data, fs, num_filters=26, fmin=20, fmax=8000)
                features.update(self.process_matrix(PLP, 'PLP'))
                
                # Line Spectral Pairs
                lspFreq = calculate_lsp_freqs_for_frames(data, fs, window_length_ms, window_step_ms, order=8)
                features.update(self.process_matrix(lspFreq, "LSP"))
                
                # Zero Crossing Rate
                ZCR = calculate_frame_wise_zcr(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(ZCR, "ZCR"))
                
                self.logger.info("Extracted %d spectral features", len(features))
            except Exception as e:
                self.logger.error(f"Error processing spectral features: {e}")
                
            return features
        except Exception as e:
            self.logger.error("Runtime error extracting spectral features: %s", str(e))
            return {}
    
    def extract_complexity_features(self, audio_path):
        """Extract complexity features from audio"""
        self.logger.debug("Extracting complexity features from: %s", audio_path)
        try:
            # Load audio file
            data, fs = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Define window parameters
            window_length_ms = 50
            window_step_ms = 25
            
            features = {}
            
            # Complexity features extraction
            try:
                # Higuchi Fractal Dimension
                HFD = calculate_hfd_per_frame(data, fs, window_length_ms, window_step_ms, 10)
                features.update(self.process_row(HFD, 'HFD'))
                
                # Frequency Entropy
                FREQ_ENTROPY = calculate_frequency_entropy(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(FREQ_ENTROPY, 'FREQ_ENTROPY'))
                
                # Amplitude Entropy
                AMP_ENTROPY = calculate_amplitude_entropy(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(AMP_ENTROPY, 'AMP_ENTROPY'))
                
                # Zero Crossing Rate can also be considered a complexity feature
                ZCR = calculate_frame_wise_zcr(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(ZCR, "ZCR"))
                
                self.logger.info("Extracted %d complexity features", len(features))
            except Exception as e:
                self.logger.error(f"Error processing complexity features: {e}")
                
            return features
        except Exception as e:
            self.logger.error("Runtime error extracting complexity features: %s", str(e))
            return {}
    
    def extract_frequency_features(self, audio_path):
        """Extract frequency features from audio"""
        self.logger.debug("Extracting frequency features from: %s", audio_path)
        try:
            # Load audio file
            data, fs = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Define window parameters
            window_length_ms = 50
            window_step_ms = 25
            
            features = {}
            
            # Frequency parameters extraction
            try:
                # Fundamental Frequency (F0)
                F0 = get_pitch(data, fs, window_length_ms, window_step_ms)
                F0_valid = F0[~np.isnan(F0)]
                features.update(self.process_row(F0_valid, 'F0'))
                
                # Jitter
                jitter = calculate_time_varying_jitter(F0, fs, window_length_ms, window_step_ms, window_length_ms*2, window_step_ms*2)
                features.update(self.process_row(np.array(jitter), 'Jitter'))
                
                # Formants (F1, F2, F3)
                F_formants = get_formants_frame_based(data, fs, window_length_ms, window_step_ms, [1, 2, 3])
                # Handle the case where F_formants is returned as a tuple
                if isinstance(F_formants, tuple):
                    # Extract the formant data from the tuple (first element is usually the formant data)
                    formant_data = np.array(F_formants[0])
                else:
                    formant_data = F_formants
                
                # Process each formant
                for i in range(formant_data.shape[0]):
                    features.update(self.process_row(formant_data[i, :], f'F{i+1}'))
                
                # Harmonicity can also be considered a frequency feature
                HARMONICITY = harmonicity(data, fs)
                features['HARMONICITY'] = HARMONICITY
                
                self.logger.info("Extracted %d frequency features", len(features))
            except Exception as e:
                self.logger.error(f"Error processing frequency features: {e}")
                
            return features
        except Exception as e:
            self.logger.error("Runtime error extracting frequency features: %s", str(e))
            return {}
    
    def extract_intensity_features(self, audio_path):
        """Extract intensity and loudness features from audio"""
        self.logger.debug("Extracting intensity features from: %s", audio_path)
        try:
            # Load audio file
            data, fs = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Define window parameters
            window_length_ms = 50
            window_step_ms = 25
            
            features = {}
            
            # Loudness and intensity parameters extraction
            try:
                # Root Mean Square amplitude
                RMS = rms_amplitude(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(RMS, 'RMS'))
                
                # Sound Pressure Level
                SPL = spl_per_frame(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(SPL, 'SPL'))
                
                # Peak amplitude
                PEAK = peak_amplitude(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(PEAK, 'PEAK'))
                
                # Short-time energy
                STE = short_time_energy(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(STE, 'STE'))
                
                # Intensity
                INTENSITY = intensity(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(INTENSITY, 'INTENSITY'))
                
                # Amplitude range
                APQ_range, APQ_std = amplitude_range(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(APQ_range, 'Amplitude_Range'))
                features.update(self.process_row(APQ_std, 'APQ2'))
                
                self.logger.info("Extracted %d intensity features", len(features))
            except Exception as e:
                self.logger.error(f"Error processing intensity features: {e}")
                
            return features
        except Exception as e:
            self.logger.error("Runtime error extracting intensity features: %s", str(e))
            return {}
    
    def extract_rhythmic_features(self, audio_path):
        """Extract rhythmic features from audio"""
        self.logger.debug("Extracting rhythmic features from: %s", audio_path)
        
        # Check if required models are available
        if None in (self.vad_model, self.vad_utils, self.transcription_model):
            self.logger.warning("VAD and transcription models are required for rhythmic features but not set")
            return {}
            
        try:
            features = {}
            
            # Create PauseBehavior instance for rhythmic analysis
            try:
                pause_behavior = PauseBehavior(self.vad_model, self.vad_utils, self.transcription_model)
                pause_behavior.configure(audio_path)
                
                # List of methods to extract from pause_behavior for rhythmic features
                rhythmic_methods = [
                    'syllable_rate', 'speech_rate', 'articulation_rate',
                    'mean_pause_duration', 'mean_silence_duration', 'mean_speech_duration',
                    'speech_to_pause_ratio', 'percentage_silence', 'percentage_voice'
                ]
                
                # Extract relevant rhythmic features
                for name in rhythmic_methods:
                    method = getattr(pause_behavior, name, None)
                    if method and callable(method):
                        try:
                            features[name] = method()
                        except Exception as e:
                            self.logger.error(f"Error executing PauseBehavior method {name}: {e}")
                
                self.logger.info("Extracted %d rhythmic features", len(features))
            except Exception as e:
                self.logger.error(f"Error in rhythmic feature extraction: {e}")
                
            return features
        except Exception as e:
            self.logger.error("Runtime error extracting rhythmic features: %s", str(e))
            return {}
    
    def extract_fluency_features(self, audio_path):
        """Extract speech fluency features from audio"""
        self.logger.debug("Extracting fluency features from: %s", audio_path)
        
        # Check if required models are available
        if None in (self.vad_model, self.vad_utils, self.transcription_model):
            self.logger.warning("VAD and transcription models are required for fluency features but not set")
            return {}
            
        try:
            features = {}
            
            # Create SpeechBehavior instance for fluency analysis
            try:
                # First, set up PauseBehavior to get speech segmentation
                pause_behavior = PauseBehavior(self.vad_model, self.vad_utils, self.transcription_model)
                pause_behavior.configure(audio_path)
                
                # Then, set up SpeechBehavior using data from PauseBehavior
                speech_behavior = SpeechBehavior(self.vad_model, self.vad_utils, self.transcription_model)
                
                # Copy data from pause_behavior to speech_behavior
                speech_behavior.data = pause_behavior.data
                speech_behavior.silence_ranges = pause_behavior.silence_ranges
                speech_behavior.speech_ranges = pause_behavior.speech_ranges
                speech_behavior.transcription_result = pause_behavior.transcription_result
                speech_behavior.text = pause_behavior.text
                
                # Perform phoneme alignment
                speech_behavior.phoneme_alignment(audio_path)
                
                # List of methods to extract from speech_behavior for fluency features
                fluency_methods = [
                    'phonation_rate', 'phonation_time', 'articulation_time', 
                    'mean_duration_of_bursts', 'number_of_pauses', 'number_of_filled_pauses',
                    'filled_pauses_per_min', 'mean_length_of_runs', 'hesitation_ratio'
                ]
                
                # Extract relevant fluency features
                for name in fluency_methods:
                    method = getattr(speech_behavior, name, None)
                    if method and callable(method):
                        try:
                            features[name] = method()
                        except Exception as e:
                            self.logger.error(f"Error executing SpeechBehavior method {name}: {e}")
                
                # Process regularity and PVI features
                try:
                    for i, res in enumerate(speech_behavior.regularity_of_segments()):
                        features[f"regularity_{i}"] = res
                    for i, res in enumerate(speech_behavior.alternating_regularity()):
                        features[f"PVI_{i}"] = res
                except Exception as e_reg:
                    self.logger.error(f"Error processing regularity/PVI features: {e_reg}")
                
                # Process relative sentence duration
                try:
                    relative_sentence_duration = speech_behavior.relative_sentence_duration()
                    features.update(self.process_row(np.array(relative_sentence_duration), "relative_sentence_duration"))
                except Exception as e_rel:
                    self.logger.error(f"Error processing relative sentence duration: {e_rel}")
                
                self.logger.info("Extracted %d fluency features", len(features))
            except Exception as e:
                self.logger.error(f"Error in fluency feature extraction: {e}")
                
            return features
        except Exception as e:
            self.logger.error("Runtime error extracting fluency features: %s", str(e))
            return {}
    
    def extract_voice_quality_features(self, audio_path):
        """Extract voice quality features from audio"""
        self.logger.debug("Extracting voice quality features from: %s", audio_path)
        try:
            # Load audio file
            data, fs = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Define window parameters
            window_length_ms = 50
            window_step_ms = 25
            
            features = {}
            
            # Voice quality features extraction
            try:
                # Shimmer
                SHIMMER = analyze_audio_shimmer(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(SHIMMER, 'SHIMMER'))
                
                # Harmonics-to-Noise Ratio and Noise-to-Harmonics Ratio
                HNR, NHR = calculate_frame_level_hnr(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(HNR, 'HNR'))
                features.update(self.process_row(NHR, 'NHR'))
                
                # Harmonicity
                HARMONICITY = harmonicity(data, fs)
                features['HARMONICITY'] = HARMONICITY
                
                # Cepstral Peak Prominence (also a voice quality feature)
                CPP = calculate_cpp(data, fs, window_length_ms, window_step_ms)
                features.update(self.process_row(CPP, 'CPP'))
                
                # Hammarberg Index (spectral tilt related to voice quality)
                HAMM_INDEX, _, _ = hammIndex(data, fs)
                features.update(self.process_row(HAMM_INDEX, 'HAMMARBERG_INDEX'))
                
                # Alpha Ratio (spectral tilt related to voice quality)
                ALPHA_RATIO = alpha_ratio(data, fs, window_length_ms, window_step_ms, (0, 1000), (1000, 5000))
                features['ALPHA_RATIO'] = ALPHA_RATIO
                
                # Jitter (also a voice quality feature)
                F0 = get_pitch(data, fs, window_length_ms, window_step_ms)
                jitter = calculate_time_varying_jitter(F0, fs, window_length_ms, window_step_ms, window_length_ms*2, window_step_ms*2)
                features.update(self.process_row(np.array(jitter), 'Jitter'))
                
                self.logger.info("Extracted %d voice quality features", len(features))
            except Exception as e:
                self.logger.error(f"Error processing voice quality features: {e}")
                
            return features
        except Exception as e:
            self.logger.error("Runtime error extracting voice quality features: %s", str(e))
            return {}
    
    def process_row(self, row, feature_name, index=-1):
        """
        Process a single row of data with statistical functions
        
        Parameters:
        -----------
        row : numpy.ndarray
            Row of data to process
        feature_name : str
            Name of the feature
        index : int, optional
            Index for matrix features, -1 for non-matrix features
            
        Returns:
        --------
        dict
            Dictionary of processed features
        """
        # Import all statistical functions individually
        from .acoustic_features.statistical_functions import (
            max, min, span, maxPos, minPos, amean, linregc1, linregc2, 
            linregerrA, linregerrQ, stddev, skewness, kurtosis, 
            quartile1, quartile2, quartile3, iqr1_2, iqr2_3, iqr1_3,
            percentile1, percentile99, pctlrange0_1, upleveltime75, upleveltime90
        )
        
        # Create a dictionary mapping function names to the actual function
        stat_funcs = {
            "max": max, "min": min, "span": span, "maxPos": maxPos, "minPos": minPos,
            "amean": amean, "linregc1": linregc1, "linregc2": linregc2,
            "linregerrA": linregerrA, "linregerrQ": linregerrQ, "stddev": stddev,
            "skewness": skewness, "kurtosis": kurtosis, "quartile1": quartile1,
            "quartile2": quartile2, "quartile3": quartile3, "iqr1_2": iqr1_2,
            "iqr2_3": iqr2_3, "iqr1_3": iqr1_3, "percentile1": percentile1,
            "percentile99": percentile99, "pctlrange0_1": pctlrange0_1,
            "upleveltime75": upleveltime75, "upleveltime90": upleveltime90
        }
        
        row_sma = sma(row)
        results = {}
        
        # Apply statistical functions to smoothed signal
        for func_name in function_names:
            # Get the function from our mapping instead of globals()
            func = stat_funcs.get(func_name)
            if func is None:
                self.logger.warning(f"Statistical function {func_name} not found in stat_funcs mapping.")
                continue
            try:
                result = func(row_sma)
            except Exception as e:
                self.logger.error(f"Error applying {func_name} to {feature_name}: {e}")
                result = None
                
            if index >= 0:
                name = f"{feature_name}_sma[{index}]_{func_name}"
            else:
                name = f"{feature_name}_sma_{func_name}"

            results[name] = result

        # Apply statistical functions to derivative of smoothed signal
        row_sma_de = de(row_sma)
        for func_name in function_names:
            func = stat_funcs.get(func_name)
            if func is None:
                self.logger.warning(f"Statistical function {func_name} not found in stat_funcs mapping.")
                continue
            try:
                result = func(row_sma_de)
            except Exception as e:
                self.logger.error(f"Error applying {func_name} to derivative of {feature_name}: {e}")
                result = None
                
            if index >= 0:
                name = f"{feature_name}_sma_de[{index}]_{func_name}"
            else:
                name = f"{feature_name}_sma_de_{func_name}"

            results[name] = result

        return results
    
    def process_matrix(self, matrix, feature_name):
        """
        Process a matrix of features with statistical functions
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            Matrix to process
        feature_name : str
            Name of the feature
            
        Returns:
        --------
        dict
            Dictionary of processed features
        """
        matrix_results = {}
        for i, row in enumerate(matrix):
            result = self.process_row(row, feature_name, i)
            matrix_results.update(result)

        return matrix_results
    
    def length(self, res):
        """
        Count the total number of features in a list of dictionaries
        
        Parameters:
        -----------
        res : list
            List of dictionaries
            
        Returns:
        --------
        int
            Total number of features
        """
        return sum([len(elm) for elm in res])
    
    def process_file(self, filepath):
        """
        Process a file and extract all raw acoustic features
        
        Parameters:
        -----------
        filepath : str
            Path to the audio file
            
        Returns:
        --------
        dict
            Dictionary of all extracted features
        """
        self.logger.info("Processing file for raw feature extraction: %s", filepath)
        
        try:
            data, fs = librosa.load(filepath, sr=self.sampling_rate)
        except Exception as e:
            self.logger.error(f"Failed to load audio file {filepath}: {e}")
            return {}
        
        # Define window parameters
        window_length_ms = 50
        window_step_ms = 25
        
        # --- Feature Calculation with Error Handling ---
        features = {}
        
        # Frequency parameters
        try:
            F0 = get_pitch(data, fs, window_length_ms, window_step_ms)
            F0_valid = F0[~np.isnan(F0)]
            features.update(self.process_row(F0_valid, 'F0'))
            
            jitter = calculate_time_varying_jitter(F0, fs, window_length_ms, window_step_ms, window_length_ms*2, window_step_ms*2)
            features.update(self.process_row(np.array(jitter), 'Jitter'))
            
            F_formants = get_formants_frame_based(data, fs, window_length_ms, window_step_ms, [1, 2, 3])
            # Handle the case where F_formants is returned as a tuple
            if isinstance(F_formants, tuple):
                # Extract the formant data from the tuple (first element is usually the formant data)
                formant_data = np.array(F_formants[0])
            else:
                formant_data = F_formants
                
            # Process each formant
            for i in range(formant_data.shape[0]):
                features.update(self.process_row(formant_data[i, :], f'F{i+1}'))
            self.logger.info("Processed frequency parameters")
        except Exception as e:
            self.logger.error(f"Error processing frequency parameters: {e}")
            
        # Spectral features
        try:
            msc = compute_msc(data, fs, nfft=512, window_length_ms=window_length_ms, window_step_ms=window_step_ms, num_msc=13)
            features.update(self.process_matrix(msc, 'MSC'))
            
            centroids = spectral_centriod(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(np.array(centroids), 'CENTRIODS'))
            
            LTAS, _ = ltas(data, fs, window_length_ms, window_step_ms) # Ignore freq array return
            features.update(self.process_row(LTAS, 'LTAS'))
            
            ALPHA_RATIO = alpha_ratio(data, fs, window_length_ms, window_step_ms, (0, 1000), (1000, 5000))
            # Alpha ratio is often scalar, handle differently if needed, here adding directly
            features['ALPHA_RATIO'] = ALPHA_RATIO 

            LOG_MEL_SPECTROGRAM = log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands=8, fmin=20, fmax=6500)
            features.update(self.process_matrix(LOG_MEL_SPECTROGRAM, 'LOG_MEL_SPECTROGRAM'))
            
            MFCC = mfcc(data, fs, window_length_ms, window_step_ms, melbands=26, lifter=20)[:15]
            features.update(self.process_matrix(MFCC, 'MFCC'))
            
            LPC = lpc(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_matrix(LPC, 'LPC'))
            
            LPCC = lpcc(data, fs, window_length_ms, window_step_ms, lpc_length=8)
            features.update(self.process_matrix(LPCC, 'LPCC'))
            
            ENVELOPE = spectral_envelope(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_matrix(ENVELOPE, 'ENVELOPE'))
            
            CPP = calculate_cpp(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(CPP, 'CPP'))
            
            HAMM_INDEX, _, _ = hammIndex(data, fs) # Ignore freq and Pxx return
            features.update(self.process_row(HAMM_INDEX, 'HAMMARBERG_INDEX'))
            
            PLP = plp_features(data, fs, num_filters=26, fmin=20, fmax=8000)
            features.update(self.process_matrix(PLP, 'PLP'))
            
            HARMONICITY = harmonicity(data, fs)
            features['HARMONICITY'] = HARMONICITY # Often scalar
            
            lspFreq = calculate_lsp_freqs_for_frames(data, fs, window_length_ms, window_step_ms, order=8)
            features.update(self.process_matrix(lspFreq, "LSP"))
            
            ZCR = calculate_frame_wise_zcr(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(ZCR, "ZCR"))
            self.logger.info("Processed spectral domain features")
        except Exception as e:
            self.logger.error(f"Error processing spectral features: {e}")

        # Voice quality
        try:
            SHIMMER = analyze_audio_shimmer(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(SHIMMER, 'SHIMMER'))
            
            HNR, NHR = calculate_frame_level_hnr(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(HNR, 'HNR'))
            features.update(self.process_row(NHR, 'NHR'))
            
            # Amplitude range returns range and std deviation
            APQ_range, APQ_std = amplitude_range(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(APQ_range, 'Amplitude_Range')) # Renamed from APQ_range
            features.update(self.process_row(APQ_std, 'APQ2')) # Renamed from APQ_std
            self.logger.info("Processed voice quality features")
        except Exception as e:
            self.logger.error(f"Error processing voice quality features: {e}")

        # Loudness and intensity
        try:
            RMS = rms_amplitude(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(RMS, 'RMS'))
            
            SPL = spl_per_frame(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(SPL, 'SPL'))
            
            PEAK = peak_amplitude(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(PEAK, 'PEAK'))
            
            STE = short_time_energy(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(STE, 'STE'))
            
            INTENSITY = intensity(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(INTENSITY, 'INTENSITY'))
            self.logger.info("Processed loudness and intensity parameters")
        except Exception as e:
            self.logger.error(f"Error processing loudness and intensity features: {e}")

        # Complexity
        try:
            HFD = calculate_hfd_per_frame(data, fs, window_length_ms, window_step_ms, 10)
            features.update(self.process_row(HFD, 'HFD'))
            
            FREQ_ENTROPY = calculate_frequency_entropy(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(FREQ_ENTROPY, 'FREQ_ENTROPY'))
            
            AMP_ENTROPY = calculate_amplitude_entropy(data, fs, window_length_ms, window_step_ms)
            features.update(self.process_row(AMP_ENTROPY, 'AMP_ENTROPY'))
            self.logger.info("Processed complexity features")
        except Exception as e:
            self.logger.error(f"Error processing complexity features: {e}")

        self.logger.info("Extracted a total of %d raw features", len(features))
        return features
    
    def process_file_model(self, filepath):
        """
        Process a file using VAD and transcription models
        
        Parameters:
        -----------
        filepath : str
            Path to the audio file
            
        Returns:
        --------
        dict
            Dictionary of all extracted features
            
        Raises:
        -------
        ValueError
            If models are not set
        ImportError
            If required dependencies for fluency/rhythmic features are missing
        """
        if None in (self.vad_model, self.vad_utils, self.transcription_model):
            raise ValueError("VAD and transcription models must be set using set_models() before calling this method")
            # Raise the specific import error if possible, otherwise a generic one
        try:
            # Attempting to instantiate will raise the specific error if placeholder is used
            PauseBehavior(None, None, None)
            SpeechBehavior(None, None, None)
        except Exception:
            raise ImportError("Required dependencies for fluency/rhythmic features are missing.")

        self.logger.info("Processing file with VAD and transcription models: %s", filepath)
        
        p_results = {}
        s_results = {}
        prob = {}

        try:
            # Process rhythmic features
            pause_behavior = PauseBehavior(self.vad_model, self.vad_utils, self.transcription_model)
            pause_behavior.configure(filepath)
            
            voiceProb_signal = self.vad_model.audio_forward(pause_behavior.data, sr=16000)
            voiceProb_signal = np.array(voiceProb_signal[0])
            prob = self.process_row(voiceProb_signal, "voiceProb")
            
            # List of methods to exclude
            excluded_methods_p = ['__init__', 'configure']
            
            # Iterate through all methods of the pause_behavior instance
            self.logger.debug("Extracting pause behavior features")
            for name, method in inspect.getmembers(pause_behavior, predicate=inspect.ismethod):
                if name not in excluded_methods_p:
                    try:
                        p_results[name] = method()
                    except Exception as e_method:
                        self.logger.error(f"Error executing PauseBehavior method {name}: {e_method}")
            
            # Process speech features
            speech_behavior = SpeechBehavior(self.vad_model, self.vad_utils, self.transcription_model)
            
            # Copy data from pause_behavior to speech_behavior
            speech_behavior.data = pause_behavior.data
            speech_behavior.silence_ranges = pause_behavior.silence_ranges
            speech_behavior.speech_ranges = pause_behavior.speech_ranges
            speech_behavior.transcription_result = pause_behavior.transcription_result
            speech_behavior.text = pause_behavior.text
            
            # Perform phoneme alignment
            alignment_error = speech_behavior.phoneme_alignment(filepath)
            self.logger.info("Phoneme alignment completed (total segments: %d, hypothetical: %d)", 
                             alignment_error[0], alignment_error[1])
            
            # List of methods to exclude
            excluded_methods_s = ['__init__', 'configure', 'phoneme_alignment', 'relative_sentence_duration', 'regularity_of_segments', 'alternating_regularity']
            
            # Iterate through all methods of the speech_behavior instance
            self.logger.debug("Extracting speech behavior features")
            for name, method in inspect.getmembers(speech_behavior, predicate=inspect.ismethod):
                if name not in excluded_methods_s:
                    try:
                        s_results[name] = method()
                    except Exception as e_method:
                        self.logger.error(f"Error executing SpeechBehavior method {name}: {e_method}")
            
            # Process regularity and PVI features
            self.logger.debug("Processing regularity and PVI features")
            try:
                for i, res in enumerate(speech_behavior.regularity_of_segments()):
                    s_results[f"regularity_{i}"] = res
                for i, res in enumerate(speech_behavior.alternating_regularity()):
                    s_results[f"PVI_{i}"] = res
            except Exception as e_reg:
                self.logger.error(f"Error processing regularity/PVI features: {e_reg}")
            
            # Process relative sentence duration
            try:
                relative_sentence_duration = speech_behavior.relative_sentence_duration()
                s_results.update(self.process_row(np.array(relative_sentence_duration), "relative_sentence_duration"))
            except Exception as e_rel:
                self.logger.error(f"Error processing relative sentence duration: {e_rel}")

        except Exception as e_main:
            self.logger.critical(f"Major error during model-based processing: {e_main}")
            # Depending on severity, you might want to return partial results or an empty dict
            # return {} 

        # Merge all results
        final_results = {**p_results, **prob, **s_results}
        self.logger.debug("Feature counts - pause: %d, prob: %d, speech: %d", len(p_results), len(prob), len(s_results))
        
        self.logger.info("Extracted a total of %d features with VAD and transcription models", len(final_results))
        return final_results
    
    def extract_features(self, audio_paths: Union[str, List[str], 'torch.utils.data.DataLoader'], features_to_calculate: Optional[List[str]] = None, separate_groups: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract specified features from a single audio file, a batch of files, or a PyTorch DataLoader
        
        Parameters:
        -----------
        audio_paths : str or List[str] or torch.utils.data.DataLoader
            Path(s) to the audio file(s) or a PyTorch DataLoader that yields:
            - File paths as strings
            - Tuples/lists where the first element is a file path
            - Audio tensors with shape (batch_size, num_samples) or (num_samples,)
        features_to_calculate : List[str], optional
            List of feature types to extract. If None, extracts all features.
            Valid options: 'spectral', 'complexity', 'frequency', 'intensity', 
                          'rhythmic', 'fluency', 'voice_quality', 'all',
                          'raw', 'transcription'
        separate_groups : bool, optional
            If True, organizes output by feature groups. If False (default), returns a flat dictionary.
        output_dir : str, optional
            If provided, extracted features for each file will be saved as a CSV file in this directory
            with the same name as the audio file.
            
        Returns:
        --------
        dict
            Dictionary containing extracted features
            If separate_groups=False (default):
                For a single file: {feature_name: feature_value, ...}
                For multiple files/DataLoader: {file_path/idx: {feature_name: feature_value, ...}, ...}
            If separate_groups=True:
                For a single file: {feature_group: {feature_name: feature_value, ...}, ...}
                For multiple files/DataLoader: {file_path/idx: {feature_group: {feature_name: feature_value, ...}, ...}, ...}
        """
        # Default to all features if none specified
        if features_to_calculate is None:
            features_to_calculate = ['all']
        
        self.logger.info("Extracting features: %s", ", ".join(features_to_calculate))
        
        # Check if output directory exists and create it if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Features will be saved to directory: {output_dir}")
        
        # Check if feature types are valid and available
        requested_unavailable = []
        for feature_type in features_to_calculate:
            if feature_type not in self.available_features:
                raise ValueError(f"Unknown feature type: {feature_type}. Available types: {list(self.available_features.keys())}")
                    
        if requested_unavailable:
            self.logger.warning(
                "The following requested feature types are unavailable due to missing dependencies and will be skipped: %s",
                ", ".join(requested_unavailable)
            )

        # Check if input is a PyTorch DataLoader
        try:
            import torch
            from torch.utils.data import DataLoader
            is_dataloader = isinstance(audio_paths, DataLoader)
        except ImportError:
            is_dataloader = False
        
        # Process a PyTorch DataLoader
        if is_dataloader:
            self.logger.info("Processing PyTorch DataLoader")
            batch_results = {}
            
            # Process each batch from the DataLoader
            for batch_idx, batch in enumerate(audio_paths):
                self.logger.info(f"Processing batch {batch_idx+1}")
                
                # Handle different batch formats:
                # 1. Batch of file paths (strings)
                # 2. Batch of tuples where first element is file path
                # 3. Batch of audio tensors
                
                if isinstance(batch, torch.Tensor):
                    # Batch is a tensor containing audio samples
                    if len(batch.shape) == 1:  # Single audio sample
                        self.logger.debug("Processing single audio tensor")
                        # Convert tensor to numpy array
                        audio_data = batch.cpu().numpy()
                        # Process single audio array
                        file_result = self._process_audio_data(
                            audio_data, 
                            features_to_calculate, 
                            f"tensor_batch{batch_idx}"
                        )
                        key = f"batch{batch_idx}_item0"
                        
                        if separate_groups and file_result:
                            self.logger.info("Extracted %d features total for %s", len(file_result), key)
                            grouped_file_result = self._organize_features_by_group(file_result)
                            batch_results[key] = grouped_file_result
                        else:
                            batch_results[key] = file_result
                            
                        # Save features to CSV if output_dir is provided
                        if output_dir is not None and file_result:
                            csv_path = os.path.join(output_dir, f"{key}.csv")
                            self._save_features_to_csv(file_result, csv_path)
                    
                    else:  # Batch of audio samples
                        self.logger.debug(f"Processing batch of {batch.shape[0]} audio tensors")
                        for i in range(batch.shape[0]):
                            audio_data = batch[i].cpu().numpy()
                            file_result = self._process_audio_data(
                                audio_data, 
                                features_to_calculate,
                                f"tensor_batch{batch_idx}_item{i}"
                            )
                            key = f"batch{batch_idx}_item{i}"
                            
                            if separate_groups and file_result:
                                self.logger.info("Extracted %d features total for %s", len(file_result), key)
                                grouped_file_result = self._organize_features_by_group(file_result)
                                batch_results[key] = grouped_file_result
                            else:
                                batch_results[key] = file_result
                                
                            # Save features to CSV if output_dir is provided
                            if output_dir is not None and file_result:
                                csv_path = os.path.join(output_dir, f"{key}.csv")
                                self._save_features_to_csv(file_result, csv_path)
                
                elif isinstance(batch, (list, tuple)):
                    # Batch is a list or tuple - check first element
                    if len(batch) > 0 and isinstance(batch[0], str):
                        # Batch contains file paths
                        for i, item in enumerate(batch):
                            file_result = self._process_single_file(item, features_to_calculate)
                            
                            if separate_groups and file_result:
                                self.logger.info("Extracted %d features total for %s", len(file_result), item)
                                grouped_file_result = self._organize_features_by_group(file_result)
                                batch_results[item] = grouped_file_result
                            else:
                                batch_results[item] = file_result
                                
                            # Save features to CSV if output_dir is provided
                            if output_dir is not None and file_result:
                                file_name = os.path.splitext(os.path.basename(item))[0]
                                csv_path = os.path.join(output_dir, f"{file_name}.csv")
                                self._save_features_to_csv(file_result, csv_path)
                    
                    elif len(batch) > 0 and isinstance(batch[0], (list, tuple)) and len(batch[0]) > 0 and isinstance(batch[0][0], str):
                        # Batch contains tuples/lists where first element is a file path
                        for i, item in enumerate(batch):
                            file_path = item[0]  # Get the file path (first element)
                            file_result = self._process_single_file(file_path, features_to_calculate)
                            
                            if separate_groups and file_result:
                                self.logger.info("Extracted %d features total for %s", len(file_result), file_path)
                                grouped_file_result = self._organize_features_by_group(file_result)
                                batch_results[file_path] = grouped_file_result
                            else:
                                batch_results[file_path] = file_result
                                
                            # Save features to CSV if output_dir is provided
                            if output_dir is not None and file_result:
                                file_name = os.path.splitext(os.path.basename(file_path))[0]
                                csv_path = os.path.join(output_dir, f"{file_name}.csv")
                                self._save_features_to_csv(file_result, csv_path)
                    
                    else:
                        self.logger.warning(f"Unsupported batch format. Skipping batch {batch_idx}")
                        continue
                
                else:
                    self.logger.warning(f"Unsupported batch item type: {type(batch)}. Skipping batch {batch_idx}")
                    continue
            
            self.logger.info("DataLoader processing complete. Processed %d items total.", len(batch_results))
            return batch_results

        # Process a single file
        elif isinstance(audio_paths, str):
            result = self._process_single_file(audio_paths, features_to_calculate, separate_groups)
            
            # Save features to CSV if output_dir is provided
            if output_dir is not None and result:
                file_name = os.path.splitext(os.path.basename(audio_paths))[0]
                csv_path = os.path.join(output_dir, f"{file_name}.csv")
                
                # For single file, the result might be already grouped
                if separate_groups:
                    # Flatten the grouped features for CSV output
                    flattened_features = self._flatten_grouped_features(result)
                    self._save_features_to_csv(flattened_features, csv_path)
                else:
                    self._save_features_to_csv(result, csv_path)
            
            return result
        
        # Process batch of files
        else:
            self.logger.info("Processing batch of %d files", len(audio_paths))
            batch_results = {}
            
            for i, audio_path in enumerate(audio_paths):
                self.logger.info("Processing file %d/%d: %s", i+1, len(audio_paths), audio_path)
                file_result = self._process_single_file(audio_path, features_to_calculate)
                
                # If separate_groups is True, organize features by group for each file
                if separate_groups and file_result:
                    self.logger.info("Extracted %d features total for %s", len(file_result), audio_path)
                    grouped_file_result = self._organize_features_by_group(file_result)
                    batch_results[audio_path] = grouped_file_result
                else:
                    batch_results[audio_path] = file_result
                
                # Save features to CSV if output_dir is provided
                if output_dir is not None and file_result:
                    file_name = os.path.splitext(os.path.basename(audio_path))[0]
                    csv_path = os.path.join(output_dir, f"{file_name}.csv")
                    self._save_features_to_csv(file_result, csv_path)
            
            self.logger.info("Batch processing complete")
            return batch_results
    
    def _process_single_file(self, audio_path, features_to_calculate, separate_groups=False):
        """
        Process a single audio file and extract features
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        features_to_calculate : List[str]
            List of feature types to extract
        separate_groups : bool
            If True, organizes output by feature groups
            
        Returns:
        --------
        dict
            Dictionary of extracted features
        """
        self.logger.info("Processing single file: %s", audio_path)
        result = {}
        
        for feature_type in features_to_calculate:
            # If 'all' is included, extract all available features and skip other types
            if feature_type == 'all':
                result = self.extract_all_features(audio_path)
                break
            
            # Extract each requested feature type
            # Add try-except around the call for robustness
            try:
                feature_extractor = self.available_features[feature_type]
                features = feature_extractor(audio_path)
                result.update(features)
            except Exception as e:
                self.logger.error(f"Error during extraction of '{feature_type}' features for {audio_path}: {e}")
        
        # If separate_groups is True, organize features by group
        if separate_groups and result:
            self.logger.info("Extracted %d features total for %s", len(result), audio_path)
            grouped_result = self._organize_features_by_group(result)
            self.logger.info("Extracted features organized into %d groups for %s", len(grouped_result), audio_path)
            return grouped_result
        
        self.logger.info("Extracted %d features total for %s", len(result), audio_path)
        return result
    
    def _process_audio_data(self, audio_data, features_to_calculate, identifier="audio_tensor"):
        """
        Process audio data directly from a numpy array and extract features
        
        Parameters:
        -----------
        audio_data : numpy.ndarray
            Audio samples as a numpy array
        features_to_calculate : List[str]
            List of feature types to extract
        identifier : str
            Identifier for logging purposes
            
        Returns:
        --------
        dict
            Dictionary of extracted features
        """
        self.logger.info("Processing audio data: %s", identifier)
        result = {}
        
        # Create a function that processes audio data for each feature type
        def process_audio_with_feature(feature_type):
            # Skip unsupported feature types for direct audio data
            if feature_type in ['rhythmic', 'fluency', 'transcription']:
                self.logger.warning(f"Feature type '{feature_type}' not supported for direct audio data processing")
                return {}
            
            if feature_type == 'all':
                # For 'all', just call each supported feature type
                all_features = {}
                for ft in ['spectral', 'complexity', 'frequency', 'intensity', 'voice_quality']:
                    all_features.update(process_audio_with_feature(ft))
                return all_features
                
            # Extract features directly from audio data
            try:
                # Most feature extraction functions require signal and sampling rate
                fs = self.sampling_rate
                data = audio_data
                
                window_length_ms = 50
                window_step_ms = 25
                
                features = {}
                
                if feature_type == 'spectral':
                    # Extract spectral features from data
                    try:
                        # Modulation Spectrum Coefficients
                        msc = compute_msc(data, fs, nfft=512, window_length_ms=window_length_ms, window_step_ms=window_step_ms, num_msc=13)
                        features.update(self.process_matrix(msc, 'MSC'))
                        
                        # Other spectral features...
                        # (Similar to existing code in extract_spectral_features method)
                    except Exception as e:
                        self.logger.error(f"Error processing spectral features for {identifier}: {e}")
                        
                elif feature_type == 'complexity':
                    # Extract complexity features from data
                    try:
                        # Higuchi Fractal Dimension
                        HFD = calculate_hfd_per_frame(data, fs, window_length_ms, window_step_ms, 10)
                        features.update(self.process_row(HFD, 'HFD'))
                        
                        # Other complexity features...
                        # (Similar to existing code in extract_complexity_features method)
                    except Exception as e:
                        self.logger.error(f"Error processing complexity features for {identifier}: {e}")
                        
                elif feature_type == 'frequency':
                    # Extract frequency features from data
                    try:
                        # Fundamental Frequency (F0)
                        F0 = get_pitch(data, fs, window_length_ms, window_step_ms)
                        F0_valid = F0[~np.isnan(F0)]
                        features.update(self.process_row(F0_valid, 'F0'))
                        
                        # Other frequency features...
                        # (Similar to existing code in extract_frequency_features method)
                    except Exception as e:
                        self.logger.error(f"Error processing frequency features for {identifier}: {e}")
                        
                elif feature_type == 'intensity':
                    # Extract intensity features from data
                    try:
                        # Root Mean Square amplitude
                        RMS = rms_amplitude(data, fs, window_length_ms, window_step_ms)
                        features.update(self.process_row(RMS, 'RMS'))
                        
                        # Other intensity features...
                        # (Similar to existing code in extract_intensity_features method)
                    except Exception as e:
                        self.logger.error(f"Error processing intensity features for {identifier}: {e}")
                        
                elif feature_type == 'voice_quality':
                    # Extract voice quality features from data
                    try:
                        # Shimmer
                        SHIMMER = analyze_audio_shimmer(data, fs, window_length_ms, window_step_ms)
                        features.update(self.process_row(SHIMMER, 'SHIMMER'))
                        
                        # Other voice quality features...
                        # (Similar to existing code in extract_voice_quality_features method)
                    except Exception as e:
                        self.logger.error(f"Error processing voice quality features for {identifier}: {e}")
                
                return features
            except Exception as e:
                self.logger.error(f"Error processing {feature_type} features from audio data: {e}")
                return {}
        
        # Process each feature type
        for feature_type in features_to_calculate:
            if feature_type == 'all':
                # Handle 'all' feature type
                all_features = process_audio_with_feature('all')
                result.update(all_features)
                break
            else:
                # Handle specific feature type
                features = process_audio_with_feature(feature_type)
                result.update(features)
        
        self.logger.info("Extracted %d features total for %s", len(result), identifier)
        return result
    
    def _organize_features_by_group(self, features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Organize features by their group
        
        Parameters:
        -----------
        features : Dict[str, Any]
            Dictionary of features
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary of features organized by group
        """
        if not features:
            return {}
            
        # Group features by their category
        feature_groups = self._group_features(list(features.keys()))
        
        # Create output dictionary organized by groups
        grouped_features = {}
        
        for group_name, feature_names in feature_groups.items():
            group_dict = {}
            for name in feature_names:
                if name in features:
                    group_dict[name] = features[name]
            
            if group_dict:  # Only include groups with features
                grouped_features[group_name] = group_dict
        
        # Check for any ungrouped features and add them to 'Other'
        all_grouped_names = [name for group in feature_groups.values() for name in group]
        ungrouped_features = {name: value for name, value in features.items() 
                             if name not in all_grouped_names}
        
        if ungrouped_features:
            if 'Other' not in grouped_features:
                grouped_features['Other'] = {}
            grouped_features['Other'].update(ungrouped_features)
            
        return grouped_features
    
    def _flatten_grouped_features(self, grouped_features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Flatten a dictionary of grouped features into a single-level dictionary
        
        Parameters:
        -----------
        grouped_features : Dict[str, Dict[str, Any]]
            Dictionary of features organized by group
            
        Returns:
        --------
        Dict[str, Any]
            Flattened dictionary of features
        """
        flattened = {}
        for group_name, group_dict in grouped_features.items():
            for feature_name, feature_value in group_dict.items():
                flattened[feature_name] = feature_value
        return flattened
    
    def _save_features_to_csv(self, features: Dict[str, Any], csv_path: str) -> None:
        """
        Save extracted features to a CSV file
        
        Parameters:
        -----------
        features : Dict[str, Any]
            Dictionary of extracted features
        csv_path : str
            Path to save the CSV file
        """
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Feature', 'Value'])
                
                for feature_name, feature_value in features.items():
                    # Handle different types of feature values
                    if isinstance(feature_value, (int, float)) or feature_value is None:
                        writer.writerow([feature_name, feature_value])
                    elif isinstance(feature_value, (list, np.ndarray)):
                        # For arrays, convert to string representation
                        if isinstance(feature_value, np.ndarray):
                            # Convert to Python list for better CSV compatibility
                            feature_list = feature_value.tolist()
                        else:
                            feature_list = feature_value
                            
                        # Limit array size in CSV to prevent excessive file sizes
                        if len(feature_list) > 100:
                            self.logger.warning(f"Feature {feature_name} has {len(feature_list)} elements, truncating to 100 for CSV output")
                            feature_list = feature_list[:100]
                            
                        # Join array elements with commas for CSV
                        writer.writerow([feature_name, str(feature_list)])
                    else:
                        # For other types, use string representation
                        writer.writerow([feature_name, str(feature_value)])
                        
            self.logger.info(f"Saved features to {csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving features to CSV {csv_path}: {e}")
    
    def _group_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Group features by their category based on name patterns
        
        Parameters:
        -----------
        feature_names : List[str]
            List of feature names to group
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary mapping category names to lists of feature names
        """
        groups = {}
        
        # Define common feature prefixes and their corresponding groups
        # More specific prefixes first
        prefix_groups = {
            'LOG_MEL_SPECTROGRAM': 'Spectral/Mel',
            'MFCC': 'Spectral/Cepstral',
            'LPCC': 'Spectral/Cepstral',
            'LPC': 'Spectral/LPC',
            'LSP': 'Spectral/LSP',
            'PLP': 'Spectral/PLP',
            'F0': 'Pitch',
            'F1': 'Formants',
            'F2': 'Formants',
            'F3': 'Formants',
            'Jitter': 'Voice Quality/Perturbation',
            'Shimmer': 'Voice Quality/Perturbation',
            'APQ': 'Voice Quality/Perturbation',
            'HNR': 'Voice Quality/Harmonicity',
            'NHR': 'Voice Quality/Harmonicity',
            'HARMONICITY': 'Voice Quality/Harmonicity',
            'CPP': 'Voice Quality/CPP',
            'HAMMARBERG_INDEX': 'Voice Quality/Spectral Tilt',
            'ALPHA_RATIO': 'Voice Quality/Spectral Tilt',
            'MSC': 'Spectral/Modulation',
            'CENTRIODS': 'Spectral/Shape',
            'LTAS': 'Spectral/Shape',
            'ENVELOPE': 'Spectral/Shape',
            'RMS': 'Intensity/Amplitude',
            'PEAK': 'Intensity/Amplitude',
            'Amplitude_Range': 'Intensity/Amplitude',
            'SPL': 'Intensity/SPL',
            'STE': 'Intensity/Energy',
            'INTENSITY': 'Intensity/Energy',
            'HFD': 'Complexity',
            'FREQ_ENTROPY': 'Complexity',
            'AMP_ENTROPY': 'Complexity',
            'ZCR': 'Complexity',
            'voiceProb': 'Speech Activity',
            'relative_sentence_duration': 'Speech Fluency/Timing',
            'regularity': 'Speech Fluency/Regularity',
            'PVI': 'Speech Fluency/Regularity'
            # Add more specific groups as needed
        }
        
        # Assign each feature to a group
        for name in feature_names:
            assigned = False
            # Check specific prefixes first
            for prefix, group in prefix_groups.items():
                # Match start or common statistical patterns
                if name.startswith(prefix) or \
                   f'_{prefix}_sma' in name or \
                   f'_{prefix}_de' in name:
                    if group not in groups:
                        groups[group] = []
                    groups[group].append(name)
                    assigned = True
                    break
            
            # Assign to broader category if no specific match
            if not assigned:
                broad_category = 'Other'
                if 'spectral' in name.lower() or 'spec' in name.lower(): broad_category = 'Spectral/Other'
                elif 'voice' in name.lower() or 'hnr' in name.lower() or 'jitter' in name.lower() or 'shimmer' in name.lower(): broad_category = 'Voice Quality/Other'
                elif 'freq' in name.lower() or 'pitch' in name.lower() or 'formant' in name.lower(): broad_category = 'Frequency/Other'
                elif 'intens' in name.lower() or 'loud' in name.lower() or 'amp' in name.lower() or 'spl' in name.lower() or 'peak' in name.lower() or 'rms' in name.lower(): broad_category = 'Intensity/Other'
                elif 'complex' in name.lower() or 'entropy' in name.lower() or 'hfd' in name.lower(): broad_category = 'Complexity/Other'
                elif 'fluency' in name.lower() or 'rhythm' in name.lower() or 'pause' in name.lower() or 'speech' in name.lower(): broad_category = 'Timing/Fluency/Other'
                
                if broad_category not in groups:
                    groups[broad_category] = []
                groups[broad_category].append(name)
        
        return groups 