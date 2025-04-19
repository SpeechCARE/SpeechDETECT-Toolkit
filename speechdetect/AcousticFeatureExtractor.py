"""
AcousticFeatureExtractor - Main interface for extracting acoustic features from speech recordings
"""
import sys
import os
import numpy as np
import inspect
import librosa
import logging
from typing import List, Dict, Union, Any, Optional, Tuple

# Import from the acoustic_features package
from acoustic_features import (
    # Voice quality features
    calculate_APQ_from_peaks, calculate_frame_based_APQ, shimmer, analyze_audio_shimmer,
    calculate_frame_level_hnr, get_voice_quality_metrics, amplitude_range,
    
    # Statistical functions
    sma, de, max, min, span, maxPos, minPos, amean, linregc1, linregc2,
    linregerrA, linregerrQ, stddev, skewness, kurtosis, quartile1, quartile2,
    quartile3, iqr1_2, iqr2_3, iqr1_3, percentile1, percentile99, pctlrange0_1,
    upleveltime75, upleveltime90,
    
    # Speech fluency features
    SpeechBehavior, calculate_duration_ms, remove_subranges,
    
    # Rhythmic structure features
    PauseBehavior,
    
    # Loudness and intensity features
    rms_amplitude, spl_per, peak_amplitude, ste_amplitude, intensity,
    
    # Frequency parameters
    get_pitch, calculate_time_varying_jitter, get_formants_frame_based,
    
    # Spectral features
    compute_msc, spectral_centriod, ltas, alpha_ratio, log_mel_spectrogram,
    mfcc, lpc, lpcc, spectral_envelope, calculate_cpp, hammIndex,
    plp_features, harmonicity, calculate_lsp_freqs_for_frames, calculate_frame_wise_zcr,
    
    # Complexity features
    calculate_hfd_per_frame, calculate_frequency_entropy, calculate_amplitude_entropy
)

# Import functions for API compatibility
from acoustic_features.cepstral_coefficients_and_spectral_features import extract_spectral_features
from acoustic_features.complexity import extract_complexity_features
from acoustic_features.frequency_parameters import extract_frequency_features
from acoustic_features.loudness_and_intensity import extract_intensity_features
from acoustic_features.rhythmic_structure import extract_rhythmic_features
from acoustic_features.speech_fluency_and_speech_production_dynamics import extract_fluency_features
from acoustic_features.voice_quality import extract_voice_quality_features

# Import function_names for statistical processing
from acoustic_features.statistical_functions import function_names


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
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)
        
        # Available feature types
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
        
        # Extract all feature sets
        features.update(self.extract_spectral_features(audio_path))
        features.update(self.extract_complexity_features(audio_path))
        features.update(self.extract_frequency_features(audio_path))
        features.update(self.extract_intensity_features(audio_path))
        features.update(self.extract_rhythmic_features(audio_path))
        features.update(self.extract_fluency_features(audio_path))
        features.update(self.extract_voice_quality_features(audio_path))
        
        self.logger.info("Extracted %d features in total", len(features))
        return features
    
    def extract_spectral_features(self, audio_path):
        """Extract spectral features from audio"""
        self.logger.debug("Extracting spectral features from: %s", audio_path)
        return extract_spectral_features(audio_path, self.sampling_rate)
    
    def extract_complexity_features(self, audio_path):
        """Extract complexity features from audio"""
        self.logger.debug("Extracting complexity features from: %s", audio_path)
        return extract_complexity_features(audio_path, self.sampling_rate)
    
    def extract_frequency_features(self, audio_path):
        """Extract frequency features from audio"""
        self.logger.debug("Extracting frequency features from: %s", audio_path)
        return extract_frequency_features(audio_path, self.sampling_rate)
    
    def extract_intensity_features(self, audio_path):
        """Extract intensity and loudness features from audio"""
        self.logger.debug("Extracting intensity features from: %s", audio_path)
        return extract_intensity_features(audio_path, self.sampling_rate)
    
    def extract_rhythmic_features(self, audio_path):
        """Extract rhythmic features from audio"""
        self.logger.debug("Extracting rhythmic features from: %s", audio_path)
        return extract_rhythmic_features(audio_path, self.sampling_rate)
    
    def extract_fluency_features(self, audio_path):
        """Extract speech fluency features from audio"""
        self.logger.debug("Extracting fluency features from: %s", audio_path)
        return extract_fluency_features(audio_path, self.sampling_rate)
    
    def extract_voice_quality_features(self, audio_path):
        """Extract voice quality features from audio"""
        self.logger.debug("Extracting voice quality features from: %s", audio_path)
        return extract_voice_quality_features(audio_path, self.sampling_rate)
    
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
        row_sma = sma(row)
        results = {}
        
        # Apply statistical functions to smoothed signal
        for func_name in function_names:
            # Use globals() to access the imported functions
            func = globals()[func_name]
            try:
                result = func(row_sma)
            except:
                result = None
                
            if index >= 0:
                name = f"{feature_name}_sma[{index}]_{func_name}"
            else:
                name = f"{feature_name}_sma_{func_name}"

            results[name] = result

        # Apply statistical functions to derivative of smoothed signal
        row_sma_de = de(row_sma)
        for func_name in function_names:
            # Use globals() to access the imported functions
            func = globals()[func_name]
            try:
                result = func(row_sma_de)
            except:
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
        
        # Load audio file
        data, fs = librosa.load(filepath, sr=self.sampling_rate)
        
        # Define window parameters
        window_length_ms = 50
        window_step_ms = 25
        
        # Process frequency parameters
        F0 = get_pitch(data, fs, window_length_ms, window_step_ms)
        F0 = F0[~np.isnan(F0)]
        jitter = calculate_time_varying_jitter(F0, fs, window_length_ms, window_step_ms, window_length_ms*2, window_step_ms*2)
        F1 = get_formants_frame_based(data, fs, window_length_ms, window_step_ms, [1, 2, 3])
        self.logger.info("Processed frequency parameters")
        
        # Process spectral features
        msc = compute_msc(data, fs, nfft=512, window_length_ms=window_length_ms, window_step_ms=window_step_ms, num_msc=13)
        centroids = spectral_centriod(data, fs, window_length_ms, window_step_ms)
        LTAS = ltas(data, fs, window_length_ms, window_step_ms)[0]
        ALPHA_RATIO = alpha_ratio(data, fs, window_length_ms, window_step_ms, (0, 1000), (1000, 5000))
        LOG_MEL_SPECTROGRAM = log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands=8, fmin=20, fmax=6500)
        MFCC = mfcc(data, fs, window_length_ms, window_step_ms, melbands=26, lifter=20)[:15]
        LPC = lpc(data, fs, window_length_ms, window_step_ms)
        LPCC = lpcc(data, fs, window_length_ms, window_step_ms, lpc_length=8)
        ENVELOPE = spectral_envelope(data, fs, window_length_ms, window_step_ms)
        CPP = calculate_cpp(data, fs, window_length_ms, window_step_ms)
        HAMM_INDEX = hammIndex(data, fs)[0]
        PLP = plp_features(data, fs, num_filters=26, fmin=20, fmax=8000)
        HARMONICITY = harmonicity(data, fs)
        lspFreq = calculate_lsp_freqs_for_frames(data, fs, window_length_ms, window_step_ms, order=8)
        ZCR = calculate_frame_wise_zcr(data, fs, window_length_ms, window_step_ms)
        self.logger.info("Processed spectral domain features")
        
        # Process voice quality
        APQ = None  # Not used in the original code
        SHIMMER = analyze_audio_shimmer(data, fs, window_length_ms, window_step_ms)
        HNR, NHR = calculate_frame_level_hnr(data, fs, window_length_ms, window_step_ms)
        APQ_range, APQ_std = amplitude_range(data, fs, window_length_ms, window_step_ms)
        self.logger.info("Processed voice quality features")
        
        # Process loudness and intensity
        RMS = rms_amplitude(data, fs, window_length_ms, window_step_ms)
        SPL = spl_per(data, fs, window_length_ms, window_step_ms)
        PEAK = peak_amplitude(data, fs, window_length_ms, window_step_ms)
        STE = ste_amplitude(data, fs, window_length_ms, window_step_ms)
        INTENSITY = intensity(data, fs, window_length_ms, window_step_ms)
        self.logger.info("Processed loudness and intensity parameters")
        
        # Process complexity
        HFD = calculate_hfd_per_frame(data, fs, window_length_ms, window_step_ms, 10)
        FREQ_ENTROPY = calculate_frequency_entropy(data, fs, window_length_ms, window_step_ms)
        AMP_ENTROPY = calculate_amplitude_entropy(data, fs, window_length_ms, window_step_ms)
        self.logger.info("Processed complexity features")
        
        # Compile results
        results = []
        
        # Append frequency parameters
        results.append(self.process_row(F0, 'F0'))
        results.append(self.process_row(np.array(jitter), 'Jitter'))
        results.append(self.process_row(F1[0, :], 'F1'))
        results.append(self.process_row(F1[1, :], 'F2'))
        results.append(self.process_row(F1[2, :], 'F3'))
        self.logger.debug("Frequency parameters: %d features", self.length(results))
        
        # Append spectral domain
        results.append(self.process_matrix(msc, 'MSC'))
        results.append(self.process_row(np.array(centroids), 'CENTRIODS'))
        results.append(self.process_row(LTAS, 'LTAS'))
        results.append(self.process_matrix(LOG_MEL_SPECTROGRAM, 'LOG_MEL_SPECTROGRAM'))
        results.append(self.process_matrix(MFCC, 'MFCC'))
        results.append(self.process_matrix(LPC, 'LPC'))
        results.append(self.process_matrix(LPCC, 'LPCC'))
        results.append(self.process_matrix(ENVELOPE, 'ENVELOPE'))
        results.append(self.process_row(CPP, 'CPP'))
        results.append(self.process_matrix(PLP, 'PLP'))
        results.append(self.process_matrix(lspFreq, "LSP"))
        self.logger.debug("Spectral domain: %d features", self.length(results))
        
        # Append voice quality
        results.append(self.process_row(APQ_std, 'APQ2'))
        results.append(self.process_row(SHIMMER, 'SHIMMER'))
        results.append(self.process_row(HNR, 'HNR'))
        results.append(self.process_row(NHR, 'NHR'))
        results.append({'ALPHA_RATIO': ALPHA_RATIO})
        results.append(self.process_row(HAMM_INDEX, 'HAMMARBERG_INDEX'))
        results.append({'HARMONICITY': HARMONICITY})
        self.logger.debug("Voice quality: %d features", self.length(results))
        
        # Append loudness and intensity
        results.append(self.process_row(APQ_range, 'Amplitude_Range'))
        results.append(self.process_row(RMS, 'RMS'))
        results.append(self.process_row(SPL, 'SPL'))
        results.append(self.process_row(PEAK, 'PEAK'))
        results.append(self.process_row(STE, 'STE'))
        results.append(self.process_row(INTENSITY, 'INTENSITY'))
        self.logger.debug("Loudness: %d features", self.length(results))
        
        # Append complexity
        results.append(self.process_row(HFD, 'HFD'))
        results.append(self.process_row(FREQ_ENTROPY, 'FREQ_ENTROPY'))
        results.append(self.process_row(AMP_ENTROPY, 'AMP_ENTROPY'))
        results.append(self.process_row(ZCR, "ZCR"))
        self.logger.debug("Complexity: %d features", self.length(results))
        
        # Merge all results
        final_results = {}
        for res in results:
            final_results.update(res)
            
        self.logger.info("Extracted a total of %d features", len(final_results))
        return final_results
    
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
        """
        if None in (self.vad_model, self.vad_utils, self.transcription_model):
            raise ValueError("VAD and transcription models must be set using set_models() before calling this method")
        
        self.logger.info("Processing file with VAD and transcription models: %s", filepath)
        
        # Process rhythmic features
        p_results = {}
        pause_behavior = PauseBehavior(self.vad_model, self.vad_utils, self.transcription_model)
        pause_behavior.configure(filepath)
        
        voiceProb_signal = self.vad_model.audio_forward(pause_behavior.data, sr=16000)
        voiceProb_signal = np.array(voiceProb_signal[0])
        prob = self.process_row(voiceProb_signal, "voiceProb")
        
        # List of methods to exclude
        excluded_methods = ['__init__', 'configure']
        
        # Iterate through all methods of the pause_behavior instance
        self.logger.debug("Extracting pause behavior features")
        for name, method in inspect.getmembers(pause_behavior, predicate=inspect.ismethod):
            # Check if the method is not in the excluded list
            if name not in excluded_methods:
                # Invoke each method and store the result
                p_results[name] = method()
        
        # Process speech features
        s_results = {}
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
        excluded_methods = ['__init__', 'configure', 'phoneme_alignment', 'relative_sentence_duration', 'regularity_of_segments', 'alternating_regularity']
        
        # Iterate through all methods of the speech_behavior instance
        self.logger.debug("Extracting speech behavior features")
        for name, method in inspect.getmembers(speech_behavior, predicate=inspect.ismethod):
            # Check if the method is not in the excluded list
            if name not in excluded_methods:
                # Invoke each method and store the result
                s_results[name] = method()
                
        # Process regularity and PVI features
        self.logger.debug("Processing regularity and PVI features")
        for i, res in enumerate(speech_behavior.regularity_of_segments()):
            s_results[f"regularity_{i}"] = res
        for i, res in enumerate(speech_behavior.alternating_regularity()):
            s_results[f"PVI_{i}"] = res
        
        # Process relative sentence duration
        relative_sentence_duration = speech_behavior.relative_sentence_duration()
        s_results.update(self.process_row(np.array(relative_sentence_duration), "relative_sentence_duration"))
        
        # Merge all results
        p_results.update(prob)
        self.logger.debug("Feature counts - pause: %d, speech: %d", len(p_results), len(s_results))
        p_results.update(s_results)
        
        self.logger.info("Extracted a total of %d features with VAD and transcription models", len(p_results))
        return p_results
    
    def extract_features(self, audio_paths: Union[str, List[str]], features_to_calculate: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract specified features from a single audio file or a batch of files
        
        Parameters:
        -----------
        audio_paths : str or List[str]
            Path(s) to the audio file(s)
        features_to_calculate : List[str], optional
            List of feature types to extract. If None, extracts all features.
            Valid options: 'spectral', 'complexity', 'frequency', 'intensity', 
                          'rhythmic', 'fluency', 'voice_quality', 'all',
                          'raw', 'transcription'
            
        Returns:
        --------
        dict
            Dictionary containing extracted features
            For a single file: {feature_name: feature_value, ...}
            For multiple files: {file_path: {feature_name: feature_value, ...}, ...}
        """
        # Default to all features if none specified
        if features_to_calculate is None:
            features_to_calculate = ['all']
        
        self.logger.info("Extracting features: %s", ", ".join(features_to_calculate))
        
        # Check if feature types are valid
        for feature_type in features_to_calculate:
            if feature_type not in self.available_features:
                raise ValueError(f"Unknown feature type: {feature_type}. Available types: {list(self.available_features.keys())}")
        
        # Process a single file
        if isinstance(audio_paths, str):
            self.logger.info("Processing single file: %s", audio_paths)
            result = {}
            
            for feature_type in features_to_calculate:
                # If 'all' is included, extract all features and skip other types
                if feature_type == 'all':
                    return self.extract_all_features(audio_paths)
                
                # Extract each requested feature type
                feature_extractor = self.available_features[feature_type]
                features = feature_extractor(audio_paths)
                result.update(features)
            
            self.logger.info("Extracted %d features total", len(result))
            return result
        # Process batch of files
        else:
            self.logger.info("Processing batch of %d files", len(audio_paths))
            batch_results = {}
            
            for i, audio_path in enumerate(audio_paths):
                self.logger.info("Processing file %d/%d: %s", i+1, len(audio_paths), audio_path)
                # Apply the same logic as for single file
                file_result = {}
                
                for feature_type in features_to_calculate:
                    if feature_type == 'all':
                        file_result = self.extract_all_features(audio_path)
                        break
                    
                    feature_extractor = self.available_features[feature_type]
                    features = feature_extractor(audio_path)
                    file_result.update(features)
                
                batch_results[audio_path] = file_result
            
            self.logger.info("Batch processing complete")
            return batch_results 