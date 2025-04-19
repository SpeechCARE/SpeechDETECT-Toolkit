import textstat
import whisperx
import statistics
import nltk
from nltk.tokenize import word_tokenize
from functools import cached_property

from .speech_fluency_and_speech_production_dynamics import calculate_duration_ms, remove_subranges

# Download NLTK data only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class PauseBehavior:
    # Define filler words as class constant
    FILLER_WORDS = {
        "um", "uh", "ah", "oh", "like", "you know", "so", "actually", "basically", "seriously",
        "literally", "i mean", "you see", "well", "okay", "right", "sort of", "kind of",
        "i guess", "you know what i mean", "believe me", "to be honest", "i think",
        "i suppose", "in a sense", "anyway", "and all that", "at the end of the day",
        "that said", "you know what?", "i feel like", "i don't know"
    }
    
    SAMPLING_RATE = 16000
    DEFAULT_DEVICE = "cuda"
    
    def __init__(self, vad_model, vad_model_utils, transcription_model):
        self.vad_model = vad_model
        self.vad_model_utils = vad_model_utils
        self.transcription_model = transcription_model
        self.silence_ranges = []
        self.speech_ranges = []
        self.text = ""
        self.data = None
        self.transcription_result = None
        self._cached_values = {}  # Cache for expensive calculations

    def configure(self, filename):
        (get_speech_timestamps, 
         save_audio, 
         read_audio, 
         VADIterator, 
         collect_chunks) = self.vad_model_utils

        # Clear cache when configuring with new file
        self._cached_values = {}
        
        # Process audio with VAD
        self.data = read_audio(filename, sampling_rate=self.SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            self.data, 
            self.vad_model, 
            sampling_rate=self.SAMPLING_RATE, 
            threshold=0.5
        )
        
        # Process speech ranges
        self.speech_ranges = [(ts["start"], ts["end"]) for ts in speech_timestamps]
        self.silence_ranges = remove_subranges(0, len(self.data), self.speech_ranges)

        # Transcription processing
        audio = whisperx.load_audio(filename)
        self.transcription_result = self.transcription_model.transcribe(audio, batch_size=8)
        
        # Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=self.transcription_result["language"], 
            device=self.DEFAULT_DEVICE
        )
        self.transcription_result = whisperx.align(
            self.transcription_result["segments"], 
            model_a, 
            metadata, 
            audio, 
            self.DEFAULT_DEVICE, 
            return_char_alignments=False
        )
        
        # Combine segment texts
        self.text = " ".join(segment["text"].strip() for segment in self.transcription_result["segments"])

    @cached_property
    def total_duration_seconds(self):
        """Total duration in seconds."""
        if self.data is None:
            return 0
        return len(self.data) / self.SAMPLING_RATE
    
    @cached_property
    def silence_durations_ms(self):
        """Calculate silence durations in milliseconds."""
        return calculate_duration_ms(self.silence_ranges, self.SAMPLING_RATE)
    
    @cached_property
    def speech_durations_ms(self):
        """Calculate speech durations in milliseconds."""
        return calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
    
    @cached_property
    def total_silence_duration_ms(self):
        """Total silence duration in milliseconds."""
        return sum(self.silence_durations_ms)
    
    @cached_property
    def total_speech_duration_ms(self):
        """Total speech duration in milliseconds."""
        return sum(self.speech_durations_ms)
    
    @cached_property
    def word_count(self):
        """Count words in transcription."""
        return len(self.text.split()) if self.text else 0
    
    @cached_property
    def token_count(self):
        """Count tokens in transcription."""
        return len(word_tokenize(self.text)) if self.text else 0
    
    @cached_property
    def syllable_count(self):
        """Count syllables in transcription."""
        return textstat.syllable_count(self.text) if self.text else 0

    def count_pause_segments(self):
        """Normalized count of pause segments per second."""
        if not self.total_duration_seconds:
            return 0
        return len(self.silence_ranges) / self.total_duration_seconds
    
    def pause_length(self):
        """Ratio of total pause duration to total audio duration."""
        if not self.total_duration_seconds:
            return 0
        # Convert from ms to seconds for consistent units
        return (self.total_silence_duration_ms / 1000) / self.total_duration_seconds
    
    def vocalic_interval(self):
        """Ratio of speech duration to total duration."""
        if not self.total_duration_seconds:
            return 0
        # Convert from ms to seconds for consistent units
        return (self.total_speech_duration_ms / 1000) / self.total_duration_seconds

    def pause_lengths_avg(self):
        """Average duration of pauses in milliseconds."""
        if not self.silence_durations_ms:
            return 0
        return statistics.mean(self.silence_durations_ms)
    
    def pause_speech_ratio(self):
        """Ratio of number of pauses to number of speech segments."""
        if not self.speech_ranges:
            return 0
        return len(self.silence_ranges) / len(self.speech_ranges)
    
    def pause_speech_duration_ratio(self):
        """Ratio of total pause duration to total speech duration."""
        if not self.total_speech_duration_ms:
            return 0
        return self.total_silence_duration_ms / self.total_speech_duration_ms
    
    def pause_totallength_ratio(self):
        """Ratio of pause duration to total duration."""
        # This is essentially the same as pause_length() but keeping for backward compatibility
        return self.pause_length()
    
    def num_words_to_num_pauses(self):
        """Ratio of word count to pause count."""
        n_pauses = len(self.silence_ranges)
        return self.word_count / (n_pauses + 1)
    
    def pause_to_syllable(self):
        """Ratio of pause count to syllable count."""
        number_of_pauses = len(self.silence_ranges)
        return number_of_pauses / (self.syllable_count + 1)
    
    def pause_to_tokens(self):
        """Ratio of pause count to token count."""
        number_of_pauses = len(self.silence_ranges)
        return number_of_pauses / (self.token_count + 1)
    
    def hesitation_rate(self, num_pauses=0):
        """Calculate the rate of filler words and hesitations."""
        if not self.text:
            return 0
            
        lower_transcription = self.text.lower()
        num_fillerwords = sum(lower_transcription.count(filler) for filler in self.FILLER_WORDS)
        
        if num_pauses:
            num_fillerwords += num_pauses

        word_count = len(lower_transcription.split())
        return num_fillerwords / (word_count + 1)