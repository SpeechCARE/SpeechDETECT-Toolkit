import textstat
import whisperx
from pocketsphinx import Decoder, AudioFile, Segmenter
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import soundfile as sf
import wave
from functools import cached_property


# Download NLTK data only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)

# Helper functions moved from speech_fluency_and_speech_production_dynamics_helper.py
def calculate_duration_ms(ranges, sr):
    """Calculate durations in milliseconds from time ranges."""
    return [(rng[1] - rng[0]) / sr * 1000 for rng in ranges]

def remove_subranges(full_range_start, full_range_end, subranges):
    """Remove subranges from a full range and return remaining ranges."""
    remaining_ranges = [(full_range_start, full_range_end)]

    for s, e in subranges:
        new_remaining_ranges = []
        for r_start, r_end in remaining_ranges:
            # If subrange is completely outside the current range, ignore it
            if e <= r_start or s >= r_end:
                new_remaining_ranges.append((r_start, r_end))
            else:
                # Add the part before the subrange
                if s > r_start:
                    new_remaining_ranges.append((r_start, s))
                # Add the part after the subrange
                if e < r_end:
                    new_remaining_ranges.append((e, r_end))
        remaining_ranges = new_remaining_ranges

    return remaining_ranges

def read_wav_segment_new(file_path, sample_rate, start_time, end_time):
    """Read a specific segment from a WAV file using soundfile."""
    data, sample_rate = sf.read(file_path, dtype='int16')
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)
    return data[start_frame:end_frame].tobytes(), sample_rate

def read_wav_segment(file_path, start_time, end_time):
    """Read a specific segment from a WAV file using wave."""
    with wave.open(file_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        wav.setpos(start_frame)
        frame_count = end_frame - start_frame
        segment_data = wav.readframes(frame_count)
        return segment_data, sample_rate

def extract_syllables(phonetic_transcription):
    """Extract syllables from phonetic transcription."""
    vowel_sounds = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    syllables = []
    current_syllable = []

    for phone in phonetic_transcription:
        if len(current_syllable) == 0:
            start_time = phone["start"]

        phoneme = phone["name"]
        current_syllable.append(phoneme)

        # When a vowel is encountered, it marks the end of a syllable
        if phoneme in vowel_sounds:
            syllables.append({"syllable": ' '.join(current_syllable), "start": start_time, "end": phone["end"]})
            current_syllable = []

    # Adding any remaining consonants as a syllable (for cases like final consonants)
    if current_syllable:
        if syllables:
            syllables[-1]["syllable"] += ' '.join(current_syllable)
            syllables[-1]["end"] = phonetic_transcription[-1]["end"]
        else:
            syllables.append({"syllable": ' '.join(current_syllable), "start": start_time, "end": phone["end"]})

    return syllables

def calculate_statistics(segments):
    """Calculate statistical measures for a list of segments."""
    if not segments:
        return 0, 0, 0
    mean = np.mean(segments)
    std_dev = np.std(segments)
    coefficient_of_variation = (std_dev / mean) * 100 if mean else 0
    return mean, std_dev, coefficient_of_variation

def calculate_raw_pvi(segments):
    """Calculate raw Pairwise Variability Index."""
    if len(segments) < 2:
        return 0
    differences = np.abs(np.diff(segments))
    return np.mean(differences)

def calculate_normalized_pvi(segments):
    """Calculate normalized Pairwise Variability Index."""
    if len(segments) < 2:
        return 0
    
    nPVI_values = []
    for i in range(len(segments) - 1):
        dur_i, dur_i_1 = segments[i], segments[i + 1]
        # Avoid division by zero
        if dur_i + dur_i_1 == 0:
            continue
        nPVI = abs(dur_i - dur_i_1) / ((dur_i + dur_i_1) / 2) * 100
        nPVI_values.append(nPVI)
    
    return np.mean(nPVI_values) if nPVI_values else 0

def calculate_silence_segments(speech_segments):
    """Calculate silence segments between speech segments."""
    if len(speech_segments) < 2:
        return []
        
    silence_segments = []
    for i in range(len(speech_segments) - 1):
        end_current_speech = speech_segments[i][1]
        start_next_speech = speech_segments[i + 1][0]
        silence_segments.append((end_current_speech, start_next_speech))
    return silence_segments

def calculate_alternating_durations(speech_segments, sampling_rate):
    """Calculate alternating durations of speech and silence."""
    speech_durations = calculate_duration_ms(speech_segments, sampling_rate)
    silence_segments = calculate_silence_segments(speech_segments)
    silence_durations = calculate_duration_ms(silence_segments, sampling_rate)

    alternating_durations = []
    for i in range(len(speech_durations)):
        alternating_durations.append(speech_durations[i])
        if i < len(silence_durations):
            alternating_durations.append(silence_durations[i])

    return alternating_durations

class SpeechBehavior:
    """Analyze speech behavior based on audio input and transcription."""
    
    SAMPLING_RATE = 16000
    DEFAULT_DEVICE = "cuda"
    VOWEL_SOUNDS = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    
    def __init__(self, vad_model, vad_model_utils, transcription_model):
        """Initialize the SpeechBehavior analyzer.
        
        Args:
            vad_model: Voice Activity Detection model
            vad_model_utils: Utility functions for VAD
            transcription_model: Model for transcribing speech
        """
        self.vad_model = vad_model
        self.vad_model_utils = vad_model_utils
        self.transcription_model = transcription_model
        self.transcription_result = []
        self.silence_ranges = []
        self.speech_ranges = []
        self.text = ""
        self.all_phonemes = []
        self.data = None
        self._cached_values = {}  # Cache for expensive calculations

    def configure(self, filename):
        """Configure the analyzer with an audio file.
        
        Args:
            filename: Path to the audio file
        """
        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = self.vad_model_utils

        # Clear cache for new file
        self._cached_values = {}
        
        # Process audio with VAD
        self.data = read_audio(filename, sampling_rate=self.SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            self.data, 
            self.vad_model, 
            sampling_rate=self.SAMPLING_RATE, 
            threshold=0.5
        )

        # Extract speech and silence ranges
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

    def phoneme_alignment(self, file_name):
        """Align phonemes with the audio file.
        
        Args:
            file_name: Path to the audio file
            
        Returns:
            tuple: (Total segments, Number of hypothetical alignments)
        """
        arpabet = nltk.corpus.cmudict.dict()
        self.all_phonemes = []
        n_hypothesis = 0

        for segment in self.transcription_result["segments"]:
            text = segment["text"].translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower().strip()
            start_time = segment["start"]
            end_time = segment["end"]

            # Read the audio segment
            data, sr = read_wav_segment_new(file_name, self.SAMPLING_RATE, start_time, end_time)
            decoder = Decoder(samprate=sr, bestpath=False)
            
            try:
                # Try to perform phoneme alignment
                decoder.set_align_text(text)
                decoder.start_utt()
                decoder.process_raw(data, full_utt=True)
                decoder.end_utt()
                decoder.set_alignment()
                decoder.start_utt()
                decoder.process_raw(data, full_utt=True)
                decoder.end_utt()
            except:
                # Fallback method when alignment fails
                for word_dict in segment["words"]:
                    word = word_dict["word"].translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower().strip()
                    try:    
                        start, end = word_dict["start"], word_dict["end"]
                    except:
                        continue

                    try:
                        phonemes = arpabet[word][0]
                    except:
                        word_normalize = word.replace("'", "")
                        if word_normalize in arpabet:
                            phonemes = arpabet[word_normalize][0]
                        else:
                            phonemes = word_normalize

                    if isinstance(phonemes, str):
                        # If phonemes is just a string (not in dictionary), skip
                        continue

                    # Evenly distribute phonemes across the word duration
                    duration = (end - start) / len(phonemes)
                    word_entry = {"text": word, "start": start, "end": end, "phonemes": []}

                    for i, phone in enumerate(phonemes):
                        phoneme_start = round(start+i*duration, 3)
                        phoneme_end = round(start+(i+1)*duration, 3)
                        stripped_phone = phone.translate(str.maketrans('', '', string.digits))
                        word_entry["phonemes"].append({"name": stripped_phone, "start": phoneme_start, "end": phoneme_end})

                    self.all_phonemes.append(word_entry)
                n_hypothesis += 1
            else:
                # Process successful alignment
                for word in decoder.get_alignment():
                    if word.name == "<sil>":
                        continue
                    word_entry = {
                        "text": word.name, 
                        "start": round(start_time+word.start/100, 3), 
                        "end": round(start_time+(word.start+word.duration)/100, 3), 
                        "phonemes": []
                    }
                    for phone in word:
                        name = phone.name
                        start = start_time + (phone.start / 100)
                        end = start_time + ((phone.start + phone.duration) / 100)
                        start = round(start, 3)
                        end = round(end, 3)
                        word_entry["phonemes"].append({"name": name, "start": start, "end": end})
                    self.all_phonemes.append(word_entry)

        return (len(self.transcription_result["segments"]), n_hypothesis)
    
    @cached_property
    def total_duration_seconds(self):
        """Total duration of the audio in seconds."""
        return len(self.data) / self.SAMPLING_RATE if self.data is not None else 0
    
    @cached_property
    def speech_durations_ms(self):
        """Calculate speech durations in milliseconds."""
        return calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
    
    @cached_property
    def silence_durations_ms(self):
        """Calculate silence durations in milliseconds."""
        return calculate_duration_ms(self.silence_ranges, self.SAMPLING_RATE)
    
    @cached_property
    def total_speech_duration_ms(self):
        """Total speech duration in milliseconds."""
        return sum(self.speech_durations_ms)
    
    @cached_property
    def total_speech_duration_min(self):
        """Total speech duration in minutes."""
        return self.total_speech_duration_ms / (60 * 1000)
    
    @cached_property
    def word_count(self):
        """Count words in transcription."""
        return len(self.text.split()) if self.text else 0
    
    @cached_property
    def token_count(self):
        """Count tokens in transcription."""
        return len(word_tokenize(self.text)) if self.text else 0
    
    @cached_property
    def syllable_count_total(self):
        """Total syllable count in transcription."""
        return textstat.syllable_count(self.text) if self.text else 0
        
    def articulation_rate(self):
        """Calculate articulation rate (phonemes per second).
        
        Returns:
            float: Number of phonemes divided by duration
        """
        if not self.all_phonemes:
            return 0
            
        num_phonemes = 0
        duration = 0
        for word in self.all_phonemes:
            duration += (word["end"] - word["start"])
            num_phonemes += len(word["phonemes"])

        return num_phonemes / (duration + 0.1)  # Adding small constant to prevent division by zero
    
    def mean_inter_syllabic_pauses(self):
        """Calculate mean duration of pauses between syllables.
        
        Returns:
            float: Mean duration of inter-syllabic pauses, or None if no silence found
        """
        if not self.all_phonemes:
            return None
            
        num_silence = 0
        sum_silence = 0
        for word in self.all_phonemes:
            ranges = [(x["start"], x["end"]) for x in word["phonemes"]]
            sil_ranges = remove_subranges(word["start"], word["end"], ranges)
            num_silence += len(sil_ranges)
            sum_silence += sum((x[1] - x[0]) for x in sil_ranges)
            
        return sum_silence / num_silence if num_silence > 0 else None
    
    def num_of_syllables(self):
        """Calculate syllables per second.
        
        Returns:
            float: Number of syllables divided by total duration
        """
        if not self.all_phonemes or not self.total_duration_seconds:
            return 0
            
        syllable_count = 0
        for word in self.all_phonemes:
            syllable_count += len(extract_syllables(word["phonemes"]))
        return syllable_count / self.total_duration_seconds

    def syllabic_interval_duration(self):
        """Calculate ratio of syllabic interval duration to total duration.
        
        Returns:
            float: Ratio of syllable duration to total duration
        """
        if not self.all_phonemes or not self.total_duration_seconds:
            return 0
            
        duration = 0
        for word in self.all_phonemes:
            for syllable in extract_syllables(word["phonemes"]):
                duration += (syllable["end"] - syllable["start"])

        return duration / (self.total_duration_seconds + 0.1)

    def vowel_duration(self):
        """Calculate ratio of vowel duration to total duration.
        
        Returns:
            float: Ratio of vowel duration to total duration
        """
        if not self.all_phonemes or not self.total_duration_seconds:
            return 0
            
        vwl_dur = 0
        for word in self.all_phonemes:
            for phone in word["phonemes"]:
                if phone["name"] in self.VOWEL_SOUNDS:
                    vwl_dur += (phone["end"] - phone["start"])

        return vwl_dur / (self.total_duration_seconds + 0.1)

    def phonation_time(self):
        """Calculate total phonation time (sum of all word durations).
        
        Returns:
            float: Total phonation time in seconds
        """
        if not self.all_phonemes:
            return 0
        return sum((x["end"] - x["start"]) for x in self.all_phonemes)

    def percentage_phonation_time(self):
        """Calculate percentage of phonation time relative to total duration.
        
        Returns:
            float: Percentage of phonation time
        """
        if not self.total_duration_seconds:
            return 0
            
        phonation_time = self.phonation_time()
        return round(phonation_time / self.total_duration_seconds, 3) * 100

    def transformed_phonation_rate(self):
        """Calculate transformed phonation rate using arcsine square root transformation.
        
        Returns:
            float: Transformed phonation rate
        """
        phonation_rate = self.percentage_phonation_time() / 100
        return np.arcsin(np.sqrt(max(0, min(1, phonation_rate))))  # Ensure value is in [0,1] range

    def tlt(self):
        """Calculate Total Locution Time ratio.
        
        Returns:
            float: Ratio of locution time to total speech duration
        """
        if not self.speech_ranges or not self.total_speech_duration_ms:
            return 0
            
        locution_time_ms = (self.speech_ranges[-1][1] - self.speech_ranges[0][0]) / self.SAMPLING_RATE * 1000
        return locution_time_ms / self.total_speech_duration_ms

    def token_duration(self):
        """Get the total speech duration in milliseconds.
        
        Returns:
            float: Total speech duration in milliseconds
        """
        return self.total_speech_duration_ms

    def syllable_count(self):
        """Calculate syllable count per second.
        
        Returns:
            float: Syllables per second
        """
        if not self.total_duration_seconds:
            return 0
            
        return self.syllable_count_total / self.total_duration_seconds

    def count_tokens(self):
        """Calculate tokens per second.
        
        Returns:
            float: Tokens per second
        """
        if not self.total_duration_seconds:
            return 0
            
        return self.token_count / self.total_duration_seconds

    def speech_rate_words(self):
        """Calculate speech rate in words per minute.
        
        Returns:
            float: Words per minute
        """
        if not self.total_speech_duration_min or not self.word_count:
            return 0
            
        return self.word_count / self.total_speech_duration_min

    def speech_rate_syllable(self):
        """Calculate speech rate in syllables per minute.
        
        Returns:
            float: Syllables per minute
        """
        if not self.total_speech_duration_min:
            return 0
            
        return self.syllable_count_total / self.total_speech_duration_min

    def phonation_to_syllable(self):
        """Calculate phonation time per syllable.
        
        Returns:
            float: Phonation time per syllable
        """
        if not self.syllable_count_total:
            return 0
            
        return self.phonation_time() / self.syllable_count_total

    def average_num_of_speech_segments(self):
        """Calculate average number of speech segments per second.
        
        Returns:
            float: Number of speech segments per second
        """
        if not self.total_duration_seconds:
            return 0
            
        return len(self.speech_ranges) / self.total_duration_seconds

    def mean_words_in_utterance(self):
        """Calculate mean words per utterance.
        
        Returns:
            float: Mean words per utterance
        """
        if not self.transcription_result or "segments" not in self.transcription_result:
            return 0
            
        number_of_utterances = len(self.transcription_result["segments"])
        if number_of_utterances == 0:
            return 0
            
        total_words = sum(len(segment["text"].split()) for segment in self.transcription_result["segments"])
        return total_words / number_of_utterances

    def mean_length_sentence(self):
        """Calculate mean length of sentences in tokens.
        
        Returns:
            float: Mean sentence length in tokens
        """
        if not self.text:
            return 0
            
        sentences = sent_tokenize(self.text)
        if not sentences:
            return 0
            
        return (self.token_count * self.total_duration_seconds) / len(sentences)

    def relative_sentence_duration(self):
        """Calculate relative duration of each sentence/segment.
        
        Returns:
            list: List of relative durations for each segment
        """
        if not self.transcription_result or not self.total_speech_duration_ms:
            return []
            
        total_speech_duration_s = self.total_speech_duration_ms / 1000
        if total_speech_duration_s == 0:
            return []
            
        return [(segment["end"] - segment["start"]) / total_speech_duration_s 
                for segment in self.transcription_result["segments"]]

    def regularity_of_segments(self):
        """Calculate regularity metrics for speech and silence segments.
        
        Returns:
            list: Metrics for speech and silence segments regularity
        """
        # Get speech and silence durations
        speech_durations = self.speech_durations_ms
        silence_durations = self.silence_durations_ms

        # Skip processing if no data
        if not speech_durations:
            return [0] * 10

        # Adjusting the silence segments if needed
        if self.silence_ranges and self.speech_ranges:
            # Exclude silence before first speech segment
            if len(silence_durations) > 0 and self.silence_ranges[0][1] <= self.speech_ranges[0][0]:
                silence_durations = silence_durations[1:]
            # Exclude silence after last speech segment
            if len(silence_durations) > 0 and self.silence_ranges[-1][0] >= self.speech_ranges[-1][1]:
                silence_durations = silence_durations[:-1]

        # Calculate statistics
        speech_mean, speech_std_dev, speech_cv = calculate_statistics(speech_durations)
        silence_mean, silence_std_dev, silence_cv = calculate_statistics(silence_durations)

        # Calculate PVI metrics
        speech_pvi = calculate_raw_pvi(speech_durations)
        speech_npvi = calculate_normalized_pvi(speech_durations)
        silence_pvi = calculate_raw_pvi(silence_durations)
        silence_npvi = calculate_normalized_pvi(silence_durations)

        return [
            speech_pvi, speech_npvi, speech_mean, speech_std_dev, speech_cv,
            silence_pvi, silence_npvi, silence_mean, silence_std_dev, silence_cv
        ]
    
    def alternating_regularity(self):
        """Calculate regularity metrics for alternating speech and silence.
        
        Returns:
            list: PVI and nPVI for alternating durations
        """
        if not self.speech_ranges:
            return [0, 0]
            
        alternating_durations = calculate_alternating_durations(self.speech_ranges, self.SAMPLING_RATE)
        speech_pvi = calculate_raw_pvi(alternating_durations)
        speech_npvi = calculate_normalized_pvi(alternating_durations)

        return [speech_pvi, speech_npvi]



