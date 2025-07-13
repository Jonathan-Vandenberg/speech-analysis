import numpy as np
import librosa
import scipy.signal
import scipy.ndimage
import io
import wave
import tempfile
import os
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

def calculate_audio_hash(audio_bytes: bytes) -> str:
    """Calculate hash of audio data for deduplication and caching"""
    import hashlib
    return hashlib.sha256(audio_bytes).hexdigest()

def convert_audio_to_wav(audio_bytes: bytes, audio_format: str) -> bytes:
    """
    Convert audio bytes to WAV format with enhanced error handling and validation
    """
    try:
        import pydub
        
        # Create audio segment from bytes
        if audio_format.lower() in ['wav', 'wave']:
            # Already WAV, but validate and normalize
            audio_segment = pydub.AudioSegment.from_wav(io.BytesIO(audio_bytes))
        elif audio_format.lower() in ['mp3', 'mpeg']:
            audio_segment = pydub.AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        elif audio_format.lower() in ['m4a', 'mp4', 'aac']:
            audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
        elif audio_format.lower() in ['ogg', 'oga']:
            audio_segment = pydub.AudioSegment.from_ogg(io.BytesIO(audio_bytes))
        elif audio_format.lower() == 'webm':
            audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        elif audio_format.lower() == 'flac':
            audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes), format="flac")
        else:
            # Try to auto-detect format
            audio_segment = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Ensure mono and proper sample rate
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        
        # Target sample rate for speech analysis (16kHz is optimal for most models)
        target_sample_rate = 16000
        if audio_segment.frame_rate != target_sample_rate:
            audio_segment = audio_segment.set_frame_rate(target_sample_rate)
        
        # Normalize volume to prevent clipping and improve SNR
        # Target -12dB to leave headroom while maximizing signal strength
        target_dbfs = -12.0
        change_in_dbfs = target_dbfs - audio_segment.dBFS
        if abs(change_in_dbfs) > 1.0:  # Only normalize if significant difference
            audio_segment = audio_segment + change_in_dbfs
        
        # Export as WAV
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format="wav")
        return output_buffer.getvalue()
        
    except Exception as e:
        # Fallback to basic conversion
        print(f"⚠️ Audio conversion failed with pydub: {e}, using fallback")
        return audio_bytes

def prepare_audio_data(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Enhanced audio preparation with robust preprocessing pipeline
    """
    try:
        # Load audio data
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(wav_bytes)
            temp_file.flush()
            
            try:
                # Load with librosa for better quality and preprocessing options
                audio_data, samplerate = librosa.load(
                    temp_file.name, 
                    sr=None,  # Keep original sample rate initially
                    mono=True,  # Ensure mono
                    dtype=np.float32
                )
                
                # Remove the temporary file
                os.unlink(temp_file.name)
                
            except Exception as librosa_error:
                print(f"⚠️ Librosa failed: {librosa_error}, using wave fallback")
                # Fallback to wave module
                os.unlink(temp_file.name)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file2:
                    temp_file2.write(wav_bytes)
                    temp_file2.flush()
                    
                    with wave.open(temp_file2.name, 'rb') as wav_file:
                        frames = wav_file.readframes(-1)
                        samplerate = wav_file.getframerate()
                        channels = wav_file.getnchannels()
                        sampwidth = wav_file.getsampwidth()
                        
                        if sampwidth == 2:
                            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sampwidth == 4:
                            audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        
                        if channels > 1:
                            audio_data = audio_data.reshape(-1, channels)
                            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
                    
                    os.unlink(temp_file2.name)
                
        # Apply enhanced preprocessing pipeline
        audio_data, samplerate = apply_enhanced_preprocessing(audio_data, samplerate)
        
        # Final validation
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty after preprocessing")
        
        return audio_data, samplerate
        
    except Exception as e:
        print(f"❌ Error preparing audio data: {e}")
        # Return minimal valid audio data to prevent total failure
        return np.array([0.0] * 1000, dtype=np.float32), 16000

def apply_enhanced_preprocessing(audio_data: np.ndarray, samplerate: int) -> Tuple[np.ndarray, int]:
    """
    Apply comprehensive audio preprocessing pipeline for improved speech analysis reliability
    """
    try:
        original_length = len(audio_data)
        print(f"🔧 Preprocessing audio: {original_length} samples at {samplerate}Hz")
        
        # Step 1: Trim silence from beginning and end (but preserve some context)
        audio_data = trim_silence_smart(audio_data, samplerate)
        
        # Step 2: Resample to optimal rate for speech analysis if needed
        target_sr = 16000
        if samplerate != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=target_sr)
            samplerate = target_sr
            print(f"🔄 Resampled to {target_sr}Hz")
        
        # Step 3: Apply intelligent noise reduction
        audio_data = reduce_noise_conservative(audio_data, samplerate)
        
        # Step 4: Normalize audio level with smart peak detection
        audio_data = normalize_audio_smart(audio_data)
        
        # Step 5: Apply gentle high-pass filter to remove low-frequency noise
        audio_data = apply_highpass_filter(audio_data, samplerate, cutoff=80)
        
        # Step 6: Apply gentle pre-emphasis to improve consonant clarity
        audio_data = apply_preemphasis(audio_data, alpha=0.95)
        
        # Step 7: Ensure reasonable duration (not too short, not too long)
        audio_data = ensure_reasonable_duration(audio_data, samplerate)
        
        print(f"✅ Preprocessing complete: {len(audio_data)} samples ({len(audio_data)/samplerate:.2f}s)")
        
        return audio_data, samplerate
        
    except Exception as e:
        print(f"⚠️ Preprocessing failed: {e}, returning original audio")
        return audio_data, samplerate

def trim_silence_smart(audio_data: np.ndarray, samplerate: int, 
                      margin_start: float = 0.1, margin_end: float = 0.1) -> np.ndarray:
    """
    Smart silence trimming that preserves some context for better analysis
    """
    try:
        # Use librosa's trim function with conservative settings
        trimmed_audio, _ = librosa.effects.trim(
            audio_data, 
            top_db=25,  # More conservative - don't cut quiet speech
            frame_length=1024,
            hop_length=256
        )
        
        # Add small margins back to preserve attack/decay
        start_margin_samples = int(margin_start * samplerate)
        end_margin_samples = int(margin_end * samplerate)
        
        # Find original trim points
        if len(trimmed_audio) < len(audio_data):
            # Calculate where trim happened
            start_diff = np.where(np.array_equal(audio_data[i:i+len(trimmed_audio)], trimmed_audio) 
                                for i in range(len(audio_data) - len(trimmed_audio) + 1))[0]
            if len(start_diff) > 0:
                trim_start = start_diff[0]
                # Add back margins if available
                new_start = max(0, trim_start - start_margin_samples)
                new_end = min(len(audio_data), trim_start + len(trimmed_audio) + end_margin_samples)
                return audio_data[new_start:new_end]
        
        return trimmed_audio
        
    except Exception:
        # Fallback: simple energy-based trimming
        return simple_trim_silence(audio_data, threshold=0.01)

def simple_trim_silence(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Simple energy-based silence trimming fallback"""
    # Calculate frame energy
    frame_length = 1024
    energy = np.array([
        np.sum(audio_data[i:i+frame_length]**2) 
        for i in range(0, len(audio_data) - frame_length, frame_length//2)
    ])
    
    # Find speech boundaries
    speech_frames = energy > (threshold * np.max(energy))
    if not np.any(speech_frames):
        return audio_data
    
    start_frame = np.argmax(speech_frames)
    end_frame = len(speech_frames) - np.argmax(speech_frames[::-1]) - 1
    
    start_sample = start_frame * frame_length // 2
    end_sample = end_frame * frame_length // 2 + frame_length
    
    return audio_data[max(0, start_sample):min(len(audio_data), end_sample)]

def reduce_noise_conservative(audio_data: np.ndarray, samplerate: int) -> np.ndarray:
    """
    Conservative noise reduction that preserves speech quality
    """
    try:
        # Simple spectral subtraction approach
        # Calculate noise profile from first and last 0.2 seconds (likely silence/background)
        noise_duration = min(0.2, len(audio_data) / samplerate / 4)
        noise_samples = int(noise_duration * samplerate)
        
        if noise_samples > 10:
            # Estimate noise from beginning and end
            noise_start = audio_data[:noise_samples]
            noise_end = audio_data[-noise_samples:]
            noise_profile = np.concatenate([noise_start, noise_end])
            
            # Calculate noise floor
            noise_floor = np.std(noise_profile) * 2
            
            # Apply gentle noise gate
            audio_data = np.where(np.abs(audio_data) < noise_floor, 
                                audio_data * 0.1,  # Reduce but don't eliminate
                                audio_data)
        
        return audio_data
        
    except Exception:
        return audio_data

def normalize_audio_smart(audio_data: np.ndarray, target_level: float = 0.7) -> np.ndarray:
    """
    Smart audio normalization that avoids clipping and preserves dynamics
    """
    try:
        # Calculate percentile-based peak to avoid normalizing to noise spikes
        peak_level = np.percentile(np.abs(audio_data), 99.5)
        
        if peak_level > 1e-6:  # Avoid division by very small numbers
            # Normalize to target level with headroom
            scale_factor = min(target_level / peak_level, 3.0)  # Cap at 3x amplification
            audio_data = audio_data * scale_factor
            
            # Gentle compression if we're still clipping
            if np.max(np.abs(audio_data)) > 0.95:
                audio_data = np.tanh(audio_data * 0.8) * 0.9
        
        return audio_data
        
    except Exception:
        return audio_data

def apply_highpass_filter(audio_data: np.ndarray, samplerate: int, cutoff: float = 80) -> np.ndarray:
    """
    Apply gentle high-pass filter to remove low-frequency noise
    """
    try:
        # Design high-pass filter
        nyquist = samplerate / 2
        normal_cutoff = cutoff / nyquist
        
        # Use a gentle 2nd order Butterworth filter
        b, a = scipy.signal.butter(2, normal_cutoff, btype='high', analog=False)
        
        # Apply filter with padding to reduce edge artifacts
        padded_length = min(len(audio_data) // 4, samplerate // 10)  # 0.1 second padding max
        filtered_audio = scipy.signal.filtfilt(b, a, audio_data, padlen=padded_length)
        
        return filtered_audio.astype(np.float32)
        
    except Exception:
        return audio_data

def apply_preemphasis(audio_data: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """
    Apply pre-emphasis filter to enhance high-frequency components (improves consonant detection)
    """
    try:
        # Pre-emphasis: y[n] = x[n] - α*x[n-1]
        emphasized = np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
        return emphasized.astype(np.float32)
        
    except Exception:
        return audio_data

def ensure_reasonable_duration(audio_data: np.ndarray, samplerate: int, 
                             min_duration: float = 0.5, max_duration: float = 30.0) -> np.ndarray:
    """
    Ensure audio has reasonable duration for analysis
    """
    duration = len(audio_data) / samplerate
    
    # Handle too short audio
    if duration < min_duration:
        print(f"⚠️ Audio too short ({duration:.2f}s), padding to {min_duration}s")
        target_samples = int(min_duration * samplerate)
        padding_needed = target_samples - len(audio_data)
        # Pad with low-level noise to avoid artifacts
        padding = np.random.normal(0, 0.001, padding_needed).astype(np.float32)
        audio_data = np.concatenate([audio_data, padding])
    
    # Handle too long audio
    elif duration > max_duration:
        print(f"⚠️ Audio too long ({duration:.2f}s), truncating to {max_duration}s")
        max_samples = int(max_duration * samplerate)
        audio_data = audio_data[:max_samples]
    
    return audio_data

def assess_audio_quality(audio_data: np.ndarray, samplerate: int) -> Dict[str, float]:
    """
    Comprehensive audio quality assessment for reliability indicators
    """
    try:
        quality_metrics = {}
        
        # 1. Signal-to-Noise Ratio estimation
        quality_metrics['snr_estimate'] = estimate_snr(audio_data, samplerate)
        
        # 2. Clipping detection
        quality_metrics['clipping_ratio'] = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)
        
        # 3. Dynamic range
        quality_metrics['dynamic_range'] = np.percentile(np.abs(audio_data), 95) - np.percentile(np.abs(audio_data), 5)
        
        # 4. Silence ratio
        silence_threshold = np.percentile(np.abs(audio_data), 10)
        quality_metrics['silence_ratio'] = np.sum(np.abs(audio_data) < silence_threshold) / len(audio_data)
        
        # 5. Frequency response quality
        quality_metrics['frequency_quality'] = assess_frequency_response(audio_data, samplerate)
        
        # 6. Overall quality score (0-100)
        quality_metrics['overall_quality'] = calculate_overall_quality_score(quality_metrics)
        
        return quality_metrics
        
    except Exception as e:
        print(f"⚠️ Quality assessment failed: {e}")
        return {'overall_quality': 50.0, 'snr_estimate': 10.0, 'clipping_ratio': 0.0}

def estimate_snr(audio_data: np.ndarray, samplerate: int) -> float:
    """Estimate Signal-to-Noise Ratio"""
    try:
        # Use voice activity detection approach
        frame_length = int(0.025 * samplerate)  # 25ms frames
        hop_length = int(0.010 * samplerate)   # 10ms hop
        
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        
        # Assume top 30% energy frames are speech, bottom 30% are noise
        energy_sorted = np.sort(frame_energy)
        noise_energy = np.mean(energy_sorted[:len(energy_sorted)//3])
        signal_energy = np.mean(energy_sorted[-len(energy_sorted)//3:])
        
        if noise_energy > 0:
            snr_linear = signal_energy / noise_energy
            snr_db = 10 * np.log10(snr_linear)
            return min(40.0, max(0.0, snr_db))  # Cap between 0-40 dB
        else:
            return 30.0  # High SNR estimate if no noise detected
            
    except Exception:
        return 15.0  # Default moderate SNR

def assess_frequency_response(audio_data: np.ndarray, samplerate: int) -> float:
    """Assess frequency response quality for speech"""
    try:
        # Calculate power spectral density
        freqs, psd = scipy.signal.welch(audio_data, samplerate, nperseg=1024)
        
        # Check key speech frequency bands
        # 300-3400 Hz is traditional telephone bandwidth
        # 85-300 Hz: Low frequencies (male fundamental)
        # 300-3400 Hz: Core speech intelligibility range
        # 3400-8000 Hz: High frequency content (consonants)
        
        low_band = (freqs >= 85) & (freqs <= 300)
        mid_band = (freqs >= 300) & (freqs <= 3400)
        high_band = (freqs >= 3400) & (freqs <= 8000)
        
        low_power = np.mean(psd[low_band]) if np.any(low_band) else 0
        mid_power = np.mean(psd[mid_band]) if np.any(mid_band) else 0  
        high_power = np.mean(psd[high_band]) if np.any(high_band) else 0
        
        # Good speech should have strong mid-band, moderate low-band, some high-band
        if mid_power > 0:
            # Score based on balance between bands
            low_ratio = low_power / mid_power if mid_power > 0 else 0
            high_ratio = high_power / mid_power if mid_power > 0 else 0
            
            # Ideal ratios for speech
            ideal_low_ratio = 0.3   # Low frequencies should be present but not dominant
            ideal_high_ratio = 0.15 # High frequencies important for consonants
            
            low_score = 1.0 - min(1.0, abs(low_ratio - ideal_low_ratio) / ideal_low_ratio)
            high_score = 1.0 - min(1.0, abs(high_ratio - ideal_high_ratio) / ideal_high_ratio)
            
            frequency_quality = (low_score + high_score) / 2 * 100
            return min(100.0, max(0.0, frequency_quality))
        else:
            return 0.0
            
    except Exception:
        return 50.0  # Default moderate quality

def calculate_overall_quality_score(metrics: Dict[str, float]) -> float:
    """Calculate overall quality score from individual metrics"""
    try:
        # Weights for different quality factors
        snr_score = min(100, max(0, metrics.get('snr_estimate', 15) * 5))  # 0-20 dB -> 0-100
        clipping_penalty = metrics.get('clipping_ratio', 0) * 100  # 0-1 -> 0-100 penalty
        dynamic_range_score = min(100, metrics.get('dynamic_range', 0.1) * 200)  # 0-0.5 -> 0-100
        silence_penalty = max(0, metrics.get('silence_ratio', 0.3) - 0.5) * 100  # Penalty for >50% silence
        frequency_score = metrics.get('frequency_quality', 50)
        
        # Weighted combination
        overall = (
            snr_score * 0.3 +
            (100 - clipping_penalty) * 0.2 +
            dynamic_range_score * 0.2 +
            (100 - silence_penalty) * 0.1 +
            frequency_score * 0.2
        )
        
        return min(100.0, max(0.0, overall))
        
    except Exception:
        return 50.0

def detect_speech_segments(audio_data: np.ndarray, samplerate: int) -> List[Tuple[float, float]]:
    """
    Detect speech segments in audio for better analysis targeting
    """
    try:
        # Use librosa's voice activity detection approach
        frame_length = int(0.025 * samplerate)  # 25ms
        hop_length = int(0.010 * samplerate)    # 10ms
        
        # Calculate RMS energy for each frame
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate spectral centroid (brightness) 
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=samplerate, hop_length=hop_length)[0]
        
        # Combine energy and spectral features for voice activity detection
        energy_threshold = np.percentile(rms, 30)  # Bottom 30% is likely silence
        centroid_threshold = 500  # Hz - below this is likely noise/silence
        
        # Detect voice activity
        voice_activity = (rms > energy_threshold) & (spectral_centroid > centroid_threshold)
        
        # Convert frame indices to time segments
        frame_times = librosa.frames_to_time(np.arange(len(voice_activity)), sr=samplerate, hop_length=hop_length)
        
        # Find continuous speech segments
        segments = []
        start_time = None
        
        for i, (is_speech, time) in enumerate(zip(voice_activity, frame_times)):
            if is_speech and start_time is None:
                start_time = time
            elif not is_speech and start_time is not None:
                # End of speech segment
                if time - start_time > 0.1:  # Minimum 100ms segment
                    segments.append((start_time, time))
                start_time = None
        
        # Handle case where speech continues to end
        if start_time is not None:
            segments.append((start_time, frame_times[-1]))
        
        return segments
        
    except Exception as e:
        print(f"⚠️ Speech segment detection failed: {e}")
        # Return whole audio as single segment
        return [(0.0, len(audio_data) / samplerate)] 