import logging
import numpy as np

def apply_noise_gate(audio_bytes: bytes, rate: int, channels: int, threshold_db: float = -40.0, attenuation: float = 0.1, *, strict: bool = False) -> bytes:
    """
    Legacy callers get the original audio back on error; strict re-raises.
    The neural fallback path needs to know when the gate did not run at all,
    so it passes strict=True and handles the announcement itself.

    Applies an Automatic Gain Control (AGC) and Dynamic Noise Gate filter
    to raw 16-bit PCM audio bytes using NumPy.
    This attenuates background noise during silences (noise gate) and
    levels/compresses the voice volume during speech (AGC), producing
    a clean, WebRTC-like compressed voice sound.
    """
    try:
        # Load raw bytes as 16-bit signed PCM samples
        samples = np.frombuffer(audio_bytes, dtype=np.int16).copy()
        
        # 20ms block size
        block_size = int(rate * 0.02) * channels
        if block_size <= 0:
            return audio_bytes
            
        num_samples = len(samples)
        
        # Compute RMS for all blocks first to estimate noise floor
        rms_values = []
        for i in range(0, num_samples, block_size):
            block = samples[i:i+block_size]
            if len(block) == 0:
                continue
            rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
            rms_values.append(rms)

        if not rms_values:
            return audio_bytes

        # Use the 15th percentile of RMS to find the background noise floor (pauses)
        noise_floor = float(np.percentile(rms_values, 15))
        
        # Threshold is 2.5x the noise floor, bounded between -45dB (180) and -22dB (2600)
        threshold = max(180.0, min(2600.0, noise_floor * 2.5))
        
        logging.info("[audio_processing] Dynamic AGC + Noise Gate: noise floor RMS = %.2f, threshold = %.2f", noise_floor, threshold)
        
        # Envelope parameters for gain smoothing (faster attack/release response)
        gain = 1.0
        attack = 0.35
        release = 0.15
        
        for i in range(0, num_samples, block_size):
            block = samples[i:i+block_size]
            if len(block) == 0:
                continue
            
            rms = rms_values[i // block_size] if (i // block_size) < len(rms_values) else np.sqrt(np.mean(block.astype(np.float32) ** 2))
            
            # Determine target gain factor
            if rms >= threshold:
                # Speech detected: Auto Gain Control to target 2000 RMS (comfortable volume)
                target_gain = 2000.0 / (rms + 1e-5)
                # Bound target gain between 0.5 (compression) and 2.5 (boosting)
                target_gain = max(0.5, min(2.5, target_gain))
            else:
                # Silence/noise detected: attenuate
                target_gain = attenuation
            
            # Smooth the gain transition (prevents audio clicking)
            if target_gain > gain:
                gain = gain + attack * (target_gain - gain)
            else:
                gain = gain + release * (target_gain - gain)
                
            # Apply smoothed gain to the block and prevent clipping
            # 16-bit PCM limit is [-32768, 32767]
            processed_block = np.clip(block.astype(np.float32) * gain, -32768.0, 32767.0)
            samples[i:i+block_size] = processed_block.astype(np.int16)
            
        return samples.tobytes()
    except Exception as e:
        logging.error("[audio_processing] Error applying dynamic AGC + Noise Gate: %s", e)
        if strict:
            raise
        return audio_bytes

def detect_silence_spans(pcm_bytes: bytes, rate: int, channels: int = 1,
                         threshold_db: float = -45.0, min_silence_s: float = 0.7,
                         pad_s: float = 0.15, window_s: float = 0.05) -> list:
    """Silent spans of a 16-bit PCM buffer, as (start_s, end_s) pairs.

    Used by voice-message playback to skip over dead air ("skip silences"
    setting). A window whose RMS stays under threshold_db counts as silence;
    runs of silent windows at least min_silence_s long become a span, shrunk
    by pad_s on each side so the skip never clips a word's attack or release.
    Spans never cover the first or last pad_s of the audio, so playback still
    starts and ends on real content. Mono downmix is the caller's job when
    channels > 1. Any unexpected input returns [] — skipping nothing is the
    safe failure for a playback convenience.
    """
    try:
        if not pcm_bytes or rate <= 0 or channels <= 0:
            return []
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        if channels > 1:
            usable = audio.size - (audio.size % channels)
            if usable <= 0:
                return []
            audio = audio[:usable].reshape(-1, channels).mean(axis=1)
        audio = audio.astype(np.float32) / 32768.0
        win = max(1, int(rate * window_s))
        n_win = audio.size // win
        if n_win == 0:
            return []
        frames = audio[: n_win * win].reshape(n_win, win)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1e-10))
        silent = db < threshold_db
        min_windows = max(1, int(round(min_silence_s / window_s)))
        spans = []
        run_start = None
        for i, is_silent in enumerate(silent):
            if is_silent and run_start is None:
                run_start = i
            elif not is_silent and run_start is not None:
                if i - run_start >= min_windows:
                    spans.append((run_start * window_s, i * window_s))
                run_start = None
        if run_start is not None and n_win - run_start >= min_windows:
            spans.append((run_start * window_s, n_win * window_s))
        total = audio.size / rate
        padded = []
        for start, end in spans:
            s = min(max(start + pad_s, pad_s), total)
            e = min(max(end - pad_s, s), total - pad_s)
            if e - s >= 0.2:
                padded.append((s, e))
        return padded
    except Exception:
        logging.exception("[audio_processing] detect_silence_spans failed")
        return []

