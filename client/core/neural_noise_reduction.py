"""Offline DPDFNet enhancement for outgoing WinZapp voice messages.

This path is intentionally separate from the real-time call suppressor.
Voice messages are complete recordings, so the batch API can process them
without adding latency to live calls. The model runs locally through ONNX
Runtime and uses the 48 kHz DPDFNet-2 high-resolution model.

The model is downloaded on first use into DPDFNet's user cache. Callers must
treat any exception as a request to keep/send the original audio.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "dpdfnet2_48khz_hr"
ATTENUATION_LIMIT_DB = 12.0


def _load_enhancer():
    """Load the documented Python API lazily so WinZapp starts without model setup."""
    from dpdfnet.api import enhance

    return enhance


def apply_dpdfnet_noise_reduction(
    audio_bytes: bytes,
    sample_rate: int,
    channels: int,
) -> bytes:
    """Enhance interleaved signed 16-bit PCM and return the same PCM format.

    Audio stays at its original sample rate and channel layout at the API
    boundary. DPDFNet internally uses the selected 48 kHz model; non-48 kHz
    inputs are resampled by DPDFNet and returned at their original rate.

    Raises on malformed PCM, unavailable dependencies/model downloads, invalid
    model output, or inference failures. The recording pipeline catches these
    exceptions and safely retains the original bytes.
    """
    if not audio_bytes:
        return audio_bytes
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels not in (1, 2):
        raise ValueError("Only mono and stereo PCM are supported")
    if len(audio_bytes) % (2 * channels):
        raise ValueError("PCM byte length is not aligned to the channel frame size")

    pcm = np.frombuffer(audio_bytes, dtype="<i2")
    if pcm.size == 0:
        return audio_bytes

    frames = pcm.reshape(-1, channels)
    source = frames.astype(np.float32) / 32768.0
    enhanced = np.empty_like(source, dtype=np.float32)
    enhance = _load_enhancer()

    # DPDFNet's public batch API intentionally downmixes stereo. Process each
    # channel separately so an opted-in stereo voice message remains stereo.
    for channel in range(channels):
        output = np.asarray(
            enhance(
                source[:, channel],
                sample_rate=int(sample_rate),
                model=MODEL_NAME,
                attn_limit_db=ATTENUATION_LIMIT_DB,
                verbose=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        if output.size != source.shape[0]:
            raise ValueError("DPDFNet returned a different number of samples")
        if not np.all(np.isfinite(output)):
            raise ValueError("DPDFNet returned non-finite audio samples")
        enhanced[:, channel] = output

    output_pcm = np.rint(np.clip(enhanced, -1.0, 32767.0 / 32768.0) * 32768.0)
    return output_pcm.astype("<i2").tobytes()
