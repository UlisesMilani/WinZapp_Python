"""Offline neural noise reduction for recorded WinZapp voice messages.

DPDFNet is used here as a post-recording processor, not in the live capture
callback. The 48 kHz model avoids forcing recordings through a 16 kHz model.
The package downloads its ONNX model on first use and caches it locally.
"""

from __future__ import annotations

import numpy as np

MODEL_NAME = "dpdfnet2_48khz_hr"
ATTENUATION_LIMIT_DB = 12.0


def apply_neural_noise_reduction(
    audio_bytes: bytes,
    sample_rate: int,
    channels: int,
) -> bytes:
    """Denoise interleaved signed 16-bit PCM while preserving its format.

    Each channel is enhanced independently because DPDFNet's public `enhance`
    API downmixes multichannel input to mono. The returned PCM preserves the
    original sample rate, channel count, sample count, and interleaving. DPDFNet
    performs any necessary resampling internally (for example, for 44.1 kHz
    device captures).

    Raises on invalid input or failed/invalid model output. Callers can then
    retain the original bytes rather than accidentally sending partial output.
    """
    if not audio_bytes:
        return audio_bytes
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")

    bytes_per_frame = channels * 2  # signed 16-bit PCM
    if len(audio_bytes) % bytes_per_frame:
        raise ValueError("PCM byte length is not aligned to complete audio frames")

    # Import lazily so that a packaging or runtime issue with the optional
    # processor never prevents WinZapp itself from starting.
    import dpdfnet

    input_pcm = np.frombuffer(audio_bytes, dtype="<i2").reshape(-1, channels)
    input_float = input_pcm.astype(np.float32) / 32768.0
    output_float = np.empty_like(input_float, dtype=np.float32)

    for channel in range(channels):
        enhanced = dpdfnet.enhance(
            audio=np.ascontiguousarray(input_float[:, channel]),
            sample_rate=int(sample_rate),
            model=MODEL_NAME,
            attn_limit_db=ATTENUATION_LIMIT_DB,
        )
        enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)
        if enhanced.size != input_float.shape[0]:
            raise RuntimeError(
                "DPDFNet changed the sample count "
                f"({input_float.shape[0]} -> {enhanced.size})"
            )
        if not np.isfinite(enhanced).all():
            raise RuntimeError("DPDFNet returned non-finite audio samples")
        output_float[:, channel] = enhanced

    # Convert back to signed PCM16 without changing channel order or clipping
    # values outside the representable range.
    output_float = np.clip(output_float, -1.0, 32767.0 / 32768.0)
    output_pcm = np.rint(output_float * 32768.0).astype("<i2")
    return output_pcm.reshape(-1).tobytes()
