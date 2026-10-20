"""Offline neural noise reduction for recorded WinZapp voice messages.

DPDFNet runs after recording on a worker thread. WinZapp pins the pretrained
48 kHz ONNX artifact and verifies its SHA-256 before use, rather than accepting
a model that silently changes on the upstream repository's default branch.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from pathlib import Path
from urllib.request import urlopen

import numpy as np

MODEL_NAME = "dpdfnet2_48khz_hr"
MODEL_FILENAME = "dpdfnet2_48khz_hr.onnx"
MODEL_REVISION = "c7ac7b249ff5e17fa606794dc4f68ed9a544834f"
MODEL_URL = (
    "https://huggingface.co/Ceva-IP/DPDFNet/resolve/"
    f"{MODEL_REVISION}/onnx/{MODEL_FILENAME}?download=true"
)
MODEL_SIZE_BYTES = 10_493_337
MODEL_SHA256 = "7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b"

_DOWNLOAD_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)
_MASK_EPS = 1e-7
_STFT_FRAME_LEN = 512
_STFT_HOP_LEN = _STFT_FRAME_LEN // 2
_SINE_WINDOW = np.sin(
    np.pi * (np.arange(_STFT_FRAME_LEN, dtype=np.float32) + 0.5) / _STFT_FRAME_LEN
)


def _model_path() -> Path:
    """Store one verified model for this WinZapp installation, across accounts."""
    from app_paths import global_dir

    return Path(global_dir("models", MODEL_FILENAME))


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _is_valid_model(path: Path) -> bool:
    try:
        digest, size = _file_sha256(path)
    except OSError:
        return False
    return size == MODEL_SIZE_BYTES and digest == MODEL_SHA256


def _ensure_model_file() -> Path:
    """Return the pinned ONNX model; download and verify it on first use."""
    path = _model_path()
    with _DOWNLOAD_LOCK:
        if _is_valid_model(path):
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{MODEL_FILENAME}.",
            suffix=".part",
            dir=str(path.parent),
        )
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        size = 0
        try:
            _LOGGER.info("Downloading the pinned DPDFNet 48 kHz model")
            with os.fdopen(fd, "wb") as output, urlopen(MODEL_URL, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())

            actual_hash = digest.hexdigest()
            if size != MODEL_SIZE_BYTES or actual_hash != MODEL_SHA256:
                raise RuntimeError(
                    "Downloaded DPDFNet model failed integrity verification "
                    f"(size={size}, sha256={actual_hash})"
                )

            os.replace(temp_path, path)
            _LOGGER.info("Pinned DPDFNet model downloaded and verified")
            return path
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise


def _enhance_mono(
    dpdfnet_module,
    mono_audio: np.ndarray,
    sample_rate: int,
    model_path: Path,
) -> np.ndarray:
    """Run one mono DPDFNet inference pass and validate the returned buffer."""
    enhanced = dpdfnet_module.enhance(
        audio=np.ascontiguousarray(mono_audio, dtype=np.float32),
        sample_rate=int(sample_rate),
        model=MODEL_NAME,
        onnx_path=str(model_path),
        attn_limit_db=None,
    )
    enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)
    if enhanced.size != mono_audio.shape[0]:
        raise RuntimeError(
            "DPDFNet changed the sample count "
            f"({mono_audio.shape[0]} -> {enhanced.size})"
        )
    if not np.isfinite(enhanced).all():
        raise RuntimeError("DPDFNet returned non-finite audio samples")
    return enhanced


def _stereo_reference_signal(input_float: np.ndarray) -> np.ndarray:
    """Build a mono reference signal for stereo/multichannel gain estimation.

    Uses the Mid (average) signal by default, but falls back to the Side
    difference when channels are predominantly out of phase so anti-phase
    signals do not cancel to silence before model inference.
    """
    mid = np.mean(input_float, axis=1, dtype=np.float32)
    if input_float.shape[1] == 2:
        side = 0.5 * (input_float[:, 0] - input_float[:, 1])
        if float(np.dot(mid, mid)) < float(np.dot(side, side)):
            return np.ascontiguousarray(side, dtype=np.float32)
    return np.ascontiguousarray(mid, dtype=np.float32)


def _apply_shared_spectral_mask(
    input_float: np.ndarray,
    ref: np.ndarray,
    enhanced_ref: np.ndarray,
) -> np.ndarray:
    """Apply the mono DPDFNet suppression mask equally to all channels.

    Uses a 50%-overlapped Sine-window STFT/iSTFT (satisfies exact Princen-Bradley
    perfect reconstruction, w^2[n] + w^2[n + N/2] = 1) to compute a real-valued
    magnitude mask from the mono reference and apply it identically to every
    channel. This preserves inter-channel phase and stereo panning while
    requiring only a single mono neural inference pass.
    """
    num_samples, num_channels = input_float.shape
    frame_len = _STFT_FRAME_LEN
    hop_len = _STFT_HOP_LEN
    window = _SINE_WINDOW

    pad_left = hop_len
    min_total = pad_left + num_samples + hop_len
    num_frames = int(np.ceil((min_total - frame_len) / hop_len)) + 1
    padded_len = (num_frames - 1) * hop_len + frame_len
    pad_right = padded_len - pad_left - num_samples

    ref_pad = np.pad(ref, (pad_left, pad_right), mode="constant")
    enh_pad = np.pad(enhanced_ref, (pad_left, pad_right), mode="constant")
    ch_pad = np.pad(input_float, ((pad_left, pad_right), (0, 0)), mode="constant")

    starts = (
        np.arange(num_frames)[:, None] * hop_len
        + np.arange(frame_len)[None, :]
    )

    z_ref = np.fft.rfft(ref_pad[starts] * window, axis=1)
    z_enh = np.fft.rfft(enh_pad[starts] * window, axis=1)
    gain_mask = np.clip(np.abs(z_enh) / (np.abs(z_ref) + _MASK_EPS), 0.0, 1.0)

    ch_frames = ch_pad[starts] * window[:, None]
    z_ch = np.fft.rfft(ch_frames, axis=1)
    z_ch_clean = z_ch * gain_mask[:, :, None]
    ch_frames_clean = (
        np.fft.irfft(z_ch_clean, n=frame_len, axis=1) * window[:, None]
    )

    out_pad = np.zeros((padded_len, num_channels), dtype=np.float32)
    for idx in range(num_frames):
        s = idx * hop_len
        out_pad[s : s + frame_len] += ch_frames_clean[idx]

    return out_pad[pad_left : pad_left + num_samples]


def apply_neural_noise_reduction(
    audio_bytes: bytes,
    sample_rate: int,
    channels: int,
) -> bytes:
    """Denoise interleaved signed 16-bit PCM while preserving its audio format.

    The 48 kHz high-resolution DPDFNet model is selected explicitly. DPDFNet
    handles any resampling internally (for example, for a 44.1 kHz capture) and
    returns the original sample rate. Mono audio is enhanced directly; stereo
    audio runs a single mono inference pass and applies the resulting real-valued
    time-frequency suppression mask across both channels to preserve stereo phase
    coherence and halve CPU cost. Errors are raised so the caller can send the
    untouched original recording instead of partially processed audio.
    """
    if not audio_bytes:
        return audio_bytes
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")

    bytes_per_frame = channels * 2
    if len(audio_bytes) % bytes_per_frame:
        raise ValueError("PCM byte length is not aligned to complete audio frames")

    input_pcm = np.frombuffer(audio_bytes, dtype="<i2").reshape(-1, channels)
    if input_pcm.size == 0:
        return audio_bytes

    # Keep digital silence byte-identical and avoid downloading the model for it.
    if int(np.max(np.abs(input_pcm.astype(np.int32)))) <= 2:
        return audio_bytes

    # Import lazily so a runtime or packaging problem never prevents startup.
    import dpdfnet

    model_path = _ensure_model_file()
    input_float = input_pcm.astype(np.float32) / 32768.0

    if channels == 1:
        output_float = _enhance_mono(
            dpdfnet, input_float[:, 0], sample_rate, model_path
        ).reshape(-1, 1)
    else:
        ref = _stereo_reference_signal(input_float)
        enhanced_ref = _enhance_mono(dpdfnet, ref, sample_rate, model_path)
        output_float = _apply_shared_spectral_mask(
            input_float, ref, enhanced_ref
        )

    output_float = np.clip(output_float, -1.0, 32767.0 / 32768.0)
    output_pcm = np.rint(output_float * 32768.0).astype("<i2")
    return output_pcm.reshape(-1).tobytes()
