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
ATTENUATION_LIMIT_DB = 12.0
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


def apply_neural_noise_reduction(
    audio_bytes: bytes,
    sample_rate: int,
    channels: int,
) -> bytes:
    """Denoise interleaved signed 16-bit PCM while preserving its audio format.

    The 48 kHz high-resolution DPDFNet model is selected explicitly. DPDFNet
    handles any resampling internally (for example, for a 44.1 kHz capture) and
    returns the original sample rate. Stereo channels are processed separately
    to avoid downmixing. Errors are raised so the caller can send the untouched
    original recording instead of partially processed audio.
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
    output_float = np.empty_like(input_float, dtype=np.float32)

    for channel in range(channels):
        enhanced = dpdfnet.enhance(
            audio=np.ascontiguousarray(input_float[:, channel]),
            sample_rate=int(sample_rate),
            model=MODEL_NAME,
            onnx_path=str(model_path),
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

    output_float = np.clip(output_float, -1.0, 32767.0 / 32768.0)
    output_pcm = np.rint(output_float * 32768.0).astype("<i2")
    return output_pcm.reshape(-1).tobytes()
