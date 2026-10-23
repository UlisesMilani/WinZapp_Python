from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import core.neural_noise_reduction as denoiser
from core.neural_noise_reduction import (
    MODEL_NAME,
    apply_neural_noise_reduction,
)


def _install_fake_dpdfnet(monkeypatch, enhance, tmp_path):
    model_path = tmp_path / denoiser.MODEL_FILENAME
    model_path.write_bytes(b"test model placeholder")
    monkeypatch.setattr(denoiser, "_ensure_model_file", lambda: model_path)
    monkeypatch.setitem(sys.modules, "dpdfnet", SimpleNamespace(enhance=enhance))
    return model_path


def test_mono_processing_uses_full_pinned_48khz_enhancement_and_preserves_pcm_length(monkeypatch, tmp_path):
    calls = []

    def enhance(audio, sample_rate, *, model, onnx_path, attn_limit_db):
        calls.append((sample_rate, model, onnx_path, attn_limit_db, np.asarray(audio).copy()))
        return np.asarray(audio, dtype=np.float32) * 0.5

    model_path = _install_fake_dpdfnet(monkeypatch, enhance, tmp_path)
    pcm = np.array([1000, -2000, 32767, -32768], dtype="<i2")
    result = apply_neural_noise_reduction(pcm.tobytes(), 48000, 1)

    expected = np.array([500, -1000, 16384, -16384], dtype="<i2")
    np.testing.assert_array_equal(np.frombuffer(result, dtype="<i2"), expected)
    assert len(result) == len(pcm.tobytes())
    assert len(calls) == 1
    assert calls[0][0] == 48000
    assert calls[0][1] == MODEL_NAME == "dpdfnet2_48khz_hr"
    assert calls[0][2] == str(model_path)
    assert calls[0][3] is None  # no attenuation limit: use full DPDFNet enhancement


def test_44100hz_input_keeps_its_original_sample_rate(monkeypatch, tmp_path):
    calls = []

    def enhance(audio, sample_rate, *, model, onnx_path, attn_limit_db):
        calls.append((sample_rate, model, onnx_path))
        return np.asarray(audio, dtype=np.float32) * 0.5

    _install_fake_dpdfnet(monkeypatch, enhance, tmp_path)
    pcm = np.array([1000, -2000, 12000, -12000], dtype="<i2")
    result = apply_neural_noise_reduction(pcm.tobytes(), 44100, 1)

    assert len(result) == len(pcm.tobytes())
    assert calls[0][0] == 44100
    assert calls[0][1] == "dpdfnet2_48khz_hr"


def test_stereo_uses_single_mono_inference_and_shared_spectral_mask(monkeypatch, tmp_path):
    seen_calls = []

    def enhance(audio, sample_rate, *, model, onnx_path, attn_limit_db):
        seen_calls.append(np.asarray(audio).copy())
        return np.asarray(audio, dtype=np.float32) * 0.5

    _install_fake_dpdfnet(monkeypatch, enhance, tmp_path)
    pcm = np.array([[1000, -1000], [2000, -2000]], dtype="<i2")
    result = apply_neural_noise_reduction(pcm.reshape(-1).tobytes(), 48000, 2)

    expected = np.array([[500, -500], [1000, -1000]], dtype="<i2")
    np.testing.assert_array_equal(np.frombuffer(result, dtype="<i2").reshape(-1, 2), expected)
    assert len(seen_calls) == 1


def test_stereo_spectral_mask_preserves_panning_and_interchannel_phase(monkeypatch, tmp_path):
    calls = []
    rate = 48000
    t = np.linspace(0.0, 0.25, rate // 4, endpoint=False)
    voice = np.sin(2.0 * np.pi * 440.0 * t)
    left = (0.6 * voice * 32767.0).astype("<i2")
    right = (0.3 * voice * 32767.0).astype("<i2")
    stereo_pcm = np.column_stack([left, right])

    def enhance(audio, sample_rate, *, model, onnx_path, attn_limit_db):
        calls.append(np.asarray(audio).copy())
        return np.asarray(audio, dtype=np.float32) * 0.5

    _install_fake_dpdfnet(monkeypatch, enhance, tmp_path)
    result = apply_neural_noise_reduction(stereo_pcm.reshape(-1).tobytes(), rate, 2)
    out = np.frombuffer(result, dtype="<i2").reshape(-1, 2).astype(np.float32)

    assert len(calls) == 1
    np.testing.assert_allclose(out[:, 0], left.astype(np.float32) * 0.5, atol=2.0)
    np.testing.assert_allclose(out[:, 1], right.astype(np.float32) * 0.5, atol=2.0)


def test_digital_silence_is_returned_unchanged_without_model_download(monkeypatch):
    def should_not_download():
        pytest.fail("the model should not be downloaded for digital silence")

    monkeypatch.setattr(denoiser, "_ensure_model_file", should_not_download)
    pcm = np.zeros(960, dtype="<i2")
    assert apply_neural_noise_reduction(pcm.tobytes(), 48000, 1) == pcm.tobytes()


def test_malformed_pcm_is_rejected_before_model_import():
    with pytest.raises(ValueError, match="aligned"):
        apply_neural_noise_reduction(b"\x01", 48000, 1)


def test_wrong_length_model_output_is_rejected(monkeypatch, tmp_path):
    _install_fake_dpdfnet(
        monkeypatch,
        lambda audio, sample_rate, *, model, onnx_path, attn_limit_db: np.zeros(1, dtype=np.float32),
        tmp_path,
    )
    pcm = np.array([100, 200, 300], dtype="<i2")
    with pytest.raises(RuntimeError, match="sample count"):
        apply_neural_noise_reduction(pcm.tobytes(), 48000, 1)


def test_non_finite_model_output_is_rejected(monkeypatch, tmp_path):
    _install_fake_dpdfnet(
        monkeypatch,
        lambda audio, sample_rate, *, model, onnx_path, attn_limit_db: np.full_like(audio, np.nan),
        tmp_path,
    )
    pcm = np.array([100, 200, 300], dtype="<i2")
    with pytest.raises(RuntimeError, match="non-finite"):
        apply_neural_noise_reduction(pcm.tobytes(), 48000, 1)


def test_pinned_model_download_rejects_wrong_hash_and_cleans_temporary_file(monkeypatch, tmp_path):
    final_path = tmp_path / denoiser.MODEL_FILENAME
    monkeypatch.setattr(denoiser, "_model_path", lambda: final_path)
    monkeypatch.setattr(denoiser, "urlopen", lambda _url, timeout: io.BytesIO(b"wrong hash"))
    with pytest.raises(RuntimeError, match="integrity verification"):
        denoiser._ensure_model_file()
    assert not final_path.exists()
    assert not list(tmp_path.glob("*.part"))


def test_model_hash_matches_pinned_revision_metadata():
    assert len(denoiser.MODEL_SHA256) == 64
    assert denoiser.MODEL_SIZE_BYTES == 10_493_337
    assert denoiser.MODEL_REVISION in denoiser.MODEL_URL
