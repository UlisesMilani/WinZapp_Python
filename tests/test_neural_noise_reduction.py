from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.neural_noise_reduction import (
    ATTENUATION_LIMIT_DB,
    MODEL_NAME,
    apply_neural_noise_reduction,
)


def _install_fake_dpdfnet(monkeypatch, enhance):
    monkeypatch.setitem(sys.modules, "dpdfnet", SimpleNamespace(enhance=enhance))


def test_mono_processing_uses_48khz_model_and_preserves_pcm_length(monkeypatch):
    calls = []

    def enhance(audio, sample_rate, *, model, attn_limit_db):
        calls.append((sample_rate, model, attn_limit_db, np.asarray(audio).copy()))
        return np.asarray(audio, dtype=np.float32) * 0.5

    _install_fake_dpdfnet(monkeypatch, enhance)
    pcm = np.array([1000, -2000, 32767, -32768], dtype="<i2")
    result = apply_neural_noise_reduction(pcm.tobytes(), 48000, 1)

    expected = np.array([500, -1000, 16384, -16384], dtype="<i2")
    np.testing.assert_array_equal(np.frombuffer(result, dtype="<i2"), expected)
    assert len(result) == len(pcm.tobytes())
    assert len(calls) == 1
    assert calls[0][0] == 48000
    assert calls[0][1] == MODEL_NAME == "dpdfnet2_48khz_hr"
    assert calls[0][2] == ATTENUATION_LIMIT_DB == 12.0


def test_stereo_channels_are_enhanced_independently_and_reinterleaved(monkeypatch):
    seen_channels = []

    def enhance(audio, sample_rate, *, model, attn_limit_db):
        seen_channels.append(np.asarray(audio).copy())
        return np.asarray(audio, dtype=np.float32) * 0.5

    _install_fake_dpdfnet(monkeypatch, enhance)
    pcm = np.array([[1000, -1000], [2000, -2000]], dtype="<i2")
    result = apply_neural_noise_reduction(pcm.reshape(-1).tobytes(), 48000, 2)

    expected = np.array([[500, -500], [1000, -1000]], dtype="<i2")
    np.testing.assert_array_equal(
        np.frombuffer(result, dtype="<i2").reshape(-1, 2),
        expected,
    )
    assert len(seen_channels) == 2
    np.testing.assert_allclose(seen_channels[0], pcm[:, 0] / 32768.0)
    np.testing.assert_allclose(seen_channels[1], pcm[:, 1] / 32768.0)


def test_malformed_pcm_is_rejected_before_model_import():
    with pytest.raises(ValueError, match="aligned"):
        apply_neural_noise_reduction(b"\x01", 48000, 1)


def test_wrong_length_model_output_is_rejected(monkeypatch):
    _install_fake_dpdfnet(
        monkeypatch,
        lambda audio, sample_rate, *, model, attn_limit_db: np.zeros(1, dtype=np.float32),
    )
    pcm = np.array([100, 200, 300], dtype="<i2")
    with pytest.raises(RuntimeError, match="sample count"):
        apply_neural_noise_reduction(pcm.tobytes(), 48000, 1)


def test_non_finite_model_output_is_rejected(monkeypatch):
    _install_fake_dpdfnet(
        monkeypatch,
        lambda audio, sample_rate, *, model, attn_limit_db: np.full_like(audio, np.nan),
    )
    pcm = np.array([100, 200, 300], dtype="<i2")
    with pytest.raises(RuntimeError, match="non-finite"):
        apply_neural_noise_reduction(pcm.tobytes(), 48000, 1)
