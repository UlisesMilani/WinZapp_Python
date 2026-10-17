from __future__ import annotations

import numpy as np
import pytest

from core import neural_noise_reduction as nr


def test_empty_audio_is_returned_without_loading_model(monkeypatch):
    monkeypatch.setattr(
        nr, "_load_enhancer",
        lambda: pytest.fail("empty audio must not initialize DPDFNet"),
    )
    assert nr.apply_dpdfnet_noise_reduction(b"", 48000, 1) == b""


@pytest.mark.parametrize(
    ("sample_rate", "channels", "data"),
    [
        (0, 1, b"\x00\x00"),
        (48000, 3, b"\x00\x00\x00\x00\x00\x00"),
        (48000, 2, b"\x00\x00"),
    ],
)
def test_invalid_pcm_metadata_is_rejected(sample_rate, channels, data):
    with pytest.raises(ValueError):
        nr.apply_dpdfnet_noise_reduction(data, sample_rate, channels)


def test_mono_processing_uses_48khz_model_and_preserves_length(monkeypatch):
    calls = []

    def fake_enhance(audio, **kwargs):
        calls.append(kwargs)
        return np.asarray(audio, dtype=np.float32) * 0.5

    monkeypatch.setattr(nr, "_load_enhancer", lambda: fake_enhance)
    samples = np.array([1000, -1000, 2000, -2000], dtype="<i2")

    result = nr.apply_dpdfnet_noise_reduction(samples.tobytes(), 48000, 1)

    out = np.frombuffer(result, dtype="<i2")
    np.testing.assert_array_equal(out, np.array([500, -500, 1000, -1000], dtype="<i2"))
    assert len(out) == len(samples)
    assert calls == [{
        "sample_rate": 48000,
        "model": "dpdfnet2_48khz_hr",
        "attn_limit_db": 12.0,
        "verbose": False,
    }]


def test_stereo_channels_are_processed_independently_and_preserved(monkeypatch):
    calls = []

    def fake_enhance(audio, **kwargs):
        calls.append(np.asarray(audio).copy())
        return np.asarray(audio, dtype=np.float32) * 0.5

    monkeypatch.setattr(nr, "_load_enhancer", lambda: fake_enhance)
    stereo = np.array([[1000, -2000], [3000, -4000], [5000, -6000]], dtype="<i2")

    result = nr.apply_dpdfnet_noise_reduction(stereo.tobytes(), 48000, 2)

    out = np.frombuffer(result, dtype="<i2").reshape(-1, 2)
    np.testing.assert_array_equal(
        out,
        np.array([[500, -1000], [1500, -2000], [2500, -3000]], dtype="<i2"),
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], stereo[:, 0] / 32768.0)
    np.testing.assert_array_equal(calls[1], stereo[:, 1] / 32768.0)


def test_non_finite_model_output_is_rejected(monkeypatch):
    monkeypatch.setattr(
        nr, "_load_enhancer",
        lambda: lambda audio, **_kwargs: np.full_like(audio, np.nan, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="non-finite"):
        nr.apply_dpdfnet_noise_reduction(np.array([1, 2], dtype="<i2").tobytes(), 48000, 1)


def test_channel_length_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(
        nr, "_load_enhancer",
        lambda: lambda audio, **_kwargs: np.zeros(len(audio) + 1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="different number of samples"):
        nr.apply_dpdfnet_noise_reduction(np.array([1, 2], dtype="<i2").tobytes(), 48000, 1)
