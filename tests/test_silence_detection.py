"""detect_silence_spans(): dead-air detection for the "skip silences"
voice-message playback option. Pure numpy — no wx, no window."""

import numpy as np

from core.audio_processing import detect_silence_spans

RATE = 8000


def _pcm(segments, channels=1):
    """Build 16-bit PCM from (seconds, "tone"|"silence") segments."""
    parts = []
    for seconds, kind in segments:
        n = int(RATE * seconds)
        if kind == "tone":
            t = np.arange(n) / RATE
            wave = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        else:
            wave = np.zeros(n, dtype=np.int16)
        parts.append(wave)
    mono = np.concatenate(parts)
    if channels > 1:
        mono = np.repeat(mono[:, None], channels, axis=1)
    return mono.tobytes()


class TestSilenceDetection:
    def test_continuous_speech_has_no_spans(self):
        assert detect_silence_spans(_pcm([(5.0, "tone")]), RATE) == []

    def test_middle_silence_is_found_with_padding(self):
        pcm = _pcm([(2.0, "tone"), (2.0, "silence"), (2.0, "tone")])
        spans = detect_silence_spans(pcm, RATE)
        assert len(spans) == 1
        start, end = spans[0]
        # The raw silence is 2.0–4.0 s; padding shrinks it on both sides.
        assert 2.0 <= start < 2.3
        assert 3.7 < end <= 4.0

    def test_a_short_pause_is_kept(self):
        pcm = _pcm([(2.0, "tone"), (0.3, "silence"), (2.0, "tone")])
        assert detect_silence_spans(pcm, RATE) == []

    def test_stereo_is_downmixed(self):
        pcm = _pcm([(1.0, "tone"), (2.0, "silence"), (1.0, "tone")], channels=2)
        spans = detect_silence_spans(pcm, RATE, channels=2)
        assert len(spans) == 1

    def test_garbage_and_empty_input_skip_nothing(self):
        assert detect_silence_spans(b"", RATE) == []
        assert detect_silence_spans(b"\x00\x01", RATE) == []
        assert detect_silence_spans(_pcm([(0.01, "tone")]), RATE) == []

    def test_spans_never_cover_the_whole_clip(self):
        pcm = _pcm([(3.0, "silence"), (1.0, "tone")])
        for start, end in detect_silence_spans(pcm, RATE):
            assert start >= 0.1
            assert end <= 4.0
