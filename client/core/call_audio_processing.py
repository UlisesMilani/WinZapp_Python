"""Echo cancellation and noise suppression for the outgoing call microphone.

Both run in WebRTC's own audio processing module (AEC3 and its noise
suppressor), the engine Chrome, Google Meet and WhatsApp's own clients use,
through the ``livekit`` package's binding. The page's WebRTC stack cannot do
it for us: WhatsApp Web only ever sees WinZapp's synthetic microphone track
(callMediaBridge.ts never opens the physical microphone), and the speaker is
played by Python, so Chrome's AEC has neither signal. Python owns both and
must cancel here.

This replaced a hand-written numpy canceller (a frequency-domain Kalman filter
behind a correlation gate) that measured 5-11 dB of echo reduction on the
same synthetic speech where AEC3 gives ~32 dB within the first seconds and
~70 dB once converged. On a laptop speaker or a mixer loopback the echo is
nearly as loud as the user's voice, so 10 dB left it plainly audible: "it
cancels nothing", as testers put it. AEC3 also carries the residual echo
suppressor and its own delay estimator, the two parts the old one lacked.

The module processes exactly 10 ms frames. Far-end audio (what the speaker
played) goes in through ``push_reference`` and near-end audio through
``process``, both on the microphone sender thread, in the order they happened.
"""

from __future__ import annotations

import logging

import numpy as np

APM_SAMPLE_RATE = 48_000
APM_FRAME = APM_SAMPLE_RATE // 100  # 10 ms, the only size the module accepts


def _default_apm_factory(*, echo_cancellation: bool, noise_suppression: bool):
    from livekit import rtc

    return rtc.AudioProcessingModule(
        echo_cancellation=echo_cancellation,
        noise_suppression=noise_suppression,
        # Rumble and DC offset below speech; WebRTC runs it in front of AEC3.
        high_pass_filter=True,
        # Off: it pumps the level between words, and the user already chose
        # their microphone gain in Windows.
        auto_gain_control=False,
    )


def _frame(samples: np.ndarray):
    from livekit import rtc

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return rtc.AudioFrame(bytearray(pcm.tobytes()), APM_SAMPLE_RATE, 1, APM_FRAME)


def _frame_samples(frame) -> np.ndarray:
    return np.frombuffer(bytes(frame.data), dtype="<i2").astype(np.float32) / 32768.0


class CallAudioProcessor:
    """Streaming microphone cleanup: push far-end audio, process near-end audio.

    ``apm_factory`` and ``frame_factory``/``frame_reader`` exist for the tests;
    the call always uses livekit's module. If the module cannot be built (the
    package missing from a broken install) the microphone passes through
    untouched rather than failing the call.
    """

    def __init__(self, *, echo_cancellation: bool, noise_suppression: bool,
                 apm_factory=None, frame_factory=None, frame_reader=None):
        self._echo_cancellation = bool(echo_cancellation)
        self._noise_suppression = bool(noise_suppression)
        self._apm_factory = apm_factory or _default_apm_factory
        self._frame_factory = frame_factory or _frame
        self._frame_reader = frame_reader or _frame_samples
        self._delay_ms = 0
        self._apm = None
        self._unavailable = False
        self.reset()

    @property
    def available(self) -> bool:
        return self._apm is not None

    def reset(self) -> None:
        """Forget the learned echo path; the next call starts from scratch."""
        self._mic = np.empty(0, dtype=np.float32)
        self._ref = np.empty(0, dtype=np.float32)
        self._frames = 0
        self._reference_frames = 0
        self._errors = 0
        self._apm = None
        if self._unavailable:
            return
        try:
            self._apm = self._apm_factory(
                echo_cancellation=self._echo_cancellation,
                noise_suppression=self._noise_suppression,
            )
        except Exception:
            # Once is enough: every reset would otherwise log the same failure.
            self._unavailable = True
            logging.exception("[call_audio] WebRTC audio processing unavailable; "
                              "microphone is sent unprocessed")
            return
        self._apply_delay()

    def set_stream_delay_ms(self, delay_ms) -> None:
        """Hint how late the echo reaches the microphone (output + input latency).

        AEC3 finds the real delay itself; the hint only gives its estimator a
        starting point.
        """
        self._delay_ms = max(0, int(round(delay_ms or 0)))
        self._apply_delay()

    def _apply_delay(self) -> None:
        if self._apm is None or not self._echo_cancellation:
            return
        try:
            self._apm.set_stream_delay_ms(self._delay_ms)
        except Exception:
            logging.debug("[call_audio] could not set the echo delay hint", exc_info=True)

    def push_reference(self, samples: np.ndarray) -> None:
        """Far-end audio (48 kHz mono float32) as the speaker played it."""
        if self._apm is None or not self._echo_cancellation:
            return
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not data.size:
            return
        self._ref = np.concatenate((self._ref, data))
        while self._ref.size >= APM_FRAME:
            chunk, self._ref = self._ref[:APM_FRAME], self._ref[APM_FRAME:]
            try:
                self._apm.process_reverse_stream(self._frame_factory(chunk))
                self._reference_frames += 1
            except Exception:
                self._errors += 1

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Return the cleaned microphone audio.

        Output comes in whole 10 ms frames, so it can be shorter than the input
        (the remainder waits for the next call); length is conserved over the
        stream. Without a module, the input comes back as it is.
        """
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._apm is None:
            return data
        self._mic = np.concatenate((self._mic, data))
        out = []
        while self._mic.size >= APM_FRAME:
            chunk, self._mic = self._mic[:APM_FRAME], self._mic[APM_FRAME:]
            try:
                frame = self._frame_factory(chunk)
                self._apm.process_stream(frame)
                out.append(self._frame_reader(frame))
            except Exception:
                # A frame the module refused is sent as captured, so the other
                # person hears a moment of echo rather than a gap.
                self._errors += 1
                out.append(chunk)
            self._frames += 1
        if not out:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(out)

    def diagnostics(self) -> dict:
        """What the processor is doing, for the call log."""
        return {
            "engine": "webrtc" if self._apm is not None else "unavailable",
            "echo_cancellation": self._echo_cancellation,
            "noise_suppression": self._noise_suppression,
            "delay_hint_ms": self._delay_ms,
            "frames": self._frames,
            "reference_frames": self._reference_frames,
            "errors": self._errors,
        }
