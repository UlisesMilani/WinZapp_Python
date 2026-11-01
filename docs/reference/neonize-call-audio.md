# Neonize call audio

The bridge keeps its public PCM contract: microphone input is s16le mono
48 kHz; the native source/sink is mono 16 kHz in 60 ms frames. This limits
audio bandwidth to 8 kHz. No upsampling or codec switch restores fullband
audio, and these source fixes do not establish parity with Telegram or Signal.

## Answer and stream ownership

Native `Active`/`ready` can arrive during `Answer`, before its return. Audio
reader ownership is independent of the state-change notification, so answering
still starts exactly one reader. Relay readiness alone cannot enable a ringing
call's microphone. A reader returning after end/rejoin cannot emit its old PCM
or release the replacement reader's claim. Microphone FIR history and partial
frames belong to one call record and are cleared on ownership changes.

A reentrant lifecycle gate orders model transitions, terminal events, PCM
emissions and microphone writes. Native audio reads stay outside that gate;
returning PCM is rechecked inside it. Terminal removal precedes emission, so
a callback rejoining the same ID cannot have its replacement removed. No model
mutex is held across the transport callback.

## PCM and clocks

`call_audio_pcm.py` uses a 97-tap streaming Blackman FIR before 3:1 decimation
(1 ms group delay). It preserves sample count, packet-boundary continuity and
speech level; it adds no AGC, compressor or noise suppression. Synthetic tests
measure passband level and alias attenuation. The existing client processing
and echo cancellation remain responsible for capture/playback processing.

Native `callPCMQueue` retains multi-frame decoder output across short C reads
instead of dropping bytes beyond the caller's capacity. Capacity is eight
60 ms frames; overload drops oldest samples. Microphone capacity is three
frames (180 ms); a missing capture frame yields on-time silence instead of
blocking a 60 ms sender tick for 150 ms.

## Negotiated codec and packet handling

Absent settings still select MLow. Explicit `use_mlow_codec_v1=false` selects
Opus for the real encoder and decoder, not just a log label. Both use 16 kHz
PCM. Opus encodes 60 ms SILK wideband at 24 kbit/s VBR, using Pion Opus pinned
to `v0.1.1-0.20261026214112-e7b8aa55508c` (MIT); no new native codec DLL is
installed. Opus RTP timestamps advance at 48 kHz (2880 per 60 ms), while MLow
uses 16 kHz. Receive timestamps are converted by accumulated deltas, including
uint32 wrap and fractional carry. Reception statistics retain the wire clock.

Both media directions read the authoritative settings at packet boundaries
under one codec mutex. Late accept settings therefore replace encoder and
per-participant decoder/reorder state and change the RTP stride together,
without replacing SRTP keys, sequence or ROC. There is no startup-hook window
that can lose an update. Old decoded PCM remains valid PCM; the direct playout
buffer resets when the new timestamp origin arrives. Authentication diagnostics
describe each wire arrival once; concealed PCM is labelled separately.

Only SRTP-authenticated packets enter a per-participant bounded reorder queue.
Late/duplicate packets never reach the stateful decoder twice. A missing
predecessor is released when another packet arrives after 60 ms, or the queue
reaches three packets; there is no timer that invents packets while idle.
Opus uses decoder PLC for gaps up to 120 ms before decoding the following
packet. The MLow port has no PLC API: its fallback is a short decay to silence,
not FEC or reconstructed speech. Large discontinuities retain the existing
playout reset. Group rosters use the negotiated codec for current and future
receivers; existing SRTP keys, ROC handling and participant gates are preserved.

## Evidence and limits

Tests use generated PCM, fake call events and synthetic authenticated RTP.
They cover answer ordering, end/rejoin ownership, FIR packet splits and
stopband, multi-frame C reads, clipping, queue bounds, codec output, packet
order/loss/duplicates and sequence/timestamp wrap. Where FFmpeg is available,
an independent libopus decoder checks an Ogg stream made by the call encoder.
PR CI also builds the Windows DLL and runs the bridge suite against that build.

These checks are source, synthetic audio and build evidence. They do not
measure real network quality, hardware latency, microphone/speaker behavior,
echo or end-to-end call acceptance. A published alpha and live user acceptance
are separate gates; no codec is forced against WhatsApp negotiation.
