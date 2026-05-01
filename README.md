# oxideav-midi

Pure-Rust **MIDI** — Standard MIDI File (`.mid` / SMF) parser + transport
metadata + soft-synth scaffold. Zero C dependencies, zero FFI, zero
`*-sys`.

External instruments (SoundFont 2 `.sf2`, SFZ, DLS Level 1/2) are loaded
from disk at runtime; nothing is bundled in the binary. A pure-tone
oscillator fallback lets the synth produce some output even when no
instrument bank is installed.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Status

- `smf` — full SMF (Type 0 / 1 / 2) parser. Header (`MThd`), tracks
  (`MTrk`), variable-length quantities (bounded to 4 bytes per spec),
  every channel-voice message, sysex (`F0` / `F7`), and the common
  meta events (tempo, time signature, key signature, text, marker,
  end-of-track, SMPTE offset, sequencer-specific). Running status is
  honoured; chunk lengths are validated against remaining bytes; total
  events per file are capped at 1 M to keep malformed input bounded.
- `paths` — per-OS SoundFont/SFZ/DLS search paths plus the
  `OXIDEAV_SOUNDFONT_PATH` env-var override.
- `instruments::sf2` — full SoundFont 2 RIFF reader and voice
  generator. Walks `RIFF/sfbk` → `LIST INFO` / `LIST sdta` (smpl) /
  `LIST pdta` (phdr / pbag / pgen / inst / ibag / igen / shdr); cross-
  resolves the preset → instrument → zone → sample chain; honours the
  `keyRange` / `velRange` filters and the `sampleID` / `instrument` /
  `sampleModes` / `*Tune` / `overridingRootKey` generators; plays
  16-bit PCM with linear-interpolation resampling and an optional
  loop. Chunk lengths and array indices are bounds-checked against the
  loaded data; total samples capped at 256 Mi frames, total pdta
  records capped at 16 Mi, so malformed files cannot allocate beyond
  the spec ceiling. Stereo linking, modulators, sm24 (24-bit lower
  bytes), and full envelopes/filters are round-4.
- `instruments::sfz` / `instruments::dls` — magic-byte detector stubs;
  `make_voice` returns `Error::Unsupported`. Loader work is round-4.
- `instruments::tone` — pure-tone fallback (sine / triangle / saw /
  square) so the synth produces *something* even with no on-disk
  bank.
- `mixer` — **round-3** polyphonic voice pool (32 voices) with
  stereo mixdown, per-channel volume / pan / sustain pedal handling,
  and oldest-voice preemption when the pool is full.
- `scheduler` — **round-3** SMF event scheduler. Merges every track
  into a single time-ordered stream, converts ticks → samples against
  the current tempo + division (`samples_per_tick = us_per_quarter *
  sample_rate / (1_000_000 * ticks_per_quarter)`), and dispatches
  every event into the mixer at the right audio sample.
- `downloader` — stub that names a planned default bank (TimGM6mb) but
  currently returns `Error::Unsupported`.

The decoder factory is registered under codec id `"midi"`. Round-3
wires SMF events end-to-end: `send_packet` parses the SMF and primes
the scheduler; `receive_frame` returns interleaved S16 stereo PCM
(1024 samples per channel at 44 100 Hz) until both the event stream
and the voice pool have run dry. Without an on-disk bank the
registry-built decoder uses the pure-tone fallback; for SoundFont 2
playback build a `MidiDecoder` directly with an `Sf2Instrument`.

Round-4 is full ADSR envelopes, modulator generators, pitch bend,
SFZ + DLS loaders, and stereo SF2 sample linking.
