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
- `instruments::sf2` — **round-2** SoundFont 2 RIFF reader and voice
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
  bytes), and full envelopes/filters are round-3.
- `instruments::sfz` / `instruments::dls` — magic-byte detector stubs;
  `make_voice` returns `Error::Unsupported`. Loader work is round-3.
- `instruments::tone` — pure-tone fallback (sine / triangle / saw /
  square) so the synth produces *something* even with no on-disk
  bank.
- `downloader` — stub that names a planned default bank (TimGM6mb) but
  currently returns `Error::Unsupported`.

The decoder factory is registered under codec id `"midi"` but
`send_packet` still returns `Error::Unsupported` — round-3 wires the
SMF event stream through the SF2 voice generator (note-on/off
dispatch, tempo/division scheduling, voice mixing into PCM frames).
