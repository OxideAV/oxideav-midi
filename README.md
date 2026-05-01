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

Round-1 ships:

- `smf` — full SMF (Type 0 / 1 / 2) parser. Header (`MThd`), tracks
  (`MTrk`), variable-length quantities (bounded to 4 bytes per spec),
  every channel-voice message, sysex (`F0` / `F7`), and the common
  meta events (tempo, time signature, key signature, text, marker,
  end-of-track, SMPTE offset, sequencer-specific). Running status is
  honoured; chunk lengths are validated against remaining bytes; total
  events per file are capped at 1 M to keep malformed input bounded.
- `paths` — per-OS SoundFont/SFZ/DLS search paths plus the
  `OXIDEAV_SOUNDFONT_PATH` env-var override.
- `instruments` — `Instrument` trait, magic-byte detector stubs for
  `sf2` / `sfz` / `dls`, and a working pure-tone fallback so the synth
  produces *something* even with no on-disk bank.
- `downloader` — stub that names a planned default bank (TimGM6mb) but
  currently returns `Error::Unsupported`.

The decoder factory is registered under codec id `"midi"` but returns
`Error::Unsupported("MIDI synthesis not yet implemented; round 2")` for
now — the crate is a parser + path-finder + scaffold.

Round-2 will fill in: SoundFont 2 / SFZ / DLS sample loading, voice
mixing, SMF event → voice on/off scheduling driven by tempo and
division, and the actual `oxideav-http` integration in the downloader.
