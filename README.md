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
  generator. Walks `RIFF/sfbk` → `LIST INFO` / `LIST sdta` (smpl +
  optional sm24) / `LIST pdta` (phdr / pbag / pgen / inst / ibag /
  igen / shdr); cross-resolves the preset → instrument → zone →
  sample chain. Honours the `keyRange` / `velRange` filters; the
  `sampleID` / `instrument` / `sampleModes` / `*Tune` /
  `overridingRootKey` generators; the volume DAHDSR envelope (gens
  33-38) and `initialAttenuation` (gen 48); the modulation DAHDSR
  envelope (gens 25-30) routed into pitch (gen 7) and filter cutoff
  (gen 11); the initial low-pass biquad filter (gens 8/9); and the
  exclusive-class drum cut (gen 57). PCM storage is signed 24-bit
  (`i32`) — `sm24` lower bytes are combined with `smpl`'s 16-bit
  upper bytes when present, otherwise the 16-bit value is widened.
  Stereo zones (`LEFT` / `RIGHT` `sample_type` + cross-linked
  `sample_link`) render natively in stereo, bypassing the mixer's
  mono-pan law. Chunk lengths and array indices are bounds-checked
  against the loaded data; total samples capped at 256 Mi frames,
  total pdta records capped at 16 Mi, so malformed files cannot
  allocate beyond the spec ceiling.
- `instruments::sfz` — text patch reader **plus voice generator**.
  Tokenises SFZ syntax (line + block comments, headers, opcode
  `name=value` pairs with space-bearing values), walks `<control>` /
  `<global>` / `<master>` / `<group>` / `<region>` sections, flattens
  inheritance into one fully-resolved opcode map per region, and (via
  `SfzInstrument::open`) reads every referenced sample off disk against
  the SFZ file's directory + the active `default_path`. Strongly-typed
  fields: `lokey` / `hikey` / `lovel` / `hivel`, `pitch_keycenter`,
  `key` (sets all three), `loop_start` / `loop_end` / `loop_mode`,
  `tune` / `transpose`, `volume`, `pan`, `trigger`. Note names (`C4`,
  `c#4`, `Db5`) parse alongside decimal MIDI keys. Voice generation
  decodes the WAV sample bytes (8/16/24/32-bit PCM and IEEE_FLOAT) into
  mono f32, picks the matching region by (key, velocity), shifts pitch
  off `pitch_keycenter` + `tune` + `transpose`, applies a DAHDSR
  amplitude envelope from `ampeg_*` opcodes, and runs a vibrato LFO
  from `lfo01_freq` / `lfo01_pitch` / `lfo01_delay`. `#include` is
  rejected with `Error::Unsupported`; `#define` is preserved verbatim.
- `instruments::dls` — DLS Level 1 + 2 RIFF reader **plus voice
  generator**. Walks the `RIFF/DLS ` form (`colh` / `vers` / `ptbl` /
  `lins-list` / `wvpl-list` / `INFO-list`), surfaces the parsed bank
  with instrument → region → wave-pool topology, `wsmp` loop / pitch /
  gain headers, `wlnk` cue-table references, and `art1` / `art2`
  articulation connection blocks (raw — connection enums not
  interpreted). Voice generation picks the matching instrument by MIDI
  program, picks a region by (key, velocity), resolves the `wlnk` →
  `ptbl` → wave-pool entry, decodes the PCM (8/16-bit WAV-shaped) into
  mono f32, shifts pitch off the `wsmp.unity_note`, and plays the
  sample through the shared sample-playback voice. Loop modes: forward
  loop (`WLOOP_TYPE_FORWARD`, DLS1) and release loop
  (`WLOOP_TYPE_RELEASE`, DLS2). `art1`/`art2` connection-block
  evaluation is deferred — the parsed blocks remain on the bank for
  round-2 callers.
- `instruments::tone` — pure-tone fallback (sine / triangle / saw /
  square) so the synth produces *something* even with no on-disk
  bank.
- `mixer` — polyphonic voice pool (32 voices) with stereo mixdown,
  per-channel volume / pan / sustain pedal handling, oldest-voice
  preemption when the pool is full, channel/poly aftertouch routed
  to per-voice pressure, RPN 0 (pitch-bend range) handling, and
  exclusive-class drum cuts. Native stereo voices (SF2 stereo zones)
  are rendered through their own L/R buses, bypassing the mono-pan
  law.
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

Coverage today (round 9): full SF2 voice with sm24 24-bit samples,
stereo zones, DAHDSR volume + modulation envelopes, low-pass biquad
filter (gens 8/9), modEnv→pitch / modEnv→filter routing (gens 7/11),
exclusive-class drum cuts (gen 57), pitch bend with RPN 0 range,
channel/poly aftertouch; **SFZ voice generator** with DAHDSR amplitude
envelope (`ampeg_*`) and vibrato LFO (`lfo01_freq` / `lfo01_pitch`);
**DLS Level 1 + 2 voice generator** (parser-driven region selection,
wsmp loop info, unmodulated sample playback — articulation `art1`/
`art2` block interpretation is round 2). All three instrument paths
share one `SamplePlayer` voice for sample playback + DAHDSR + vibrato +
pitch bend.
