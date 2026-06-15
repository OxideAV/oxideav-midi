# oxideav-midi

Pure-Rust **MIDI** — Standard MIDI File (`.mid` / SMF) parser + writer,
transport metadata, and a soft-synth. Zero C dependencies, zero FFI,
zero `*-sys`.

External instruments (SoundFont 2 `.sf2`, SFZ, DLS Level 1/2) are
loaded from disk at runtime; nothing is bundled in the binary. A
pure-tone oscillator fallback lets the synth produce output even when
no instrument bank is installed.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## SMF (`smf`)

Full SMF (Type 0 / 1 / 2) parser + writer. Header (`MThd`), tracks
(`MTrk`), variable-length quantities (bounded to 4 bytes per spec),
every channel-voice message, sysex (`F0` / `F7`), and the meta events
(tempo, time/key signature, text, marker, end-of-track, SMPTE offset,
sequencer-specific). Running status honoured; chunk lengths validated;
total events capped at 1 M to keep malformed input bounded.

- **Writer** — `SmfFile::to_bytes()` serialises a parsed file back to a
  complete SMF byte stream (`MThd` + one `MTrk` per `Track`);
  `Track::to_bytes_chunk()` emits a single `MTrk` chunk. Output uses
  explicit status bytes and round-trips byte-for-byte through `parse`.
  `MAX_VLQ_VALUE = 0x0FFF_FFFF` is the public 4-byte VLQ cap;
  out-of-range values surface `Error::InvalidData` at encode time.
- **Meta-event iterators** — one typed, absolute-tick, stably-merged
  (track 0 before track 1 at the same tick) iterator per meta kind:
  `tempo_map`, `time_signatures`, `key_signatures`, `markers`,
  `lyrics`, `cue_points`, `track_names`, `instrument_names`, `texts`,
  `copyrights`, `smpte_offsets`, `sequencer_specifics`,
  `sequence_numbers`, `midi_ports`, `channel_prefixes`. Tempo /
  time-sig / key-sig values are decoded (BPM, `1 << dd`, circle-of-
  fifths label); text payloads expose `text_bytes()` + a
  `text_lossy()` `Cow<str>`; SMPTE offsets decode the packed
  `FrameRate`.
- **Channel-voice iterators** — `program_changes`, `control_changes`
  (with `is_channel_mode()`), `pitch_bends` (`signed_value()` /
  `is_centre()`), `channel_pressures`, `poly_aftertouches`, and the
  piano-roll `notes()` (Note-On/Off pairing with velocity-0 = Off
  convention, FIFO re-strike) plus `active_notes_at(tick)`.
- **RPN / NRPN decoder** — `parameter_data_entries()` folds the
  CC 6 / 38 Data Entry pump and CC 96 / 97 Increment / Decrement against
  each channel's running RPN (CC 101 / 100) / NRPN (CC 99 / 98) selector,
  emitting one resolved `ParameterDataEntry` per pump action. Registered
  parameters are classified per Table 3a (`RegisteredParameter`: Pitch
  Bend Sensitivity, Channel Fine / Coarse Tuning, Tuning Program /
  Bank, Modulation Depth Range, MPE Config, the nine RP-049 3D Sound
  Controllers, Null, plus a `Reserved` catch-all); NRPNs surface as raw
  14-bit numbers. The Null Function Number and the power-up default
  disable the pump; one active parameter per channel (RPN supersedes
  NRPN and vice versa).
- **SysEx** — `sysex_events()` surfaces both `F0` and `F7` flavours
  with `manufacturer_id()` / `ends_with_eox()` / `is_complete_message()`
  helpers; `universal_sysex_events()` + `SysExEvent::
  universal_classification()` decode the Universal SysEx Table 4
  vocabulary (`UniversalSysEx` / `UniversalSubId1`), realm-aware
  (Non-RT `0x7E` vs RT `0x7F`).
- **Channel-state snapshot** — `SmfChannelSnapshot` +
  `channel_snapshot_at(channel, tick)` /
  `channel_snapshots_at(tick)` replay channel-voice events up to a
  tick (in scheduler order) for seek initialisation, folding Program
  Change, Pitch Bend, and the snapshot-tracked CCs; `apply()` is
  exposed for custom replay.

## Instruments

- `instruments::sf2` — full SoundFont 2 RIFF reader + voice generator.
  Resolves preset → instrument → zone → sample; honours key/vel
  ranges, the sample / tune / root-key generators, volume + modulation
  DAHDSR envelopes, modEnv→pitch / modEnv→filter routing, the initial
  low-pass biquad, and exclusive-class drum cuts. 24-bit `sm24`
  samples and native stereo zones supported. Bounds-checked with spec
  ceilings on sample / record counts.
- `instruments::sfz` — text patch reader + voice generator. Tokenises
  the SFZ syntax, flattens `<global>`/`<master>`/`<group>`/`<region>`
  inheritance, reads referenced WAV samples (8/16/24/32-bit PCM +
  IEEE_FLOAT), and drives a DAHDSR amplitude envelope, vibrato LFO,
  and a `fil_type`-aware biquad filter envelope. `#include` is
  rejected.
- `instruments::dls` — DLS Level 1 + 2 RIFF reader + voice generator
  with `art1`/`art2` articulation interpretation and an EG2 + 2-pole
  resonant low-pass filter. Forward and release loop modes.
- `instruments::articulation` — DLS connection-block evaluator backed
  by the MMA DLS1/DLS2 tables; named `CONN_*` constants, the common
  source→destination default + modulator routings, and the standard
  unit conversions.
- `instruments::tone` — pure-tone fallback (sine / triangle / saw /
  square).

## Synthesis

- `mixer` — 32-voice polyphonic pool with stereo mixdown, per-channel
  volume / pan / sustain, oldest-voice preemption, channel/poly
  aftertouch, RPN handling, exclusive-class drum cuts, and native
  stereo voices. Supports RPN 0/1/2/5/6 (pitch-bend range, channel
  fine/coarse tune, mod-depth range, MPE config), Data Inc/Dec
  (CC 96/97), mod-wheel + MPE timbre routing, Master Volume / Balance
  / Fine / Coarse Tuning (Universal Real-Time SysEx), and GM2 Global
  Parameter Control (reverb/chorus parameters decoded, not yet applied
  as a DSP send).
- `mixer::MpeZone` / `MpeRole` — MIDI Polyphonic Expression v1.1: MCM
  zone configuration, per-note bend / pressure / CC 74 on Member
  Channels, Member+Manager bend combining, default PB sensitivities,
  PKP drop on Members, and sounding-note reset on reconfiguration.
- `scheduler` — merges every track into one time-ordered stream,
  converts ticks → samples against tempo + division, and dispatches
  events into the mixer at the right sample. Wires the Universal
  SysEx surface (GM 1/2 System On/Off, Master Volume/Balance/Tuning,
  MIDI Tuning Standard, Data Inc/Dec, GM2 GPC).
- `tuning` — MIDI Tuning Standard (MTS) microtuning state + Universal
  SysEx decoders (key-based + scale/octave tables, signed cents added
  to equal temperament; drum channel exempt).
- `paths` — per-OS SoundFont/SFZ/DLS search paths +
  `OXIDEAV_SOUNDFONT_PATH` override.
- `downloader` — stub naming a planned default bank; currently returns
  `Error::Unsupported`.

The decoder factory registers under codec id `"midi"`: `send_packet`
parses the SMF and primes the scheduler; `receive_frame` returns
interleaved S16 stereo PCM (1024 samples/channel at 44.1 kHz) until
the event stream and voice pool run dry. Without an on-disk bank the
registry-built decoder uses the pure-tone fallback; for SoundFont 2
playback build a `MidiDecoder` directly with an `Sf2Instrument`.

## Fuzzing

A `cargo-fuzz` harness covers every attacker-facing parser:

```
cargo +nightly fuzz run smf    # smf::parse + iterators
cargo +nightly fuzz run sf2    # instruments::sf2::Sf2Bank::parse
cargo +nightly fuzz run dls    # instruments::dls::DlsBank::parse
cargo +nightly fuzz run sfz    # instruments::sfz::parse_str
```

Each target asserts arbitrary bytes return a `Result` with no panic /
OOM / overflow / OOB. Curated seed corpora live under
`fuzz/corpus/<target>/`.

## Profiling

`benches/synth_render.rs` (`harness = false`) is a repeatable SMF→PCM
wall-clock harness over a dense 8-channel / 32-voice score through an
in-memory SF2 bank, plus a `--corpus` PCM-hash mode and a `--spin
SECS` sampling-profiler loop.

## License

MIT — see `LICENSE`.
