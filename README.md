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
  `SmfFile::to_bytes_running_status()` /
  `Track::to_bytes_chunk_running_status()` emit the same stream with
  **running-status compression** — a channel-voice event whose status
  byte equals the previous emitted one drops the redundant status byte,
  with the running-status buffer cleared by every meta / sysex event
  (the parser's reset points), so the compressed output still
  round-trips byte-for-byte.
- **Builder** — `SmfBuilder` / `TrackBuilder` assemble a file from
  scratch without hand-computing delta-times. `TrackBuilder` places
  events at **absolute ticks** in any order, stably sorts them
  (insertion order breaks ties), computes deltas at `build()`, and
  auto-appends an `EndOfTrack` unless one is already present; the
  `note(tick, duration, channel, key, velocity)` helper emits a Note
  On/Off pair. `SmfBuilder` keeps `ntrks` in sync with the track count
  (defaults to format 1, 480 ticks-per-quarter) so the result always
  passes `to_bytes`'s header-consistency check.
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
- **Channel-mode classifier** — `ControlChangeEvent::channel_mode()`
  decodes a `120..=127` controller into a typed `ChannelModeMessage`
  (All Sound Off, Reset All Controllers, Local Control on/off, All Notes
  Off, Omni Off / On, Mono On with channel count, Poly On);
  `is_all_notes_off()` flags 123–127. `channel_mode_messages()` is the
  stably-merged absolute-tick iterator over that subset.
- **Effects-depth classifier** — `ControlChangeEvent::effect_depth()`
  decodes an `91..=95` controller into a typed `EffectDepth` (Reverb
  Send / Tremolo / Chorus Send / Celeste / Phaser, per MIDI 1.0 *Control
  Change Messages* Table 3), each carrying the `0..=127` level via
  `.level()` and its source `.controller()`. CC 91 / CC 93 are the GM
  Reverb / Chorus sends the synth's effects bus consumes.
  `effect_depths()` is the stably-merged absolute-tick iterator over that
  subset.
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
  helpers. `reassembled_sysex_messages()` folds the `F0`-opener +
  `F7`-continuation packet stream into complete logical
  `ReassembledSysEx` messages (per-track continuation state machine,
  trailing-`F7` stripped from `body`, `complete` / `packet_count` /
  `id_byte()` / `is_universal()` accessors; standalone `F7` escapes and
  unterminated chains surface as `complete == false`).
  `universal_sysex_events()` + `SysExEvent::
  universal_classification()` decode the Universal SysEx Table 4
  vocabulary (`UniversalSysEx` / `UniversalSubId1`), realm-aware
  (Non-RT `0x7E` vs RT `0x7F`). `UniversalSysExEvent::
  mtc_full_message()` decodes the Real-Time MIDI Time Code **Full
  Message** body (`F0 7F dev 01 01 hr mn sc fr F7`) into a typed
  `MtcFullMessage` — the SMPTE `hr/mn/sc/fr` quartet plus a decoded
  `FrameRate` (24 / 25 / 30-drop / 30-non-drop) and hours count;
  `None` on non-Full-Message packets or a quartet truncated mid-stream.
  `UniversalSysExEvent::mtc_user_bits()` decodes the Real-Time MTC
  **User Bits Message** body (`F0 7F dev 01 02 u1..u9 F7`) into a typed
  `MtcUserBits` — the eight SMPTE/EBU Binary Groups plus the two Binary
  Group Flag Bits (`flag_i` / `flag_j`) and the `reassembled()` 32-bit
  value (`hhhhgggg ffffeeee ddddcccc bbbbaaaa` nibble order); `None` on
  non-User-Bits packets or a payload truncated before all nine bytes.
  `UniversalSysExEvent::notation_bar_number()` decodes the Notation
  Information **Bar Number** message (`F0 7F dev 03 01 aa aa F7`) into a
  `NotationBarNumber` — the signed 14-bit (lsb-first) field plus a
  `state()` classifier (`NotationBarState`: NotRunning `0x2000`, the
  negative-through-zero CountIn range, the positive BarInSong range, and
  RunningUnknown `0x1F7F`). `notation_time_signature()` decodes the
  **Time Signature** message in both Immediate (`03 02`) and Delayed
  (`03 42`, `is_delayed()`) forms into a `NotationTimeSignature` — the
  leading `nn dd cc bb` quartet (mirroring the `FF 58` meta event) plus
  every compound `nn dd` pair (`is_compound()`); `denominator()` decodes
  `1 << dd`. Both return `None` on a wrong-classification packet or a
  body truncated before the declared bytes arrive.
- **Tick → wall-clock time** — `tempo_timeline()` folds the tempo map
  against the header `Division` into a `TempoTimeline`; its
  `tick_to_seconds(tick)` resolves any absolute tick to elapsed seconds
  in `O(log n)` (binary search over tempo segments). `tick_to_seconds()`
  is the one-shot convenience and `duration_seconds()` reports the
  scheduled event span (max end tick across tracks). Musical divisions
  integrate piecewise across Set Tempo changes; SMPTE divisions use the
  fixed `1/(fps × ticks_per_frame)` rate and ignore tempo events —
  matching the scheduler's tick→sample arithmetic so reported duration
  agrees with rendered length.
- **Channel-state snapshot** — `SmfChannelSnapshot` +
  `channel_snapshot_at(channel, tick)` /
  `channel_snapshots_at(tick)` replay channel-voice events up to a
  tick (in scheduler order) for seek initialisation, folding Program
  Change, Pitch Bend, and the snapshot-tracked CCs; `apply()` is
  exposed for custom replay.

## UMP / MIDI 2.0 (`ump`)

Universal MIDI Packet container and MIDI 2.0 Protocol vocabulary, per
the MIDI Association *UMP Format and MIDI 2.0 Protocol* spec
(M2-104-UM v1.1.2). Transport-independent: the module operates purely on
32-bit `u32` words (on-wire byte order is out of scope per spec §2.1.1).

- **`ump::packet`** — the [`Ump`] word container. Decodes the Message
  Type (MT) nibble and derives packet size (1/2/3/4 words) from Table 4,
  so even unmodelled / Reserved Message Types are sized correctly.
  Extracts group / status / opcode / channel; Utility (MT 0x0) and UMP
  Stream (MT 0xF) are Groupless. `UmpStream` walks a flat `&[u32]` into
  self-delimiting packets of mixed sizes, surfacing a trailing partial
  packet as a single `Err`.
- **`ump::message`** — typed `decode` + `encode` for four Message Types:
  Utility (NOOP, JR Clock, JR Timestamp, Delta Clockstamp TPQ, Delta
  Clockstamp 20-bit); System Common / System Real Time (SPP LSB-first,
  `0xF0`/`0xF7` rejected since SysEx rides MT 0x3); MIDI 1.0 Channel
  Voice (all 7 opcodes, 2-byte messages zero-fill); and the full MIDI
  2.0 Channel Voice set — registered/assignable per-note controllers,
  registered/assignable + relative controllers, per-note pitch bend,
  16-bit-velocity Note On/Off with attribute type/data, 32-bit poly /
  channel pressure, per-note management D/S flags, Program Change with
  the Bank Valid flag, and 32-bit Pitch Bend. `UmpMessage::decode`
  dispatches on MT; Data / Flex / Stream / Reserved surface as
  `Unhandled`.
- **`ump::scaling`** — the spec Appendix D bit-scaling primitives:
  Min-Center-Max upscaling (smooth shift below center, bit-repeat above)
  and truncating downscaling, with 7/14 ⇄ 16/32 helpers. Verified
  against the §D.1.3 numerical examples and §D.1.2 center-value table;
  7⇄32 and 14⇄32 round-trips proven lossless.
- **Translation** — `Midi1ChannelVoice::to_midi2` / `Midi2ChannelVoice::
  to_midi1` implement the Default Translation Mode (§D.2/D.3): Note On
  velocity-0 ⇄ Note Off 0x8000, velocity floored to 1 on downscale,
  pitch-bend 14⇄32 with LSB-first packing, Program Change bank handling,
  and `None` for the messages with no counterpart (special CCs that
  belong to compound RPN/NRPN sequences; per-note / relative / per-note-
  management on the way down).

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
  Parameter Control.
- `mixer` **system effects bus** — the GM2 Reverb + Chorus parameters
  (CA-024) drive a real stereo DSP send, not just decoded state.
  Per-channel **CC 91** (Reverb Send) and **CC 93** (Chorus Send)
  scale each voice's post-pan signal into the bus. The reverb is a
  Schroeder design (parallel feedback comb bank → series allpass
  diffusers) whose comb feedback is computed from the CA-024 Reverb
  Time so the −60 dB decay matches the spec; the chorus is a
  sine-modulated delay line driven by the CA-024 Mod Rate / Mod Depth /
  Feedback, and the chorus→reverb send (CA-024 chorus `pp=4`) routes
  the wet chorus into the reverb input. Both sends default to 0 so a
  dry score renders bit-identically to the pre-effects path; the delay
  lines size to the output rate via `Mixer::set_sample_rate`, and GM
  System On/Off flushes the tails.
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
