# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Round 381 — synth hot-path: skip the effects bus on the dry path

- `Mixer::mix_stereo` now gates the entire Reverb + Chorus effects bus
  behind a latched `EffectsBus::active` flag. While every channel's
  reverb / chorus send (CC 91 / CC 93) is zero the bus delay lines are
  provably all-zero, so the per-sample send accumulation **and** the
  Schroeder comb/allpass reverb + modulated-delay chorus DSP (including a
  per-sample `sin` in each chorus `ModDelay`) are skipped — their wet
  return into the mix would be exactly `0.0`. The first non-zero send
  latches the bus on and it stays on so the reverb/chorus tail still
  rings out after the sends fall back to zero; `reset_gm_effects` /
  GM-reset paths call `EffectsBus::clear`, which re-zeroes the state and
  the flag.
- Measured on the `benches/synth_render.rs` dense 8-channel / 32-voice
  dry SF2 score: best-of-9 wall clock 87.6 ms → 66.3 ms (≈24 % faster).
  Output is **bit-identical** — every `--corpus` PCM FNV-1a hash is
  unchanged.
- New regression test `fx_gate_inactive_blocks_leave_no_residue` proves
  the skip is exactly equivalent to running the DSP over all-zero input:
  a mixer that rendered several dry idle blocks before its first
  reverb-send note produces byte-identical output, from the send note
  onward, to a freshly-constructed mixer.

### Round 378 — synth pitch-bend / portamento coexistence fix

- `Mixer::set_pitch_bend` now sums each voice's in-progress portamento
  glide offset into the pitch it pushes, so a live pitch-bend arriving
  mid-glide no longer clobbers the glide (both the non-MPE and MPE-Manager
  broadcast paths). New regression test asserts a +100-cent bend during a
  −400-cent glide lands at −300 cents.

### Round 378 — synth Portamento (CC 5 / 65 / 84)

- The mixer now performs portamento pitch glides. `set_portamento` (CC 65
  on/off), `set_portamento_time` (CC 5), and `set_portamento_control`
  (CC 84) track the per-channel glide state; a note struck with portamento
  active starts displaced by `(from_key − target_key) × 100` cents and
  glides linearly to the target over a CC 5-controlled span
  (`PORTAMENTO_MAX_MS` at CC 5 = 127), advanced per rendered block by
  `advance_portamento` and summed with the live channel bend / tuning via
  `reapply_pitch_for_slot`. The glide **source** is the pending CC 84
  Portamento Control key (always glides, ignoring CC 65, and resets after
  one note-on per the MIDI 1.0 spec) or, when CC 65 is on, the channel's
  previously-played key. Drum channel (10) never glides; CC 5 = 0 or a
  same-key repeat is instantaneous. Reset All Controllers clears the CC 65
  switch + pending CC 84 source. Scheduler routes CC 5 / 65 / 84. 5 new
  tests (off = no glide, on glides from previous note with a mid-trajectory
  + settle check, CC 84 glides ignoring on/off, CC 84 one-note reset, zero
  time = instant). The exact CC 5 → glide-time curve is implementation-
  defined per the spec; the chosen linear ms map is documented on
  `PORTAMENTO_MAX_MS`.
- Provenance: `docs/audio/midi/midi-1.0/M1_v4-2-1_MIDI_1-0_Detailed_Specification_96-1-4.pdf`
  §"PORTAMENTO CONTROLLER" + `Control-Change-Messages-Data-Bytes.pdf`
  (CC 5 / 65 / 84).

### Round 378 — synth All Sound Off / All Notes Off split (CC 120/123-127)

- The scheduler previously conflated CC 120 (All Sound Off) and CC 123
  (All Notes Off) into a single global hard-stop. They are now distinct
  **per-channel** Channel Mode messages per the MIDI 1.0 table:
  `Mixer::all_sound_off(channel)` silences the channel's voices
  immediately (no release, ignoring the Sustain / Sostenuto pedals — the
  "kill now" panic), while `Mixer::all_notes_off_channel(channel)` turns
  the channel's notes off via the normal release envelope and honours a
  held pedal exactly as an explicit Note Off would. CC 124/125 (Omni
  Off/On) and CC 126/127 (Mono/Poly On) route to All-Notes-Off per their
  "+ all notes off" definition; CC 122 (Local Control) is recognised as a
  no-op (no local keyboard to gate in a software synth). 4 new tests
  (immediate-cut vs. sustain, release-path vs. sustain, per-channel
  scope, release-path note-off).
- Provenance: `docs/audio/midi/midi-1.0/Control-Change-Messages-Data-Bytes.pdf`
  (Channel Mode Messages 120-127).

### Round 378 — synth Soft Pedal (CC 67, una corda)

- New `Mixer::set_soft_pedal(channel, value)` models CC 67: a note struck
  while the pedal is down renders at the `SOFT_PEDAL_GAIN` (≈ −3.5 dB)
  *una corda* attenuation, captured per-voice at note-on (new `VoiceSlot::
  note_gain` folded into the mix-time gain alongside Volume × Expression).
  Notes already sounding when the pedal moves are unaffected — the pedal
  shifts the action for the next strike, not the current vibration. Reset
  All Controllers clears it (RP-015 resets the pedals). The scheduler
  routes CC 67. 3 new tests (struck-while-down attenuation, already-
  sounding immunity, RAC clear).
- Provenance: `docs/audio/midi/midi-1.0/Control-Change-Messages-Data-Bytes.pdf`
  (CC 67 = Soft Pedal On/Off).

### Round 378 — synth Sostenuto Pedal (CC 66)

- New `Mixer::set_sostenuto(channel, value)` models the Sostenuto pedal
  per the MIDI 1.0 Control Change table: on press it **captures** only the
  notes that are still sounding at that instant; a later NoteOff on a
  captured voice defers its release until the pedal lifts, while notes
  started *after* the press are unaffected. Sostenuto and Sustain (CC 64)
  hold independently — a deferred voice releases only when **both** pedals
  are up, via the shared `release_deferred_voices` re-evaluation. The
  `VoiceSlot` gains a `sostenuto_captured` flag and a `released` marker
  (still-held vs. release-tail). Reset All Controllers and `all_notes_off`
  clear the new state; the scheduler routes CC 66. 4 new tests (capture
  semantics, post-press exclusion, still-held survival across lift, and
  Sustain+Sostenuto independence).
- Provenance: `docs/audio/midi/midi-1.0/Control-Change-Messages-Data-Bytes.pdf`
  (CC 66 = Sostenuto On/Off).

### Round 378 — synth Reset All Controllers (CC 121, RP-015)

- New `Mixer::reset_all_controllers(channel)` implements the RP-015
  "Response to Reset All Controllers" recommended practice exactly: it
  resets Expression (→ 127), Modulation (→ 0), the Pedals (Sustain /
  Portamento / Sostenuto / Soft → 0), the RPN/NRPN selector (→ null),
  Pitch Bend (→ centre), and Channel + Polyphonic Pressure (→ 0), while
  leaving Bank Select, Channel Volume, Pan, Program, the Effect
  Controllers, the Sound Controllers, and the *values* of registered /
  non-registered parameters untouched. Lifting Sustain fires the release
  on any voices the pedal was holding, and the centred bend / cleared
  mod-wheel / cleared pressure re-apply to every held voice immediately.
  The scheduler routes CC 121 to it. 3 new tests (full RP-015 reset +
  preserve list, sustain-release, held-voice bend recentre).
- Provenance: `docs/audio/midi/recommended-practices/rp15.pdf` (RP-015).

### Round 378 — synth CC 11 Expression Controller

- The mixer now tracks **CC 11 (Expression Controller)** per channel and
  folds it into the mix-time gain. Expression is a *percentage of Channel
  Volume* per the MIDI 1.0 Control Change table, so the two multiply
  (`volume × expression / 127²`); a sequence can shape dynamics inside the
  headroom Channel Volume reserves. `ChannelState::expression` defaults to
  127 (transparent) so a score that never sends CC 11 renders identically
  to the pre-Expression path. The scheduler routes CC 11 to the new field.
  2 new mixer tests (multiplicative scaling vs. a half-Expression channel;
  default-127 transparency).
- Provenance: `docs/audio/midi/midi-1.0/Control-Change-Messages-Data-Bytes.pdf`
  (CC 11 = Expression Controller).

### Round 374 — MMC LOCATE [TARGET] Standard Time decoder

- New `MmcCommand::locate_target()` decodes the LOCATE [TARGET] command
  body (`44 06 01 hr mn sc fr ff`, RP-013 §5 Format 2) into an
  `MmcStandardTime { hours_raw, minutes, seconds, frames, subframes }`.
  `MmcStandardTime::frame_rate()` recovers the SMPTE rate from bits 5-6 of
  the `hr` byte (shared with the MTC Standard Time encoding) and
  `hours_count()` the hours from bits 0-4. `None` for the LOCATE [I/F]
  form (sub-command `0x00`), a non-LOCATE command, or a truncated body.
  3 new tests.
- Provenance: `docs/audio/midi/recommended-practices/RP-013_v1-0_MIDI_Machine_Control_Specification_96-1-4.pdf` §"STANDARD TIME CODE".

### Round 374 — Universal SysEx round-trip fidelity test

- New `round_trip_universal_sysex_families_recover_decoders` test builds a
  track carrying an MMC Command, an MSC message, an Identity Reply, a GM 1
  System On, and a Sample Dump Request, drives it through `to_bytes()` →
  `parse()`, and asserts (a) exact structural equality and (b) that every
  new typed body decoder recovers the same value from the re-parsed file.
  Closes the parse→write→parse loop for the round-374 SysEx families.

### Round 374 — General MIDI System On / Off typed decoder

- New `UniversalSysExEvent::general_midi_system()` decodes the General
  MIDI System messages (`F0 7E <dev> 09 0n F7`) into a `GeneralMidiSystem`
  enum: `Level1On` (`0x01`, RP-003), `Off` (`0x02`), `Level2On` (`0x03`,
  General MIDI 2). `None` for any other packet.
- New `SmfFile::general_midi_system_messages()` absolute-tick iterator.
  3 new tests cover the three message types, the non-GM negative, and a
  reset-then-disable two-message ordering.
- Provenance: `docs/audio/midi/recommended-practices/RP-003_General_MIDI_System_Level_1_Specification_96-1-4_0.1.pdf` and the General MIDI 2 specification under `docs/audio/midi/`.

### Round 374 — Sample Dump Standard header / request body decoders

- New `UniversalSysExEvent::sample_dump_header()` decodes a Sample Dump
  Header (`F0 7E <dev> 01 ss ss ee ff ff ff gg gg gg hh hh hh ii ii ii jj
  F7`) into `SampleDumpHeader { sample_number, sample_format,
  sample_period, sample_length, loop_start, loop_end, loop_type }`. The
  21-bit numeric fields unpack their three 7-bit groups LSB-first; the
  loop-type byte classifies into `LoopType` (`Forward` / `BackwardForward`
  / `Off` / `Other`).
- New `UniversalSysExEvent::sample_dump_request()` decodes a Sample Dump
  Request (`F0 7E <dev> 03 ss ss F7`) into the requested 14-bit sample
  number. 4 new tests cover all header fields, the loop-type variants,
  header truncation, and the request.
- Provenance: `docs/audio/midi/midi-1.0/M1_v4-2-1_MIDI_1-0_Detailed_Specification_96-1-4.pdf` §"DUMP HEADER" / §"DUMP REQUEST".

### Round 374 — Identity Reply body decoder (`UniversalSysExEvent::identity_reply`)

- New `UniversalSysExEvent::identity_reply()` decodes a Non-Real-Time
  General Information Identity Reply (`F0 7E <dev> 06 02 mm ff ff dd dd
  ss ss ss ss F7`) into `IdentityReply { manufacturer, family,
  family_member, software_revision }`. The manufacturer ID is a
  `ManufacturerId::Single(mm)` or, when `mm == 0x00`, the three-byte
  `ManufacturerId::Extended(b1, b2)` form, with the family / member /
  revision fields shifting accordingly. Family and family-member codes
  are 14-bit LSB-first.
- New `SmfFile::identity_replies()` absolute-tick iterator. 4 new tests
  cover the single-byte and extended manufacturer forms, the
  Identity-Request / truncation negatives, and the iterator.
- Provenance: `docs/audio/midi/midi-1.0/M1_v4-2-1_MIDI_1-0_Detailed_Specification_96-1-4.pdf` §"GENERAL INFORMATION".

### Round 374 — MIDI Show Control body decoder (`UniversalSysExEvent::show_control`)

- New `UniversalSysExEvent::show_control()` decodes a Real-Time MIDI Show
  Control packet (`F0 7F <dev> 02 <command_format> <command> <data> F7`,
  RP-002-014) into `ShowControlMessage { command_format, command, data }`.
  The bytes after the `0x02` `<msc>` Sub-ID are the Command Format and
  Command (not Universal Sub-ID #2), so the decoder reads them directly.
- `ShowControlFormat` names the General-category formats (Lighting / Sound
  / Machinery / Video / Projection / Process Control / Pyrotechnics) plus
  All-types (`0x7F`); Specific formats pass through as `Other(u8)`.
  `ShowControlCommand` names the full RP-002-014 §5 command set (`Go` …
  `Abort`, `0x01`..=`0x0B`, `0x10`..=`0x1E`, `0x20`..=`0x26`); unknown
  opcodes pass through as `Other(u8)`.
- New `SmfFile::show_control_messages()` absolute-tick iterator. 6 new
  tests cover Lighting GO, the cue-number data field, every command-format
  and command classification, the non-MSC / truncation negatives, and the
  iterator merge.
- Provenance: `docs/audio/midi/recommended-practices/RP-002-014_v1-1-1_MIDI_Show_Control_Specification_96-1-4.pdf`.

### Round 374 — MIDI Machine Control command / response body decoders

- New `UniversalSysExEvent::mmc_command()` decodes a Real-Time MMC
  Command packet (`F0 7F <dev> 06 <command…> F7`, RP-013 §5) into a typed
  `MmcCommand { command: MmcCommandType, operands: Vec<u8> }`. The byte
  after the `0x06` `<mcc>` Sub-ID is the first command opcode (not a
  Universal Sub-ID #2), so the decoder reads the command stream directly.
  `MmcCommandType` names the full RP-013 §5 set (`Stop`/`Play`/`Rewind`/
  `Locate`/`Write`/`Step`/`Resume`/…); unknown opcodes pass through as
  `Other(u8)`. `MmcCommandType::is_motion_control()` flags the operand-
  less transport commands (`0x01`..=`0x0D`, `Wait`, `Resume`).
- New `UniversalSysExEvent::mmc_response()` decodes a Real-Time MMC
  Response packet (`F0 7F <dev> 07 <info-field…> F7`, RP-013 §6) into
  `MmcResponse { field: MmcInformationField, operands }`, typing the
  common timecode / status Information Fields.
- New `SmfFile::mmc_commands()` / `SmfFile::mmc_responses()` absolute-tick
  iterators mirror the `device_controls()` merge-by-tick contract. 14 new
  tests cover transport opcodes, `Locate` operands, high opcodes, unknown
  pass-through, the Non-Real-Time / Device-Control negative cases,
  truncation, and the multi-track iterator merge.
- Provenance: `docs/audio/midi/recommended-practices/RP-013_v1-0_MIDI_Machine_Control_Specification_96-1-4.pdf`.

### Round 368 — Device Control SysEx body decoder (`UniversalSysExEvent::device_control`)

- New `UniversalSysExEvent::device_control()` typed body decoder for the
  Real-Time Device Control family (`F0 7F <dev> 04 nn …`), mirroring the
  MTC / Notation Information body decoders. Returns a `DeviceControl`
  enum: `MasterVolume { value14 }` and `MasterBalance { value14 }` decode
  the 14-bit lsb-first value per the MIDI 1.0 Detailed Specification
  §"DEVICE CONTROL — MASTER VOLUME AND MASTER BALANCE" (`0x2000` =
  balance centre, `0x3FFF` = max / hard-right).
- `None` for non-Device-Control packets and for a correctly-classified
  packet truncated before the value pair arrives (no half-decoded value).
  Five tests cover Volume, Balance, the negative cases, and an
  end-to-end parse at an absolute tick.
- `DeviceControl::MasterFineTuning { value14 }` and `MasterCoarseTuning
  { msb, lsb }` decode the CA-025 (Master Fine/Coarse Tuning) messages.
  `DeviceControl::fine_tuning_cents()` returns the displacement in cents
  from A440 (`100/8192 × (value14 − 8192)`, `0x2000` = 0 cents) and
  `coarse_tuning_semitones()` the signed `msb − 64` semitone shift
  (`0x40` = 0, `−64 … +63`); both `None` for the wrong variant. Four
  added tests cover the CA-025 table endpoints, truncation, and a parsed
  SMF fine-tuning value.
- `DeviceControl::GlobalParameterControl { body }` surfaces the CA-024
  Global Parameter Control body bytes (sub-id2-onward, trailing `F7`
  stripped) for callers that walk the CA-024 slot-path structure
  themselves; the scheduler's GM2 effect bus owns the detailed walk.
- New `SmfFile::device_controls()` — the stably-merged absolute-tick
  iterator over the Device Control subset, one `DeviceControlEvent`
  (`tick` / `track` / decoded `DeviceControl`) per packet, matching the
  existing `channel_mode_messages()` / `effect_depths()` iterator family.
  Three added tests: GPC body decode, two-track tick-ordered merge, and
  the empty-result case.

### Round 361 — Reverb/Chorus Type select loads CA-024 table defaults

- `set_gm_reverb_param(0, type)` now also resets the Reverb Time to the
  selected type's CA-024 table default — the spec says "When a Reverb
  Type is selected, the default Reverb Time from the table below for
  that Reverb Type should be set" (Type 0 → 44, 1 → 50, 2 → 56, 3 → 64,
  4 → 64, 8 → 50). Previously the type select stored the type but left
  the time at its prior value.
- `set_gm_chorus_param(0, type)` now loads the whole Mod Rate / Mod
  Depth / Feedback / Send-to-Reverb row for the selected Chorus Type
  (CA-024 "pp = 0 : Chorus Type … Sets Chorus parameters as listed
  below"; the six-row Chorus table). An explicit `pp=1..4` edit sent
  afterward still overrides the row default.
- New `GmEffects::reverb_type_default_time_val` /
  `chorus_type_default_row` / `apply_chorus_type_defaults` table
  helpers back this. Two tests: reverb type → default time (with later
  pp=1 override), chorus type → parameter row (with later pp=3
  override).

### Round 361 — Effects-depth controller classifier + iterator (CC 91–95)

- New `smf::EffectDepth` enum + `ControlChangeEvent::effect_depth() ->
  Option<EffectDepth>`: classifies an "Effects N Depth" controller
  (CC 91–95) into Reverb Send / Tremolo / Chorus Send / Celeste /
  Phaser per the MIDI 1.0 *Control Change Messages — Data Bytes*
  document (Table 3), each carrying the `0..=127` value via `.level()`
  and reporting its source `.controller()`. CC 91 (default Reverb Send
  Level) and CC 93 (default Chorus Send Level) are the two sends the
  mixer's CA-024 effects bus consumes. Returns `None` outside 91–95.
- New `smf::EffectDepthEvent` + `SmfFile::effect_depths() ->
  Vec<EffectDepthEvent>`: the typed, absolute-tick, stably-merged
  (track 0 before track 1 at the same tick) iterator over the
  effects-depth subset of `control_changes`, with `channel()` /
  `depth()` accessors. Empty when no track carries CC 91–95.
- Tests: full 91–95 classification, none-outside-range, cross-track
  merge with non-effect CCs filtered, empty-result.

### Round 361 — GM2 Reverb + Chorus DSP send bus (CA-024)

- The system Reverb + Chorus parameters decoded from the GM2 Global
  Parameter Control SysEx (CA-024) are now **applied** as real DSP, not
  just stored. `mixer::Mixer` gains a stereo effects bus: a Schroeder
  reverb (parallel feedback comb bank → series allpass diffusers, comb
  feedback derived from the CA-024 Reverb Time so the −60 dB decay
  tracks the spec) and a sine-modulated delay-line chorus (rate / depth /
  feedback from the CA-024 chorus table). The chorus→reverb send
  (CA-024 chorus `pp=4`) routes the wet chorus output into the reverb
  input.
- New per-channel `ChannelState::reverb_send` (CC 91, Effects 1 Depth)
  and `chorus_send` (CC 93, Effects 3 Depth), wired through the
  scheduler's control-change dispatch. CA-024 specifies "the send levels
  to the Reverb and Chorus effects are controlled with Control Changes
  #91 and #93." Both default to 0 (fully dry) so a score that never
  touches the controllers renders bit-identically to the pre-effects
  path.
- `Mixer::set_sample_rate()` / `sample_rate()` size the effects delay
  lines for the output rate (the decoder calls it at construction);
  `Mixer::clear_effects()` flushes the tails, and GM System On/Off
  (`reset_gm_effects`) now flushes them too.
- Tests: dry-by-default bit-identity, reverb + chorus tails outlasting
  the dry note, longer Reverb Time retaining more tail energy,
  chorus→reverb routing, tail flush on GM reset, and sample-rate
  resize.

### Round 345 — typed Channel Mode Message classifier

- New `ChannelModeMessage` enum + `ControlChangeEvent::channel_mode() ->
  Option<ChannelModeMessage>`: classifies a channel-mode controller
  (`Bn cc vv`, `cc` in `120..=127`) into All Sound Off, Reset All
  Controllers, Local Control (`on` decoded from the value-byte switch:
  `< 64` off, `>= 64` on), All Notes Off, Omni Off, Omni On, Mono On
  (`channels` = the value byte `m`), or Poly On — MIDI 1.0 Detailed
  Specification §"Channel Mode Messages". `is_all_notes_off()` flags
  messages 123–127, which the spec says also act as All Notes Off.
  Returns `None` for a continuous controller (`0..=119`).
- New `ChannelModeEvent` struct + `SmfFile::channel_mode_messages() ->
  Vec<ChannelModeEvent>`: the typed, absolute-tick, stably-merged
  iterator over the channel-mode subset of the `Bn` stream (continuous
  controllers filtered out), with `channel()` / `message()` accessors.

### Round 345 — multi-packet SysEx reassembly

- New `ReassembledSysEx` struct + `SmfFile::reassembled_sysex_messages()
  -> Vec<ReassembledSysEx>`: folds the per-track `F0` / `F7` SysEx
  packet stream into complete logical messages. An `F0` opener whose
  payload does not end in `0xF7` is continued by `F7` packets until one
  ends in `0xF7` (EOX); the helper runs that continuation state machine
  per track and emits one message per chain, in opener-tick order. A
  self-contained `F0 … F7` is a 1-packet message; a `F7` packet with no
  open chain is surfaced as a standalone escape; a new `F0` opener (or
  end of track) while a chain is still open flushes the prior chain as
  `complete == false`. `body` concatenates the manufacturer payloads
  with the final `0xF7` stripped; `id_byte()` / `is_universal()` expose
  the opening manufacturer / Universal-realm byte; `complete` and
  `packet_count` report chain termination + length. (MIDI 1.0 Detailed
  Specification, "EOX — End of Exclusive".)

### Round 345 — SMF builder API (`SmfBuilder` / `TrackBuilder`)

- New `TrackBuilder`: assemble a `Track` from events placed at
  **absolute ticks** in any order, deferring the SMF delta-time
  arithmetic to `build()`. Events are stably sorted by tick (insertion
  order breaks ties, so a Program Change added before a Note On at the
  same tick stays before it); each event's delta is `tick −
  previous_tick`; a trailing `MetaEvent::EndOfTrack` is appended
  automatically (at the last tick) unless the caller already added one,
  so the output is always a legal `MTrk`. `push` / `channel` / `meta`
  place events; the `note(tick, duration, channel, key, velocity)`
  helper emits a Note On + matching Note Off (`8n key 0`) pair. An empty
  builder yields a lone End-of-Track (the minimum legal track).
- New `SmfBuilder`: assemble a complete `SmfFile`, keeping `ntrks` in
  sync with the track count so the result always satisfies
  `SmfFile::to_bytes`'s header-consistency check. `format` / `division`
  override the defaults (format 1, 480 ticks-per-quarter); `add_track`
  appends a finished `Track`, and `track(|t| …)` builds a `TrackBuilder`
  inline. Round-trips through `to_bytes` / `to_bytes_running_status`.

### Round 345 — running-status writer

- New `SmfFile::to_bytes_running_status() -> Result<Vec<u8>>` and
  `Track::to_bytes_chunk_running_status() -> Result<Vec<u8>>`:
  serialise with **running-status compression** (MIDI 1.0 Detailed
  Specification, "Running Status"). A channel-voice event whose status
  byte equals the immediately-preceding emitted status omits the
  redundant status byte; the running-status buffer is cleared by every
  meta event and every sysex (`F0` / `F7`) event — exactly the reset
  points the parser observes — so the compressed stream still
  round-trips byte-for-byte through `parse`. The existing `to_bytes` /
  `to_bytes_chunk` remain the explicit-status default (bit-stable
  output). Same strict validation as the explicit writer. Status-byte
  compression only: the encoder never rewrites a Note On velocity-0
  into a running Note Off or vice versa, preserving caller event shapes.

### Round 337 — Notation Information Bar Number + Time Signature body decoders

- New `NotationBarNumber` struct + `UniversalSysExEvent::
  notation_bar_number() -> Option<NotationBarNumber>`: decodes a
  Real-Time Notation Information **Bar Number** message
  (`F0 7F <dev> 03 01 aa aa F7`, MIDI 1.0 Detailed Specification
  §"Notation Information — Bar Marker") into its signed 14-bit `aa aa`
  field (assembled lsb-first from the two 7-bit data bytes).
  `raw14` is the unsigned field, `value()` the sign-extended `i16`
  (`−8192 ..= 8191`), and `state()` classifies it into the documented
  regions via the new `NotationBarState` enum: `NotRunning` (`0x2000`),
  `CountIn(i16)` (the negative-through-zero range, bar `0` = last count-
  in bar), `BarInSong(i16)` (`1 ..= 8062`), and `RunningUnknown`
  (`0x1F7F`). Returns `None` unless the packet classifies as
  `RtNotationBarNumber` and carries both `aa` bytes.
- New `NotationTimeSignature` + `NotationTimeSignaturePair` structs +
  `UniversalSysExEvent::notation_time_signature() ->
  Option<NotationTimeSignature>`: decodes a Real-Time Notation
  Information **Time Signature** message in both forms — Immediate
  (`03 02`, effective on receipt) and Delayed (`03 42`, effective on the
  next Bar Marker; `is_delayed()`). The body `ln nn dd cc bb [nn dd ...]`
  duplicates the `FF 58` Standard MIDI File Time Signature meta event
  layout, extended with compound-signature pairs. `pairs` carries every
  `numerator` / `denominator_pow2` pair in wire order (`primary()`,
  `is_compound()`); `NotationTimeSignaturePair::denominator()` decodes
  `1 << denominator_pow2`. `clocks_per_click` / `thirty_seconds_per_
  quarter` are the leading pair's metronome bytes. Returns `None` when
  `ln` is below the 4-byte minimum, is not of the `4 + 2k` form, or the
  body is truncated before `ln` bytes arrive.
- Ten new unit tests (in-song / count-in / flag regions / truncation /
  parsed-SMF for Bar Number; simple / delayed / compound / invalid-ln /
  parsed-SMF for Time Signature). Full lib suite 586 → 596 tests, zero
  ignored.

### Round 332 — MTC User Bits Message body decoder

- New `MtcUserBits` struct + `UniversalSysExEvent::mtc_user_bits()
  -> Option<MtcUserBits>`: decodes a Real-Time MIDI Time Code **User
  Bits Message** (`F0 7F <dev> 01 02 u1..u9 F7`, RP-004/008 §"User
  Bits") into its eight SMPTE/EBU Binary Groups (`groups: [u8; 8]`,
  each masked to the low nibble of its `0000xxxx` payload byte) plus
  the `u9` Binary Group Flag-Bits byte. Returns `None` unless the
  packet classifies as `UniversalSubId2::RtMtcUserBits` *and* carries
  all nine `u1..u9` bytes (truncated streams yield `None`).
- `MtcUserBits::flag_i()` / `flag_j()` isolate the two flag bits
  (`i` = `u9` bit 0 = SMPTE bit 43 / EBU bit 27; `j` = `u9` bit 1 =
  SMPTE bit 59 / EBU bit 43). `reassembled()` returns the 32-bit
  value the eight groups form per the November-1991 redefinition
  (`hhhhgggg ffffeeee ddddcccc bbbbaaaa`: Group 1 in the least-
  significant nibble, Group 8 in the most-significant).

### Round 327 — MTC Full Message body decoder

- New `MtcFullMessage` struct + `UniversalSysExEvent::mtc_full_message()
  -> Option<MtcFullMessage>`: decodes a Real-Time MIDI Time Code **Full
  Message** (`F0 7F <dev> 01 01 hr mn sc fr F7`, RP-004/008 §"Full
  Message") into its SMPTE `hr / mn / sc / fr` quartet. Returns `None`
  unless the packet classifies as `UniversalSubId2::RtMtcFullMessage`
  *and* carries all four time bytes (truncated streams yield `None`
  rather than a half-decoded time).
- `MtcFullMessage::frame_rate()` decodes bits 5-6 of the `hr` byte
  through the shared `FrameRate::from_hours_byte`; `hours_count()`
  returns the low five bits (`0..=23`). Counter fields are surfaced
  verbatim — no per-rate clamping — so pathological generators stay
  visible to the caller.

### Round 320 — `SmfFile` tick → wall-clock-seconds conversion

- New `SmfFile::tempo_timeline() -> TempoTimeline`: precomputes the
  per-segment mapping from absolute division ticks to elapsed seconds by
  folding `tempo_map()` against the header `Division`. The returned
  `TempoTimeline::tick_to_seconds(tick)` resolves any tick in `O(log n)`
  via binary search over tempo segments; `TempoTimeline::segments()`
  exposes the `(start_tick, seconds_at_start, seconds_per_tick)` anchors.
- New `SmfFile::tick_to_seconds(tick) -> f64` one-shot convenience and
  `SmfFile::duration_seconds() -> f64` (elapsed time at the max end tick
  across all tracks; `0.0` for an empty file).
- New public `const DEFAULT_TEMPO_US_PER_QUARTER: u32 = 500_000` (the SMF
  120-BPM default assumed before the first Set Tempo).
- Conversion semantics mirror the scheduler's tick→sample arithmetic:
  for `Division::TicksPerQuarter`, `seconds_per_tick =
  (usec_per_quarter / 1_000_000) / division`, integrated piecewise
  across each `FF 51` Set Tempo change (a change repeated at the same
  tick keeps its last value); for `Division::Smpte`, the rate is fixed
  at `1 / (frames_per_second × ticks_per_frame)` and Set Tempo events do
  not apply. The `frames_per_second` byte is used verbatim (29 treated
  literally, as the scheduler does). Result is monotonically
  non-decreasing in tick, so reported duration agrees with rendered
  output length.
- Seven new unit tests: default-120-BPM / initial-tempo-change /
  piecewise-across-changes (+ monotonicity) / SMPTE-ignores-tempo /
  duration-max-end-tick / empty-file-zero / same-tick-last-wins. Full
  lib suite 569 → 576 tests, zero ignored.

### Round 316 — `SmfFile::parameter_data_entries()` RPN / NRPN Data Entry pump decoder

- New `SmfFile::parameter_data_entries() -> Vec<ParameterDataEntry>`:
  folds the CC 6 / CC 38 (Data Entry MSB / LSB) and CC 96 / CC 97 (Data
  Increment / Decrement) pump against each channel's running RPN
  (CC 101 / CC 100) / NRPN (CC 99 / CC 98) selector state, emitting one
  resolved parameter-write event per pump action. The read-side analogue
  of the runtime RPN/NRPN state machine and the resolving companion to
  `control_changes()`, which deliberately leaves the pump un-decoded.
- New public types: `ParameterDataEntry` (tick / track / channel /
  resolved parameter / action), `SelectedParameter`
  (`Registered { number, param }` vs `NonRegistered { number }` with
  `number()` / `is_registered()` / `is_null()`), `DataEntryAction`
  (`EntryMsb` / `EntryLsb` / `Increment` / `Decrement`), and
  `RegisteredParameter` classifying Table 3a: Pitch Bend Sensitivity,
  Channel Fine / Coarse Tuning, Tuning Program Change / Bank Select,
  Modulation Depth Range, MPE Configuration, the nine RP-049 3D Sound
  Controllers, the Null Function Number, and a `Reserved(number)`
  catch-all.
- Selector semantics per MIDI 1.0 *Control Change Messages — Data
  Bytes* (Table 3 / 3a + the "set or change a Registered Parameter"
  procedure): one active parameter per channel (RPN supersedes NRPN and
  vice versa), lone MSB / LSB rewrites preserve the other byte, the Null
  Function Number (packed `0x3FFF`) and the power-up default disable the
  pump (no event emitted), and inc/dec drop their "don't care" value
  byte. State evolves along the globally merged `(tick, track, order)`
  stream so a conductor-track selector governs a later part-track entry.
- Twelve new unit tests: empty / no-selection-drop / RPN0 MSB+LSB /
  NRPN raw number / inc-dec value-drop / null-disables / RPN-supersedes-
  NRPN / lone-LSB-rewrite / 3D-controller packing / reserved-RPN /
  per-channel-independence / cross-track selector. Full lib suite
  557 → 569 tests, zero ignored.

## [0.0.4](https://github.com/OxideAV/oxideav-midi/compare/v0.0.3...v0.0.4) - 2026-06-14

### Other

- midi r307: SmfFile::active_notes_at() sounding-note seek lens
- add SmfFile::poly_aftertouches() per-key aftertouch iterator
- add SmfFile::notes() Note On/Off pairing into sounding-note spans
- run-segmented volume-envelope evaluation — bit-identical, ~20% faster SMF→PCM synthesis
- add SmfFile::channel_pressures() — Dn-pp mono-aftertouch iteration helper
- SMF SmfFile::pitch_bends() — En-lsb-msb channel-voice pitch-bend iteration helper
- SmfFile::control_changes() — Bn-cc-vv channel-voice continuous-controller / channel-mode iteration helper
- SmfFile::program_changes() — Cn-pp channel-voice patch-select iteration helper
- SmfFile::universal_sysex_events() — Table-4-classified file-wide iteration helper
- SysExEvent::universal_classification() — Table 4 Universal SysEx classifier
- SmfFile::sysex_events() iteration helper (F0 / F7)
- SmfFile::channel_prefixes() iteration helper (FF 20 01 cc)
- SmfFile::to_bytes() / Track::to_bytes_chunk() mux-side writer
- SmfFile::midi_ports() iteration helper (FF 21 01 pp)
- SmfFile::sequence_numbers() iteration helper (FF 00 02 ssss)
- SmfFile::sequencer_specifics() iteration helper (FF 7F)
- SmfFile::channel_snapshot_at / channel_snapshots_at (channel-state seek primitive)
- SmfFile::smpte_offsets() iteration helper + FrameRate decoder (FF 54)
- SmfFile::texts() + copyrights() iteration helpers (FF 01 + FF 02)
- SmfFile::instrument_names() iteration helper (FF 04)

### Round 307 — `SmfFile::active_notes_at()` sounding-note seek lens

- New `SmfFile::active_notes_at(tick) -> Vec<Note>`: every `Note` span
  sounding at the absolute `tick` — the piano-roll / seek companion to
  `notes()`, and the note-level analogue of the channel-state
  `channel_snapshot_at` primitive. Where the snapshot answers "what
  controller / program / bend state does a channel carry at tick T?",
  this answers "which keys are held down at tick T?" — exactly the set a
  DAW must re-trigger (or a renderer must prime into the voice pool) when
  seeking into the middle of a file rather than playing from the top.
- A note is sounding when `start_tick <= tick && end_tick > tick` — the
  half-open interval `[start_tick, end_tick)`. The onset tick is
  inclusive (the snapshot reflects state immediately after that tick's
  events fire, the same convention as `channel_snapshot_at`); the release
  tick is exclusive (the key has come up). A zero-duration note
  (`start_tick == end_tick`) is sounding at no tick. Hanging onsets and
  unmatched releases — already dropped by `notes()` — cannot be reported.
  The result preserves the `notes()` `(start_tick, track)` order so chord
  notes stay grouped and track 0 precedes track 1.
- Seven new unit tests: empty-when-silent, half-open boundary
  (onset-inclusive / release-exclusive / mid-span / after-release),
  before-onset silence, zero-duration never sounds, chord returns all
  held keys in onset order, staggered overlap window (only-n1 / both /
  only-n2), and hanging-note never sounds. Full lib suite 550 → 557
  tests, zero ignored.

### Round 301 — `SmfFile::poly_aftertouches()` Polyphonic Key Pressure stream

- New `SmfFile::poly_aftertouches() -> Vec<PolyAftertouchEvent>`: every
  `An kk pp` Polyphonic Key Pressure (per-key aftertouch) channel-voice
  event as a `PolyAftertouchEvent { tick, track, channel, key, pressure
  }` pinned to the absolute tick on its parent track, stably merged
  across tracks (track 0 before track 1 at the same tick) under the same
  convention every existing iteration helper and the scheduler use.
- `An` was the only channel-voice status nibble without a dedicated
  typed extraction helper. Distinct from Channel Pressure (`Dn`,
  surfaced by `channel_pressures()`) — `An` carries a per-key `kk` byte,
  so per-voice aftertouch automation can be rebuilt. Accessors
  `channel()` / `key()` / `pressure()` return the decoded fields.
- Nine new unit tests: tick-zero decode, low-nibble channel index,
  running-status (key, pressure) pair chaining, late-position absolute
  tick, two-track stable sort, cross-track tick merge, filter exclusion
  of every other channel-voice kind, `to_bytes()`/`parse()` round trip,
  and empty-when-none. Full lib suite 541 → 550 tests, zero ignored.

### Round 292 — `SmfFile::notes()` Note On / Note Off pairing into sounding-note spans

- New `SmfFile::notes() -> Vec<Note>`: pairs every Note On
  (`9n key vel`, `vel > 0`) with the Note Off that releases it and
  returns one `Note { start_tick, end_tick, track, channel, key,
  velocity, off_velocity }` span per sounding note, ordered by onset.
  Where the channel-voice helpers (`program_changes` /
  `control_changes` / `pitch_bends` / `channel_pressures`) surface one
  value per *wire* event, this helper joins the two wire events that
  bracket a note into a single span carrying its duration — the
  primitive a piano-roll / DAW note-lane view consumes directly.
- Honours the MIDI 1.0 *Summary of MIDI Messages* Table 1 velocity-0
  convention: a `9n key 0` is treated as a Note Off and closes the
  earliest open note of that pitch (FIFO), with `off_velocity == 0`.
  An explicit `8n key off_vel` carries its release velocity through to
  `Note::off_velocity`.
- Matches over the *globally* merged event stream sorted by
  `(absolute tick, track, in-track position)` — the same stable-merge
  convention every other iteration helper and the scheduler use — so a
  Note Off on a different track from its Note On still pairs correctly.
  Overlapping notes of the same `(channel, key)` are matched FIFO; an
  unmatched release or a hanging onset is dropped from the span list.
- `Note` accessors: `channel()` / `key()` / `velocity()` /
  `off_velocity()` / `duration_ticks()` (`end_tick - start_tick`).
- 13 new unit tests: single-note pairing, velocity-0 close, channel
  decode, FIFO overlap, chord (distinct pitches + cross-track
  ordering), cross-track off pairing, hanging-on / unmatched-off drop,
  zero-duration span, sibling-helper isolation, and a `to_bytes()` /
  `parse` structural round trip.

### Round 285 — synthesis profiling + run-segmented volume envelope (bit-identical, ~20 % faster)

- New `benches/synth_render.rs` (`harness = false`) — repeatable
  SMF→PCM wall-clock harness: dense 8-channel / 32-voice score with
  pitch-bend sweeps + volume/pan CCs rendered through an in-memory
  looping SF2 bank. `--corpus` hashes (FNV-1a-64) the PCM for every
  in-tree fixture SMF through both the SF2 bank and the tone
  fallback; `--spin SECS` loops the render as a sampling-profiler
  target; default mode prints per-iteration wall time + output hash.
- Profiling ranked `Sf2Voice::render` at ~89 % of the synthesis wall
  clock; within it the per-sample DAHDSR volume-envelope stage walk
  (an `Option` test + up to four stage comparisons + an f32 divide,
  serialised behind the phase walk) at ~31 % of total, ahead of
  sample fetch + linear interpolation (~15 %).
- `Sf2Voice::render` now evaluates the volume envelope in
  stage-segmented runs (`envelope_run`) into a 256-entry stack
  buffer: delay / hold / sustain become slice fills, attack / decay /
  release become element-wise loops with no loop-carried dependency
  that the compiler vectorises. Every per-sample expression is kept
  verbatim from `envelope_at`, so the rendered PCM is bit-identical —
  corpus hashes are unchanged and the new
  `envelope_run_matches_envelope_at_per_sample` test pins
  `to_bits()` equality across stage boundaries, the release tail,
  and the `elapsed`-wrap fallback. Dense-score render: 80.2 ms →
  64.2 ms (-20 %) on an Apple-silicon dev box.

### Round 275 — `SmfFile::channel_pressures()` — Dn-pp channel-voice mono-aftertouch iteration helper

- New `SmfFile::channel_pressures(&self) -> Vec<ChannelPressureEvent>`
  surfaces every `Dn pp` Channel Pressure (mono aftertouch)
  channel-voice event on every track, pinned to the absolute tick at
  which it fires, in time order. Each entry is a
  `ChannelPressureEvent { tick, track, channel, pressure }` with the
  status nibble's low four bits decoded into the spec's `0..=15`
  channel index and the single data byte `pp` (`0..=127`) carrying
  "the single greatest pressure value (of all the current depressed
  keys)" per the MIDI 1.0 *Summary of MIDI Messages* Table 1.
- The new `ChannelPressureEvent` struct exposes `channel()` /
  `pressure()` accessors. The helper stays routing-agnostic — the
  pressure value's musical effect (volume / vibrato depth / filter
  cutoff) is left to the receiving instrument.
- Only `Dn` is selected; polyphonic key pressure (`An`, per-key) keeps
  its own surface, and the neighbouring CC (`Bn`) / program (`Cn`) /
  pitch-bend (`En`) / note (`8n` / `9n`) channel-voice events stay
  isolated. Per-track sequences are stably merged by absolute tick
  (track 0 before track 1 at the same tick), the same convention as
  every meta-event, SysEx, and channel-voice helper.
- 8 new unit tests cover tick-zero decode, channel-index nibble,
  running-status chains (single-data-byte status), late-position
  absolute tick, stable same-tick cross-track sort, cross-track tick
  merge, cross-kind filtering, and a `to_bytes()` / `parse` round trip.

### Round 267 — `SmfFile::pitch_bends()` — En-lsb-msb channel-voice pitch-bend iteration helper

- New `SmfFile::pitch_bends(&self) -> Vec<PitchBendEvent>` surfaces
  every `En lsb msb` Pitch Bend channel-voice event on every track,
  pinned to the absolute tick at which it fires, in time order. Each
  entry is a `PitchBendEvent { tick, track, channel, value }` with the
  status nibble's low four bits decoded into the spec's `0..=15`
  channel index and the two data bytes combined into the 14-bit code
  `(msb << 7) | lsb`, `0..=0x3FFF`, no-bend centre `0x2000` (the parser
  assembles the value at decode time).
- The new `PitchBendEvent` struct carries the same `tick` / `track`
  pair the existing channel-voice iteration helpers use plus the
  decoded `channel` and the assembled 14-bit `value`. The `channel()`
  / `value()` accessors return the fields with mnemonic names; a
  `signed_value() -> i16` accessor returns the displacement from centre
  in `-8192..=8191` (`value as i32 - 0x2000`, so `0x2000` → `0`,
  `0x0000` → `-8192`, `0x3FFF` → `8191`); an `is_centre()` predicate
  flags the no-bend position for bend-lane collapse / wheel-release
  detection.
- Resolving the 14-bit code to an actual pitch displacement requires
  the channel's Pitch Bend Sensitivity (RPN 0, default ±2 semitones),
  which is intentionally left to the receiving application: the helper
  stays sensitivity-agnostic and surfaces the raw code so callers pick
  their own bend-range policy.
- Per-track sequences are stably merged by absolute tick — track 0's
  events fire before track 1's at the same tick — the same convention
  used by every existing iteration helper (`control_changes()`,
  `program_changes()`, the SysEx and meta-event families, …) and the
  scheduler's merged event list.
- Companion to the wire-state primitive `SmfFile::channel_snapshot_at`,
  which folds the *last* pitch bend per channel into
  `SmfChannelSnapshot::pitch_bend` at the seek point. Where the
  snapshot answers "what is the wheel position on channel N at tick
  T?", `pitch_bends()` answers "give me every bend in song order" —
  the typed accessor a DAW bend-lane editor or a glissando / vibrato
  curve renderer reads.
- 10 new tests cover the empty-input case, the centre value at tick
  zero, the `lsb` / `msb` 14-bit combine at both extremes (`0x0000`
  signed `-8192`, `0x3FFF` signed `+8191`), channel-index decode across
  the `0..=15` range, running-status reuse on the `En` two-data-byte
  status (three chained bends sharing one status byte), late-position
  absolute-tick tracking, stable-sort merge of same-tick events across
  tracks, time-ordered merge of different-tick events across tracks,
  filtering against the other six channel-voice status kinds (`8n`,
  `9n`, `An`, `Bn`, `Cn`, `Dn`), a cross-check against
  `channel_snapshot_at` confirming the snapshot's `pitch_bend` field
  tracks the iterator at each change point and on the silence between
  changes, and a `to_bytes()` / `parse()` round trip confirming the
  bend stream survives the mux-side writer.

### Round 260 — `SmfFile::control_changes()` — Bn-cc-vv channel-voice continuous-controller / channel-mode iteration helper

- New `SmfFile::control_changes(&self) -> Vec<ControlChangeEvent>`
  surfaces every `Bn cc vv` Control Change channel-voice event on
  every track, pinned to the absolute tick at which it fires, in time
  order. Each entry is a `ControlChangeEvent { tick, track, channel,
  controller, value }` with the status nibble's low four bits decoded
  into the spec's `0..=15` channel index and both data bytes `cc` /
  `vv` surfaced as raw `0..=127` payloads.
- The new `ControlChangeEvent` struct carries the same `tick` /
  `track` pair the existing iteration helpers use plus the decoded
  `channel`, `controller`, and `value` bytes. The
  `channel()` / `controller()` / `value()` accessor methods return the
  same fields with mnemonic names so call sites driving a
  controller-automation view read `cc.controller()` / `cc.value()`
  rather than the bare field access. A `is_channel_mode()` predicate
  flags the channel-mode family (`controller` in `120..=127`) — All
  Sound Off (`120`), Reset All Controllers (`121`), Local Control
  (`122`), All Notes Off (`123`), Omni Mode Off / On (`124` / `125`),
  Mono / Poly Mode On (`126` / `127`) — so a player reset-detector
  can route on the predicate without re-checking the controller
  range manually.
- Resolution against a controller vocabulary (the MIDI 1.0
  *Control Change Messages — Data Bytes* table's 14-bit MSB / LSB
  pairing for controllers `0..=31` plus `32..=63`, the on-off
  threshold `value >= 64` for switch controllers `64..=69`, the
  CC-6 / CC-38 Data Entry pump that drives RPN / NRPN parameter writes
  selected through CC-100 / CC-101 RPN and CC-98 / CC-99 NRPN pairs)
  is intentionally left to the receiving application: the helper
  stays controller-agnostic and surfaces the raw value byte so
  callers pick their own controller-vocabulary policy.
- Per-track sequences are stably merged by absolute tick — track 0's
  events fire before track 1's at the same tick — the same convention
  used by every existing iteration helper (`program_changes()`,
  `sysex_events()`, `universal_sysex_events()`, the meta-event
  family, …) and the scheduler's merged event list.
- Companion to the wire-state primitive `SmfFile::channel_snapshot_at`,
  which folds the *last* value of the six snapshot-tracked
  controllers (Bank MSB / LSB, Modulation, Volume, Pan, Expression,
  Sustain) into the snapshot at the seek point. Where the snapshot
  answers "what value is CC-7 / CC-10 / … at tick T?",
  `control_changes()` answers "give me every controller change in
  song order, including the controllers the snapshot doesn't track"
  — the typed accessor a DAW lane-editor or a CC-1 modulation-curve
  renderer reads.
- 12 new tests cover the empty-input case, single-controller decode,
  the full `0..=127` value range, channel-index decode across the
  `0..=15` range, running-status reuse on the `Bn` two-data-byte
  status (three chained CCs sharing one status byte), late-position
  absolute-tick tracking, stable-sort merge of same-tick events
  across tracks, time-ordered merge of different-tick events across
  tracks, filtering against the other six channel-voice status kinds
  (`8n`, `9n`, `An`, `Cn`, `Dn`, `En`), the channel-mode family
  (`120..=127`) flagged by `is_channel_mode()` alongside a continuous
  CC-7 surface that doesn't, a cross-check against
  `channel_snapshot_at` confirming the snapshot's `volume` field
  tracks the iterator at each change point and on the silence
  between changes, and a `to_bytes()` / `parse()` round trip
  confirming the CC stream survives the mux-side writer. Total
  lib-test count: **508** (up from 496).

### Round 254 — `SmfFile::program_changes()` — Cn-pp channel-voice patch-select iteration helper

- New `SmfFile::program_changes(&self) -> Vec<ProgramChangeEvent>`
  surfaces every `Cn pp` Program Change channel-voice event on every
  track, pinned to the absolute tick at which it fires, in time order.
  Each entry is a `ProgramChangeEvent { tick, track, channel, program }`
  with the status nibble's low four bits decoded into the spec's
  `0..=15` channel index and the single data byte `pp` surfaced as the
  `0..=127` patch number.
- The new `ProgramChangeEvent` struct carries the same `tick` / `track`
  pair the existing iteration helpers use plus the decoded `channel` and
  raw `program` bytes. `channel()` and `program()` accessor methods
  return the same fields with mnemonic names so call sites driving an
  instrument-list view read `pc.channel()` / `pc.program()` rather than
  the bare field access.
- Resolution against a patch list (General MIDI 1 / 2, GS, XG, …) is
  intentionally left to the receiving application: the helper stays
  bank-agnostic and surfaces the raw program byte so callers can pick
  their own bank-select (CC 0 / CC 32) policy.
- Per-track sequences are stably merged by absolute tick — track 0's
  events fire before track 1's at the same tick — the same convention
  used by every existing iteration helper
  (`sysex_events()`, `universal_sysex_events()`, the meta-event family,
  …) and the scheduler's merged event list.
- Companion to the wire-state primitive `SmfFile::channel_snapshot_at`,
  which folds the *last* Program Change per channel into
  `SmfChannelSnapshot::program` for seek initialisation. Where the
  snapshot answers "what patch is this channel on at tick T?",
  `program_changes()` answers "give me every patch change in song
  order" — the typed accessor a DAW track-inspector view (highlighting
  the bar each instrument enters) reads.
- 10 new tests cover the empty-input case, single-patch decode, the
  full `0..=127` program range, channel-index decode across the
  `0..=15` range, running-status reuse on the `Cn` single-data-byte
  status, late-position absolute-tick tracking, stable-sort merge of
  same-tick events across tracks, time-ordered merge of different-tick
  events across tracks, filtering against the other six channel-voice
  status kinds (`8n`, `9n`, `An`, `Bn`, `Dn`, `En`), and a cross-check
  against `channel_snapshot_at` confirming the snapshot's `program`
  field tracks the iterator at each change point and on the silence
  between changes. Total lib-test count: **496** (up from 486).

### Round 251 — `SmfFile::universal_sysex_events()` — Table-4-classified file-wide iteration helper

- New `SmfFile::universal_sysex_events(&self) -> Vec<UniversalSysExEvent>`
  surfaces every Universal System Exclusive packet on every track —
  `F0 7E …` (Non-Real-Time) and `F0 7F …` (Real-Time) only — pinned
  to the absolute tick at which it fires, in time order, with the
  Table-4 classification already resolved. Each entry is a
  `UniversalSysExEvent { tick, track, classification, data }`.
- The new `UniversalSysExEvent` struct carries the same `tick` /
  `track` pair the existing iteration helpers use, the parsed
  `UniversalSysEx { realm, device_id, sub_id1 }` classification
  the round-246 per-event classifier returns, and the verbatim
  payload bytes from the wire (leading `<realm>` byte through
  trailing `0xF7` when present) so callers reading Sub-ID #2-
  derived arguments — Master Volume's 14-bit value, MTC Full
  Message's `hr / mn / se / fr` quartet, MTS Single Note Tuning's
  note + tuning triple, … — don't have to re-walk
  `SmfFile::sysex_events()` alongside the typed list.
- Manufacturer-prefixed `F0` packets (Roland `0x41`, Yamaha `0x43`,
  any leading byte other than `0x7E` / `0x7F`) and `F7`
  continuation / escape packets are filtered out — callers
  interested in those route through `SmfFile::sysex_events()` and
  `SysExEvent::manufacturer_id()` directly. `F0` packets truncated
  before the Sub-ID #1 byte (payload shorter than 3 bytes) are also
  filtered, matching the contract of the underlying
  `SysExEvent::universal_classification()` classifier so the two
  views agree on which packets are universal.
- Per-track sequences are stably merged by absolute tick — track 0's
  universal packets fire before track 1's at the same tick — the
  same convention used by `SmfFile::sysex_events()` /
  `tempo_map()` / `time_signatures()` / `key_signatures()` /
  `markers()` / `lyrics()` / `cue_points()` / `track_names()` /
  `instrument_names()` / `texts()` / `copyrights()` /
  `smpte_offsets()` / `sequencer_specifics()` /
  `sequence_numbers()` / `midi_ports()` / `channel_prefixes()`
  and `scheduler.rs` §"merged event list, sorted by absolute tick".
- The helper re-uses `SysExEvent::universal_classification()` for the
  per-packet decode so any future tweak to the Table 4 vocabulary
  lands in exactly one place; a regression test walks `sysex_events()`
  in parallel and confirms the typed view stays byte-for-byte aligned
  with the per-event classifier across a mixed packet sample.
- 7 new unit tests cover the empty-file case, the manufacturer-
  prefixed + escape filter, both-realm classification, cross-track
  stable merging, same-tick stable-sort ordering (track 0 wins),
  truncated-universal-packet drop, and the per-event classifier
  parity check.

### Round 246 — `SysExEvent::universal_classification()` — Table 4 Universal SysEx classifier

- New `SysExEvent::universal_classification(&self) -> Option<UniversalSysEx>`
  classifies a Universal System Exclusive packet against Table 4 of the
  MIDI 1.0 *Universal System Exclusive Messages* document, returning a
  `UniversalSysEx { realm, device_id, sub_id1 }` for every `F0` packet
  whose leading byte is `0x7E` (Universal Non-Real-Time) or `0x7F`
  (Universal Real-Time). The Sub-ID #1 byte is parsed against the
  Table 4 vocabulary into a `UniversalSubId1` enum that names every
  category in the document; categories that branch on Sub-ID #2 carry
  a `UniversalSubId2` payload that names every Sub-ID #2 value the
  document defines for the category-realm pair. Sub-ID #1 / Sub-ID #2
  bytes outside the Table 4 vocabulary the classifier knows about
  surface through `UniversalSubId1::Other(raw)` /
  `UniversalSubId2::Other(raw)` so callers with deeper, more recent
  vocabulary can still route the packet.
- The classifier is realm-aware: the same `(sub_id1, sub_id2)` byte
  pair names different messages in the two realms (e.g. `0x09 0x01`
  is `GeneralMidi1SystemOn` in Non-Real-Time and
  `ControllerDestinationChannelPressure` in Real-Time, per Table 4);
  the classifier decodes against the parsed `realm` before matching.
  Singleton Sub-ID #1 categories (`0x7B` End of File through `0x7F`
  ACK) are singleton-shaped and report as unit variants of
  `UniversalSubId1`. `F7` continuation / escape packets, manufacturer-
  prefixed `F0` packets (any leading byte other than `0x7E` / `0x7F`),
  and `F0` packets truncated before Sub-ID #1 (fewer than 3 payload
  bytes) return `None` so callers route them through
  `SysExEvent::manufacturer_id` or the raw `SysExEvent::data` instead.
- The `UniversalSubId1` enum surfaces every Sub-ID #1 byte the
  document defines: Sample Dump Header / Data Packet / Request (the
  three Non-Real-Time singletons at `0x01..=0x03`), MIDI Time Code
  (Non-RT Setup at `0x04` / RT Quarter-Frame at `0x01`), Sample Dump
  Extensions, General Information, File Dump, MIDI Tuning Standard,
  General MIDI / Controller Destination Setting, Downloadable Sounds /
  Key-Based Instrument Control, File Reference / Scalable Polyphony
  MIP, MIDI Visual Control / Mobile Phone Control, MIDI Capability
  Inquiry, MIDI Show Control, Notation Information, Device Control,
  Real-Time MTC Cueing — plus the five Non-Real-Time singletons (End
  of File, Wait, Cancel, NAK, ACK).
- The `UniversalSubId2` enum surfaces every Sub-ID #2 byte the
  document defines for those categories: 15 Non-Real-Time MTC Setup
  sub-categories (Special, Punch In / Out Points, Delete Punch In /
  Out Point, Event Start / Stop Point, the four "with additional
  info" variants, Cue Points, Delete Cue Point, Event Name with
  additional info), 7 Sample Dump Extension sub-categories (Loop
  Points / Sample Name transmission and request, Extended Dump
  Header, Extended Loop Points transmission and request), the
  Identity Request / Reply pair, File Dump Header / Data Packet /
  Request, 10 MTS sub-categories (Bulk Dump Request / Reply, Tuning
  Dump Request, Key-Based Tuning Dump, Scale/Octave Tuning Dump in 1-
  and 2-byte formats, Single Note Tuning Change with Bank Select,
  Scale/Octave Tuning in 1- and 2-byte formats, RT Single Note
  Tuning Change), GM 1 / GM Off / GM 2 System On, the DLS quartet
  (Turn DLS On / Off / Voice Allocation Off / On), the File
  Reference quartet (Open File, Select or Reselect Contents, Open
  File and Select Contents, Close File), MTC Full Message / User
  Bits, MSC Extensions, Notation Bar Number / Time Signature
  Immediate / Time Signature Delayed, the five Device Control
  variants (Master Volume / Balance / Fine Tuning / Coarse Tuning /
  Global Parameter Control), 10 RT MTC Cueing sub-categories (the
  Real-Time mirror of the Non-Real-Time MTC Setup family), the
  Controller Destination triple (Channel Pressure, Polyphonic Key
  Pressure, Control Change), Key-Based Instrument Control, Scalable
  Polyphony MIP Message, Mobile Phone Control Message.
- 30 new unit tests cover: every Sub-ID #1 vocabulary slot, every
  Sub-ID #2 vocabulary slot across both realms, realm disambiguation
  for the shared `0x09 0x01` / `0x02` / `0x07` / `0x08` / `0x09`
  byte pairs, the four "this is not a Universal packet" return-`None`
  paths (`F7` continuation, manufacturer-prefixed `F0`, expanded
  three-byte manufacturer ID, payload shorter than 3 bytes), the
  `device_id` byte preservation across the `0x00..=0x7F` range
  (specific-device vs broadcast `0x7F` target), the `Other(raw)`
  fallback paths for unknown Sub-ID #1 and Sub-ID #2 values, and an
  end-to-end pass that parses an SMF carrying GM 1 System On +
  Master Volume + Master Fine Tuning, walks `SmfFile::sysex_events()`,
  classifies each entry, and confirms the realm split + Sub-ID #1 /
  Sub-ID #2 decoding all the way through the parser → iteration
  helper → classifier pipeline.
- Builds on the existing `SysExEvent` surface (`tick`, `track`,
  `is_escape`, `data`, `ends_with_eox`, `is_complete_message`,
  `manufacturer_id`) — `universal_classification` is the realm- and
  category-decoded view of the same payload `manufacturer_id` returns
  the leading byte of, so a caller routing by either accessor stays
  in sync with the verbatim `SysExEvent::data` buffer the SMF
  parser produces.

### Round 243 — `SmfFile::sysex_events()` (F0 / F7) iteration helper

- New `SmfFile::sysex_events(&self) -> Vec<SysExEvent>` surfaces
  every System Exclusive event — both the `F0` start and the `F7`
  continuation / escape flavours — as a `SysExEvent { tick, track,
  is_escape, data }` with the absolute tick on the parent track,
  stably merged across tracks (track 0 before track 1 at the same
  tick) under the same merge rule as every existing iteration
  helper and the scheduler. Standard MIDI File Specification 1.0
  §"System Exclusive Events" defines `F0 <varlen> <payload>` (a
  complete-or-starting SysEx message; the trailing `F7` end marker,
  when present, is included as the final byte of `<payload>`) and
  `F7 <varlen> <payload>` (a continuation packet for a previously-
  started `F0` message *or* an arbitrary escape sequence whose
  payload is shipped verbatim to the wire). Both forms surface
  through the helper; the `is_escape` flag distinguishes them and
  the `data` payload is reproduced verbatim so a writer can
  round-trip the helper output through `SmfFile::to_bytes` without
  re-synthesising the SysEx framing.
- `SysExEvent::ends_with_eox()` returns `true` when the payload
  terminates with the `0xF7` end marker; `SysExEvent::is_complete_message()`
  is sugar for `!is_escape && ends_with_eox()`, the common universal-
  SysEx case (GM-on `F0 7E 7F 09 01 F7`, Master Volume, Master
  Tuning) where a caller routes the whole packet in one step.
  `SysExEvent::manufacturer_id()` returns the leading byte of an
  `F0` payload (the manufacturer ID, or `0x7E` non-real-time /
  `0x7F` real-time for universal SysEx) and `None` for an `F7`
  packet or an empty payload; expanded three-byte IDs (`0x00`-
  prefixed) are surfaced as `Some(0x00)` and the caller inspects
  `data[0..=2]` for the full ID.
- The `FF 7F` `SequencerSpecific` channel — surfaced through
  `SmfFile::sequencer_specifics()` — is *not* selected here; the
  two channels carry different semantics (SysEx travels to the
  MIDI wire; `FF 7F` is file-private metadata that does not) and a
  file may carry both an `F0 7E 7F 09 01 F7` Universal Non-Real-Time
  GM-On packet on the conductor track and a private `FF 7F`
  plugin-state blob alongside it. The helper surfaces empty
  payloads (`F0 00` / `F7 00`) as `data.is_empty()` rather than
  filtering them out — the spec permits a zero-length packet.
- 9 new unit tests cover: empty case (no `F0` / `F7` in any track);
  universal GM-on at tick zero (`F0 7E 7F 09 01 F7`, complete
  message, manufacturer `0x7E`); `F0` without trailing `F7` marking
  a multi-packet opener (Roland `0x41`); `F7` continuation pairing
  after an `F0` opener (with EOX terminator on the continuation
  packet); empty payload `F0 00` surfacing as `data.is_empty()`;
  multi-track merge by absolute tick; stable sort keeping
  `(track, in-track)` order at the same tick; filter-purity against
  `F0` + `F7` alongside `FF 03 / FF 01 / FF 21 / FF 20 / FF 51 /
  FF 7F / B0 / 90 / FF 2F` plus a cross-check that
  `sequencer_specifics()` surfaces its single `FF 7F` entry
  untouched; and a parser → `to_bytes` → parser round-trip
  exercising the writer's `F0` + `F7` paths so the helper can't
  drift out of sync with the mux.
- Surfaces the SysEx channel alongside the 15-helper meta-event
  family; the meta-event helper count itself stays at 15
  (`SmfFile::{tempo_map, time_signatures, key_signatures, markers,
  lyrics, cue_points, track_names, instrument_names, texts,
  copyrights, smpte_offsets, sequencer_specifics, sequence_numbers,
  midi_ports, channel_prefixes}`), since SysEx is a wire-event
  family rather than a meta-event family.

### Round 240 — `SmfFile::channel_prefixes()` (FF 20 01 cc) iteration helper

- New `SmfFile::channel_prefixes(&self) -> Vec<ChannelPrefixEvent>`
  surfaces every `FF 20 01 cc` Channel Prefix meta event as a
  `ChannelPrefixEvent { tick, track, channel }` with the absolute
  tick on the parent track, stably merged across tracks (track 0
  before track 1 at the same tick) under the same merge rule as
  every existing iteration helper and the scheduler. `FF 20` carries
  the channel-binding hint for non-channel events that follow on the
  same track: the single payload byte names the MIDI channel
  (`0..=15`) the following meta / sysex events should be associated
  with — text, lyric, marker, cue point, sysex — until another
  `FF 20` arrives, the next channel-voice event arrives and
  supersedes the binding, or the track ends. The Standard MIDI File
  Specification 1.0 lists the event as part of the meta-event
  vocabulary; modern authoring tools prefer explicit per-track
  channel-voice streams plus `FF 21` port hints, but older files
  still emit it and a round-trip workflow must preserve it.
- `ChannelPrefixEvent::channel()` returns the spec-clamped channel
  index as `Option<u8>`: `Some(c)` when `c < 16`, `None` otherwise.
  The raw byte stays available on the `channel` field so files with
  out-of-spec values (a single bit set in the high nibble, etc.)
  still round-trip; the helper returns `None` rather than mask the
  byte because masking would silently route to an unintended channel.
- Only `FF 20` is selected — the neighbouring `FF 21` port-hint
  sibling (different routing semantics: per-track physical port
  assignment versus per-message channel override) stays on its own
  `SmfFile::midi_ports()` helper so callers reconstructing the
  channel association of surrounding non-channel events get a clean
  time-ordered list independent of the other meta streams.
- 8 new unit tests cover: empty case (no `FF 20` in any track);
  single binding at tick zero; full spec `0..=15` round trip (one
  case per `cc`); out-of-spec `cc = 0x20` surfaces raw with
  `channel() == None`; multi-track merge by absolute tick; stable
  sort keeping `(track, in-track)` order at the same tick;
  filter-purity against `FF 21 / FF 00 / FF 01 / FF 03 / FF 05 /
  FF 51 / FF 54 / FF 58 / FF 59 / FF 7F`; and a parser → `to_bytes`
  → parser round-trip that exercises the writer's `FF 20` path so
  the helper can't drift out of sync with the mux.
- Lifts the SMF meta-event iterator family from 14 to **15** total:
  `SmfFile::{tempo_map, time_signatures, key_signatures, markers,
  lyrics, cue_points, track_names, instrument_names, texts,
  copyrights, smpte_offsets, sequencer_specifics, sequence_numbers,
  midi_ports, channel_prefixes}`.

### Round 234 — `SmfFile::to_bytes()` SMF mux-side writer

- New `SmfFile::to_bytes(&self) -> Result<Vec<u8>>` serialises a
  parsed SMF back to a complete byte stream (`MThd` header + one
  `MTrk` chunk per `Track`), suitable to hand back to `parse` for
  a structural round-trip. Companion `Track::to_bytes_chunk(&self)
  -> Result<Vec<u8>>` emits one self-contained `MTrk` chunk so
  callers building a multi-track file from independent track
  sources can splice the chunks under a single `MThd` header.
- Every channel-voice variant (`NoteOn` / `NoteOff` /
  `PolyAftertouch` / `ControlChange` / `ProgramChange` /
  `ChannelAftertouch` / `PitchBend`), every concrete meta variant
  (`SequenceNumber` / `Text` / `ChannelPrefix` / `Port` /
  `EndOfTrack` / `Tempo` / `SmpteOffset` / `TimeSignature` /
  `KeySignature` / `SequencerSpecific` / passthrough `Unknown`),
  and both sysex forms (`F0` start, `F7` continuation/escape) emit
  on the wire format the parser already accepts. `Unknown` with
  `type_byte: 0x2F` is rejected so the End-of-Track marker cannot
  be smuggled in past the placement check.
- Output uses **explicit status bytes** throughout — the spec
  permits but does not require running-status compression, and
  the explicit form keeps the writer deterministic regardless of
  internal track ordering. A reader that does not honour running
  status can still consume the output unchanged.
- New top-level `MAX_VLQ_VALUE = 0x0FFF_FFFF` constant — the
  largest VLQ-encodable value per the SMF spec's 4-byte cap (the
  same cap the parser's `MAX_VLQ_BYTES = 4` enforces on the read
  side). Delta-times, meta payload lengths, and sysex payload
  lengths must all fit; the writer surfaces `Error::InvalidData`
  for anything larger.
- Strict validation at encode-time, with descriptive
  `Error::InvalidData` messages that name the offending field /
  track / event index so caller-side debugging stays local:
  `header.ntrks` must match `tracks.len()`; each track must end
  with exactly one `MetaEvent::EndOfTrack` as its final event;
  channel-voice data bytes must have the high bit clear; pitch-bend
  values must fit 14 bits; `KeySignature.mode` must be 0 or 1;
  `Tempo` must fit 24 bits; SMPTE `frames_per_second` must be one
  of `{24, 25, 29, 30}`; `TicksPerQuarter` must be `1..=0x7FFF`;
  `Text.kind` must be `0x01..=0x0F`.
- Lifts the SMF surface from read-only to **read + write**: the
  full event vocabulary the parser materialises (channel-voice,
  meta, sysex) now round-trips through `to_bytes` -> `parse`
  byte-for-byte. The 17 new writer tests cover spec-VLQ encoding
  (10 worked examples), every meta variant, every channel-voice
  variant, both sysex forms, a multi-track Format-1 file, the
  long-VLQ (4-byte) delta branch, all five validation rejections,
  and the per-track chunk helper.

### Round 230 — `SmfFile::midi_ports()` (FF 21)

- New `smf::MidiPortEvent { tick, track, port }` value type pinned to
  the absolute tick at which an `FF 21 01 pp` MIDI Port Meta-Event
  fires on its parent track. `MidiPortEvent::port()` returns the
  physical port byte (`0..=127`).
- New `SmfFile::midi_ports() -> Vec<MidiPortEvent>` iterator helper
  that walks every track, sums the per-track cumulative deltas, and
  stably merges the matching `FF 21` events across tracks (track 0
  before track 1 at the same tick) under the same merge rule the
  other 13 meta-event helpers and `scheduler.rs` §"merged event
  list, sorted by absolute tick" already use.
- The `FF 21` Meta-Event carries an unofficial routing hint: the
  single payload byte names the physical output port the
  surrounding channel-voice messages on this track should be
  dispatched through. The Standard MIDI File Specification 1.0 caps
  a single channel-voice stream at 16 channels (the four status
  nibbles `8x..Ex` combined with the four channel bits `x0..xF`);
  the port hint lets a multi-port DAW back-end multiply that ceiling
  by however many physical outputs it wires up, by labelling each
  track with the port it routes to. The convention is one `FF 21`
  near the start of a track (delta zero, before the first
  channel-voice event), but the helper surfaces every occurrence
  rather than enforcing the placement rule so files that re-route
  mid-track still round-trip the hint.
- Only `FF 21` is selected — the neighbouring `FF 20` channel-prefix
  hint (different routing semantics: per-message channel override
  versus per-track physical port assignment) stays on its own so the
  port-routing layer gets a clean time-ordered list independent of
  the surrounding meta streams.
- Lifts the SMF meta-event iterator family from 13 to **14** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets,sequencer_specifics,sequence_numbers,midi_ports}`).
- 8 new unit tests covering: empty case, single hint at tick 0, full
  7-bit range (`0x7F`) round-trip, per-track routing in a format-1
  multi-port file, same-tick stable sort (track 0 before track 1),
  cross-track merge sorted by tick, late-position absolute-tick
  tracking after a channel-voice event, and filter exclusion against
  the surrounding `FF 00` / `FF 01` / `FF 03` / `FF 05` / `FF 20` /
  `FF 51` / `FF 54` / `FF 58` / `FF 59` / `FF 7F` events (the
  `FF 20` sibling test pins the channel-prefix-vs-port distinction).

### Round 224 — `SmfFile::sequence_numbers()` (FF 00)

- New `smf::SequenceNumberEvent { tick, track, number }` value type
  pinned to the absolute tick at which an `FF 00 02 ssss` Sequence
  Number Meta-Event fires on its parent track.
  `SequenceNumberEvent::number()` returns the 16-bit identifier (the
  `ssss` payload is decoded big-endian).
- New `SmfFile::sequence_numbers() -> Vec<SequenceNumberEvent>`
  iterator helper that walks every track, sums the per-track
  cumulative deltas, and stably merges the matching `FF 00` events
  across tracks (track 0 before track 1 at the same tick) under the
  same merge rule the other 12 meta-event helpers and `scheduler.rs`
  §"merged event list, sorted by absolute tick" already use.
- The Standard MIDI File Specification 1.0 reserves the event for
  delta-time zero (the first event of a track) and uses it to label
  a format-2 pattern so it can be cued from a Song Select, but the
  helper surfaces every occurrence rather than enforcing the
  placement rule so files that carry the label later in a track
  still round-trip. Format-1 / format-0 files conventionally place
  the event on track 0 to label the file as a whole.
- Lifts the SMF meta-event iterator family from 12 to **13** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets,sequencer_specifics,sequence_numbers}`).
- 8 new unit tests covering: empty case, single label at tick 0,
  big-endian decode (`0x1234`), full `0xFFFF` round-trip, format-2
  per-pattern labels, same-tick stable sort (track 0 before track
  1), late-position absolute-tick tracking, and filter exclusion
  against the surrounding `FF 01` / `FF 03` / `FF 05` / `FF 51` /
  `FF 54` / `FF 58` / `FF 59` / `FF 7F` events.

### Round 219 — `SmfFile::sequencer_specifics()` (FF 7F)

- New `smf::SequencerSpecificEvent { tick, track, data }` value type
  pinned to the absolute tick at which an `FF 7F len data`
  Sequencer-Specific Meta-Event fires on its parent track.
  `SequencerSpecificEvent::data_bytes()` borrows the raw payload.
- New `SmfFile::sequencer_specifics() -> Vec<SequencerSpecificEvent>`
  iterator helper that walks every track, sums the per-track
  cumulative deltas, and stably merges the matching `FF 7F` events
  across tracks (track 0 before track 1 at the same tick) under the
  same merge rule the other 11 meta-event helpers and `scheduler.rs`
  §"merged event list, sorted by absolute tick" already use. Only
  `FF 7F` is selected — channel-message `F0` / `F7` SysEx events
  (which travel through the scheduler's SysEx pump rather than the
  meta-event family) stay out of the list.
- `FF 7F` payloads are surfaced verbatim: the parser does not
  interpret the SysEx-style manufacturer-ID convention (where
  `data[0]`, or `data[0..=2]` when `data[0] == 0x00`, holds the ID),
  so a caller routing by manufacturer can inspect
  `SequencerSpecificEvent::data` directly while a generic player can
  ignore per the SMF spec's "unknown meta events SHOULD be ignored"
  rule. Empty payloads (`FF 7F 00`) are surfaced as
  `data.is_empty()` rather than filtered out — the spec permits a
  zero-length blob.
- Lifts the SMF meta-event iterator family from 11 to **12** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets,sequencer_specifics}`).
- 8 new unit tests covering: empty case, single event at tick 0,
  zero-length payload, multiple within one track, cross-track merge
  sorted by tick, same-tick stable sort (track 0 before track 1),
  filter excluding text / tempo / SMPTE / time / key meta kinds,
  and absolute-tick tracking after channel-voice events.

### Round 213 — `SmfFile::channel_snapshot_at` / `channel_snapshots_at` (channel-state seek primitive)

- New `smf::SmfChannelSnapshot { program, bank_msb, bank_lsb, volume,
  pan, expression, modulation, sustain, pitch_bend }` capturing the
  on-the-wire channel state of one MIDI channel at one absolute SMF
  tick. Fields that no event has touched stay at the SMF + GM 1
  (RP-003) recommended defaults: `volume = 100`, `pan = 64`,
  `expression = 127`, `modulation = 0`, `sustain = false`,
  `pitch_bend = 0x2000`; `program` / `bank_msb` / `bank_lsb` stay
  `None` so a seek-time initialiser can skip emitting CCs the file
  never wrote.
- New `SmfFile::channel_snapshot_at(channel, tick) ->
  SmfChannelSnapshot` and `SmfFile::channel_snapshots_at(tick) ->
  [SmfChannelSnapshot; 16]`. Each replays every channel-voice event
  from every track up to and including the requested tick, in
  scheduler order — a stable merge by `(tick, track, in-track)`,
  track 0 winning over track 1 at the same tick, the same
  convention `tempo_map` / `time_signatures` / `key_signatures` /
  the eight text-meta helpers / `smpte_offsets` and `scheduler.rs`
  §"merged event list, sorted by absolute tick" use.
- The fold updates the snapshot for Program Change, Pitch Bend,
  and Control Change on the seven channel-state CCs (CC 0 Bank
  Select MSB, CC 1 Modulation Wheel, CC 7 Channel Volume, CC 10
  Pan, CC 11 Expression, CC 32 Bank Select LSB, CC 64 Sustain
  Pedal). Notes / poly + channel aftertouch are ignored — they
  affect voice state, not channel state — so the snapshot is
  cheap to compute for any tick without enumerating sounding
  notes. Sustain pedal decodes the CC 64 value with the spec
  threshold (`value >= 64` = on, `< 64` = off).
- Events at exactly `tick` are *included* in the replay — the
  snapshot reflects state immediately after that tick fires.
  Channels `>= 16` (not a legal MIDI channel) fall through to the
  default snapshot rather than panic / index-out.
- Bulk accessor `channel_snapshots_at` pools every track's events
  into a single merge + replay pass so initialising all 16
  channels at a seek target is single-pass rather than 16-pass —
  the natural primitive for a DAW or player seeking into the
  middle of a file.
- `SmfChannelSnapshot::apply(&ChannelBody)` is also exposed
  publicly so callers running custom replay (e.g. against a
  custom track ordering or a filtered event set) can reuse the
  same wire semantics.
- This is the first SMF-file accessor that moves beyond the
  "iterate every meta event" lens introduced by `tempo_map` /
  `time_signatures` / `key_signatures` / the eight text-meta
  helpers / `smpte_offsets`. The iteration helpers answer "when
  does X fire?"; the snapshot answers "what is the wire state at
  T?" — the two are complementary primitives for SMF seeking and
  inspection.
- 15 new dedicated tests: default snapshot when no events,
  Program Change at tick 0, the four CC dimensions
  (volume / pan / expression / modulation), CC 64 threshold at 63
  vs. 64, Bank Select MSB + LSB independence, 14-bit Pitch Bend
  decode, tick-filter inclusion of events at exact tick (90 →
  100 → 199 → 200 → ∞ walk), running-status Program Change
  series replayed, multi-channel independence, notes /
  aftertouch leave the snapshot at defaults, unknown CCs
  (`CC 99` / `CC 100`) ignored, invalid channel returns default,
  bulk `channel_snapshots_at` returns 16 independent states
  consistent with per-channel calls, multi-track merge at same
  tick honours track 0 → track 1 stable order (last-writer-wins
  on the wire), public `SmfChannelSnapshot::apply` round-trips
  every event-kind.

### Round 208 — `SmfFile::smpte_offsets()` iteration helper + `FrameRate` decoder (`FF 54`)

- New `smf::SmpteOffsetEvent { tick, track, hours_raw, minutes,
  seconds, frames, subframes }` plus `SmfFile::smpte_offsets() ->
  Vec<SmpteOffsetEvent>`. Collects every SMPTE Offset meta event
  (`FF 54 05 hr mn se fr ff`, the wall-clock cue declaring when a
  track's first event is meant to fire) from every track, pins each
  one to the absolute tick of its parent track via cumulative
  `TrackEvent::delta` sums, then merges the per-track sequences
  with a stable sort by `tick` — track 0 wins over track 1 at the
  same tick, matching the same merge rule used by the ten existing
  text + rhythmic helpers (`tempo_map` / `time_signatures` /
  `key_signatures` / `markers` / `lyrics` / `cue_points` /
  `track_names` / `instrument_names` / `texts` / `copyrights`) and
  by `scheduler.rs` §"merged event list, sorted by absolute tick".
- New `smf::FrameRate` enum (`Fps24` / `Fps25` / `Fps30DropFrame`
  / `Fps30NonDrop`) decoded from the packed `hr` byte per the
  MIDI Time Code spec, RP-004/008 §"HOURS COUNT": bits 5-6 hold
  the rate type (`00=24fps`, `01=25fps`, `10=30fps drop-frame`,
  `11=30fps non-drop`); bits 0-4 hold the hours count; bit 7 is
  reserved. `FrameRate::from_hours_byte(hr)` exposes the raw
  decode; `frames_per_second()` returns the nominal counter rate
  (drop-frame still numbers 30 frames per wall-second);
  `is_drop_frame()` distinguishes the two 30-Hz variants.
- `SmpteOffsetEvent::frame_rate()` / `hours_count()` /
  `seconds_total()` surface the SMPTE-cueing semantics without
  forcing callers to re-mask the `hr` byte. `seconds_total()`
  returns the wall-clock offset as `h*3600 + m*60 + s + (frames +
  subframes/100) / fps` — drop-frame uses the same 30 Hz divisor
  since the counter compensates for the 29.97 Hz playback, so the
  helper stays rate-independent across the four MTC types (callers
  needing strict NTSC-accurate timing should re-derive from the
  tempo map).
- Out-of-range values (hours > 23, minutes > 59, seconds > 59,
  frames above the rate's nominal count, subframes > 99) are
  preserved as-is rather than clamped — the helper surfaces the
  raw counter so callers can inspect malformed files.
- Only `FF 54` is selected; the rhythmic / text meta events stay
  uncontaminated (asserted by a dedicated cross-kind filter test).
- Lifts the SMF meta-event iterator family from 10 to **11** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets}`), covering every rhythmic + text + SMPTE-cueing
  meta event the spec defines a per-event "when it fires" lens for.
- 10 new dedicated tests: empty, single event with 24 fps decode,
  all-four-frame-rates decode at hours=1, multi-track merge sorted
  by tick, stable sort track-0-before-track-1 at same tick,
  filter excludes other rhythmic + text meta kinds (with
  sibling-helper uncontamination check), `seconds_total()` at 24
  fps with non-zero sub-frames, `seconds_total()` at 30-non-drop
  origin, absolute-tick accounting through running-status channel
  events, and `FrameRate::from_hours_byte()` bit-mask coverage
  (bit 7 reserved, bits 5-6 select the rate, bits 0-4 don't leak
  into the rate decode).

### Round 202 — `SmfFile::texts()` + `SmfFile::copyrights()` iteration helpers (`FF 01` + `FF 02`)

- New `smf::TextEvent { tick, track, text }` plus
  `SmfFile::texts() -> Vec<TextEvent>`. Collects every free-form /
  general text meta event (`FF 01 len text`, the catch-all annotation
  kind used for production notes, "do not edit", version stamps,
  mix-engineer comments, …) from every track, pins each one to the
  absolute tick of its parent track via cumulative
  `TrackEvent::delta` sums, then merges the per-track sequences with
  a stable sort by `tick` — track 0 wins over track 1 at the same
  tick, matching the same merge rule used by the seven existing
  text-meta helpers (`markers` / `lyrics` / `cue_points` /
  `track_names` / `instrument_names`) and the rhythmic helpers
  (`tempo_map` / `time_signatures` / `key_signatures`) and by
  `scheduler.rs` §"merged event list, sorted by absolute tick".
- New `smf::CopyrightEvent { tick, track, text }` plus
  `SmfFile::copyrights() -> Vec<CopyrightEvent>`. Collects every
  copyright-notice meta event (`FF 02 len text`) from every track
  under the same merge rule. The SMF specification recommends
  placing the notice on the first track at tick 0 so players can
  surface authorship without scanning the whole file; this helper
  surfaces every occurrence in time order regardless, so callers
  that only want the first notice can take `.next()` on the
  iterator while callers that want the full history can read the
  whole `Vec`.
- Only `FF 01` / `FF 02` are selected by their respective helpers.
  The five sibling text-kind meta events (`FF 03` track name,
  `FF 04` instrument name, `FF 05` lyric, `FF 06` marker, `FF 07`
  cue point) are filtered out so callers populating the relevant
  metadata get a clean per-track stream without having to
  discriminate themselves. A dedicated round-trip test asserts
  that a single track may legally carry both `FF 01` and `FF 02`
  and that the two helpers surface them independently — and the
  cross-kind filter tests now assert that all seven text-meta
  helpers stay uncontaminated when every text-kind event is
  present.
- Lifts the SMF text-meta iterator family from 8 to **10**
  helpers (`SmfFile::{tempo_map,time_signatures,key_signatures,
  markers,lyrics,cue_points,track_names,instrument_names,texts,
  copyrights}`), covering every `FF 01..=07` text-flavour meta
  event the spec defines.
- Same accessor shape as the other eight text-meta helpers:
  `TextEvent::text_bytes()` / `CopyrightEvent::text_bytes()` for
  the raw payload (encoding is spec-unspecified — historically
  Latin-1, modern files emit UTF-8), `text_lossy()` for a
  `Cow<str>` UTF-8 decode with `U+FFFD` substitutes for invalid
  sequences.
- 19 new dedicated tests (10 for `texts()`, 9 for `copyrights()`):
  empty, single event, multiple within one track, multi-track
  merge sorted by tick, stable sort track-0-before-track-1 at
  same tick, filter excludes other text kinds (with sibling-
  helper uncontamination check), absolute-tick accounting
  through running-status channel events, `text_lossy()`
  invalid-UTF-8 substitution, and a cross-kind coexistence test
  proving `FF 01` and `FF 02` on the same track surface
  independently.

### Round 196 — `SmfFile::instrument_names()` iteration helper (`FF 04`)

- New `smf::InstrumentNameEvent { tick, track, text }` plus
  `SmfFile::instrument_names() -> Vec<InstrumentNameEvent>`. Collects
  every instrument-name meta event (`FF 04 len text`, the per-track
  voice / patch label distinct from the `FF 03` track-list label) from
  every track, pins each one to the absolute tick of its parent track
  via cumulative `TrackEvent::delta` sums, then merges the per-track
  sequences with a stable sort by `tick` — track 0 wins over track 1
  at the same tick, matching the same merge rule used by
  `SmfFile::track_names()` / `cue_points()` / `markers()` / `lyrics()`
  / `tempo_map()` / `time_signatures()` / `key_signatures()` and by
  `scheduler.rs` §"merged event list, sorted by absolute tick".
- Only `FF 04` is selected. The other text-kind meta events
  (`FF 01` general text, `FF 02` copyright, `FF 03` track name,
  `FF 05` lyric, `FF 06` marker, `FF 07` cue point) are filtered out
  so callers populating per-track instrument metadata get a clean
  per-track stream without having to discriminate themselves. A
  dedicated round-trip test asserts that a single track may legally
  carry both `FF 03` and `FF 04` and that the two helpers surface
  them independently.
- Lifts the SMF text-meta iterator family from 7 to 8 helpers
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names}`).
- Same accessor shape as the other six text-meta helpers:
  `InstrumentNameEvent::text_bytes()` for the raw payload (encoding
  is spec-unspecified — historically Latin-1, modern files emit
  UTF-8), `text_lossy()` for a `Cow<str>` UTF-8 decode with `U+FFFD`
  substitutes for invalid sequences.
- 10 dedicated tests: empty, single event, per-track in format-1,
  multiple within one track, multi-track merge sorted by tick, stable
  sort track-0-before-track-1 at same tick, filter excludes other
  text kinds (including a sanity check that the sibling helpers stay
  uncontaminated), absolute-tick accounting through running-status
  channel events, coexistence with `FF 03` on the same track, and
  `text_lossy()` invalid-UTF-8 substitution.

## [0.0.3](https://github.com/OxideAV/oxideav-midi/compare/v0.0.2...v0.0.3) - 2026-05-30

### Other

- SmfFile::track_names() iteration helper (FF 03)
- SmfFile::cue_points() iteration helper (FF 07)
- SmfFile::lyrics() iteration helper (FF 05)
- SmfFile::markers() iteration helper (FF 06)
- add cargo-fuzz harness over smf / sf2 / dls / sfz parsers (round 172)
- fix parse_key integer overflow on huge octave magnitudes
- add SmfFile::key_signatures() iteration helper
- SmfFile::tempo_map() iteration helper (FF 51)
- SmfFile::time_signatures() iteration helper (FF 58)

### Round 192 — `SmfFile::track_names()` iteration helper (`FF 03`)

- New `smf::TrackNameEvent { tick, track, text }` plus
  `SmfFile::track_names() -> Vec<TrackNameEvent>`. Collects every
  track-name meta event (`FF 03 len text`, the DAW-track-list
  convention from the Standard MIDI File 1.0 specification) from
  every track, pins each one to the absolute tick of its parent
  track via cumulative `TrackEvent::delta` sums, then merges the
  per-track sequences with a stable sort by `tick` — track 0 wins
  over track 1 at the same tick, matching the same merge rule used
  by `SmfFile::cue_points()` / `SmfFile::markers()` /
  `SmfFile::lyrics()` / `SmfFile::tempo_map()` /
  `SmfFile::time_signatures()` / `SmfFile::key_signatures()` and by
  `scheduler.rs` §"merged event list, sorted by absolute tick".
- Only `FF 03` is selected. Other text-kind meta events
  (`FF 01` general text, `FF 02` copyright, `FF 04` instrument
  name, `FF 05` lyric, `FF 06` marker, `FF 07` cue point) are
  filtered out so callers populating a DAW track-list label don't
  have to discriminate themselves.
- Authoring tools conventionally emit at most one `FF 03` per track
  at tick 0, but the spec does not constrain count or placement.
  The helper surfaces every occurrence so callers that only want
  the first name per track can collect into a
  `HashMap<usize, TrackNameEvent>` keyed on `TrackNameEvent::track`,
  while callers tracking renames over time read the full `Vec`. On
  a format-0 file the single track's `FF 03` is conventionally read
  as the sequence title.
- `TrackNameEvent::text_bytes()` borrows the raw `text` payload
  unchanged (the SMF spec leaves the encoding unspecified —
  historically Latin-1, modern DAWs emit UTF-8). `text_lossy()`
  returns `Cow<str>` using `String::from_utf8_lossy`, so invalid
  UTF-8 surfaces as `U+FFFD` replacement characters rather than
  panicking — convenient default for callers that only need the
  human-readable track label.
- Cost is linear in the total event count and bounded above by the
  parser's existing `MAX_EVENTS_PER_FILE` cap; the helper does not
  introduce a new allocation ceiling.
- 6 new unit tests in `src/smf.rs::tests` cover: empty input,
  single name at tick 0, per-track names on a format-1 two-track
  file (`Drums` / `Bass`), two `FF 03` events on one track in time
  order (`Intro` at tick 0, `Main` at tick 480), stable sort at the
  same tick across two tracks, filtering against the full
  text-meta neighbourhood (`FF 01` general text, `FF 02` copyright,
  `FF 04` instrument name, `FF 05` lyric, `FF 06` marker, `FF 07`
  cue — with cross-checks that the marker / lyric / cue helpers
  stay uncontaminated), and `text_lossy()` resilience against
  non-UTF-8 bytes. Brings the in-crate unit suite from 333 to 339
  unit tests, all passing under `cargo test`.
- Docstring cross-links: `SmfFile::lyrics()` and
  `SmfFile::cue_points()` now point at `SmfFile::track_names` in
  their "distinct from" enumerations, so the doc graph between the
  six text-meta helpers stays bidirectionally connected.

### Round 186 — `SmfFile::cue_points()` iteration helper (`FF 07`)

- New `smf::CueEvent { tick, track, text }` plus
  `SmfFile::cue_points() -> Vec<CueEvent>`. Collects every cue-point
  meta event (`FF 07 len text`, the Standard MIDI File 1.0
  film-score / theatrical sync convention) from every track, pins
  each one to the absolute tick of its parent track via cumulative
  `TrackEvent::delta` sums, then merges the per-track sequences
  with a stable sort by `tick` — track 0 wins over track 1 at the
  same tick, matching the same merge rule used by
  `SmfFile::markers()` / `SmfFile::lyrics()` /
  `SmfFile::tempo_map()` / `SmfFile::time_signatures()` /
  `SmfFile::key_signatures()` and by `scheduler.rs` §"merged event
  list, sorted by absolute tick".
- Only `FF 07` is selected. Other text-kind meta events
  (`FF 01` general text, `FF 02` copyright, `FF 03` track name,
  `FF 04` instrument name, `FF 05` lyric, `FF 06` marker, …) are
  filtered out so callers driving external synchronisation (scene
  change, SFX trigger, video cue) don't have to discriminate
  themselves.
- `CueEvent::text_bytes()` borrows the raw `text` payload unchanged
  (the SMF spec leaves the encoding unspecified — historically
  Latin-1, modern editors emit UTF-8). `text_lossy()` returns
  `Cow<str>` using `String::from_utf8_lossy`, so invalid UTF-8
  surfaces as `U+FFFD` replacement characters rather than panicking
  — convenient default for callers that only need the human-readable
  cue name.
- Cost is linear in the total event count and bounded above by
  the parser's existing `MAX_EVENTS_PER_FILE` cap; the helper does
  not introduce a new allocation ceiling.
- 8 new unit tests in `src/smf.rs::tests` cover: empty input,
  single cue at tick 0, three-cue in-order sequence (the
  Intro / SceneA / SceneB shape), multi-track merge order,
  stable sort at the same tick, filtering against neighbouring
  text kinds (`FF 03` track name, `FF 05` lyric, `FF 06` marker —
  with a cross-check that the marker and lyric helpers stay
  uncontaminated), absolute-tick accounting through running-status
  channel events, and `text_lossy()` resilience against non-UTF-8
  bytes. Brings the in-crate suite from 323 to 331 unit tests, all
  passing under `cargo test -p oxideav-midi`.
- Docstring cross-links: `SmfFile::markers()` and `SmfFile::lyrics()`
  now point at `SmfFile::cue_points()` for the film-score sync
  stream so callers searching either doc find the cue companion.

### Round 182 — `SmfFile::lyrics()` iteration helper (`FF 05`)

- New `smf::LyricEvent { tick, track, text }` plus
  `SmfFile::lyrics() -> Vec<LyricEvent>`. Collects every lyric meta
  event (`FF 05 len text`, the karaoke `.kar` syllable convention)
  from every track, pins each one to the absolute tick of its
  parent track via cumulative `TrackEvent::delta` sums, then merges
  the per-track sequences with a stable sort by `tick` — track 0
  wins over track 1 at the same tick, matching the same merge rule
  used by `SmfFile::markers()` / `SmfFile::tempo_map()` /
  `SmfFile::time_signatures()` / `SmfFile::key_signatures()` and
  by `scheduler.rs` §"merged event list, sorted by absolute tick".
- Only `FF 05` is selected. Other text-kind meta events
  (`FF 01` general text, `FF 02` copyright, `FF 03` track name,
  `FF 04` instrument name, `FF 06` marker, `FF 07` cue point, …)
  are filtered out so karaoke callers iterating syllables don't
  have to discriminate themselves.
- `LyricEvent::text_bytes()` borrows the raw `text` payload
  unchanged (the SMF spec leaves the encoding unspecified —
  historically Latin-1, modern files emit UTF-8). `text_lossy()`
  returns `Cow<str>` using `String::from_utf8_lossy`, so invalid
  UTF-8 surfaces as `U+FFFD` replacement characters rather than
  panicking — convenient default for callers that only need the
  human-readable text.
- Cost is linear in the total event count and bounded above by
  the parser's existing `MAX_EVENTS_PER_FILE` cap; the helper
  does not introduce a new allocation ceiling.
- 8 new unit tests in `src/smf.rs::tests` cover: empty input,
  single syllable at tick 0, four-syllable in-order sequence
  (the "Twinkle, Twinkle" `.kar` shape), multi-track merge order,
  stable sort at the same tick, filtering against neighbouring
  text kinds (`FF 03` track name, `FF 06` marker), absolute-tick
  accounting through running-status channel events, and
  `text_lossy()` resilience against non-UTF-8 bytes. Brings the
  in-crate suite from 315 to 323 unit tests, all passing under
  `cargo test -p oxideav-midi`.
- Docstring cross-link: `SmfFile::markers()` now points at
  `SmfFile::lyrics()` for the karaoke syllable stream so callers
  searching the marker docs find the lyric companion.

### Round 176 — `SmfFile::markers()` iteration helper (`FF 06`)

- New `smf::MarkerEvent { tick, track, text }` plus
  `SmfFile::markers() -> Vec<MarkerEvent>`. Collects every marker
  meta event (`FF 06 len text`, the DAW song-section convention)
  from every track, pins each one to the absolute tick of its
  parent track via cumulative `TrackEvent::delta` sums, then merges
  the per-track sequences with a stable sort by `tick` — track 0
  wins over track 1 at the same tick, matching the same merge
  rule used by `SmfFile::tempo_map()` /
  `SmfFile::time_signatures()` / `SmfFile::key_signatures()` and
  by `scheduler.rs` §"merged event list, sorted by absolute tick".
- Only `FF 06` is selected. Other text-kind meta events
  (`FF 03` track name, `FF 05` lyric, `FF 07` cue point, …)
  are filtered out so callers iterating section labels don't have
  to discriminate themselves.
- `MarkerEvent::text_bytes()` borrows the raw `text` payload
  unchanged (the SMF spec leaves the encoding unspecified —
  historically Latin-1, modern DAWs emit UTF-8). `text_lossy()`
  returns `Cow<str>` using `String::from_utf8_lossy`, so invalid
  UTF-8 surfaces as `U+FFFD` replacement characters rather than
  panicking — convenient default for callers that only need the
  human-readable label.
- Cost is linear in the total event count and bounded above by
  the parser's existing `MAX_EVENTS_PER_FILE` cap; the helper
  does not introduce a new allocation ceiling.
- 8 new unit tests in `src/smf.rs::tests` cover: empty input,
  single marker at tick 0, multiple per-track markers in order,
  multi-track merge order, stable sort at the same tick,
  filtering against neighbouring text kinds (`FF 03` / `FF 05`),
  absolute-tick accounting through running-status channel
  events, and `text_lossy()` resilience against non-UTF-8 bytes.
  Brings the in-crate suite from 307 to 315 unit tests, all
  passing under `cargo test -p oxideav-midi`.

### Round 172 — cargo-fuzz harness over every attacker-facing parser

- New `fuzz/` crate (its own `[workspace]` so it doesn't drag the
  umbrella) with four libfuzzer-sys targets covering every parser
  that takes attacker-controlled bytes end-to-end:
  - `smf` exercises `oxideav_midi::smf::parse` plus the three public
    iteration helpers (`tempo_map`, `time_signatures`,
    `key_signatures`) on every successful parse, so the
    cumulative-tick accounting + meta-event extraction paths cover
    fuzz-discovered shapes too.
  - `sf2` exercises `instruments::sf2::Sf2Bank::parse` (the full
    `RIFF/sfbk` walker + LIST INFO / LIST sdta / LIST pdta
    cross-link resolution).
  - `dls` exercises `instruments::dls::DlsBank::parse` (the
    `RIFF/DLS<space>` walker + `colh` / `ptbl` / `lins-list` /
    `wvpl-list` / per-instrument `rgn ` / `rgn2` / `wsmp` / `wlnk` /
    `art1` / `art2` chain).
  - `sfz` exercises `instruments::sfz::parse_str` (the comment
    stripper + tokenizer + `<global>` / `<master>` / `<group>` /
    `<region>` header walker + opcode flattening + every typed
    field parser).
- Each target asserts the contract every parser advertises: arbitrary
  bytes return a `Result`, with no panic / OOM / integer overflow
  (debug) / out-of-bounds index on any path. The return value is
  intentionally discarded.
- Curated seed corpora under `fuzz/corpus/<target>/` give the fuzzer
  a head start across the well-formed, partial-but-legal, and
  known-edge shapes. The `sfz` corpus also keeps the round-172
  regression input (`regression_r172_octave_overflow.sfz.bin`) so the
  fixed crash stays under perpetual fuzzer pressure.
- **Fuzz-discovered bug fix**: `sfz::parse_key` (note-name → MIDI key
  conversion) used to panic with `attempt to multiply with overflow`
  in a debug build when handed an octave whose magnitude approached
  `i32::MAX` (e.g. `lokey=C-2011420400`, the libfuzzer crash sample).
  The `(octave + 1) * 12 + note_idx + accidental` chain now uses
  `checked_add` / `checked_mul` and falls out to `None` on overflow,
  matching the existing `xyz` / `c100` / `c-100` rejection paths.
  New `parse_key_octave_extremes_do_not_overflow` lib test pins the
  fix against `C-2011420400`, `i32::MAX`, `i32::MIN`, the `g`
  note-name variant, and the previously-tested moderate
  out-of-MIDI-range pair (`c100` / `c-100`).
- Initial 4×~45 s runs cleared **30+ million inputs** across smf /
  sf2 / dls and **2 M inputs** across sfz with zero remaining
  crashes. The harness can run indefinitely; CI does not gate on
  fuzz time.
- 306 → 307 lib tests, 14 → 14 integration tests, 0 ignored.

### Round 128 — `SmfFile::key_signatures()` iteration helper

- New `smf::KeySignatureChange { tick, track, sharps_flats, mode }`
  pins one decoded `FF 59 02 sf mi` Key Signature meta event to the
  absolute tick (cumulative delta-sum) at which it fires on its parent
  track. `sharps_flats` is the signed `-7..=+7` accidental count
  (negative = flats); `mode` is `0` major or `1` minor.
  `KeySignatureChange::tonic_name()` returns the tonic spelling
  (`"C"`, `"F#"`, `"Bb"`, …) and `name()` returns the full key name
  (`"C major"`, `"A minor"`, …). Both helpers consult a 15-entry
  lookup keyed by `sf + 7` and return `None` for out-of-range `sf`
  or any `mode` other than 0 / 1, so junk payloads stay observable
  but never panic. `is_major()` / `is_minor()` mirror the mode bit.
- New `SmfFile::key_signatures()` walks every track, sums per-track
  deltas into absolute ticks, collects every
  `MetaEvent::KeySignature`, and returns the merged stream sorted by
  tick. The sort is stable so two changes at the same tick keep the
  per-track insertion order — track 0 wins over track 1 at the same
  tick, matching the scheduler's merge convention and the existing
  `tempo_map` / `time_signatures` helpers.
- 8 new lib tests (`smf::tests`): empty-when-no-meta-event;
  single-change-at-tick-zero (C major, all four fields + the two
  display helpers); three changes within one track (C major →
  A major → C minor); merge across two tracks sorted by tick; stable
  sort keeps track 0 before track 1 at the same tick; absolute-tick
  accounting after running-status channel events; the full 30-entry
  circle-of-fifths name table (both modes, every `sf` in `-7..=+7`);
  out-of-range `sf` / unknown `mode` produce `None`.
- 298 → 306 lib tests, 14 → 14 integration tests, 0 ignored.

### Round 125 — `SmfFile::tempo_map()` iteration helper

- New `smf::TempoChange { tick, track, microseconds_per_quarter_note,
  bpm }` pins one decoded `FF 51 03 tt tt tt` Set Tempo meta event to
  the absolute tick (cumulative delta-sum) at which it fires on its
  parent track. `bpm` is pre-computed as
  `60_000_000.0 / microseconds_per_quarter_note`;
  `microseconds_per_quarter_note == 0` maps to `f64::INFINITY` so a
  degenerate payload can't divide-by-zero. `TempoChange::new` is the
  public constructor that does the pre-computation.
- New `SmfFile::tempo_map()` walks every track, sums per-track deltas
  into absolute ticks, collects every `MetaEvent::Tempo`, and returns
  the merged stream sorted by tick. The sort is stable so two changes
  at the same tick keep the per-track insertion order — track 0 wins
  over track 1 at the same tick, matching the scheduler's merge
  convention and the existing `SmfFile::time_signatures()` helper.
- 7 new lib tests (`smf::tests`): empty-when-no-meta-event;
  single-change-at-tick-zero (with the BPM cross-check); three
  changes within one track; merge across two tracks sorted by tick;
  stable sort keeps track 0 before track 1 at the same tick;
  absolute-tick accounting after running-status channel events;
  zero µs/qn maps to `+INF` BPM without panic.
- 291 → 298 lib tests, 14 → 14 integration tests, 0 ignored.

### Round 122 — `SmfFile::time_signatures()` iteration helper

- New `smf::TimeSignatureChange { tick, track, numerator,
  denominator_pow2, clocks_per_click, notated_32nd_per_quarter }`
  pins one decoded `FF 58 04 nn dd cc bb` meta event to the absolute
  tick (cumulative delta-sum) at which it fires on its parent track.
  `TimeSignatureChange::denominator()` returns `1 << dd`, saturated
  at `u32::MAX` so a spec-illegal `dd >= 32` can't overflow the
  shift.
- New `SmfFile::time_signatures()` walks every track, sums per-track
  deltas into absolute ticks, collects every
  `MetaEvent::TimeSignature`, and returns the merged stream sorted
  by tick. The sort is stable so two changes at the same tick keep
  the per-track insertion order — track 0 wins over track 1 at the
  same tick, matching the scheduler's merge convention.
- 7 new lib tests (`smf::tests`): empty-when-no-meta-event;
  single-change-at-tick-zero (all six fields); three changes within
  one track; merge across two tracks sorted by tick; stable sort
  keeps track 0 before track 1 at the same tick; absolute-tick
  accounting after running-status channel events; denominator
  saturates on a pathological `dd >= 32`.
- 284 → 291 lib tests, 9 → 9 integration tests, 0 ignored.

## [0.0.2](https://github.com/OxideAV/oxideav-midi/compare/v0.0.1...v0.0.2) - 2026-05-24

### Added

- *(midi)* GM2 Global Parameter Control (Universal Real-Time SysEx 04 05)
- *(midi)* Master Balance (Universal Real-Time SysEx 04 02)
- *(midi)* Data Increment / Decrement (CC 96/97) per RP-018

### Other

- round 98 — MIDI Tuning Standard (MTS) microtuning
- Round 95: SFZ-side filter envelope + fil_type + cutoff wiring
- rewrite release-envelope comment to drop a named third-party-impl reference
- EG2 + 2-pole resonant low-pass filter on the shared SamplePlayer (round 91)
- DLS art1/art2 articulation interpretation (round 80)
- round 75 — MPE + RPN 1/2/5 + CA-25 master tuning + master volume SysEx
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder

### Round 114 — GM2 Global Parameter Control (Universal Real-Time SysEx `04 05`)

- New `mixer::GmEffects` carries the GM2 system-wide Reverb + Chorus
  parameters edited by the **Global Parameter Control** Universal
  Real-Time SysEx message (`F0 7F <dev> 04 05 …`, MMA CA-024). The two
  GM2-reserved slots are `0101` (Reverb) and `0102` (Chorus). Each raw
  7-bit parameter value is decoded to its engineering unit with the
  CA-024 "Recommended Practice for Reverb and Chorus Parameters (from
  General MIDI Level 2)" formulas:
  - Reverb `pp=0` Type (select), `pp=1` Time
    `rt = exp((val − 40) · 0.025)` s.
  - Chorus `pp=0` Type (select), `pp=1` Mod Rate `mr = val · 0.122` Hz,
    `pp=2` Mod Depth `md = (val + 1) / 3.2` ms, `pp=3` Feedback
    `fb = val · 0.763` %, `pp=4` Send-to-Reverb `ctr = val · 0.787` %.
- New `Mixer::set_gm_reverb_param(pp, val)` /
  `Mixer::set_gm_chorus_param(pp, val)` apply one parameter-value pair;
  unrecognised parameter ids are ignored per CA-024 ("only that
  parameter-value pair should be ignored"). `Mixer::gm_effects()`
  exposes the current state and `Mixer::reset_gm_effects()` restores
  the GM2 recommended initial defaults (Reverb Type 4 Large Hall,
  Chorus Type 2 Chorus 3, with the per-type table values).
- The scheduler's Universal Real-Time dispatch now routes sub-ID#2
  `05` (`dispatch_global_parameter_control`): it parses the Slot Path
  Length / Parameter-ID Width / Value Width header, requires the
  GM2-reserved slot path (length 1, Slot MSB 1), reads the MSB-first
  parameter ids and LSB-first values across the pair list to EOX, and
  routes each into the reverb/chorus setters by Slot LSB. Non-GM2 slot
  paths (Slot MSB ≠ 1 or length ≠ 1) are ignored.
- GM 1 / GM 2 System On / GM System Off now also reset the GM2 effect
  parameters to their CA-024 defaults.
- Tests: GM2 defaults; reverb type+time decode; all five chorus
  parameter decodes; non-GM2 slot ignored; unknown-parameter-in-a-pair
  ignored (rest applied); GM-on resets effects. (+7 lib tests.)
- The decoded parameters are observable program state; a reverb/chorus
  DSP send is intentionally deferred to a later round.

### Round 105 — Master Balance (Universal Real-Time SysEx `04 02`)

- New `Mixer::set_master_balance_14(value)` /
  `Mixer::master_balance_14()` carry the device-level Master Balance
  scalar from the MIDI 1.0 *Detailed Specification* v4.2.1 §"DEVICE
  CONTROL — MASTER VOLUME AND MASTER BALANCE" (p.57). The 14-bit
  value is stored verbatim with `0x0000` = hard left,
  `0x2000` = centre, `0x3FFF` = hard right; the setter clamps inputs
  above `0x3FFF` to the spec maximum.
- New `Mixer::master_balance_gains()` returns the per-side
  multipliers `(left, right)` that the mix loop folds into every
  voice's gain — `(1.0, 1.0)` at centre, `(1.0, 0.0)` at hard left,
  `(0.0, 1.0)` at hard right, and a linear ramp on the *far* side
  between centre and each extreme (the *near* side stays at unity).
  This is the textbook "balance between two sound sources" law M1
  v4.2.1 §"BALANCE" describes for CC 8, applied here as the
  device-level analog.
- The stereo + mono branches of `Mixer::mix_stereo` now multiply
  every voice by these master-balance gains. The values are hoisted
  out of the per-slot loop, alongside the existing master-volume
  scalar, so the per-voice arithmetic gains a single extra `f32`
  multiply per side and the default `0x2000` setting produces an
  output buffer byte-identical to the pre-round-105 mix (asserted by
  the new `master_balance_centre_matches_pre_balance_output` test).
- `scheduler::dispatch_universal_sysex` recognises `04 02 lsb msb`
  in the Universal Real-Time area and forwards the combined 14-bit
  value via `set_master_balance_14`. GM 1 / GM 2 System On / GM
  System Off resets now also restore Master Balance to centre
  (`0x2000`), matching the rest of the master-state reset surface.
- 12 new tests (9 `mixer`, 3 `scheduler`): default-centre +
  unity-gains, hard-left mutes right, hard-right mutes left,
  half-left/half-right ramp arithmetic, the clamp-above-14-bit
  guard, the per-side zeroing of the mix output at each extreme, the
  centre-equals-default-output regression, and the three scheduler
  SysEx routings (centre / hard left / hard right). The existing
  `universal_gm_on_sysex_resets_state` test gained an additional
  Master-Balance-set-then-reset assertion.

### Round 102 — Data Increment / Data Decrement (RP-018)

- New `Mixer::data_inc_dec(channel, step)` implements the Data Increment
  (CC 96) / Data Decrement (CC 97) response from the MMA *Response to
  Data Inc/Dec Controllers* recommended practice
  (`docs/audio/midi/recommended-practices/rp18.pdf`, RP-018). Per the
  spec the controller's value byte is *don't care*; the scheduler passes
  a fixed `+1` step for CC 96 and `-1` for CC 97. Each step adjusts the
  sub-field RP-018 prescribes for the currently-selected RPN:
  - **RPN 0** (Pitch Bend Sensitivity): step the LSB (cents). Because
    the mixer stores the combined `pitch_bend_range_cents`
    (= semitones·100 + cents), `±1` performs the spec's
    "LSB-wraps-into-MSB at 100" carry automatically (RP-018 worked
    example: two CC 96 = +2 cents; 200 → 199 borrows down into 1
    semitone + 99 cents). Clamped to `>= 1` so the range never reaches
    zero, and the live pitch bend is re-applied to held voices.
  - **RPN 1** (Channel Fine Tuning): step the LSB of the 14-bit
    fine-tune accumulator; the cents view is re-derived and routed to
    held voices.
  - **RPN 2** (Channel Coarse Tuning): step the MSB (= one semitone) per
    the 4.2-Addendum rule RP-018 cites, clamped to the CA-25 signed
    range −64..=+63.
  - **RPN 5** (Modulation Depth Range, CA-26): step the cents field, the
    RP-018 default for future Registered Parameters, clamped to the
    existing 0..=2400 envelope.
  - **RPN Null** (`0x3FFF`) and any unmodelled / NRPN selection are a
    no-op, mirroring `set_data_entry`'s null guard. NRPNs (CC 98/99) are
    not modelled, so a step issued under an NRPN selection does nothing.
- `scheduler::dispatch` routes CC 96 → `data_inc_dec(ch, 1)` and CC 97 →
  `data_inc_dec(ch, -1)`.
- 11 new tests (10 `mixer`, 1 `scheduler`): RPN-0 cent-step +
  LSB-wraps-into-MSB carry, RPN-0 decrement borrow, value-byte-ignored
  contract, RPN-1 fine-tune LSB step, RPN-2 semitone step + signed-range
  clamp, RPN-5 cent step, RPN-Null no-op, RPN-0 clamp-above-zero, held-
  voice bend re-apply on range widen, and the scheduler CC 96/97 routing
  (with a deliberately nonsense data byte to prove it is ignored).

### Round 98 — MIDI Tuning Standard (MTS) microtuning

- New `tuning` module implements the retuning surface from the MMA
  *MIDI Tuning Messages* specification
  (`docs/audio/midi/extensions/MIDI-Tuning-Updated-Specification.pdf`,
  incorporating CA-020 / CA-021 / RP-020). `TuningTable` holds two
  layers of microtuning state — a global 128-entry key-based table and
  per-channel 12-entry scale/octave tables — both expressed as signed
  cents added to a key's 12-tone-equal-temperament pitch. Defaults to
  equal temperament everywhere, so a synth that never receives an MTS
  message renders bit-identically to the pre-MTS path.
- Data-format decoders, each with worked-example unit tests against the
  spec's tables: `freq_word_to_cents_offset` (3-byte
  `semitone + fraction14/16384` frequency word → cents offset from the
  addressed key, with the reserved `7F 7F 7F` "no change" word
  returning `None`); `scale_octave_1byte_to_cents` (`00 = -64 c`,
  `40 = 0 c`, `7F = +63 c`); `scale_octave_2byte_to_cents` (14-bit,
  `0x0000 = -100 c`, `0x2000 = 0 c`, `0x3FFF = +100 c`); and
  `scale_octave_channel_mask` (the `ff gg hh` 3-byte channel bitmap,
  with `ff` bits 2–6 reserved → must not light any channel).
- `Mixer` carries a `TuningTable`; the per-key offset is folded into
  every voice-pitch composition site (note-on + the two
  `set_pitch_bend` re-apply paths). Drum channel (MIDI 10 = index 9) is
  exempt from retuning, matching the existing CA-25 master-tuning
  exemption. New public API: `set_key_tuning_word`,
  `set_scale_octave_tuning`, `reset_tuning`, `tuning()`. The real-time
  message forms re-apply pitch to sounding voices (`live = true`); the
  non-real-time "setup" forms update only the stored table.
- `scheduler::dispatch_universal_sysex` now routes sub-ID#1 `08` (MIDI
  Tuning Standard) in both the Universal Real-Time (`7F`) and
  Non-Real-Time (`7E`) areas: Single-Note Tuning Change (sub-ID#2
  `02`) and its bank form (`07`), and Scale/Octave Tuning 1-byte
  (`08`) and 2-byte (`09`) forms. Multi-change single-note messages
  (`ll` entries) and truncated/over-promised buffers are bounds-checked
  so malformed input cannot read past the payload. GM 1 / GM 2 System
  On / GM System Off now also reset MTS tuning to equal temperament.
- 25 new tests (12 `tuning` unit, 7 `mixer`, 6 `scheduler`) covering
  the decoders, table summation, live vs. setup re-apply, drum-channel
  exemption, pitch-bend summation, channel-mask selection, GM reset,
  and truncated-message safety.

### Round 95 — SFZ-side filter envelope + `fil_type` + `cutoff` wiring

- New `FilterType` enum on `instruments::sample_voice` covers the six
  SFZ v1 `fil_type` values documented in
  `docs/audio/midi/instrument-formats/sfz-legacy.html` "Filter type"
  table: `lpf_1p` / `hpf_1p` (one-pole, 6 dB/oct, resonance ignored)
  and `lpf_2p` / `hpf_2p` / `bpf_2p` / `brf_2p` (two-pole, 12 dB/oct).
  Default is `TwoPoleLowPass`, which preserves round-91 SF2 / DLS
  behaviour exactly. `FilterType::parse_sfz()` honours the
  case-insensitive opcode string + falls back to `lpf_2p` on unknown
  values per the SFZ default convention.
- `FilterParams` gains a `kind: FilterType` field. SF2 / DLS construct
  the struct without overriding it (they inherit `TwoPoleLowPass`); the
  `instruments::articulation::Articulation::filter()` helper also pins
  the kind explicitly to make the SF2/DLS commitment to a single shape
  permanent against any future `Default` flip.
- `SamplePlayer::update_filter_coeffs` switches on `filter_kind`. The
  one-pole arms compute `tan(ω/2)`-prewarped bilinear coefficients with
  `b2 = a2 = 0` so the same direct-form-1 `tick()` math still applies
  (no per-sample branch). The 2-pole arms keep the round-91 RBJ-cookbook
  denominators (`a0 = 1 + α`, `a1 = -2·cos(ω)`, `a2 = 1 - α`) and switch
  numerators per shape: low-pass `(1-cos)/2, 1-cos, (1-cos)/2`,
  high-pass `(1+cos)/2, -(1+cos), (1+cos)/2`, band-pass
  `sin/2, 0, -sin/2`, band-reject `1, -2cos, 1`. All derivations are
  the project's own RBJ math (bilinear transform of the analog
  prototypes) — SF2 §8.1.3 + §9.7 + SFZ-legacy `fil_type` row only
  specify the response shape and slope, not the discrete realisation.
- `SamplePlayer::new`'s `needs_filter` gate now respects non-low-pass
  shapes: a `fil_type` of `hpf_*` / `bpf_2p` / `brf_2p` is not "open"
  at the SF2 13500-cents sentinel (it would still attenuate the audible
  band), so the biquad allocation also fires whenever
  `cfg.filter.kind` is anything other than the two low-pass variants.
- `instruments::sfz::build_filter_from_opcodes()` parses `cutoff=`
  (Hz, with a `cents = 1200·log2(fc_hz / 8.176)` bridge into SF2
  absolute cents — clamped into the §8.1.3 useful range `1500..=13500`),
  `resonance=` (dB → centibels at `cb = dB · 10`, clamped to the
  SFZ-spec range `0..=40 dB`), and `fil_type=` (via
  `FilterType::parse_sfz`). Aliases (`cutoff2`, `resonance2`,
  `filtype`) are honoured per the opcode index.
- `instruments::sfz::build_mod_env_from_opcodes()` parses the SFZ v1
  `fileg_*` Envelope-Generator opcodes (`fileg_delay` /
  `fileg_attack` / `fileg_hold` / `fileg_decay` / `fileg_sustain` /
  `fileg_release` + `fileg_depth` for the routing depth) and their
  SFZ v2 `fil_*` aliases (`fil_delay` etc.). `fileg_sustain` is in
  percent and maps to the SamplePlayer's `0..=1` fraction;
  `fileg_depth` is clamped into the documented `-12000..=12000`
  range and dropped into `ModEnvParams::to_filter_cents`.
- `instruments::sfz::build_config_for_region` calls the two helpers
  instead of plumbing `Default::default()` for `mod_env` / `filter`.
  Regions without `fileg_*` / `fil_type` / `cutoff` / `resonance`
  opcodes still render bit-identically to the round-91 path — the
  SF2 "filter open" sentinel (13500 cents) + the inert `ModEnvParams`
  default fall straight through `build_filter_from_opcodes()` /
  `build_mod_env_from_opcodes()`.
- 18 new tests in the crate:
  - `instruments::sample_voice::tests::high_pass_attenuates_below_cutoff`
  - `instruments::sample_voice::tests::band_pass_peaks_at_cutoff`
  - `instruments::sample_voice::tests::band_reject_kills_signal_at_cutoff`
  - `instruments::sample_voice::tests::one_pole_low_pass_attenuates_high_frequencies`
  - `instruments::sfz::tests::hz_to_filter_cents_round_trips_known_values`
  - `instruments::sfz::tests::build_filter_from_opcodes_honours_cutoff_resonance_filtype`
  - `instruments::sfz::tests::build_filter_from_opcodes_defaults_to_open_lpf_2p`
  - `instruments::sfz::tests::build_filter_from_opcodes_clamps_resonance_into_range`
  - `instruments::sfz::tests::filter_type_parse_covers_all_sfz_v1_values`
  - `instruments::sfz::tests::build_mod_env_from_opcodes_honours_fileg_set`
  - `instruments::sfz::tests::build_mod_env_from_opcodes_aliases_v2_names`
  - `instruments::sfz::tests::build_mod_env_default_is_inert`
  - `instruments::sfz::tests::build_mod_env_clamps_depth`
  - `instruments::sfz::tests::full_template_smoke_drops_cutoff_into_sample_player`
  - `instruments::sfz::tests::sfz_with_low_cutoff_attenuates_high_frequencies`
  - `instruments::sfz::tests::sfz_high_pass_inverts_attenuation`
  - `instruments::sfz::tests::sfz_fileg_sweeps_cutoff_open_during_attack`
  - `instruments::sfz::tests::build_config_for_region_default_filter_open`

  All 228 lib tests + 14 integration tests pass (242 total, 0 ignored).
- Sources: `docs/audio/midi/instrument-formats/sfz-opcodes-index.html`
  (Aria opcode reference: `cutoff`, `resonance`, `fil_type`, `fileg_*`
  + `fil_*` aliases with documented defaults / ranges / SFZ-version
  tags), `docs/audio/midi/instrument-formats/sfz-legacy.html`
  ("Filter type" table enumerating the six `fil_type` values + the
  "filter disabled" semantics for an absent `cutoff` opcode),
  `docs/audio/midi/instrument-formats/sf2-spec-2.04.pdf` §8.1.3
  (`initialFilterFc` / `initialFilterQ` unit conventions reused
  unchanged from round 91).

### Round 91 — EG2 + 2-pole resonant low-pass filter on the shared `SamplePlayer`

- New `ModEnvParams` + `FilterParams` structs on
  `instruments::sample_voice`. `ModEnvParams` mirrors the existing
  `EnvelopeParams` DAHDSR shape but treats `sustain_level` as the
  *post-decrease linear fraction* per SF2 v2.04 §8.1.3 `sustainModEnv`
  ("decrease in level expressed in 0.1 % units", with 0 = peak and 1000
  = silence) and carries `to_filter_cents` — the per-mod-env-unit
  contribution to the filter cutoff per SF2 gen 11 `modEnvToFilterFc`.
  `FilterParams` carries `cutoff_cents` (SF2 absolute cents re. 8.176
  Hz; default 13500 = "filter open" sentinel from §8.1.3 gen 8) and
  `q_centibels` (0 = Butterworth, per §8.1.3 gen 9).
- `SamplePlayer::new` allocates a per-voice 2-pole resonant low-pass
  biquad only when the bank actually wants the filter (cutoff <
  13000 cents *or* mod-env-to-filter routing depth > 200 cents).
  Otherwise the per-sample biquad work is skipped entirely so
  unconfigured SFZ / DLS regions render bit-identically to the
  round-80 pre-filter path.
- Biquad coefficients computed from the SF2 §8.1.3 cents reference
  `fc_hz = 8.176 * 2^(cents/1200)` (clamped to the spec useful range
  `1500..=13500`, then clipped at `0.99 * Nyquist`) and the RBJ
  cookbook low-pass formulas: `a0 = 1 + α`, `a1 = -2·cos(ω)`,
  `a2 = 1 - α`, `b0 = b2 = (1 - cos(ω))/2`, `b1 = 1 - cos(ω)`, where
  `α = sin(ω)/(2Q)`. Q (centibels) → linear `q_lin = √(½) · 10^(cb/200)`
  with a `0.1..=16.0` clamp so runaway resonance can't produce NaN
  coefficients. The RBJ derivation is the project's own clean-room
  math (bilinear transform of the analog 2-pole low-pass
  `H(s) = ω₀² / (s² + (ω₀/Q)s + ω₀²)`) — SF2 §8.1.3 explicitly
  leaves the filter implementation to the renderer per §9.7.
- `SamplePlayer::render` evaluates the mod-env DAHDSR each sample,
  adds `mod_env_level * mod_env_to_filter_cents` to the initial
  cutoff, and lazily recomputes biquad coefficients only when the
  live cutoff drifts > 50 cents from the last computed value (cheap
  perceptual gate that keeps the inner loop multiplication-only
  during the delay / hold / sustain phases when the mod-env is
  constant). The filter sits between sample fetch and amplitude
  envelope, so EG1 amplitude shaping is post-filter as expected.
- `SamplePlayer::release` now captures the EG2 release-start level
  separately from the EG1 release-start level so the filter cutoff
  tails off from wherever the mod-env was at note-off, not from
  peak — a discontinuity-free release for both amplitude and filter.
- New `Articulation::mod_env() -> ModEnvParams` +
  `Articulation::filter() -> FilterParams` helpers on
  `instruments::articulation`. `mod_env()` converts the DLS EG2
  destinations (`CONN_DST_EG2_DELAYTIME` / `_ATTACKTIME` /
  `_HOLDTIME` / `_DECAYTIME` / `_SUSTAINLEVEL` / `_RELEASETIME`
  plus the `SRC_EG2 → DST_FILTER_CUTOFF` routing depth) into the
  SamplePlayer-side `ModEnvParams`; `filter()` maps
  `CONN_DST_FILTER_CUTOFF` / `CONN_DST_FILTER_Q` into `FilterParams`,
  falling back to the SF2 "filter open" sentinel (13500 cents) when
  the region carries no filter blocks.
- `instruments::dls::build_dls_config` plumbs `art.mod_env()` +
  `art.filter()` into the `SamplePlayerConfig`. A DLS bank with no
  `art2` filter blocks produces the same audio it did pre-round-91;
  a bank with an `art2` `SRC_EG2 → DST_FILTER_CUTOFF` block sweeps
  the cutoff exactly as the spec describes.
- `instruments::sfz::build_config_for_region` populates the new
  fields with `Default::default()` — SFZ `fileg_*` / `fil_type` /
  `cutoff` opcodes are not yet plumbed; SFZ banks render
  bit-identically to the round-80 path until a future round wires
  the SFZ filter opcodes.
- 12 new tests across the crate:
  - `instruments::sample_voice::tests::default_filter_is_inert_no_biquad_allocated`
  - `instruments::sample_voice::tests::low_cutoff_filter_attenuates_high_frequencies`
  - `instruments::sample_voice::tests::mod_env_dahdsr_shape_matches_spec`
  - `instruments::sample_voice::tests::eg2_filter_sweep_changes_spectrum_over_note`
  - `instruments::sample_voice::tests::high_q_filter_resonates_at_cutoff`
  - `instruments::sample_voice::tests::release_captures_mod_env_level`
  - `instruments::articulation::tests::default_mod_env_is_inert`
  - `instruments::articulation::tests::default_filter_is_open_sentinel`
  - `instruments::articulation::tests::eg2_to_filter_routing_lands_on_mod_env`
  - `instruments::articulation::tests::filter_cutoff_override_lands_on_filter`
  - `instruments::articulation::tests::eg2_attack_time_lands_on_mod_env_attack`
  - `tests::dls_articulation::dls_articulation_eg2_sweeps_filter_cutoff_open`
    (integration: builds a DLS bank with `SRC_EG2 → DST_FILTER_CUTOFF`
    + slow EG2 attack, renders through the full `MidiDecoder`, asserts
    the late-window RMS is > 1.6× the early-window RMS).
- Test count moves from 212 → 224.
- Spec backing: SF2 v2.04 §8.1.3 generators 8 + 9 + 11 + 25–30
  (`docs/audio/midi/instrument-formats/sf2-spec-2.04.pdf`); DLS Level
  2.2 v1.0 Amendment 2 §1.5.2 + Tables 5–6 + 8–10 + 1.13 + 1.14
  (`docs/audio/midi/instrument-formats/dls2amd2(all)a(pub).pdf`); DLS
  Level 1 v1.1b "Device Architecture" + "Articulation"
  (`docs/audio/midi/instrument-formats/dls1v11b.pdf`).

### Round 80 — DLS `art1` / `art2` articulation interpretation

- New module `instruments::articulation` interpreting DLS Level 1 and
  Level 2 connection blocks at voice-build time. Backed by MMA DLS
  Level 1 v1.1b (`docs/audio/midi/instrument-formats/dls1v11b.pdf`,
  Table 1 + 2 of the Device Architecture section) and MMA DLS Level 2.2
  v1.0 Amendment 2 (`docs/audio/midi/instrument-formats/dls2amd2(all)a(pub).pdf`,
  Tables 5–10).
- Named constants for every `CONN_SRC_*` / `CONN_DST_*` / `CONN_TRN_*`
  enum from DLS2 Tables 8 + 9 + 10, plus `ABSOLUTE_ZERO` sentinel,
  exported from `instruments::articulation` so callers can build
  blocks programmatically (used by the new
  `crates/oxideav-midi/tests/dls_articulation.rs` integration test).
- New `Articulation::evaluate(region_blocks, instrument_blocks) ->
  Articulation` walks the region-level then instrument-level
  connection lists, overlaying every recognised connection on top of
  the spec defaults (DLS2 Tables 5 + 6). Per the DLS spec, a region
  block overrides the corresponding instrument-level block.
- Supported `SRC_NONE → DST_x` connections (the "absolute default
  override" branch): Vol EG (EG1) delay / attack / hold / decay /
  sustain / release; Mod EG (EG2) delay / attack / hold / decay /
  sustain / release (raw, surfaced for a later round); modulator LFO
  frequency + start delay; vibrato LFO frequency + start delay;
  filter cutoff (DST_FILTER_CUTOFF) and Q; tuning (DST_PITCH);
  per-region gain (DST_GAIN); pan (DST_PAN).
- Supported modulator routings: `SRC_LFO → DST_PITCH` (vibrato depth
  on DLS1-style banks where LFO and vibrato share a source);
  `SRC_LFO → DST_GAIN` (tremolo depth); `SRC_VIBRATO → DST_PITCH`
  (dedicated DLS2 vibrato depth — wins over `SRC_LFO → DST_PITCH`
  when both are present); `SRC_EG2 → DST_PITCH` and `SRC_EG2 →
  DST_FILTER_CUTOFF` (mod-env routings, raw); `SRC_KEYONVELOCITY →
  DST_EG1_ATTACKTIME` (velocity-dependent attack, raw).
- `DlsInstrument::make_voice` now calls
  `Articulation::evaluate(&region.articulation, &inst.articulation)`
  and folds the resulting [`EnvelopeParams`] + [`VibratoParams`] +
  tuning cents + gain multiplier into the `SamplePlayerConfig`. An
  empty `lart` list still falls back to the SamplePlayer defaults so
  banks with no articulation are bit-identical to round-75 output.
- Unit conversions: time-cents → seconds (DLS §1.14.3, clamped at
  60 s), absolute-pitch → cents (DLS §1.14.1, clamped at ±14 400),
  absolute-pitch → Hz (LFO frequency, clamped at 50 Hz), gain → linear
  amplitude (DLS §1.14.4, clamped at -96..+48 dB), sustain-percent
  → 0..=1, pan-percent → ±50.
- New integration test `tests/dls_articulation.rs` exercises the full
  SMF → scheduler → DLS bank → articulation → SamplePlayer → PCM path:
  a `SRC_NONE → DST_PITCH @ +1200 cents` block produces an audibly
  different rendering than the same bank with no `lart` list (avg
  pointwise PCM diff > 50 LSB); a `SRC_NONE → DST_EG1_RELEASETIME`
  block at +2 s sustains a louder release tail than the default 100 ms.
- 9 new unit tests in `instruments::articulation`: spec-default
  fallback when both lists are empty; region overrides instrument;
  instrument-level fallback when region is silent; tuning cents
  conversion; LFO → pitch routes to the vibrato depth; DLS2 vibrato
  source takes precedence over the mod LFO source; gain destination
  attenuates correctly (-6 dB → 0.5012 linear); ABSOLUTE_ZERO sentinel
  is skipped without overriding the default; unrecognised connections
  are dropped silently.

Total test count: 199 lib + 13 integration = 212 (up from 190 + 11 in
round 75).

### Round 75 — MPE + RPN expansion + Master Tuning / Master Volume SysEx

- **MIDI Polyphonic Expression (MPE) v1.1** end-to-end
  (`docs/audio/midi/extensions/M1-100-UM_v1-1_MIDI_Polyphonic_Expression_Specification.pdf`):
  the MPE Configuration Message (RPN 0x0006 on channel 0 = Lower
  Manager / channel 15 = Upper Manager) configures one or two zones
  via [`Mixer::set_mpe_zone`]. Per §2.2.5 the receiver sets Manager
  Channel PB Sensitivity to 2 semitones and every Member Channel to
  48 semitones at MCM time. Per §2.2.7 Polyphonic Key Pressure on a
  Member is silently dropped (the spec says it "shall not be sent").
  Per Appendix C, Member Channel Pitch Bend sums in cents with the
  Manager Channel's bend before reaching the held voice. Per §2.2.3,
  a zone reconfiguration stops every Sounding Note on the affected
  channels and resets their controllers. New types
  [`Mixer::MpeZone`] / [`Mixer::MpeRole`] / [`Mixer::MpeZoneKind`]
  surface the zone topology to callers.
- **CC #74 ("third dimension of control")** routed through the new
  [`Voice::set_timbre`] hook. MPE Manager broadcasts to every voice
  in the zone; Member only reaches its own voice. Non-MPE channels
  route plainly.
- **RPN 1 (Channel Fine Tuning)** — 14-bit data-entry value maps
  linearly to ±100 cents (centre 0x40/0x00). Stored as both the raw
  accumulator and the derived cents view on `ChannelState` so an
  MSB-then-LSB sequence composes bit-exact.
- **RPN 2 (Channel Coarse Tuning)** — CC 6 MSB sets signed semitone
  offset centred on 0x40 (-64..=+63). CC 38 LSB ignored per spec
  ("the LSB is always 0").
- **RPN 5 (Modulation Depth Range)** per
  `docs/audio/midi/recommended-practices/ca26-RPN05-Modulation-Depth-Range.pdf`:
  CC 6 sets whole-cent range, CC 38 sets fractional cents (0..=99).
  Default 50 cents matches the GM 2 recommended practice. Clamped
  to ≤ 2400 cents (±2 octaves) so a stray CC 6 = 127 can't pop the
  timbre out of audibility.
- **CC 1 (Modulation Wheel)** routed via
  [`Mixer::set_mod_wheel`] → [`Voice::set_mod_depth_cents`] using
  the channel's RPN-5 range. Applied to every held voice plus
  picked up at note-on for new voices.
- **Universal Real-Time SysEx — Master Volume** (`F0 7F <dev> 04 01
  lsb msb F7`): the 14-bit value applies as a multiplicative global
  gain at mix time.
- **Universal Real-Time SysEx — Master Fine Tuning / Master Coarse
  Tuning** per
  `docs/audio/midi/recommended-practices/ca25-Master-Fine-Coarse-Tuning-SysEx-Message.pdf`:
  Fine is ±100 cents centred on 0x40/0x00 (formula
  `100/8192 × (value - 0x2000)`); Coarse is a signed semitone count
  centred on 0x40 with the LSB always 0 per spec. Both sum with the
  per-channel RPN-1 / RPN-2 tuning + the live pitch bend into the
  effective cents pushed to each voice. Drum channel (MIDI 10 =
  index 9) is exempt from all tuning per CA-25's "MUST NOT result in
  MIDI note-shifting" clause.
- **Universal Non-Real-Time SysEx — GM 1 / GM 2 System On + GM
  System Off** (sub-IDs `09 01` / `09 02` / `09 03`) reset the
  mixer's master state to GM defaults: master volume → max, master
  fine / coarse tuning → centre, all sounding notes off.
- **Voice trait extension**: `set_mod_depth_cents` + `set_timbre`
  default-no-op methods so existing voices keep working unchanged.
- **Per-channel state expansion**: `ChannelState` gains
  `mod_wheel`, `mod_depth_range_cents`, `channel_fine_tune_cents`,
  `channel_fine_tune_raw_14`, `channel_coarse_tune_semitones`, and
  `mpe_role` fields. All zero / centre by default so existing tests
  see unchanged routing.
- **Tests added**: 38 new — 27 lib-side mixer (RPN 1 / 2 / 5
  data-entry, channel + master fine/coarse routing, drum channel
  exemption, master volume scaling, mod-wheel routing, CC 74
  routing, MPE zone assignment + role tagging + PB sensitivity
  defaults + zero-members deactivate + Upper zone, Member + Manager
  PB combining, Manager CC 74 broadcast, Member CC 74 isolation,
  Polyphonic Key Pressure dropped on Member, Member + Manager
  pressure combining, zone-conflict resolution, MCM via the data-
  entry pathway, MCM on a non-Manager channel ignored), 6 lib-side
  scheduler (CC 1 / CC 74 / Master Volume / Master Fine / Master
  Coarse / GM-on routing + MCM via SMF), and 5 new integration
  (`tests/mpe_and_master_tuning.rs`) checking the public
  `MidiDecoder` surface. Total: 189 lib + 11 integration = 200
  passing (was 156 + 6 = 162).

Wall respected: every change in this round used only
`docs/audio/midi/extensions/M1-100-UM_v1-1_MIDI_Polyphonic_Expression_Specification.pdf`
(pages 1–28), `docs/audio/midi/recommended-practices/ca25-Master-Fine-Coarse-Tuning-SysEx-Message.pdf`,
`docs/audio/midi/recommended-practices/ca26-RPN05-Modulation-Depth-Range.pdf`,
`docs/audio/midi/midi-1.0/Universal-System-Exclusive-Messages.pdf`,
plus the in-tree crate sources + `oxideav-core`'s public API. No
external library source consulted, paraphrased, or cross-checked.


## [0.0.1](https://github.com/OxideAV/oxideav-midi/compare/v0.0.0...v0.0.1) - 2026-05-04

### Other

- SFZ + DLS voice generators (task #410)
- DLS Level 1 + 2 RIFF reader (parse + dump bank)
- SFZ text patch reader (load + dump regions)

### Round 9 — SFZ + DLS voice generators (task #410)

- **Shared sample-playback voice** (`instruments::sample_voice`). Mono
  in, mono out; the [`mixer`](src/mixer.rs) handles stereo panning.
  Covers the DAHDSR amplitude envelope (delay / attack / hold / decay /
  sustain / release), four loop modes (`NoLoop`, `OneShot`,
  `LoopContinuous`, `LoopSustain`), pitch bend via the existing
  `Voice::set_pitch_bend_cents` hook, channel/poly aftertouch via
  `Voice::set_pressure`, exclusive-class drum cuts, and a sine vibrato
  LFO with rate / depth / start-delay.
- **Minimal RIFF/WAVE PCM decoder** (`instruments::wav_pcm`). Decodes
  8-bit unsigned, 16-bit signed LE, 24-bit signed LE, 32-bit signed LE
  PCM, and 32-bit IEEE_FLOAT into mono f32. Stereo / multi-channel WAVs
  are mixed down to mono by averaging channels (round-2 voice
  generation will keep stereo intact).
- **SFZ voice generator**. `SfzInstrument::make_voice` walks the
  flattened region table for the highest-priority match on `(key,
  velocity)`, decodes the WAV bytes loaded by `SfzInstrument::open`,
  shifts pitch off `pitch_keycenter` + `tune` + `transpose`, and
  instantiates a `SamplePlayer` honoring the region's `loop_*` opcodes
  + an amplitude envelope from `ampeg_delay/attack/hold/decay/sustain/
  release` + a vibrato LFO from `lfo01_freq` / `lfo01_pitch` /
  `lfo01_delay` (with `vibrato_*` aliases).
- **DLS Level 1 + 2 voice generator**. `DlsInstrument::make_voice`
  picks the matching instrument by MIDI program (bank-MSB / LSB
  matching is round 2), picks a region by `(key, velocity)`, resolves
  `wlnk.table_index` → `ptbl` cue → wave-pool entry, decodes the PCM
  via `wav_pcm::decode_pcm_bytes`, and plays the sample through the
  shared `SamplePlayer`. Region-level `wsmp` overrides the wave-level
  default per the spec; `WLOOP_TYPE_FORWARD` (0) maps to
  `LoopContinuous`, `WLOOP_TYPE_RELEASE` (1) maps to `LoopSustain`.
  `art1`/`art2` connection-block evaluation is round 2 — the parsed
  blocks remain on the bank.
- **`InstrumentSource` builder** + `MidiDecoder::with_instrument_source`.
  Caller passes `InstrumentSource::sf2(path)` / `sfz(path)` / `dls(path)`
  / `Tone` and the decoder picks the right loader. Format detection is
  *not* by extension — the caller picks the variant.
- **Tests added**: 13 net new lib-side (5 sample_voice, 4 wav_pcm, 3
  SFZ voice-generation, 2 DLS voice-generation, minus the 2 existing
  `make_voice_returns_unsupported` tests that were replaced/upgraded
  to actually exercise the round-1 voice path) + 3 integration
  (`tests/voice_round_trip.rs`) exercising end-to-end SFZ/SF2/DLS
  rendering through `MidiDecoder` with an RMS non-silence assertion.
  Total: 156 lib + 6 integration = 162 passing (was 143 + 3 = 146).

### Round 8 — DLS Level 1 + 2 sample-loader (task #409)

- **DLS RIFF parser**: walks the `RIFF`/`DLS ` form and pulls the
  `colh` collection header, optional `vers` version stamp, `ptbl`
  pool table, `lins` instrument list, `wvpl` wave pool, and
  top-level `INFO` metadata into a fully-resolved
  [`DlsBank`](src/instruments/dls.rs). Instruments → regions →
  wave-pool samples are cross-referenced; nothing references back
  into the source bytes.
- **Wave pool**: every `wave-list` entry is parsed for its standard
  WAV `fmt ` + `data` chunks plus the optional `wsmp` per-wave loop
  / pitch / gain header. Sample bytes are kept in their on-disk
  form (8-bit unsigned or 16-bit LE signed); decode is round-2.
- **Instrument table**: each `ins ` LIST surfaces its bank/program
  (decoded into `bank_msb` / `bank_lsb` / `program_number` and the
  `is_drum()` bit-31 helper), instrument name from a per-instrument
  `INFO/INAM`, and an instrument-level articulation list parsed
  from `lart` (DLS1) or `lar2` (DLS2) sub-LISTs.
- **Regions**: `rgnh` (key + velocity range, fusOptions, key group,
  optional DLS2 `usLayer`), `wsmp` (per-region overrides), `wlnk`
  (cue-table reference), and per-region articulation. DLS2 `rgn2`
  LISTs parse alongside DLS1 `rgn ` and are flagged via
  `DlsRegion::is_level2`.
- **Articulation**: `art1-ck` and `art2-ck` connection blocks
  (12-byte records: source / control / destination / transform /
  scale) parse into `Vec<DlsArticulationBlock>` tagged with
  `DlsArtKind::{Art1, Art2}` so the round-2 voice generator picks
  the right enum table (DLS1 spec page 43 / DLS2 spec tables 8-10).
  Connection enums are stored as raw `u16`s — no interpretation in
  round 1.
- **Magic-byte stub becomes real probe + parser**: `is_dls()` and
  the new `DlsInstrument::probe()` honour the `RIFF`/`DLS ` magic;
  `DlsInstrument::open()` and `parse_bytes()` plumb through to the
  full bank parser. `make_voice()` still returns
  `Error::Unsupported` (round-2 work, same shape as the SFZ
  followup).
- **Bounds + caps**: every chunk length is checked against bytes
  remaining; pool-table, articulation, and wave-pool counts are
  capped at `MAX_RECORDS` (1 Mi); cumulative wave-data bytes capped
  at `MAX_WAVE_BYTES` (256 MiB).
- **Tests added**: 13 lib-side (magic detection, minimal-DLS
  parse + wave pool + instrument + region + articulation, DLS2
  rgnh-with-usLayer, art2 block, wsmp loop record, error paths
  for non-DLS / truncated outer / non-DLS path, drum-bit decode,
  open round-trip through disk) + 1 integration smoke
  (`tests/sfz_sf2_dls_smoke.rs`) building a 2-region DLS in
  memory and dumping the instrument + region table. Total: 143
  lib + 3 integration = 146 passing (was 130 + 2 = 132).
- **Smoke test renamed** from `tests/sfz_sf2_smoke.rs` to
  `tests/sfz_sf2_dls_smoke.rs` to reflect the wider coverage.

### Round 7 — SFZ text patch reader (task #127)

- **SFZ parser**: tokenises SFZ syntax (line `// ...` + block
  `/* ... */` comments, `<header>` sections, `name=value` opcode
  pairs with space-bearing values like sample paths) and walks the
  full `<control>` / `<global>` / `<master>` / `<group>` / `<region>`
  hierarchy. Inheritance is flattened into one fully-resolved opcode
  map per region (`global → master → group → region`, later overrides
  earlier).
- **Strongly-typed region fields** for the round-2 voice generator:
  `sample_path`, `lokey` / `hikey` / `lovel` / `hivel`,
  `pitch_keycenter`, `key` (sets lokey + hikey + pitch_keycenter),
  `loop_start` / `loop_end`, `loop_mode` (no_loop / one_shot /
  loop_continuous / loop_sustain), `transpose`, `tune` (alias
  `pitch`), `volume`, `pan`, `trigger`. Note names (`C4`, `c#4`,
  `Db5`, `c-1`) parse alongside decimal MIDI keys.
- **Sample loader**: `SfzInstrument::open` resolves every `sample=`
  path against the SFZ file's directory + the active `<control>
  default_path=` opcode and reads the bytes off disk into
  `region.sample_bytes`. Missing or unreadable samples become a hard
  parse error so the caller learns at load time. `parse_str` skips
  the filesystem hooks for in-memory tests.
- **Preprocessor**: `#include` is rejected with `Error::Unsupported`
  (round-1 reader doesn't follow includes); `#define` is stored
  verbatim in the surrounding scope's opcode map without macro
  expansion.
- **DLS reader status: docs-blocked**. The new
  `docs/audio/midi/instrument-formats/` directory contains the SFZ
  format docs (10 HTML files) plus the SoundFont 2.04 spec PDF, but
  no Microsoft DLS Level 1/2 specification. The DLS magic-byte stub
  remains in place; voice generation continues to return
  `Error::Unsupported`.
- New tests: 23 added — 22 lib-side covering tokenisation, comment
  stripping, key-name parsing, header inheritance, group reset,
  control / default_path resolution, loop opcodes, opcode-map
  preservation, `#include` rejection, sample loading + missing-file
  handling, and a tutorial-shaped template smoke; 2 integration tests
  (`tests/sfz_sf2_smoke.rs`) that dump SFZ regions + an SF2 preset
  list via the public API. Total: 130 lib + 2 integration = 132
  passing (was 111).

### Round 6 — SF2 polish (sm24 + stereo + mod-env + filter, task #139)

- **24-bit sample storage** (SF2 2.04+ `sm24` chunk). PCM is now
  stored as `Arc<[i32]>` carrying signed 24-bit values in the lower
  24 bits. When `sm24` is present its u8 lower bytes are combined
  with the 16-bit `smpl` upper bytes; otherwise the 16-bit value is
  widened by left-shift-8. Mismatched sm24 length is silently
  ignored per spec ("parsers must tolerate"). Voice fetch divides
  by 2^23 instead of 2^15.
- **Stereo SF2 zones**. Sample headers tagged `LEFT` / `RIGHT` with
  a valid bidirectional `sample_link` are detected at `resolve` time
  and produce a stereo-aware `Sf2Voice` that holds two phase
  counters and writes distinct L/R via the new `Voice::render_stereo`
  hook. The mixer routes such voices through a balance law (cos/sin
  scaled to unity at centre) rather than its mono-pan path.
- **Modulation envelope** (gens 25-30 — delay/attack/hold/decay/
  sustain/release). Same DAHDSR shape as the volume envelope but
  with `0..=1` sustain levels; release tracks the volume envelope's
  release_pos so a note-off cleanly tails both off together.
- **Mod-env routing** (gens 7 + 11). `modEnvToPitch` adds the
  envelope-scaled cents offset to the live pitch-bend cents on every
  sample. `modEnvToFilterFc` modulates the biquad cutoff in cents.
- **Initial low-pass filter** (gens 8/9). Direct-form-1 RBJ-cookbook
  biquad on the voice output. Cutoff in absolute cents (re. 8.176
  Hz), Q in centibels of resonance. Filter state is allocated only
  when the cutoff is below ~12 kHz or the mod-env routes meaningfully
  to it; bypass otherwise.
- **Exclusive class** (gen 57). Note-on with the same non-zero class
  on the same channel hard-stops every prior voice in that class —
  used for hi-hat open/closed pairs in drum kits. Implemented in
  the mixer via a new `Voice::exclusive_class` hook.
- **Pitch-wheel range RPN 0** verified end-to-end. `ChannelState::
  pitch_bend_range_cents` defaults to 200 (±2 semitones); CC 100/101
  selects RPN 0; CC 6/38 sets the semitone+cent range. Live bend is
  re-applied on range change so still-held voices pick up the new
  scale. Existing tests cover the path.
- New tests: 16 added (sm24 combine + missing-sm24 fallback +
  wrong-length tolerance; stereo resolve + render + mixer routing +
  self-link rejection; filter HF attenuation + default bypass;
  mod-env routing + brightening; exclusive class propagate + cut;
  overridingRootKey verification; 24-bit grid sanity). Total: 111
  passing (was 95).
