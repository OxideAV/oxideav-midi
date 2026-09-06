# oxideav-midi

[![CI](https://github.com/OxideAV/oxideav-midi/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-midi/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-midi.svg)](https://crates.io/crates/oxideav-midi) [![docs.rs](https://docs.rs/oxideav-midi/badge.svg)](https://docs.rs/oxideav-midi) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust **MIDI** — Standard MIDI File (`.mid` / SMF) parser + writer,
the MIDI 2.0 axis (Universal MIDI Packet, `.midi2` MIDI Clip File,
MIDI-CI), transport metadata, and a soft-synth. Zero C dependencies,
zero FFI, zero `*-sys`.

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
  convention, FIFO re-strike) plus `active_notes_at(tick)`. `notes()`
  also folds the CC 88 **High-Resolution Velocity Prefix** (CA-031)
  into the pairing: a pending `Bn 58 vv` affixes its 7 bits below the
  next Note On / Off velocity on the channel
  (`Note::on_velocity14` / `off_velocity14` / `velocity14()`), with the
  register cleared per note message and the `9n key 0` form staying a
  plain Note Off.
- **Sound-controller classifier** — `ControlChangeEvent::
  sound_controller()` types CC 70–79 per RP-021 (*Sound Controller
  Defaults (Revised)*): Sound Variation / Timbre / Release / Attack /
  Brightness plus the RP-021-named Decay Time / Vibrato Rate / Depth /
  Delay, with `level()` / `controller()` / `ordinal()`;
  `sound_controllers()` is the merged iterator.
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
  `UniversalSysExEvent::device_control()` decodes the Real-Time **Device
  Control** family (`F0 7F <dev> 04 nn …`, Table 4 Sub-ID #1 `0x04`) into
  a typed `DeviceControl`: Master Volume / Master Balance (14-bit
  lsb-first value, `0x2000` = balance centre), Master Fine Tuning
  (`fine_tuning_cents()` = `100/8192 × (value14 − 8192)` per CA-025),
  Master Coarse Tuning (`coarse_tuning_semitones()` = signed `msb − 64`,
  `0x40` = no change), and Global Parameter Control (CA-024 body bytes
  surfaced verbatim, trailing `F7` stripped); `None` on the wrong
  classification or a value pair truncated mid-stream.
  `device_controls()` is the stably-merged absolute-tick iterator over
  that subset (one `DeviceControlEvent` per packet, track 0 before track
  1 at the same tick).
- **Transport / device control SysEx** — `mmc_command()` /
  `mmc_response()` decode the Real-Time **MIDI Machine Control** families
  (`F0 7F <dev> 06|07 …`, RP-013) into typed `MmcCommandType`
  (`Stop`/`Play`/`Locate`/`Write`/`Resume`/…) and `MmcInformationField`
  opcodes plus verbatim operands; `MmcCommand::locate_target()` decodes
  the LOCATE [TARGET] body into an `MmcStandardTime` (`frame_rate()` +
  `hours_count()`). `show_control()` decodes **MIDI Show Control**
  (`F0 7F <dev> 02 <fmt> <cmd> …`, RP-002-014) into a `ShowControlFormat`
  / `ShowControlCommand` (`Go`/`Stop`/`Resume`/…). `identity_reply()`
  decodes the Non-RT **Identity Reply** (`F0 7E <dev> 06 02 mm ff ff dd
  dd ss ss ss ss F7`) into manufacturer (single / three-byte extended),
  14-bit family + member codes, and the four revision bytes.
  `general_midi_system()` maps GM System On/Off (`F0 7E <dev> 09 0n F7`,
  RP-003 / GM2) to `GeneralMidiSystem`. `sample_dump_header()` /
  `sample_dump_request()` decode the Sample Dump Standard header (21-bit
  LSB-first fields + `LoopType`) and request; `sample_dump_extension()`
  adds the five CA-019 Sample Dump Extensions (Extended Dump Header with
  28-bit fixed-point Hz rate + 35-bit word counts + the ten-value
  `ExtendedLoopType`, Extended Loop Point transmission / request, Sample
  Name transmission / request). `controller_destination()` decodes the
  CA-022 Controller Destination Setting (Channel / Poly Pressure or an
  allowed CC routed to the Pitch / Filter / Amplitude / LFO-depth
  parameter table); `key_based_instrument_control()` the CA-023
  per-key drum controller message (`0x78`/`0x79` redefined as Fine /
  Coarse Tuning, disallowed numbers reported);
  `file_reference()` the CA-018 URL-based sound-file message (Open /
  Select / Open-and-Select / Close with length validation, typed
  DLS/SF2 instrument-map + WAV select views, and the CA-028
  map-entire-file `bank_offset()` extension);
  `scalable_polyphony_mip()` the SP-MIDI **MIP** message (RP-034 §2.1:
  priority-ordered `{cc vv}` pairs with cumulative polyphony values,
  the §3.1.3/§3.3 validity rules enforced at decode, plus the §2.2
  Figure 1 `masked_channels()` algorithm); and
  `midi_visual_control()` the RP-050 MIDI Visual Control Data Set
  (Parameter Address Map naming via `MvcParameter`, `is_mvc_on()` /
  `is_mvc_off()`, checksum verdict reported not rejected). Each has a
  stably-merged absolute-tick iterator (`mmc_commands()`,
  `mmc_responses()`, `show_control_messages()`, `identity_replies()`,
  `general_midi_system_messages()`, `sample_dump_extensions()`,
  `controller_destinations()`, `key_based_instrument_controls()`,
  `file_references()`, `midi_visual_controls()`). CA-019's un-annotated
  `ss ss` Sample Number is read **LSB-first** per the parent Sample
  Dump Standard — the MSB-first byte order in CA-019's own worked
  examples is a documented erratum (`docs/audio/midi/midi-errata.md`
  E1) confirming this crate's reading.
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
- **`ump::message`** — typed `decode` + `encode` for the core Message
  Types: Utility (NOOP, JR Clock, JR Timestamp, Delta Clockstamp TPQ,
  Delta Clockstamp 20-bit — status nibble at bits 20..24 per Figure
  26 / Table 26); System Common / System Real Time (SPP LSB-first,
  `0xF0`/`0xF7` rejected since SysEx rides MT 0x3); MIDI 1.0 Channel
  Voice (all 7 opcodes, 2-byte messages zero-fill); and the full MIDI
  2.0 Channel Voice set — registered/assignable per-note controllers,
  registered/assignable + relative controllers, per-note pitch bend,
  16-bit-velocity Note On/Off with attribute type/data, 32-bit poly /
  channel pressure, per-note management D/S flags, Program Change with
  the Bank Valid flag, and 32-bit Pitch Bend. `UmpMessage::decode` +
  `encode` dispatch every defined MT (Data / Flex / Stream included);
  only Reserved Message Types surface as `Unhandled`.
- **`ump::stream`** — the UMP Stream vocabulary (MT 0xF, §7.1):
  Endpoint Discovery / Info / Device Identity / Name / Product
  Instance Id notifications, Stream Configuration Request +
  Notification, Function Block Discovery / Info / Name, and the Start
  / End of Clip markers. Multi-packet text runs honour the spec byte
  caps (98 / 42 ASCII / 91) with a `StreamTextAssembler` for
  reassembly.
- **`ump::data`** — System Exclusive (7-bit, MT 0x3, §7.7) and SysEx8
  (MT 0x5, §7.8, Stream IDs + the §7.8.1 abort form), with payload
  splitters / assemblers for both; Mixed Data Set Header + Payload
  chunks (§7.9); the §7.10 16-bit Manufacturer ID translation.
- **`ump::flex`** — Flex Data (MT 0xD, §7.5): Set Tempo (10 ns units),
  Set Time Signature, Set Metronome, Set Key Signature, Set Chord Name
  (full alteration surface), and the Status Bank 0x01/0x02 metadata +
  performance text families (12-byte chunks, ≤32 UMPs, melisma
  preserved) with Channel/Group addressing.
- **`ump::scaling`** — the spec Appendix D bit-scaling primitives:
  Min-Center-Max upscaling (smooth shift below center, bit-repeat above)
  and truncating downscaling, with 7/14 ⇄ 16/32 helpers. Verified
  against the §D.1.3 numerical examples and §D.1.2 center-value table;
  7⇄32 and 14⇄32 round-trips proven lossless.
- **Translation** — `Midi1ChannelVoice::to_midi2` / `Midi2ChannelVoice::
  to_midi1` implement the Default Translation Mode (§D.2/D.3): Note On
  velocity-0 ⇄ Note Off 0x8000, velocity floored to 1 on downscale,
  pitch-bend 14⇄32 with LSB-first packing, Program Change bank handling,
  CC 96/97 as plain Control Changes, and `None` for the messages with no
  counterpart (special CCs that belong to compound RPN/NRPN sequences;
  per-note / relative / per-note-management on the way down).
  `ump::translator` adds the stateful compound layer:
  `Midi1ToMidi2Translator` folds CC 98/99/100/101 + 6/38 into single
  Registered/Assignable Controller messages on the §D.3.3 triggers and
  Bank Select CC 0/32 into Program Change bank fields (§D.3.4);
  `midi2_to_midi1_messages` performs the reverse §D.2.3/§D.2.4
  expansions.

## MIDI Clip File (`clip`)

The `.midi2` SMF2 clip format (M2-116-U v1.0) — the MIDI 2.0
counterpart of the SMF Type 0 file. Reader + writer for the full
framing: `SMF2CLIP` File Header, leading Set Profile On SysEx (no
Delta Clockstamp), `DCS(0)` + DCTPQ, Clip Configuration Header, Start
/ End of Clip markers, per-message Delta Clockstamps with the §3.2.2
`DCS + NOOP` restart for gaps beyond the 20-bit field, and the §7.3
nothing-after-End rule. The writer is a fixed point of the reader over
every UMP Message Type (`tests/writer_fixed_point.rs`).
`Scheduler::from_clip()` plays a clip **natively** (see *Native MIDI
2.0 synthesis* below) — the registered decoder does this for
`SMF2CLIP` payloads and applies the Configuration Header's Set
Profile On messages first. `ClipFile::to_smf()` remains the Appendix-D
Default Translation into an SMF (MIDI 2.0 Channel Voice via the
compound expansions, Flex tempo / meter / key / text to SMF metas,
SysEx7 reassembly to `F0` events) and `ClipFile::from_smf()` the
Appendix-A concordance direction; the two are lossy exactly once,
then stable.

## Native MIDI 2.0 synthesis

The mixer consumes MIDI 2.0 Channel Voice messages (M2-104 §7.4) at
full resolution instead of downgrading them to 7 / 14 bits:

- **Resolution anchor.** Every refinement is anchored on the spec's
  own 7 ↔ 16 / 32-bit equivalence, the §D.1.3 Min-Center-Max upscale
  grid: a value on the grid (every MIDI 1.0 value's upscale) renders
  **bit-identically** to its MIDI 1.0 counterpart in both translation
  directions, the positions between two grid points are distinct and
  monotone, `0xFFFF` / `0xFFFF_FFFF` are exactly position 127. The
  MIDI 1.0 paths are untouched (bench `--corpus` hashes unchanged for
  MIDI 1.0 content); `tests/midi2_native.rs` and `tests/clip_native.rs`
  pin identity and difference on rendered PCM.
- **Pitch** — `Voice::set_pitch_bend_fine_cents` carries fractional
  cents; `Mixer::set_pitch_bend_32` (§7.4.11) keeps the 32-bit bend, a
  sweep inside one 14-bit step is strictly monotone. Per-Note Pitch
  Bend (§7.4.12) scales by the RPN #00/07 Sensitivity of Per-Note
  Pitch Bend (Q7.25, §7.4.13; default 2.0 HCU) and sums with the
  channel bend. **Pitch 7.9** (Note On Attribute Type 3, §7.4.15.3) and
  **Pitch 7.25** (Registered Per-Note Controller #3, §7.4.15.2, live
  through the note's life) set absolute pitches — the sample is picked
  at the integer part (`Mixer::midi2_note_sample_key`) and the
  fraction is an exact offset overriding MTS; channel / master tuning,
  bends and glides still apply relatively; 7.9 out-ranks 7.25 for that
  one note.
- **Velocity** — `Mixer::note_on_midi2` refines the 7-bit-built voice
  by the ratio of the voice's own velocity curve (`Voice::velocity_gain`,
  square law by default) at the 16-bit value's continuous position
  (`midi2_velocity_position`), continuous across steps; velocity 0 is
  a Note On at the lowest velocity (§7.4.2), never a Note Off; CC 88
  is ignored in the 2.0 Protocol (§7.4.6).
- **32-bit controllers** — `set_control_change_32` keeps CC 1 / 7 /
  10 / 11 / 91 / 93 at full resolution through the same response
  curves (GM2 square law, RP-036 pan with `0x8000_0000` the true
  centre, fractional modulation depth via
  `Voice::set_mod_depth_fine_cents`); switch / table controllers take
  the §D.1.4 downscale; the §7.4.6.1 special formats (CC 84 source
  note, CC 126 channel count in the top 7 bits) and the §7.4.6 ignore
  list (CC 0 / 32 / 6 / 38 / 98–101 / 88) are honoured.
  `set_channel_pressure_32` / `set_poly_pressure_32` likewise.
- **Registered / Assignable Controllers** (§7.4.7 / §7.4.8) —
  `set_registered_controller` implements the unified RPN form for Bank
  0 #00–#02 / #05 / #06 with the Figure 57–61 field layouts and #00/07;
  `registered_controller` reads the 32-bit layout back;
  `set_relative_registered_controller` applies the two's-complement
  relative form (saturating). Assignable Controllers are recorded per
  channel with the relative form applied.
- **Per-Note Controllers** (§7.4.4, Appendix A) — per-`(channel, note)`
  state (`PerNoteState`, shared and persistent per Appendix C.1, left
  alone by Reset All Controllers per Appendix B.2): #1 Modulation
  (summed with CC 1), #3 Pitch 7.25, #7 / #11 per-note square-law
  gains, #10 per-note pan, #71–#78 Sound Controllers captured at
  note-on with #74 Brightness live, #91 / #93 per-note sends; other
  defined numbers are recorded, Reserved numbers ignored; Assignable
  Per-Note Controllers are recorded. **Per-Note Management** (§7.4.5)
  implements D (detach: sounding voices keep their values) and S
  (reset), D-then-S.
- **Program Change with Bank Valid** (§7.4.9) latches the GM2 bank pair
  then the program. Groups fold onto the single 16-channel Function
  Block; JR Timestamps are accepted and rendered at their Delta
  Clockstamp position.
- **MIDI-CI Profiles** (M2-101 §7.8 / §7.9, M2-102 §2.3 / §2.6) —
  `Mixer::set_profile_on` / `set_profile_off` implement the Device-ID
  addressing (a Channel as the Manager of a Multi-Channel Profile with
  the version-2 channel span, the Group `0x7E`, the Function Block
  `0x7F`); the scheduler routes Set Profile On / Off SysEx into it and
  `profile_enabled` / `enabled_profiles` expose the result. No Standard
  Defined Profile specification is staged, so enabling changes no
  sound parameter by itself.

## MIDI-CI (`ci`)

Typed MIDI Capability Inquiry SysEx surface (M2-101-UM v1.2.1, with
the M2-102 Profile ID rules and M2-103 Property Exchange payloads
carried as data). Parses and emits the bracket-free payload form that
rides in a UMP SysEx7 run or an SMF `F0` event: the §5.2.1 envelope
(Device ID, Sub-ID#2, Version/Format, LSB-first 28-bit MUIDs),
Management (Discovery + reply, Endpoint inquiry/reply, Invalidate
MUID, ACK, v1/v2 NAK), Profile Configuration (inquiry/reply lists,
Set On/Off, Enabled / Disabled / Added / Removed reports, Details,
Profile Specific Data), the Property Exchange base messages
(capabilities + the chunked Get/Set/Subscribe/Notify family), and
Process Inquiry (capabilities + MIDI Message Report). Wire surface
only — no session state machine.

## Instruments

- `instruments::sf2` — full SoundFont 2 RIFF reader + voice generator.
  Resolves preset → instrument → zone → sample with the §7.3 / §7.7
  **global zones** supplying defaults under the §9.4 precedence;
  honours key/vel ranges, the sample / tune / root-key generators,
  `scaleTuning`, volume + modulation DAHDSR envelopes with the
  `keynumTo…Hold/Decay` key tracking, modEnv→pitch / modEnv→filter
  routing, the **Vibrato and Modulation LFOs** (§9.1.6: delay + triangle,
  routed to pitch / filter / volume), the initial low-pass biquad,
  `pan` and the reverb / chorus **effects sends** (gens 15–17, summed
  with the channel CC 91 / 93 sends), and exclusive-class drum cuts.
  `pmod` / `imod` **modulators** are parsed and evaluated at note-on for
  the sources known then (No Controller, velocity, key number; §8.2
  direction / polarity, linear + switch types, amount source, §8.3
  absolute-value transform) with the §9.5.1 supersede / add precedence;
  the §8.4.2 velocity→cutoff default modulator is implicit and
  supersedable, and the §8.4.3 / §8.4.4 defaults put Channel Pressure
  and CC 1 on the Vibrato LFO (the GM2 Sound Controllers CC 76–78 scale
  its rate / depth / delay). Not evaluated: modulators sourced from
  live channel state (MIDI CC palette, pressure, pitch wheel), linked
  chains, and the concave / convex source types (the staged §8.2.4
  formula is not usable as printed; the §8.4.1 velocity→attenuation
  default keeps a square-law approximation). 24-bit `sm24` samples and
  native stereo zones supported. Bounds-checked with spec ceilings on
  sample / record counts.
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
  Parameter Control. CC 10 panning follows the **RP-036 Default Pan
  Formula** exactly (`cos`/`sin` of `π/2 · max(0, pan − 1)/126`): 64 is
  a true equal-power centre and 0/1 both pan hard left. **CC 7 / CC 11
  / Master Volume follow the GM2 square-law response curve** (RP-024
  §3.3.4/§3.3.6/§4.1: amplitude ∝ value², `gain[dB] = 40·log10(v/127)`,
  volume and expression composing additively in dB). The **CC 88
  High-Resolution Velocity Prefix** (CA-031) refines the next note-on's
  gain by its 14-bit velocity ratio (prefix-free scores render
  bit-identically; a Note Off expires a pending prefix).
- `mixer` **GM2 channel roles + bank select** (RP-024 §2.4/§3.3.1) —
  CC 0/32 Bank Select is *pending* until the next Program Change, which
  latches the pair and switches the channel role: MSB `78H` → Rhythm
  Channel, `79H` → Melody (required on ch 10/11 per RP-035 §2.1,
  honoured on all channels per the [optional] clause). Channel 10 boots
  as Rhythm in `78H/00H`. Every drum exemption (tuning RPNs, master
  tuning, MTS, portamento, mod wheel, sound controllers) follows the
  live role, not the fixed index. `Instrument::make_voice_banked`
  carries the latched bank to voice lookup; the SF2 backend maps
  `78H/xxH` → SoundFont percussion bank 128 and `79H/vv` → bank `vv`,
  with the GM2 §2.6 undefined-program fallbacks (kit → `(128,0)`,
  variation → the GM1 set).
- `mixer` **RP-021 Sound Controllers CC 71–78** (GM2 §3.3.11–§3.3.18) —
  relative parameters centred at 64, captured into each new voice
  (`Voice::apply_sound_controls`): Filter Resonance (±3 cb/step),
  Release / Attack / Decay Time and Vibrato Rate / Depth / Delay
  (×`2^((v−64)/32)`), and CC 74 Brightness routed **live** as a
  ±50 cents/step filter-cutoff shift on both the SF2 and SFZ/DLS voice
  types (biquad instantiated on demand when an open voice is darkened).
  Rhythm Channels record but don't respond, per the GM2 recommendation.
- `mixer` **CA-022 Controller Destination Setting** (GM2 §4.6/§3.7) —
  the Universal Real-Time `09 01/03` messages route Channel Pressure or
  one Control Change (01–1F/40–5F) per channel into the GM2 controlled-
  parameter table: Pitch (±24 semitones), Filter Cutoff (150
  cents/step), Amplitude (0–(127/64)·100 %), and LFO Pitch / Filter /
  Amplitude Depth (0–600 / 0–2400 cents / 0–100 % tremolo), summing
  with the timbre's default pressure response. Only the last message
  per channel is active; one CC routing at a time. The SFZ/DLS voice
  implements the full destination set (including a 5 Hz default LFO
  when the region has no preset vibrato — which also makes **CC 1
  modulation audible** on those voices); the SF2 voice implements the
  cutoff destination.
- `mixer` **CA-023 Key-Based Instrument Controllers** (GM2 §4.8) — the
  Universal Real-Time `0A 01` message edits individual percussion
  sounds on Rhythm Channels: Note Volume (relative, `40H` = 100 %),
  Pan (absolute, with the channel CC 10 *offsetting* the per-key
  position per §3.3.5), Reverb / Chorus Send (absolute, per-voice
  override of the channel sends), and the CA-023-redefined `78H/79H`
  Fine / Coarse Tuning (the sanctioned per-key drum tuning). Edits
  apply at the key's next note-on; a Program Change on a Rhythm
  Channel adopts the new set's presets (clears the table); Melody
  Channels don't respond.
- `instruments::percussion` + `mixer` **GM2 Percussion Sound Set**
  (RP-024 Appendix B / §2.8.1) — `DrumSet::from_program` resolves the
  Bank 78H/00H Program Change to one of the nine drum sets (STANDARD,
  ROOM, POWER, ELECTRONIC, ANALOG, JAZZ, BRUSH, ORCHESTRA, SFX) with the
  §2.5 undefined-program → STANDARD fallback; each key resolves to its
  Appendix-B instrument name, recommended preset pan, and EXC group,
  following the `@` inheritance to STANDARD (SFX is self-contained).
  The mixer wires three §2.8.1 behaviours on Rhythm Channels: the
  **mutually-exclusive Note choke** (a Note On for a member of an EXC
  group — hi-hats, whistles, guiros, cuicas, triangles, surdos,
  scratches — mutes the sounding member of the same group), **Note Off
  ignored** so drum one-shots ring out (except ORCHESTRA Note 88 and
  SFX Notes 47–84), and the **per-key preset pan** default (offset by
  CC 10 per §3.3.5). `Mixer::active_drum_set` / `drum_key_name` expose
  the resolved set + instrument name.
- `mixer` **SP-MIDI Channel Masking** (RP-034/RP-035) — the mixer
  accepts a device Polyphony Level (`set_sp_midi_polyphony`, the §3.3
  "SPn" budget) and evaluates each received MIP message through the
  §2.2 Figure 1 algorithm: masked channels have their note-ons
  filtered and their sounding voices cut; unlisted channels mask;
  invalid messages (§3.1.3) leave the previous table in force; GM/GM2
  System On restores the §3.1.1 initialized state. Unconfigured, MIP
  messages are recorded but mask nothing.
- `mixer` **continuous controllers + pedals** — **CC 11 Expression**
  multiplies Channel Volume at mix time (Expression is a percentage of
  Volume per the MIDI 1.0 Control Change table; default 127 = transparent).
  **CC 121 Reset All Controllers** follows RP-015 exactly — Expression →
  127, Modulation → 0, Pedals → 0, RPN/NRPN selector → null, Pitch Bend →
  centre, Channel + Poly Pressure → 0, while Volume / Pan / Program / Bank
  / Effects / Sound-Controllers and the RPN parameter *values* are
  preserved. The **CC 64 Sustain**, **CC 66 Sostenuto**, and **CC 67 Soft**
  pedals are modelled distinctly: Sostenuto captures only notes sounding at
  press time and holds independently of Sustain (a deferred note releases
  only when both pedals lift); the Soft pedal (*una corda*) attenuates
  notes struck while it is down, captured per-voice so sounding notes are
  unaffected. **CC 5 / 65 / 84 Portamento** glides a new note's pitch from
  the previously-played key (CC 65 on) or an explicit CC 84 source over a
  CC 5-controlled span, advanced per render block and summed with the live
  bend. **CC 120 All Sound Off** (immediate hard cut, ignoring pedals) and
  **CC 123-125 All Notes Off / Omni Off / Omni On** (normal release,
  honouring pedals; the mode itself never changes — GM2 doesn't support
  Omni) are split per the Channel Mode table; **CC 126/127 Mono / Poly
  Mode On** implement GM2 §2.5/§3.5.6/§3.5.7 **Mode 4**: a Melody
  Channel switches to one-note-at-a-time on CC 126 with M = 1 (any
  other M is invalid and the message is ignored; Rhythm Channels keep
  polyphony), each note-on releasing the previous note, and CC 127 /
  GM reset restore Mode 3; **CC 122 Local Control** is a recognised
  no-op.
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
  events into the mixer at the right sample. Routes the channel-voice
  controllers (Bank Select CC 0/32, Expression, Portamento CC 5/65/84,
  the Sound Controllers CC 71–78, the Sustain / Sostenuto / Soft pedals,
  Reset All Controllers, and the All Sound Off / All Notes Off
  channel-mode family) plus the Universal SysEx surface (GM 1/2 System
  On/Off, Master Volume/Balance/Tuning, MIDI Tuning Standard, Data
  Inc/Dec, GM2 GPC, CA-022 Controller Destination Setting, and the
  SP-MIDI MIP message).
- `tuning` — MIDI Tuning Standard (MTS) microtuning state + Universal
  SysEx decoders (key-based + scale/octave tables, signed cents added
  to equal temperament; drum channel exempt).
- `paths` — per-OS SoundFont/SFZ/DLS search paths +
  `OXIDEAV_SOUNDFONT_PATH` override.
- `downloader` — stub naming a planned default bank; currently returns
  `Error::Unsupported`.

The decoder factory registers under codec id `"midi"`: `send_packet`
parses the SMF — or a `.midi2` MIDI Clip File, scheduled natively at
full MIDI 2.0 resolution with its Configuration Header profiles
applied — and primes the scheduler; `receive_frame` returns
interleaved S16 stereo PCM (1024 samples/channel at 44.1 kHz) until
the event stream and voice pool run dry. Without an on-disk bank the
registry-built decoder uses the pure-tone fallback; for SoundFont 2
playback build a `MidiDecoder` directly with an `Sf2Instrument`.

## Fuzzing

A `cargo-fuzz` harness covers every attacker-facing parser:

```
cargo +nightly fuzz run smf    # smf::parse + iterators + both writers as fixed points
cargo +nightly fuzz run clip   # clip::parse + writer fixed point + to_smf + native render
cargo +nightly fuzz run ump    # UMP word stream: decode → encode → decode fixed point
cargo +nightly fuzz run sf2    # instruments::sf2::Sf2Bank::parse
cargo +nightly fuzz run dls    # instruments::dls::DlsBank::parse
cargo +nightly fuzz run sfz    # instruments::sfz::parse_str
```

Each target asserts arbitrary bytes return a `Result` with no panic /
OOM / overflow / OOB; `smf` and `clip` additionally require the writers
to be fixed points of the parsers (the SMF writer may refuse only the
documented reader-tolerated shapes: a header `ntrks` disagreeing with
the `MTrk` chunks present, a missing / misplaced End of Track, an
`Unknown { 0x2F }` meta), and `clip` renders a bounded slice through
the native MIDI 2.0 path. Curated seed corpora (including the
fuzz-found regressions) live under `fuzz/corpus/<target>/`; the `Fuzz`
workflow builds every target on nightly and smoke-runs each for 30 s.

## Profiling

`benches/synth_render.rs` (`harness = false`) is a repeatable SMF→PCM
wall-clock harness over a dense 8-channel / 32-voice score through an
in-memory SF2 bank, plus a `--corpus` PCM-hash mode and a `--spin
SECS` sampling-profiler loop.

The Reverb + Chorus effects bus carries a latched activation gate: while
every channel's reverb / chorus send (CC 91 / CC 93) is zero, the bus
delay lines are provably all-zero, so the mixer skips the per-sample
send accumulation **and** the whole Schroeder reverb + modulated-chorus
DSP — their wet return would be exactly `0.0`. The first non-zero send
latches the bus on (and it stays on so a tail still rings out after the
sends drop). On the dry dense-score profile this removes the comb /
allpass / chorus inner loops entirely, cutting the SMF→PCM wall clock by
~24 % (87.6 → 66.3 ms here) with bit-identical PCM (corpus hashes
unchanged).

`Sf2Voice::render` additionally hoists the mod-env→pitch / filter
decision out of the per-sample loop: those routings are fixed for a
whole render call, so the loop splits once into a slow path (filter
and/or mod-env pitch active) and a bare sample-playback fast path that
drops the per-sample mod-env evaluation, the filter-coefficient drift
check, and the biquad `filter_step`. On the dry SF2 dense score this
takes the wall clock to ~55 ms (≈37 % under the round-378 baseline),
again with every `--corpus` PCM hash unchanged.

The SF2 §8.4.2 default modulator (velocity → filter cutoff) puts most
notes on the filtered path, so the same dense score now renders in
~122 ms with the biquad engaged. Three byte-identical hoists bring it to
~90 ms (−27 %): the modulation envelope is evaluated only when a routing
depth consumes it, a filter whose cutoff nothing modulates runs its
coefficient drift check once per 256-frame block, and a static-filter
inner loop keeps the biquad coefficients and delay line in locals (same
expression order) — every `--corpus` PCM hash unchanged.

## License

MIT — see `LICENSE`.
