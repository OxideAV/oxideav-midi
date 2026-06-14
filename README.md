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
  Round 122 adds `SmfFile::time_signatures()` — a sorted iteration
  helper that returns every `FF 58 04 nn dd cc bb` change as a
  `TimeSignatureChange { tick, track, numerator, denominator_pow2,
  clocks_per_click, notated_32nd_per_quarter }` with the absolute tick
  (cumulative delta-sum) on the parent track. Per-track sequences are
  stably merged so two changes at the same tick keep `(track,
  in-track)` order — the same convention the scheduler uses.
  `TimeSignatureChange::denominator()` returns the decoded `1 << dd`,
  saturated at `u32::MAX` so pathological `dd >= 32` can't overflow.
  Round 125 adds the analogous `SmfFile::tempo_map()` — every
  `FF 51 03 tt tt tt` Set Tempo as a
  `TempoChange { tick, track, microseconds_per_quarter_note, bpm }`
  with the absolute tick on the parent track. Per-track sequences
  are stably merged by tick (track 0 before track 1 at the same
  tick). `bpm` is pre-computed as `60_000_000.0 / µs_per_qn`;
  `µs_per_qn == 0` maps to `f64::INFINITY` rather than
  divide-by-zero. Players that need an initial tempo before any
  explicit Set Tempo should assume 500 000 µs/qn = 120 BPM per
  convention.
  Round 128 adds `SmfFile::key_signatures()` — every
  `FF 59 02 sf mi` change as a
  `KeySignatureChange { tick, track, sharps_flats, mode }` with the
  absolute tick on the parent track, stably merged across tracks
  (track 0 before track 1 at the same tick).
  `KeySignatureChange::tonic_name()` / `name()` resolve the signed
  `-7..=+7` accidental count and `0`/`1` mode bit to a textbook
  circle-of-fifths label (`"C major"`, `"A minor"`, `"F# major"`,
  `"Bb minor"`, …); out-of-range `sf` or unknown `mode` returns
  `None` rather than fabricating a label.
  Round 176 adds `SmfFile::markers()` — every `FF 06 len text`
  marker meta event as a `MarkerEvent { tick, track, text }` with
  the absolute tick on the parent track, stably merged across
  tracks (track 0 before track 1 at the same tick). Only `FF 06`
  is selected — neighbouring text-kind metas (`FF 03` track name,
  `FF 05` lyric, etc.) are filtered out so the helper matches the
  DAW convention of one section-label list per song.
  `MarkerEvent::text_bytes()` returns the raw payload (the SMF spec
  leaves the encoding unspecified); `text_lossy()` returns a
  UTF-8-decoded `Cow<str>` with `U+FFFD` substitutes for invalid
  sequences so callers don't have to encode-detect themselves.
  Round 182 adds the karaoke companion `SmfFile::lyrics()` —
  every `FF 05 len text` lyric meta event as a
  `LyricEvent { tick, track, text }` with the absolute tick on
  the parent track, stably merged across tracks (track 0 before
  track 1 at the same tick) under the same merge rule as
  `markers()` / `tempo_map()` / `time_signatures()` /
  `key_signatures()`. Only `FF 05` is selected so the `.kar`
  syllable stream comes out as a clean time-ordered list,
  independent of any surrounding `FF 03` track name / `FF 06`
  marker / `FF 07` cue point events. Same accessor shape as the
  marker helper: `LyricEvent::text_bytes()` for the raw payload
  (encoding is spec-unspecified — historically Latin-1, modern
  files emit UTF-8), `text_lossy()` for a `Cow<str>` UTF-8 decode
  with `U+FFFD` substitutes for invalid sequences.
  Round 186 adds the film-score sync companion
  `SmfFile::cue_points()` — every `FF 07 len text` cue-point meta
  event as a `CueEvent { tick, track, text }` with the absolute
  tick on the parent track, stably merged across tracks (track 0
  before track 1 at the same tick) under the same merge rule as
  `markers()` / `lyrics()` / `tempo_map()` / `time_signatures()` /
  `key_signatures()`. Only `FF 07` is selected so callers driving
  external synchronisation (scene change, SFX trigger, video cue)
  get a clean time-ordered list independent of the surrounding
  `FF 03` track name / `FF 05` lyric / `FF 06` marker streams.
  Same accessor shape as the marker and lyric helpers:
  `CueEvent::text_bytes()` for the raw payload (encoding is
  spec-unspecified), `text_lossy()` for a `Cow<str>` UTF-8 decode
  with `U+FFFD` substitutes for invalid sequences.
  Round 192 adds the DAW-track-list companion
  `SmfFile::track_names()` — every `FF 03 len text` track-name meta
  event as a `TrackNameEvent { tick, track, text }` with the
  absolute tick on the parent track, stably merged across tracks
  (track 0 before track 1 at the same tick) under the same merge
  rule as `cue_points()` / `markers()` / `lyrics()` / `tempo_map()`
  / `time_signatures()` / `key_signatures()`. Only `FF 03` is
  selected so callers populating the DAW track list get a clean
  per-track label stream independent of the surrounding `FF 01`
  general text / `FF 02` copyright / `FF 04` instrument name /
  `FF 05` lyric / `FF 06` marker / `FF 07` cue point events.
  Authoring tools conventionally emit at most one `FF 03` per
  track at tick 0 (on a format-0 file the single track's `FF 03`
  is read as the sequence title); the helper surfaces every
  occurrence so callers that only want the first name per track
  can collect into a `HashMap<usize, TrackNameEvent>` keyed on
  `TrackNameEvent::track`. Same accessor shape as the cue / marker
  / lyric helpers: `TrackNameEvent::text_bytes()` for the raw
  payload (encoding is spec-unspecified — historically Latin-1,
  modern files emit UTF-8), `text_lossy()` for a `Cow<str>` UTF-8
  decode with `U+FFFD` substitutes for invalid sequences.
  Round 196 adds the voice/patch companion
  `SmfFile::instrument_names()` — every `FF 04 len text`
  instrument-name meta event as an `InstrumentNameEvent { tick,
  track, text }`, distinct from the `FF 03` track-list label so a
  single track may legally carry both. Pinned to the absolute tick
  on the parent track, stably merged across tracks (track 0 before
  track 1 at the same tick) under the same merge rule as the other
  seven text-meta helpers; only `FF 04` is selected so callers
  populating per-track patch / preset metadata get a clean
  per-track instrument stream independent of the surrounding
  `FF 01` general text / `FF 02` copyright / `FF 03` track name /
  `FF 05` lyric / `FF 06` marker / `FF 07` cue point events. Lifts
  the family from 7 to 8 helpers
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names}`). Same accessor
  shape as the others: `InstrumentNameEvent::text_bytes()` for the
  raw payload, `text_lossy()` for a `Cow<str>` UTF-8 decode with
  `U+FFFD` substitutes for invalid sequences.
  Round 202 closes the text-meta family with two more helpers:
  `SmfFile::texts()` (`FF 01` general / free-form text) and
  `SmfFile::copyrights()` (`FF 02` copyright notice), each surfacing
  the matching event kind as a `TextEvent` / `CopyrightEvent
  { tick, track, text }` value pinned to the absolute tick on the
  parent track and stably merged across tracks under the same
  track-0-before-track-1-at-the-same-tick rule as the prior six
  text-meta helpers and the scheduler. `FF 01` is the catch-all
  annotation kind (production notes, "do not edit", version
  stamps); `FF 02` declares the sequence's copyright notice (the
  SMF specification recommends placing it on the first track at
  tick 0 but the helper surfaces every occurrence in time order).
  Same accessor shape as the rest of the family:
  `TextEvent::text_bytes()` / `CopyrightEvent::text_bytes()` for
  the raw payload (encoding is spec-unspecified), `text_lossy()`
  for a `Cow<str>` UTF-8 decode with `U+FFFD` substitutes for
  invalid sequences. Lifts the family from 8 to **10** helpers
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights}`
  ), covering every `FF 01..=07` text-flavour meta event the spec
  defines.
  Round 208 adds the SMPTE wall-clock companion
  `SmfFile::smpte_offsets()` — every `FF 54 05 hr mn se fr ff`
  SMPTE Offset meta event as an `SmpteOffsetEvent { tick, track,
  hours_raw, minutes, seconds, frames, subframes }` with the
  absolute tick on the parent track, stably merged across tracks
  (track 0 before track 1 at the same tick) under the same merge
  rule as the ten text-meta / rhythmic helpers and the scheduler.
  The `hr` byte packs the SMPTE frame rate in bits 5-6 per the
  MIDI Time Code spec (RP-004/008 §"HOURS COUNT"): `00=24fps`,
  `01=25fps`, `10=30fps drop-frame`, `11=30fps non-drop`. A new
  `FrameRate` enum (`Fps24` / `Fps25` / `Fps30DropFrame` /
  `Fps30NonDrop`) plus `FrameRate::from_hours_byte(hr)` decodes
  the packed bits; `SmpteOffsetEvent::frame_rate()` /
  `hours_count()` (bits 0-4) / `seconds_total()` (wall-clock
  seconds: `h*3600 + m*60 + s + (frames + subframes/100)/fps`)
  surface the SMPTE-cueing semantics without forcing callers to
  re-mask the byte themselves. Lifts the SMF meta-event iterator
  family from 10 to **11** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets}`), covering every rhythmic + text + cueing meta
  event the spec defines a per-event "when it fires" lens for.
  Round 219 closes the meta-event helper family with the
  sequencer-private companion `SmfFile::sequencer_specifics()` —
  every `FF 7F len data` Sequencer-Specific Meta-Event as a
  `SequencerSpecificEvent { tick, track, data }` with the absolute
  tick on the parent track, stably merged across tracks (track 0
  before track 1 at the same tick) under the same merge rule as
  every existing iteration helper and the scheduler. `FF 7F` is the
  SMF escape hatch for sequencer-private or manufacturer-private
  payloads carried inline with the music data: the spec leaves the
  payload bytes opaque (by SysEx convention `data[0]` — or
  `data[0..=2]` when `data[0] == 0x00` — holds the manufacturer
  ID), and the parser preserves them verbatim so a caller routing
  by ID can decode while a generic player can ignore per the spec's
  "unknown meta events SHOULD be ignored" rule. Only `FF 7F` is
  selected — channel-message `F0` / `F7` SysEx events travel through
  the scheduler's SysEx pump rather than the meta-event family, so
  a DAW round-trip workflow (load → save) can preserve every private
  blob without re-reading the wire-event stream. Empty payloads
  (`FF 7F 00`) are surfaced as `data.is_empty()` rather than
  filtered out — the spec permits a zero-length blob.
  `SequencerSpecificEvent::data_bytes()` borrows the raw payload.
  Lifts the SMF meta-event iterator family from 11 to **12** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets,sequencer_specifics}`), covering every rhythmic +
  text + cueing + sequencer-private meta event the spec defines a
  per-event "when it fires" lens for.
  Round 224 adds the pattern-cueing companion
  `SmfFile::sequence_numbers()` — every `FF 00 02 ssss` Sequence
  Number Meta-Event as a `SequenceNumberEvent { tick, track, number
  }` with the absolute tick on the parent track, stably merged
  across tracks (track 0 before track 1 at the same tick) under the
  same merge rule as every existing iteration helper and the
  scheduler. `ssss` is decoded big-endian into a `u16`. The Standard
  MIDI File Specification 1.0 reserves the event for delta-time zero
  (the first event of a track) — on a format-2 file it labels each
  pattern so the sequence can be cued from a Song Select, and on a
  format-1 / format-0 file it labels the file as a whole — but the
  helper surfaces every occurrence rather than enforcing the
  placement rule so files that carry the label later in a track
  still round-trip. `SequenceNumberEvent::number()` returns the
  decoded identifier. Lifts the SMF meta-event iterator family from
  12 to **13** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets,sequencer_specifics,sequence_numbers}`).
  Round 230 adds the multi-port-routing companion
  `SmfFile::midi_ports()` — every `FF 21 01 pp` MIDI Port Meta-Event
  as a `MidiPortEvent { tick, track, port }` with the absolute tick
  on the parent track, stably merged across tracks (track 0 before
  track 1 at the same tick) under the same merge rule as every
  existing iteration helper and the scheduler. `pp` is the physical
  port byte (`0..=127`); the Standard MIDI File Specification 1.0
  leaves the mapping from port index to physical output up to the
  receiving application — typically `0` is the first output port,
  `1` the second, and so on. The pre-multi-port convention places
  one `FF 21` near the start of a track (delta zero, before the
  first channel-voice event) so a multi-port back-end can dispatch
  each track's channel stream through 16 × N channels, but the
  helper surfaces every occurrence rather than enforcing the
  placement rule so files that re-route mid-track still round-trip.
  `MidiPortEvent::port()` returns the decoded byte. Only `FF 21` is
  selected — the neighbouring `FF 20` channel-prefix hint stays on
  its own (different routing semantics: per-message channel
  override versus per-track physical port assignment) so the
  port-routing layer gets a clean time-ordered list independent of
  the surrounding meta streams. Lifts the SMF meta-event iterator
  family from 13 to **14** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets,sequencer_specifics,sequence_numbers,midi_ports}`).
  Round 240 adds the channel-binding companion
  `SmfFile::channel_prefixes()` — every `FF 20 01 cc` Channel Prefix
  Meta-Event as a `ChannelPrefixEvent { tick, track, channel }` with
  the absolute tick on the parent track, stably merged across tracks
  (track 0 before track 1 at the same tick) under the same merge
  rule as every existing iteration helper and the scheduler. `cc`
  carries the channel-binding hint for non-channel events that
  follow on the same track (text, lyric, marker, cue point, sysex
  — until another `FF 20`, the next channel-voice event, or end of
  track). The Standard MIDI File Specification 1.0 lists the event
  as part of the meta-event vocabulary; modern authoring tools
  prefer explicit per-track channel-voice streams plus `FF 21` port
  hints, but older files still emit it and a round-trip workflow
  must preserve it. `ChannelPrefixEvent::channel()` returns the
  spec-clamped channel as `Option<u8>` (`Some(c)` for `c < 16`,
  `None` for an out-of-spec high-nibble byte — a receiver should
  fall back to the most recent channel-voice channel rather than
  silently mask). The raw byte stays on the `channel` field so
  files with out-of-spec values still round-trip. Only `FF 20` is
  selected — the neighbouring `FF 21` port-hint sibling stays on
  its own (different routing semantics: per-track physical port
  assignment versus per-message channel override) so the
  channel-binding layer gets a clean time-ordered list independent
  of the surrounding meta streams. Lifts the SMF meta-event
  iterator family from 14 to **15** total
  (`SmfFile::{tempo_map,time_signatures,key_signatures,markers,
  lyrics,cue_points,track_names,instrument_names,texts,copyrights,
  smpte_offsets,sequencer_specifics,sequence_numbers,midi_ports,
  channel_prefixes}`).
  Round 243 surfaces the SysEx wire-event channel alongside the
  15-helper meta-event family with `SmfFile::sysex_events() ->
  Vec<SysExEvent>`: every System Exclusive event — both the `F0`
  start and the `F7` continuation / escape flavours — as a
  `SysExEvent { tick, track, is_escape, data }` with the absolute
  tick on the parent track, stably merged across tracks (track 0
  before track 1 at the same tick) under the same merge rule as
  every existing iteration helper and the scheduler. SMF spec
  §"System Exclusive Events" defines `F0 <varlen> <payload>` (a
  complete-or-starting message; the trailing `F7`, when present,
  is included as the final byte of `<payload>`) and `F7 <varlen>
  <payload>` (a continuation packet for a previously-started `F0`
  *or* an arbitrary escape sequence shipped verbatim to the wire).
  Both forms surface through the helper; `data` is reproduced
  verbatim so a writer can round-trip the output through
  `to_bytes()`. Convenience accessors: `SysExEvent::ends_with_eox()`
  flags a payload terminating with `0xF7`; `is_complete_message()`
  is sugar for `!is_escape && ends_with_eox()` (the common
  universal-SysEx case — GM-on `F0 7E 7F 09 01 F7`, Master Volume,
  Master Tuning); `manufacturer_id()` returns the leading byte of
  an `F0` payload (`0x7E` non-real-time / `0x7F` real-time for the
  universal vocabulary, otherwise a vendor ID — expanded `0x00`-
  prefixed three-byte IDs surface as `Some(0x00)` and the caller
  inspects `data[0..=2]`) and `None` for `F7` or an empty payload.
  The `FF 7F` `SequencerSpecific` channel — surfaced through
  `sequencer_specifics()` — is not selected here; the two carry
  different semantics (SysEx travels to the MIDI wire; `FF 7F` is
  file-private metadata that does not) so a file may carry both an
  `F0 7E 7F 09 01 F7` Universal Non-Real-Time GM-On packet and a
  private `FF 7F` plugin-state blob alongside it.
  Round 246 adds the Table 4 vocabulary decoder atop the same
  `SysExEvent` channel: `SysExEvent::universal_classification() ->
  Option<UniversalSysEx>` reads the `<realm> <device_id> <sub_id1>
  [<sub_id2>]` shape of a Universal SysEx `F0` packet against
  Table 4 of the MIDI 1.0 *Universal System Exclusive Messages*
  document and returns a `UniversalSysEx { realm, device_id,
  sub_id1 }` naming the parsed category. The `realm` field
  distinguishes Non-Real-Time (`0x7E`) versus Real-Time (`0x7F`);
  the `sub_id1` field is a `UniversalSubId1` enum that names every
  category in the table — Sample Dump Header / Data / Request, MIDI
  Time Code (Non-RT Setup + RT Quarter-Frame families), Sample
  Dump Extensions, General Information (Identity Request / Reply),
  File Dump, MIDI Tuning Standard, General MIDI 1 / GM Off / GM 2
  System On, Downloadable Sounds (DLS On / Off / Voice Allocation
  On / Off), File Reference Message, MIDI Visual Control, MIDI
  Capability Inquiry, MIDI Show Control, Notation Information (Bar
  Number, Time Signature Immediate / Delayed), Device Control
  (Master Volume / Balance / Fine Tuning / Coarse Tuning / Global
  Parameter Control), Real-Time MTC Cueing, Controller Destination
  Setting (Channel Pressure / Polyphonic Key Pressure / Control
  Change), Key-Based Instrument Control, Scalable Polyphony MIP,
  Mobile Phone Control — plus the five Non-Real-Time singletons
  (End of File, Wait, Cancel, NAK, ACK). The classifier is
  realm-aware: the same `(sub_id1, sub_id2)` byte pair names
  different messages in the two realms (e.g. `0x09 0x01` is
  `GeneralMidi1SystemOn` in Non-Real-Time and
  `ControllerDestinationChannelPressure` in Real-Time, per
  Table 4); the decoder resolves the realm-dependent semantics
  before matching. Sub-ID #1 / Sub-ID #2 values outside the
  Table 4 vocabulary surface through `UniversalSubId1::Other(raw)`
  / `UniversalSubId2::Other(raw)` so callers with deeper or more
  recent vocabulary can route the packet through a fallback path.
  Round 251 lifts the same Table-4 classifier to a file-wide
  iteration helper with `SmfFile::universal_sysex_events() ->
  Vec<UniversalSysExEvent>`: every `F0 7E …` / `F0 7F …` Universal
  SysEx packet on every track, pinned to its absolute tick, with
  the classification eagerly resolved into a
  `UniversalSysExEvent { tick, track, classification, data }`.
  Manufacturer-prefixed `F0` packets (Roland `0x41`, Yamaha `0x43`,
  any leading byte other than `0x7E` / `0x7F`) and `F7`
  continuation / escape packets are filtered out — callers reading
  those route through `SmfFile::sysex_events()` and
  `SysExEvent::manufacturer_id()` directly. `F0` packets truncated
  before Sub-ID #1 (payload shorter than 3 bytes) are also filtered,
  matching the underlying classifier's contract. Per-track sequences
  are stably merged by absolute tick — track 0's universal packets
  fire before track 1's at the same tick — the same convention used
  by `sysex_events()` and every meta-event helper. The verbatim
  payload bytes (leading `<realm>` byte through trailing `F7`, when
  present) are preserved on `UniversalSysExEvent::data` so callers
  reading Sub-ID #2-derived arguments (Master Volume's 14-bit value,
  MTC Full Message's `hr / mn / se / fr` quartet, MTS Single Note
  Tuning's note + tuning triple, …) don't have to re-walk
  `sysex_events()` alongside the typed list.
  Round 254 adds the channel-voice patch-change companion
  `SmfFile::program_changes() -> Vec<ProgramChangeEvent>`: every
  `Cn pp` Program Change on every track, pinned to the absolute
  tick at which it fires, with the channel index and program byte
  surfaced as a `ProgramChangeEvent { tick, track, channel,
  program }`. The status nibble's low four bits decode to the
  spec's `0..=15` channel index (channel "1" in human-facing
  tools is index `0`); the single data byte `pp` is the
  `0..=127` patch number. Resolution against a patch list
  (General MIDI 1 / 2, GS, XG, …) is left to the receiving
  application — the helper stays bank-agnostic and surfaces the
  raw program byte so callers driving an instrument-list view
  pick their own bank-select policy. Per-track sequences are
  stably merged by absolute tick — track 0's events fire before
  track 1's at the same tick — the same convention used by every
  meta-event and SysEx helper. Companion to the wire-state
  primitive `channel_snapshot_at` which folds the *last* Program
  Change per channel into `SmfChannelSnapshot::program` for seek
  initialisation: where the snapshot answers "what patch is this
  channel on at tick T?", `program_changes()` answers "give me
  every patch change in song order" — a DAW track-inspector view
  highlighting the bar each instrument enters reads the latter.
  Round 260 extends the channel-voice typed-iterator family with
  `SmfFile::control_changes() -> Vec<ControlChangeEvent>`: every
  `Bn cc vv` Control Change on every track, pinned to the absolute
  tick at which it fires, with the channel index plus the raw
  `cc` / `vv` data bytes surfaced as a `ControlChangeEvent { tick,
  track, channel, controller, value }`. The status nibble's low
  four bits decode to the spec's `0..=15` channel index; the first
  data byte `cc` is the controller number selected from the MIDI
  1.0 *Control Change Messages — Data Bytes* table (Bank Select
  MSB / LSB `0` / `32`, Modulation `1`, Volume `7`, Pan `10`,
  Expression `11`, Data Entry `6` / `38`, RPN MSB / LSB `100` /
  `101`, NRPN MSB / LSB `98` / `99`, Sustain `64`, …); the second
  data byte `vv` is the raw value. The channel-mode family
  (`controller` in `120..=127`: All Sound Off / Reset All
  Controllers / Local Control / All Notes Off / Omni Off / Omni On
  / Mono Mode On / Poly Mode On) is *not* diverted to a separate
  surface — those events ride the same `Bn` lane, and the
  `ControlChangeEvent::is_channel_mode()` predicate gives callers a
  one-line route into reset detection without re-checking the
  controller range. Resolution against a controller vocabulary
  (the 14-bit MSB / LSB pairing for controllers `0..=31` plus
  `32..=63`, the on-off threshold `value >= 64` for switch
  controllers, the Data Entry pump that drives RPN / NRPN parameter
  writes) is left to the receiving application — the helper stays
  controller-agnostic and surfaces the raw value byte so callers
  building a DAW lane editor / a CC-1 modulation-curve renderer / a
  CC-7 / CC-11 volume-expression curve view / an RPN-NRPN pair
  reassembler pick their own controller-vocabulary policy. Per-track
  sequences are stably merged by absolute tick — track 0's events
  fire before track 1's at the same tick — the same convention used
  by every meta-event and SysEx helper. Companion to the wire-state
  primitive `channel_snapshot_at` which folds the *last* value of
  the six snapshot-tracked controllers (Bank MSB / LSB, Modulation,
  Volume, Pan, Expression, Sustain) into the snapshot for seek
  initialisation: where the snapshot answers "what value is CC-7 /
  CC-10 / … at tick T?", `control_changes()` answers "give me every
  controller change in song order, including the controllers the
  snapshot doesn't track" — a DAW automation lane reads the latter.
  Round 267 extends the channel-voice typed-iterator family with
  `SmfFile::pitch_bends() -> Vec<PitchBendEvent>`: every `En lsb msb`
  Pitch Bend on every track, pinned to the absolute tick at which it
  fires, with the channel index plus the assembled 14-bit value
  surfaced as a `PitchBendEvent { tick, track, channel, value }`. The
  status nibble's low four bits decode to the spec's `0..=15` channel
  index; the two data bytes combine into `(msb << 7) | lsb`,
  `0..=0x3FFF`, with the no-bend centre at `0x2000` (the parser
  assembles the value at decode time). `PitchBendEvent::signed_value()`
  returns the displacement from centre as a signed `-8192..=8191`
  (`0x2000` → `0`, `0x0000` → `-8192`, `0x3FFF` → `8191`);
  `is_centre()` flags the no-bend position for bend-lane collapse or
  wheel-release detection. Resolving the 14-bit code to an actual pitch
  displacement requires the channel's Pitch Bend Sensitivity (RPN 0,
  default ±2 semitones), left to the receiving application — the helper
  stays sensitivity-agnostic and surfaces the raw code. Per-track
  sequences are stably merged by absolute tick — track 0's events fire
  before track 1's at the same tick — the same convention used by
  every meta-event, SysEx, and channel-voice helper. Companion to the
  wire-state primitive `channel_snapshot_at` which folds the *last*
  pitch bend per channel into `SmfChannelSnapshot::pitch_bend` for seek
  initialisation: where the snapshot answers "what is the wheel
  position on channel N at tick T?", `pitch_bends()` answers "give me
  every bend in song order" — a DAW bend-lane editor reads the latter.
  Round 275 extends the channel-voice typed-iterator family with
  `SmfFile::channel_pressures() -> Vec<ChannelPressureEvent>`: every
  `Dn pp` Channel Pressure (mono aftertouch) on every track, pinned to
  the absolute tick at which it fires, with the channel index plus the
  single pressure byte surfaced as a `ChannelPressureEvent { tick,
  track, channel, pressure }`. Per the MIDI 1.0 *Summary of MIDI
  Messages* Table 1, status nibble `1101` carries one data byte `pp`
  (`0..=127`) holding "the single greatest pressure value (of all the
  current depressed keys)" — distinct from polyphonic key pressure
  (`An`, per-key), which keeps its own surface. The status nibble's low
  four bits decode to the spec's `0..=15` channel index;
  `ChannelPressureEvent::channel()` / `pressure()` surface the decoded
  fields. The pressure value's musical effect (typically routed to
  volume, vibrato depth, or filter cutoff by the receiving instrument)
  is the receiver's concern, so the helper stays routing-agnostic and
  surfaces the raw `0..=127` byte. Per-track sequences are stably
  merged by absolute tick — track 0's events fire before track 1's at
  the same tick — the same convention used by every meta-event, SysEx,
  and channel-voice helper. A round-trip through `to_bytes()` / `parse`
  preserves the stream byte-for-byte.
  Round 292 adds the piano-roll companion `SmfFile::notes() ->
  Vec<Note>`: where every prior channel-voice helper surfaces one value
  per *wire* event, `notes()` pairs each Note On (`9n key vel`, `vel >
  0`) with the Note Off that releases it and returns one `Note {
  start_tick, end_tick, track, channel, key, velocity, off_velocity }`
  span per sounding note, ordered by onset — the primitive a DAW
  note-lane / piano-roll view consumes directly without re-deriving
  on/off pairing. The MIDI 1.0 *Summary of MIDI Messages* Table 1
  velocity-0 convention is honoured: a `9n key 0` is the running-status
  Note-Off form and closes the earliest open note of that pitch (FIFO)
  with `off_velocity == 0`, while an explicit `8n key off_vel` carries
  its release velocity through to `Note::off_velocity`. Matching walks
  the *globally* merged event stream sorted by `(absolute tick, track,
  in-track position)` — the same stable-merge convention every meta /
  SysEx / channel-voice helper and the scheduler use — so a Note Off on
  a different track from its Note On still pairs; overlapping notes of
  the same `(channel, key)` are matched FIFO, and an unmatched release
  or a hanging onset (no Note Off before EOF) is dropped from the span
  list. `Note::duration_ticks()` returns `end_tick - start_tick`
  (always non-negative, possibly zero when an off lands on the onset
  tick); `channel()` / `key()` / `velocity()` / `off_velocity()`
  surface the decoded fields.
  Round 301 completes the per-message channel-voice typed-iterator
  family with `SmfFile::poly_aftertouches() -> Vec<PolyAftertouchEvent>`:
  every `An kk pp` Polyphonic Key Pressure (per-key aftertouch) event as
  a `PolyAftertouchEvent { tick, track, channel, key, pressure }` with
  the absolute tick on the parent track, stably merged across tracks
  (track 0 before track 1 at the same tick) under the same merge rule as
  every existing iteration helper and the scheduler. `An` is the only
  channel-voice status without a dedicated extraction helper before this
  round; it is distinct from Channel Pressure (`Dn`, surfaced by
  `channel_pressures()` — the single greatest pressure over all depressed
  keys) in that `An` carries a per-key `kk` byte so per-voice aftertouch
  can be rebuilt. Only `An` is selected — Control Change (`Bn`), Program
  Change (`Cn`), pitch-bend (`En`), channel-pressure (`Dn`), and note
  (`8n` / `9n`) events stay on their own surfaces. Accessors `channel()`
  / `key()` / `pressure()` return the decoded `0..=15` / `0..=127` /
  `0..=127` fields. With this the eight channel-voice typed iterators
  (`program_changes`, `control_changes`, `pitch_bends`,
  `channel_pressures`, `poly_aftertouches`, `notes`, plus `sysex_events`
  / `universal_sysex_events`) cover every status nibble the wire defines.
  Round 307 adds the sounding-note seek lens
  `SmfFile::active_notes_at(tick) -> Vec<Note>` — the piano-roll
  companion to `notes()` and the note-level analogue of the
  channel-state `channel_snapshot_at` primitive. Where the snapshot
  answers "what controller / program / bend state does a channel carry
  at tick T?", `active_notes_at` answers "which keys are held down at
  tick T?" — exactly the set a DAW must re-trigger (or a renderer must
  prime into the voice pool) when seeking into the middle of a file
  rather than playing from the top. A note is sounding when
  `start_tick <= tick && end_tick > tick`, the half-open interval
  `[start_tick, end_tick)`: the onset tick is inclusive (the lens
  reflects state immediately after that tick's events fire, the same
  convention as `channel_snapshot_at`), the release tick is exclusive
  (the key has come up), and a zero-duration note
  (`start_tick == end_tick`) is sounding at no tick. The result reuses
  the matched spans from `notes()` verbatim — same velocity-0 Note-Off
  convention, FIFO re-strike matching, and drop of hanging onsets /
  unmatched releases — and preserves the `notes()` `(start_tick, track)`
  order so chord notes stay grouped and track 0 precedes track 1.
  Round 234 closes the SMF read-vs-write asymmetry: the parser
  has always materialised the full event vocabulary (`Channel`,
  `Sysex`, `Meta`), and round 234 adds the matching writer.
  `SmfFile::to_bytes(&self) -> Result<Vec<u8>>` serialises a
  parsed file back to a complete SMF byte stream (`MThd` + one
  `MTrk` per `Track`) suitable to hand back to `parse` for a
  structural round-trip; `Track::to_bytes_chunk(&self) ->
  Result<Vec<u8>>` emits one self-contained `MTrk` chunk so
  callers building a multi-track file from independent track
  sources can splice the chunks under a single `MThd` header.
  Output uses explicit status bytes throughout — the SMF
  specification permits but does not require running-status
  compression on the wire, and the explicit form keeps the writer
  deterministic regardless of internal track ordering. Every
  channel-voice variant, every concrete `MetaEvent` variant
  (including passthrough `Unknown` for forward compatibility),
  and both sysex forms (`F0` start / `F7` continuation-escape)
  round-trip byte-for-byte through `to_bytes` -> `parse`. The
  writer surfaces `Error::InvalidData` at encode time for any
  value that cannot fit the wire format: a new public constant
  `MAX_VLQ_VALUE = 0x0FFF_FFFF` matches the parser's 4-byte VLQ
  cap (delta-times, meta payload lengths, sysex payload lengths);
  data bytes must have the MIDI status bit clear (`<= 0x7F`);
  pitch-bend values must fit 14 bits; tempo values must fit 24
  bits; `KeySignature.mode` must be `0` or `1`; SMPTE
  `frames_per_second` must be one of `{24, 25, 29, 30}`;
  `TicksPerQuarter` must be `1..=0x7FFF`; `Text.kind` must be
  `0x01..=0x0F`; `header.ntrks` must match `tracks.len()`; each
  track must end with exactly one `MetaEvent::EndOfTrack` as its
  final event (the writer never auto-appends — the scheduler keys
  final-tempo / final-CC events off the EOT tick, so silently
  moving it would change semantics). The 17 new writer tests
  cover spec-VLQ encoding (10 worked examples from
  §"Variable-Length Quantities"), every meta variant, every
  channel-voice variant, both sysex forms, a multi-track Format-1
  file, the long-VLQ (4-byte) delta branch, and every validation
  rejection.
  Round 213 lifts the SMF-file accessor surface beyond the
  "iterate every meta event" lens with a **channel-state
  snapshot** primitive for seeking: `SmfChannelSnapshot { program,
  bank_msb, bank_lsb, volume, pan, expression, modulation,
  sustain, pitch_bend }` plus
  `SmfFile::channel_snapshot_at(channel, tick)` /
  `SmfFile::channel_snapshots_at(tick) -> [SmfChannelSnapshot; 16]`.
  Each method replays every channel-voice event from every track
  up to and including the requested tick — in scheduler order (a
  stable merge by `(tick, track, in-track)`, track 0 winning over
  track 1 at the same tick, the same convention as every existing
  iteration helper and `scheduler.rs` §"merged event list, sorted
  by absolute tick") — and folds each Program Change, Pitch Bend,
  and CC 0 / 1 / 7 / 10 / 11 / 32 / 64 into the snapshot. Fields
  that no matching event has touched stay at the SMF + GM 1
  (RP-003) recommended defaults: volume 100, pan 64, expression
  127, modulation 0, sustain off, pitch_bend `0x2000`; program /
  bank stay `None` so a seek-time initialiser can skip emitting
  CCs the file never wrote. Notes / aftertouch are ignored — they
  affect voice state, not channel state — so the snapshot is
  cheap to compute for any tick without enumerating sounding
  notes. Events at exactly `tick` are *included* (the snapshot
  reflects state immediately after that tick fires). Channels
  `>= 16` fall through to the default snapshot rather than panic.
  Sustain pedal decodes the CC 64 value with the spec threshold
  (`value >= 64` = on). The fold operation is also exposed
  publicly as `SmfChannelSnapshot::apply(&ChannelBody)` so callers
  running custom replay (e.g. against a custom track ordering)
  can reuse the same wire semantics. The bulk accessor pools
  events into one pass so initialising every channel at a seek
  target is single-pass rather than 16-pass — the natural primitive
  for a DAW seeking into the middle of a file.
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
  amplitude envelope from `ampeg_*` opcodes, runs a vibrato LFO from
  `lfo01_freq` / `lfo01_pitch` / `lfo01_delay`, and (round 95) drives
  a filter envelope from `fileg_*` opcodes through a `fil_type`-aware
  biquad — `lpf_1p` / `hpf_1p` / `lpf_2p` (default) / `hpf_2p` /
  `bpf_2p` / `brf_2p` per the SFZ-legacy `fil_type` table, with
  `cutoff=` (Hz → SF2 absolute cents) and `resonance=` (dB → centibels)
  feeding the round-91 RBJ biquad and `fileg_depth` driving the EG2 →
  cutoff routing. `#include` is rejected with `Error::Unsupported`;
  `#define` is preserved verbatim.
- `instruments::dls` — DLS Level 1 + 2 RIFF reader **plus voice
  generator** with **articulation interpretation** (round 80) and
  **EG2 + 2-pole resonant low-pass filter** wiring (round 91). Walks
  the `RIFF/DLS ` form (`colh` / `vers` / `ptbl` / `lins-list` /
  `wvpl-list` / `INFO-list`), surfaces the parsed bank with
  instrument → region → wave-pool topology, `wsmp` loop / pitch /
  gain headers, `wlnk` cue-table references, and `art1` / `art2`
  articulation connection blocks. Voice generation picks the matching
  instrument by MIDI program, picks a region by (key, velocity),
  resolves the `wlnk` → `ptbl` → wave-pool entry, decodes the PCM
  (8/16-bit WAV-shaped) into mono f32, shifts pitch off the
  `wsmp.unity_note`, evaluates the region + instrument articulation
  through `instruments::articulation::Articulation::evaluate`, and
  plays the sample through the shared sample-playback voice with the
  resolved DAHDSR envelope + vibrato LFO + tuning + gain + the
  modulation envelope (EG2) routed into a 2-pole resonant low-pass
  filter cutoff (round 91). Loop modes: forward loop
  (`WLOOP_TYPE_FORWARD`, DLS1) and release loop (`WLOOP_TYPE_RELEASE`,
  DLS2).
- `instruments::articulation` — DLS Level 1/2 connection-block
  evaluator backed by MMA DLS1 v1.1b Tables 1–2 + MMA DLS2.2 v1.0
  Amendment 2 Tables 5–10. Named constants for every `CONN_SRC_*` /
  `CONN_DST_*` / `CONN_TRN_*` enum + the `ABSOLUTE_ZERO` sentinel.
  Supported `SRC_NONE → DST_x` defaults: Vol EG DAHDSR (delay /
  attack / hold / decay / sustain / release), Mod EG DAHDSR (raw —
  surfaced for a later round), modulator + vibrato LFO frequency +
  start delay, filter cutoff + Q, tuning, gain, pan. Supported
  modulator routings: `SRC_LFO → DST_PITCH` (vibrato on DLS1),
  `SRC_LFO → DST_GAIN` (tremolo), `SRC_VIBRATO → DST_PITCH`
  (dedicated DLS2 vibrato — wins over the LFO routing),
  `SRC_EG2 → DST_PITCH` + `SRC_EG2 → DST_FILTER_CUTOFF` (mod-env,
  raw), `SRC_KEYONVELOCITY → DST_EG1_ATTACKTIME` (raw). Unit
  conversions: time-cents → seconds (clamped at 60 s), absolute-pitch
  → cents (clamped at ±14 400), absolute-pitch → Hz for LFO frequency
  (clamped at 50 Hz), gain → linear (clamped at -96..+48 dB),
  sustain-percent → 0..=1, pan-percent → ±50. Region blocks override
  instrument-level blocks per spec; an empty `lart` list falls back
  to SamplePlayer defaults so banks with no articulation are
  byte-identical to round-75 output.
- `instruments::tone` — pure-tone fallback (sine / triangle / saw /
  square) so the synth produces *something* even with no on-disk
  bank.
- `mixer` — polyphonic voice pool (32 voices) with stereo mixdown,
  per-channel volume / pan / sustain pedal handling, oldest-voice
  preemption when the pool is full, channel/poly aftertouch routed
  to per-voice pressure, RPN 0 (pitch-bend range) handling, and
  exclusive-class drum cuts. Native stereo voices (SF2 stereo zones)
  are rendered through their own L/R buses, bypassing the mono-pan
  law. Round 75 adds: **RPN 1** (channel fine tune, ±100 c) /
  **RPN 2** (channel coarse tune, ±63 semis) / **RPN 5** (modulation
  depth range, CA-26) / **RPN 6** (MPE Configuration Message — see
  below). Round 102 adds **Data Increment (CC 96) / Data Decrement
  (CC 97)** per RP-018: the value byte is ignored and each message
  steps the RP-018-prescribed sub-field of the selected RPN by one —
  the LSB (cents) for RPN 0 / 1 / 5 (with RPN 0's LSB wrapping into the
  semitone MSB at 100, the borrow falling out of the combined
  base-100 cents store) and the MSB (one semitone) for RPN 2; RPN Null
  and unmodelled / NRPN selections are a no-op. CC 1 (mod wheel) routed
  to voices through the new
  [`Voice::set_mod_depth_cents`] hook; CC 74 (MPE "third dimension" /
  brightness) routed through [`Voice::set_timbre`]. Master state on
  the mixer adds **Master Volume** (Universal Real-Time SysEx
  `7F 7F 04 01`) applied as a global gain at mix-time, and
  **Master Fine / Master Coarse Tuning** (CA-25, sub-IDs `04 03` /
  `04 04`) summed with the per-channel fine + coarse tune to derive
  the effective pitch each voice receives. Drum channel (MIDI 10 =
  index 9) is exempt from tuning per CA-25. Round 105 adds **Master
  Balance** (Universal Real-Time SysEx `7F 7F 04 02 lsb msb`) per the
  M1 v4.2.1 *Detailed Specification* §"DEVICE CONTROL — MASTER
  VOLUME AND MASTER BALANCE" (p.57): 14-bit value with
  `00 00 = hard left`, `7F 7F = hard right`, centre = `0x2000`.
  Stored verbatim and folded into the mix-time per-side gains via
  [`Mixer::master_balance_gains`] using the textbook balance law
  (the *far* side attenuates while the *near* side stays at unity, so
  a stereo source panned hard one way mutes the opposite bus without
  boosting the near bus). Default `0x2000` produces the identity
  gains `(1.0, 1.0)`, keeping the mix bit-identical to the
  pre-round-105 output until a SysEx moves balance off centre. GM 1 /
  GM 2 System On / GM System Off also reset Master Balance to centre.
  Round 114 adds the **GM2 Global Parameter Control** state
  (`mixer::GmEffects`, CA-024 Universal Real-Time SysEx `04 05`): the
  system-wide Reverb (slot `0101`) and Chorus (slot `0102`)
  parameters, decoded to engineering units via the CA-024 GM2 tables
  (Reverb Type / Time `rt = exp((val-40)·0.025)` s; Chorus Type /
  Mod-Rate `val·0.122` Hz / Mod-Depth `(val+1)/3.2` ms / Feedback
  `val·0.763`% / Send-to-Reverb `val·0.787`%) via
  [`Mixer::set_gm_reverb_param`] / [`Mixer::set_gm_chorus_param`].
  Defaults are the GM2 recommended initial settings (Reverb Type 4
  Large Hall, Chorus Type 2 Chorus 3); GM System On/Off resets them.
  The parameters are decoded and observable but not yet applied as a
  reverb/chorus DSP send — a later round can wire the effects bus
  without re-parsing the SysEx.
- `mixer::MpeZone` / `mixer::MpeRole` — MIDI Polyphonic Expression
  (M1-100-UM v1.1) support. The MCM (RPN 0x0006 on channel 0 for
  Lower, channel 15 for Upper) configures one or two zones; each
  zone's Manager Channel carries zone-wide CCs and its Member
  Channels host per-note Pitch Bend / Channel Pressure / CC 74.
  Per Appendix C the Member Channel pitch bend sums in cents with
  the Manager's bend before reaching the voice. Per §2.2.5 the
  receiver sets default PB Sensitivity to 2 semitones on the
  Manager and 48 semitones on every Member at MCM time. Per §2.2.7
  Polyphonic Key Pressure on a Member is silently dropped. Per
  §2.2.3 a zone reconfiguration stops every Sounding Note on the
  affected channels and resets their controllers.
- `scheduler` — SMF event scheduler. Merges every track into a single
  time-ordered stream, converts ticks → samples against the current
  tempo + division (`samples_per_tick = us_per_quarter * sample_rate /
  (1_000_000 * ticks_per_quarter)`), and dispatches every event into
  the mixer at the right audio sample. Round 75 wires the Universal
  Real-Time / Non-Real-Time SysEx surface: GM 1 / GM 2 System On
  (sub-IDs `09 01` / `09 03`) reset all controllers + master tuning
  + master volume; GM System Off (`09 02`) does the same; Master
  Volume (`04 01`), Master Fine Tuning (`04 03`) + Master Coarse
  Tuning (`04 04`) all route into the mixer's master-state setters.
  Round 105 routes Master Balance (`04 02`) into
  `Mixer::set_master_balance_14`.
  CC 1 / CC 74 are pumped into the new mixer hooks; the MPE
  Configuration Message (RPN 6 on the Lower / Upper Manager Channel)
  reaches the mixer via the existing RPN data-entry pipeline. Round 98
  routes sub-ID#1 `08` (MIDI Tuning Standard) in both Universal areas:
  Single-Note Tuning Change (sub-ID#2 `02` + bank form `07`) and
  Scale/Octave Tuning 1-byte (`08`) / 2-byte (`09`) forms into the
  `tuning` table; GM System On/Off additionally reset MTS tuning to
  equal temperament. Round 102 routes CC 96 / CC 97 (Data Increment /
  Decrement, RP-018) into `Mixer::data_inc_dec`. Round 114 routes the
  Global Parameter Control message (`04 05`, CA-024): it parses the
  Slot Path Length / Parameter-ID Width / Value Width header, walks the
  GM2-reserved slot path (Slot Path Length 1, Slot MSB 1; Slot LSB
  `01` = Reverb, `02` = Chorus), and applies each parameter-value pair
  (MSB-first ID, LSB-first value) into the mixer's GM2 effect setters,
  ignoring unrecognised slots/parameters per the spec.
- `tuning` — MIDI Tuning Standard (MTS) microtuning state + Universal
  SysEx data-format decoders, per the MMA *MIDI Tuning Messages*
  specification (CA-020 / CA-021 / RP-020). A `TuningTable` holds a
  global 128-entry **key-based** table (the current tuning program) and
  per-channel 12-entry **scale/octave** tables, both as signed cents
  added to a key's equal-tempered pitch (default = equal temperament
  everywhere, so untuned playback is byte-identical to the pre-MTS
  path). Decoders cover the 3-byte frequency word
  (`semitone + fraction14/16384`, with the reserved `7F 7F 7F` "no
  change" sentinel), the scale/octave 1-byte (`00=-64c / 40=0c /
  7F=+63c`) and 2-byte (14-bit, ±100 c) offsets, and the `ff gg hh`
  channel bitmap (with the reserved `ff` bits 2–6 ignored). The mixer
  folds the per-key offset into every voice-pitch composition; the
  real-time message forms retune sounding notes immediately while the
  non-real-time "setup" forms update only the stored table. Drum
  channel (MIDI 10) is exempt from retuning per CA-25's
  no-note-shifting rule.
- `downloader` — stub that names a planned default bank (TimGM6mb) but
  currently returns `Error::Unsupported`.

The decoder factory is registered under codec id `"midi"`. Round-3
wires SMF events end-to-end: `send_packet` parses the SMF and primes
the scheduler; `receive_frame` returns interleaved S16 stereo PCM
(1024 samples per channel at 44 100 Hz) until both the event stream
and the voice pool have run dry. Without an on-disk bank the
registry-built decoder uses the pure-tone fallback; for SoundFont 2
playback build a `MidiDecoder` directly with an `Sf2Instrument`.

Coverage today (round 91): full SF2 voice with sm24 24-bit samples,
stereo zones, DAHDSR volume + modulation envelopes, low-pass biquad
filter (gens 8/9), modEnv→pitch / modEnv→filter routing (gens 7/11),
exclusive-class drum cuts (gen 57); pitch bend with **RPN 0 / 1 / 2 /
5 / 6** (range, channel fine tune, channel coarse tune, modulation
depth range, MPE configuration); channel/poly aftertouch; **SFZ voice
generator** with DAHDSR amplitude envelope (`ampeg_*`) and vibrato
LFO (`lfo01_freq` / `lfo01_pitch`); **DLS Level 1 + 2 voice
generator** with `art1`/`art2` connection-block interpretation
(round 80) — Vol EG DAHDSR, vibrato LFO, tuning, gain, pan, plus the
well-known `SRC_LFO → DST_PITCH` / `SRC_VIBRATO → DST_PITCH` /
`SRC_LFO → DST_GAIN` routings; **round 91** lands EG2 + filter
rendering on the shared `SamplePlayer` — `SRC_NONE →
DST_FILTER_CUTOFF` / `DST_FILTER_Q` initialise a per-voice 2-pole
resonant low-pass biquad (RBJ low-pass against the SF2 v2.04 §8.1.3
cents reference `fc_hz = 8.176 * 2^(cents/1200)`), and the
`SRC_EG2 → DST_FILTER_CUTOFF` routing sweeps the cutoff each output
frame from the EG2 DAHDSR envelope (every `CONN_DST_EG2_*`
destination interpreted at voice-build time). All three instrument
paths share one `SamplePlayer` voice for sample playback + DAHDSR
amplitude envelope + vibrato + pitch bend + EG2 + filter (the SF2
voice keeps its own parallel filter path for compatibility with
stereo zones + 24-bit `sm24` samples; both biquads land the same
RBJ cookbook math against the SF2 §8.1.3 reference).

Round 75 also delivers the **MIDI Polyphonic Expression (MPE)** v1.1
control surface (M1-100-UM): MCM-driven Lower / Upper zone
configuration, per-note pitch bend / channel pressure / CC #74 on
Member Channels, Appendix-C combining of Member + Manager pitch
bend, §2.2.5 default 48-semi Member PB sensitivity, §2.2.7 drop of
Polyphonic Key Pressure on Member Channels, §2.2.3 sounding-note
reset on zone reconfiguration. Plus **Universal Real-Time SysEx**
Master Volume (`F0 7F <dev> 04 01 lsb msb F7`), Master Balance
(`04 02`), Master Fine / Master Coarse Tuning (CA-25, `04 03` /
`04 04`), GM2 **Global Parameter Control** (CA-024, `04 05` — Reverb
slot `0101` / Chorus slot `0102`), and GM 1 / GM 2 System On / GM
System Off (Non-Real-Time, `09 01` / `09 02` / `09 03`).

## Fuzzing

Round 172 stands up a `cargo-fuzz` (libfuzzer-sys) harness over every
attacker-facing parser:

```
cargo +nightly fuzz run smf    # smf::parse + tempo/time/key iterators
cargo +nightly fuzz run sf2    # instruments::sf2::Sf2Bank::parse
cargo +nightly fuzz run dls    # instruments::dls::DlsBank::parse
cargo +nightly fuzz run sfz    # instruments::sfz::parse_str
```

Each target asserts the contract every parser already advertises:
arbitrary bytes return a `Result`, with no panic / OOM / integer
overflow / out-of-bounds index on any path. Curated seed corpora
under `fuzz/corpus/<target>/` give the fuzzer a head start across
the well-formed, partial-but-legal, and known-edge shapes. The
`sfz` corpus also keeps the round-172 regression input
(`regression_r172_octave_overflow.sfz.bin`) so the
`parse_key("C-2011420400")` overflow stays under perpetual
fuzzer pressure after the fix.

Initial 4×~45 s runs cleared 30+ million inputs total across smf /
sf2 / dls and 2 M across sfz with zero remaining crashes. The
single round-172 finding (overflow in the SFZ key-name octave
multiplication) was fixed in-place and pinned by a new
`parse_key_octave_extremes_do_not_overflow` unit test.

## Profiling

Round 285 adds `benches/synth_render.rs` (`harness = false`) — a
repeatable SMF→PCM wall-clock harness over a dense 8-channel /
32-voice score (24 s of music, pitch-bend sweeps, volume + pan CCs)
rendered through an in-memory looping SF2 bank, plus a `--corpus`
mode that hashes (FNV-1a-64) the PCM produced for every in-tree
fixture SMF through both the SF2 bank and the tone fallback, and a
`--spin SECS` mode that loops the render as a stable sampling-profiler
target.

Profiling that harness put `Sf2Voice::render` at ~89 % of the wall
clock, and within it the per-sample DAHDSR volume-envelope stage walk
at ~31 % of the total (sample fetch + linear interpolation ~15 %).
The envelope is now evaluated in stage-segmented runs into a 256-entry
stack buffer (`envelope_run`): constant stages (delay / hold /
sustain) become slice fills and the ramp stages (attack / decay /
release) become element-wise loops with no loop-carried dependency,
which the compiler vectorises (NEON `fdiv.4s` on aarch64). Every
per-sample expression is kept verbatim from the scalar evaluator, so
the rendered PCM is **bit-identical** (corpus hashes unchanged; the
`envelope_run_matches_envelope_at_per_sample` unit test pins
`to_bits()` equality across all stage boundaries, the release tail,
and the `elapsed`-wrap fallback). Dense-score render time: 80.2 ms →
64.2 ms (-20 %) on an Apple-silicon dev box.
