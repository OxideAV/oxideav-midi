# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- rewrite release-envelope comment to remove FluidSynth citation
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
