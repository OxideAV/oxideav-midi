# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
