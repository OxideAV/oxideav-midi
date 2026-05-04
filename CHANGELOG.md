# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
