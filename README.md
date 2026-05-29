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
