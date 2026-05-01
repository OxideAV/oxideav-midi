# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
