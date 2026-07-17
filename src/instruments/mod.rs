//! External-instrument trait + per-format adapters + pure-tone
//! fallback.
//!
//! The [`Instrument`] trait describes a small surface — "give me one
//! voice (a sound source) for this MIDI program at this pitch" — so
//! the synth core can stay format-agnostic.
//!
//! - [`sf2`] is a working SoundFont 2 reader + voice generator: it
//!   loads a `.sf2` bank into memory, cross-resolves the preset →
//!   instrument → zone → sample chain, and renders sm24-aware 24-bit
//!   PCM at the requested pitch via linear interpolation. Honours the
//!   volume + modulation DAHDSR envelopes, the initial low-pass biquad
//!   filter, mod-env → pitch / filter routing, exclusive-class drum
//!   cuts, and native stereo zones.
//! - [`sfz`] is a working text patch reader **plus voice generator**.
//!   The reader strips comments, walks `<control>` / `<global>` /
//!   `<master>` / `<group>` / `<region>` sections, flattens
//!   inheritance into one fully-resolved opcode map per region, and
//!   (when constructed via [`sfz::SfzInstrument::open`]) reads every
//!   referenced sample off disk. `make_voice` decodes the WAV bytes
//!   (8/16/24/32-bit PCM and IEEE_FLOAT), picks the matching region
//!   by (key, velocity), shifts pitch off `pitch_keycenter` + `tune` +
//!   `transpose`, and runs a DAHDSR amplitude envelope (`ampeg_*`) +
//!   vibrato LFO (`lfo01_*`).
//! - [`dls`] is a working DLS Level 1 + Level 2 RIFF reader **plus
//!   voice generator**. The reader walks the `DLS ` form, pulls the
//!   `colh` / `vers` / `ptbl` / `lins` / `wvpl` chunks apart, and
//!   surfaces a fully-resolved bank ([`dls::DlsBank`]) of instruments
//!   → regions → wave-pool samples with their `wsmp` loop info,
//!   `wlnk` cue-table references, and `art1` / `art2` connection
//!   blocks. `make_voice` picks the matching instrument by program,
//!   picks a region by (key, velocity), resolves wlnk → ptbl →
//!   wave-pool, decodes the PCM, evaluates the region + instrument
//!   articulation through `articulation::Articulation` (round 80),
//!   and plays the sample through the shared
//!   `sample_voice::SamplePlayer` with the resolved DAHDSR envelope
//!   + vibrato LFO + tuning + gain applied.
//! - `articulation` is the DLS Level 1/2 connection-block evaluator
//!   used by [`dls`] at voice-build time. Honours the `SRC_NONE →
//!   DST_x` default-override connections for the Vol EG, the
//!   modulator + vibrato LFO, tuning, gain and pan, plus a handful of
//!   `SRC_x → DST_y` modulator routings — see the module's doc for
//!   the supported subset.
//! - `sample_voice` is the shared sample-playback voice both `sfz`
//!   and `dls` use. Mono in, mono out — the [`mixer`](crate::mixer)
//!   handles stereo panning. Covers DAHDSR amplitude envelope, four
//!   loop modes (no-loop / one-shot / continuous / sustain), pitch
//!   bend, and a vibrato LFO (rate/depth/delay).
//! - `wav_pcm` is a minimal RIFF/WAVE PCM decoder — 8-bit unsigned,
//!   16-bit signed LE, 24-bit signed LE, 32-bit signed LE PCM, and
//!   32-bit IEEE_FLOAT — used by the SFZ and DLS sample loaders.
//! - [`tone::ToneInstrument`] is the canary: if no SoundFont is
//!   available, the synth still produces *something*.

use oxideav_core::Result;

#[doc(hidden)] // internal: DLS connection-block/modulator-table evaluator used at voice-build time
pub mod articulation;
pub mod dls;
pub mod percussion;
#[doc(hidden)] // internal: shared sample-playback voice guts (SFZ/DLS plumbing)
pub mod sample_voice;
pub mod sf2;
pub mod sfz;
pub mod tone;
#[doc(hidden)] // internal: WAV PCM helper for the SFZ/DLS sample loaders
pub mod wav_pcm;

/// Snapshot of a channel's **Sound Controllers** (CC 71–78) at
/// note-on time — the RP-021 "Sound Controller Defaults" set with the
/// response semantics GM2 (RP-024 §3.3.11–§3.3.18) pins down: every
/// value is a *relative* parameter whose centre (null point) is 64 =
/// "no change" from the timbre's preset, below 64 decreases and above
/// 64 increases, with the exact response left to the implementation's
/// discretion.
///
/// The mixer captures the snapshot into each freshly-struck voice via
/// [`Voice::apply_sound_controls`]; a snapshot of all-64s is never
/// delivered (nothing to change), so unmodified scores render
/// bit-identically to the pre-CC-71–78 synth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundControls {
    /// CC 71 — Filter Resonance (Timbre / Harmonic Intensity), GM2
    /// §3.3.11. Strengthens/weakens the preset filter resonance.
    pub resonance: u8,
    /// CC 72 — Release Time, GM2 §3.3.12.
    pub release_time: u8,
    /// CC 73 — Attack Time, GM2 §3.3.13.
    pub attack_time: u8,
    /// CC 74 — Brightness (filter cutoff), GM2 §3.3.14. Also the MPE
    /// "third dimension"; routed live via [`Voice::set_timbre`] and
    /// captured here for new notes.
    pub brightness: u8,
    /// CC 75 — Decay Time, GM2 §3.3.15.
    pub decay_time: u8,
    /// CC 76 — Vibrato Rate, GM2 §3.3.16.
    pub vibrato_rate: u8,
    /// CC 77 — Vibrato Depth, GM2 §3.3.17.
    pub vibrato_depth: u8,
    /// CC 78 — Vibrato Delay, GM2 §3.3.18.
    pub vibrato_delay: u8,
}

impl Default for SoundControls {
    fn default() -> Self {
        // GM2 §3.3.11–§3.3.18: every Sound Controller defaults to 64
        // (40H, "no change").
        Self {
            resonance: 64,
            release_time: 64,
            attack_time: 64,
            brightness: 64,
            decay_time: 64,
            vibrato_rate: 64,
            vibrato_depth: 64,
            vibrato_delay: 64,
        }
    }
}

impl SoundControls {
    /// `true` when every controller sits at its 64 "no change" centre.
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }

    /// This synth's discretionary map from a relative Sound Controller
    /// value to a multiplicative scale: `2^((v − 64) / 32)` — centre
    /// 64 → ×1 (no change), 0 → ×0.25, 127 → ×3.9. Exponential so the
    /// audible change per step is uniform, matching how envelope times
    /// and LFO rates are perceived; GM2 leaves the exact curve to the
    /// manufacturer's discretion.
    pub fn scale(value: u8) -> f32 {
        2f32.powf((value.min(127) as f32 - 64.0) / 32.0)
    }
}

/// One voice rendered into a planar f32 buffer.
///
/// Voices are ephemeral — the synth holds them while a note is on,
/// drops them when it releases. A voice produces samples until it
/// reports `done()`; the synth then frees the slot.
pub trait Voice: Send {
    /// Render up to `out.len()` mono samples into `out`. Returns the
    /// number of samples actually written. Anything past the returned
    /// count is left untouched; callers should treat shorter writes as
    /// "voice ran out — drop it after this chunk".
    fn render(&mut self, out: &mut [f32]) -> usize;

    /// Signal note-off. The voice may keep producing samples while it
    /// runs through its release envelope.
    fn release(&mut self);

    /// `true` when the voice has nothing more to produce.
    fn done(&self) -> bool;

    /// Set the per-voice pitch-bend offset, in cents (1/100 semitone).
    /// `0` is centre. Default is a no-op for voices that don't model
    /// pitch (the round-3 / round-4 generators all support it).
    fn set_pitch_bend_cents(&mut self, _cents: i32) {}

    /// Set per-voice pressure (aftertouch), `0.0..=1.0`. Default route
    /// is a multiplicative gain on the rendered samples. Voices may
    /// override to route pressure into filter cutoff, vibrato depth,
    /// etc.; the round-4 default modulator chain just modulates volume.
    fn set_pressure(&mut self, _pressure: f32) {}

    /// Per-note modulation-wheel depth, expressed as a signed pitch
    /// offset in cents. The mixer computes this as
    /// `mod_wheel/127 * channel.mod_depth_range_cents` (per CA-26
    /// RPN 5 with GM2's default 50-cent range), and the voice routes
    /// it the same way it routes a vibrato LFO peak deviation: an
    /// additional pitch sway summed with the existing pitch-bend
    /// offset. Default is a no-op for voices that don't model
    /// modulation depth (the tone fallback ignores it).
    fn set_mod_depth_cents(&mut self, _cents: i32) {}

    /// MPE-style "third dimension of control" (Control Change #74).
    /// Per the MPE spec §2.2.8 + Appendix D, this carries timbre
    /// information that affects the live voice independently of pitch
    /// bend (CC74 = filter brightness on most receivers). The argument
    /// is the raw `0..=127` scalar; voices map it into their internal
    /// timbre parameter as they see fit. Default no-op.
    fn set_timbre(&mut self, _value_0_127: u8) {}

    /// Capture a channel's Sound Controller snapshot (CC 71–78, GM2
    /// RP-024 §3.3.11–§3.3.18) into this voice. Called by the mixer
    /// **once**, immediately after note-on and before the first
    /// render, and only when the snapshot is non-neutral — so
    /// implementations may scale their envelope / LFO / filter state
    /// in place without idempotency bookkeeping. Default no-op for
    /// voices with nothing to scale.
    fn apply_sound_controls(&mut self, _controls: &SoundControls) {}

    /// CA-022 / GM2 RP-024 §4.6 **Filter Cutoff Control** destination:
    /// an additive filter-cutoff offset in cents (GM2 range −9600 to
    /// +9450), driven by Channel Pressure or the routed Control Change.
    /// Absolute (not cumulative) — the mixer recomputes and re-sends
    /// the full offset on every controller move. Default no-op for
    /// voices without a filter.
    fn set_filter_cutoff_mod_cents(&mut self, _cents: i32) {}

    /// CA-022 / GM2 RP-024 §4.6 **LFO Filter Depth** destination: peak
    /// LFO-driven filter-cutoff sway in cents (GM2 range 0–2400).
    /// Absolute; default no-op for voices without an LFO or filter.
    fn set_lfo_filter_depth_cents(&mut self, _cents: i32) {}

    /// CA-022 / GM2 RP-024 §4.6 **LFO Amplitude Depth** destination
    /// (tremolo): `0.0..=1.0` = 0–100 % amplitude sway. Absolute;
    /// default no-op for voices without an LFO.
    fn set_lfo_amp_depth(&mut self, _depth: f32) {}

    /// `true` when this voice produces native stereo output via
    /// [`render_stereo`](Voice::render_stereo) and should bypass the
    /// mixer's mono-pan law. Default `false` — the mixer renders the
    /// mono `render` output and pans it.
    fn is_stereo(&self) -> bool {
        false
    }

    /// Render up to `out_l.len()` stereo samples into the L/R planes.
    /// Both planes must be the same length. Default impl renders the
    /// mono [`render`](Voice::render) output into `out_l` and copies it
    /// to `out_r`; voices that override [`is_stereo`](Voice::is_stereo)
    /// to `true` override this to write distinct L/R samples (e.g. a
    /// SoundFont stereo zone that pulls from a paired sample).
    fn render_stereo(&mut self, out_l: &mut [f32], out_r: &mut [f32]) -> usize {
        debug_assert_eq!(out_l.len(), out_r.len());
        let n = self.render(out_l);
        out_r[..n].copy_from_slice(&out_l[..n]);
        n
    }

    /// Non-zero exclusive-class id (SF2 generator 57). When a new
    /// voice with the same `exclusive_class` is started on the same
    /// channel, the mixer hard-stops every prior voice in that class —
    /// drum kits use this for hi-hat open/closed pairs. Default `0` =
    /// no exclusivity.
    fn exclusive_class(&self) -> u16 {
        0
    }
}

/// Source of voices for one MIDI program (a "bank").
///
/// `Send + Sync` so an `Arc<dyn Instrument>` is `Send`-able into the
/// `MidiDecoder` (which itself must be `Send` per the `Decoder` trait).
/// `make_voice` takes `&self` so concrete impls only need shared
/// references to whatever cross-cutting state they hold (sample arena
/// in `sf2::Sf2Bank` etc.).
pub trait Instrument: Send + Sync {
    /// Human-readable name for diagnostics. Implementations should
    /// return something stable — a filename, a "TimGM6mb GM Set", or
    /// `"pure-tone fallback"` for the canary.
    fn name(&self) -> &str;

    /// Allocate a voice for `program` (0..=127, the GM/MIDI program
    /// number) at MIDI key `key` (0..=127) and velocity `velocity`
    /// (0..=127). The `sample_rate` is the audio output rate the synth
    /// is rendering at — voices size their oscillator phase / sample
    /// playback rate against it.
    fn make_voice(
        &self,
        program: u8,
        key: u8,
        velocity: u8,
        sample_rate: u32,
    ) -> Result<Box<dyn Voice>>;

    /// Allocate a voice for `program` within a **GM2 bank** (RP-024
    /// §3.3.1): `bank_msb`/`bank_lsb` are the latched CC 0 / CC 32
    /// pair, `78H/xxH` selecting the Percussion Sound Set (a Rhythm
    /// Channel's drum kit) and `79H/xxH` a Melody Sound Set variation
    /// (`79H/00H` = the GM1 set). The default implementation ignores
    /// the bank and delegates to [`Self::make_voice`], so bank-unaware
    /// backends keep their exact previous behaviour; bank-aware
    /// backends (SF2, whose presets carry a bank number and whose GM
    /// convention parks drum kits in bank 128) override it.
    fn make_voice_banked(
        &self,
        bank_msb: u8,
        bank_lsb: u8,
        program: u8,
        key: u8,
        velocity: u8,
        sample_rate: u32,
    ) -> Result<Box<dyn Voice>> {
        let _ = (bank_msb, bank_lsb);
        self.make_voice(program, key, velocity, sample_rate)
    }
}
