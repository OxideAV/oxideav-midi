//! External-instrument trait + per-format adapters + pure-tone
//! fallback.
//!
//! The [`Instrument`] trait describes a small surface — "give me one
//! voice (a sound source) for this MIDI program at this pitch" — so
//! the synth core can stay format-agnostic.
//!
//! - [`sf2`] is a working SoundFont 2 reader + voice generator: it
//!   loads a `.sf2` bank into memory, cross-resolves the preset →
//!   instrument → zone → sample chain, and renders 16-bit PCM at the
//!   requested pitch via linear interpolation. (Modulators, sm24,
//!   stereo linking, and full envelopes/filters are pending.)
//! - [`sfz`] is a working text patch reader: it strips comments, walks
//!   `<control>` / `<global>` / `<master>` / `<group>` / `<region>`
//!   sections, flattens inheritance into one fully-resolved opcode map
//!   per region, and (when constructed via
//!   [`sfz::SfzInstrument::open`]) reads every referenced sample off
//!   disk. `make_voice` is round-2 — it returns `Error::Unsupported`
//!   today; the parsed [`sfz::SfzInstrument::regions`] table is the
//!   public surface round-1 surfaces.
//! - [`dls`] is still a magic-byte detector stub; `make_voice` returns
//!   `Error::Unsupported`. Loader work is blocked on the DLS Level 1/2
//!   specification landing in `docs/audio/midi/instrument-formats/`.
//! - [`tone::ToneInstrument`] is the canary: if no SoundFont is
//!   available, the synth still produces *something*.

use oxideav_core::Result;

pub mod dls;
pub mod sf2;
pub mod sfz;
pub mod tone;

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
/// in [`sf2::Sf2Bank`] etc.).
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
}
