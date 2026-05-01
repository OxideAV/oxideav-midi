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
//! - [`sfz`] and [`dls`] are still magic-byte detector stubs;
//!   `make_voice` returns `Error::Unsupported` for both. Loaders are
//!   round-3.
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
