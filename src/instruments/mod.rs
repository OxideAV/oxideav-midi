//! External-instrument trait + per-format adapter stubs + a pure-tone
//! fallback that needs no on-disk file.
//!
//! Round-1: only header/magic detection and a working sine/saw/triangle
//! oscillator. Actual sample fetching from SoundFont 2 / SFZ / DLS
//! lands in round-2.
//!
//! The [`Instrument`] trait describes a small surface — "give me one
//! voice (a sound source) for this MIDI program at this pitch" — so
//! the synth core can stay format-agnostic. The pure-tone fallback
//! ([`tone::ToneInstrument`]) is the canary: if no SoundFont is
//! available, the synth still produces *something*.

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
pub trait Instrument: Send {
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
