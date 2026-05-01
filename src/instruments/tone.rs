//! Pure-tone fallback "instrument" — the synth canary.
//!
//! This module produces a trivial oscillator (sine, sawtooth, triangle,
//! or square) at the MIDI key's nominal frequency. It needs no on-disk
//! file. Use it when no SoundFont 2 / SFZ / DLS bank is installed, so
//! the synth still produces *something* rather than silence.
//!
//! Programs are mapped onto waveforms with a tiny GM-flavoured table:
//!
//! - 0..=7  (Pianos)         → triangle (soft transient)
//! - 8..=23 (Chromatic perc.)→ triangle
//! - 24..=39 (Guitars)       → sawtooth
//! - 40..=55 (Strings)       → sawtooth
//! - 56..=71 (Brass / Reeds) → square
//! - 72..=87 (Pipes / Synth) → sine
//! - 88..=119               → triangle (catch-all melodic)
//! - 120..=127 (FX / Drums)  → square (snappier)
//!
//! It's not musical, but every channel produces a distinguishable tone.
//! Real instrument banks are loaded via the `sf2` / `sfz` / `dls`
//! adapters in the parent module.

use oxideav_core::Result;

use super::{Instrument, Voice};

/// Pure-tone instrument bank. Stateless — one shared instance can hand
/// out as many voices as you ask for.
#[derive(Default)]
pub struct ToneInstrument;

impl ToneInstrument {
    pub fn new() -> Self {
        Self
    }
}

impl Instrument for ToneInstrument {
    fn name(&self) -> &str {
        "pure-tone fallback"
    }

    fn make_voice(
        &self,
        program: u8,
        key: u8,
        velocity: u8,
        sample_rate: u32,
    ) -> Result<Box<dyn Voice>> {
        let waveform = waveform_for_program(program);
        let frequency = midi_key_to_hz(key);
        // Velocity 0..=127 maps to roughly 0..=1.0; perceptually-OK
        // square-law curve so soft notes feel softer than linear.
        let v = velocity as f32 / 127.0;
        let amplitude = v * v * 0.4; // headroom: 16 voices at peak ~ 1.0
        Ok(Box::new(ToneVoice {
            waveform,
            phase: 0.0,
            phase_inc: frequency / sample_rate.max(1) as f32,
            amplitude,
            // 250 ms attack-to-sustain, then sustained, then 100 ms release.
            attack_samples: (sample_rate as f32 * 0.05) as u32,
            release_samples: (sample_rate as f32 * 0.10) as u32,
            elapsed: 0,
            release_pos: None,
            done: false,
        }))
    }
}

/// Available oscillator shapes for the tone fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Waveform {
    Sine,
    Triangle,
    Sawtooth,
    Square,
}

/// Map a GM program number (0..=127) to a waveform — see module docs.
pub fn waveform_for_program(program: u8) -> Waveform {
    match program {
        0..=23 => Waveform::Triangle,
        24..=55 => Waveform::Sawtooth,
        56..=71 => Waveform::Square,
        72..=87 => Waveform::Sine,
        88..=119 => Waveform::Triangle,
        _ => Waveform::Square,
    }
}

/// Standard 12-tone equal-temperament conversion. A4 (key 69) = 440 Hz.
pub fn midi_key_to_hz(key: u8) -> f32 {
    440.0 * 2f32.powf((key as f32 - 69.0) / 12.0)
}

struct ToneVoice {
    waveform: Waveform,
    /// Phase in `0.0..1.0` (one full cycle).
    phase: f32,
    phase_inc: f32,
    amplitude: f32,
    attack_samples: u32,
    release_samples: u32,
    elapsed: u32,
    /// `Some(elapsed_at_release)` once `release()` was called.
    release_pos: Option<u32>,
    done: bool,
}

impl Voice for ToneVoice {
    fn render(&mut self, out: &mut [f32]) -> usize {
        if self.done {
            return 0;
        }
        for (i, slot) in out.iter_mut().enumerate() {
            // Envelope: linear attack, sustain at 1.0, linear release.
            let env = if let Some(rel_at) = self.release_pos {
                let since_release = self.elapsed.saturating_sub(rel_at);
                if since_release >= self.release_samples {
                    self.done = true;
                    // Write nothing further this render — surface the
                    // partial fill to the caller so it can drop us.
                    return i;
                }
                1.0 - (since_release as f32 / self.release_samples.max(1) as f32)
            } else if self.elapsed < self.attack_samples {
                self.elapsed as f32 / self.attack_samples.max(1) as f32
            } else {
                1.0
            };

            let osc = match self.waveform {
                Waveform::Sine => (self.phase * std::f32::consts::TAU).sin(),
                Waveform::Triangle => {
                    // Tent wave: 4|x - 0.5| - 1, x in [0, 1).
                    4.0 * (self.phase - 0.5).abs() - 1.0
                }
                Waveform::Sawtooth => 2.0 * self.phase - 1.0,
                Waveform::Square => {
                    if self.phase < 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                }
            };
            *slot = osc * env * self.amplitude;
            self.phase += self.phase_inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
            self.elapsed = self.elapsed.wrapping_add(1);
        }
        out.len()
    }

    fn release(&mut self) {
        if self.release_pos.is_none() {
            self.release_pos = Some(self.elapsed);
        }
    }

    fn done(&self) -> bool {
        self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_60_is_middle_c() {
        let f = midi_key_to_hz(60);
        // Middle C ≈ 261.63 Hz
        assert!((f - 261.6256).abs() < 0.01, "middle C frequency was {f}");
    }

    #[test]
    fn key_69_is_a4() {
        assert!((midi_key_to_hz(69) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn waveform_dispatch_covers_full_range() {
        // Every program number must map to *some* waveform — none panic.
        for p in 0u8..=127 {
            let _ = waveform_for_program(p);
        }
    }

    #[test]
    fn voice_produces_nonzero_samples() {
        let inst = ToneInstrument::new();
        let mut voice = inst.make_voice(0, 69, 100, 48_000).unwrap();
        let mut buf = [0.0f32; 1024];
        let n = voice.render(&mut buf);
        assert_eq!(n, 1024);
        let nonzero = buf.iter().filter(|s| s.abs() > 0.001).count();
        assert!(nonzero > 100, "expected non-silent output, got {nonzero}");
    }

    #[test]
    fn voice_eventually_finishes_after_release() {
        let inst = ToneInstrument::new();
        let mut voice = inst.make_voice(0, 60, 100, 48_000).unwrap();
        let mut buf = [0.0f32; 256];
        // Render some sustain.
        voice.render(&mut buf);
        voice.release();
        // Render more than the release window worth of samples.
        // 100 ms at 48 kHz = 4800 samples.
        let mut total = 0;
        for _ in 0..50 {
            let n = voice.render(&mut buf);
            total += n;
            if voice.done() {
                break;
            }
        }
        assert!(voice.done(), "voice should be done after release window");
        assert!(total > 0, "voice rendered {total} samples post-release");
    }
}
