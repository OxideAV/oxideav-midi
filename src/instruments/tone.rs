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
        let base_phase_inc = frequency / sample_rate.max(1) as f32;
        Ok(Box::new(ToneVoice {
            waveform,
            phase: 0.0,
            base_phase_inc,
            phase_inc: base_phase_inc,
            amplitude,
            pressure_gain: 1.0,
            // DAHDSR: 5 ms attack, 50 ms decay to 70 % sustain, 100 ms release.
            attack_samples: (sample_rate as f32 * 0.005).max(1.0) as u32,
            decay_samples: (sample_rate as f32 * 0.05).max(1.0) as u32,
            release_samples: (sample_rate as f32 * 0.10).max(1.0) as u32,
            sustain_level: 0.7,
            elapsed: 0,
            release_pos: None,
            release_start_level: 1.0,
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
    /// Base phase increment before pitch-bend modulation.
    base_phase_inc: f32,
    /// Live phase increment = `base_phase_inc * 2^(bend_cents / 1200)`.
    phase_inc: f32,
    amplitude: f32,
    /// Aftertouch gain in `1.0..=1.5` (rest .. full pressure).
    pressure_gain: f32,
    attack_samples: u32,
    decay_samples: u32,
    release_samples: u32,
    sustain_level: f32,
    elapsed: u32,
    /// `Some(elapsed_at_release)` once `release()` was called.
    release_pos: Option<u32>,
    /// Envelope value sampled at the moment of release.
    release_start_level: f32,
    done: bool,
}

impl ToneVoice {
    fn envelope_at(&self, t: u32) -> f32 {
        if let Some(rel_at) = self.release_pos {
            let since = t.saturating_sub(rel_at);
            if since >= self.release_samples {
                return 0.0;
            }
            // Quadratic release: starts at the release-time level,
            // decays to silence over `release_samples`.
            let x = since as f32 / self.release_samples.max(1) as f32;
            return self.release_start_level * (1.0 - x) * (1.0 - x);
        }
        // Attack.
        if t < self.attack_samples {
            return t as f32 / self.attack_samples.max(1) as f32;
        }
        let t = t - self.attack_samples;
        // Decay (exponential-ish via 1 - (1-x)^2).
        if t < self.decay_samples {
            let x = t as f32 / self.decay_samples.max(1) as f32;
            let drop = 1.0 - self.sustain_level;
            let curve = 1.0 - (1.0 - x) * (1.0 - x);
            return 1.0 - drop * curve;
        }
        // Sustain.
        self.sustain_level
    }
}

impl Voice for ToneVoice {
    fn render(&mut self, out: &mut [f32]) -> usize {
        if self.done {
            return 0;
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let env = self.envelope_at(self.elapsed);
            if self.release_pos.is_some() && env <= 0.0 {
                self.done = true;
                return i;
            }

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
            *slot = osc * env * self.amplitude * self.pressure_gain;
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
            self.release_start_level = self.envelope_at(self.elapsed).max(0.0);
            self.release_pos = Some(self.elapsed);
        }
    }

    fn done(&self) -> bool {
        self.done
    }

    fn set_pitch_bend_cents(&mut self, cents: i32) {
        let bend_ratio = (2.0f32).powf(cents as f32 / 1200.0);
        self.phase_inc = self.base_phase_inc * bend_ratio;
    }

    fn set_pressure(&mut self, pressure: f32) {
        let p = pressure.clamp(0.0, 1.0);
        self.pressure_gain = 1.0 + 0.5 * p;
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
    fn adsr_envelope_has_distinct_decay_phase() {
        let inst = ToneInstrument::new();
        // 48 kHz → 5 ms attack ≈ 240 samples, 50 ms decay ≈ 2400 samples,
        // sustain at 0.7. The attack-peak sample (~240) should be ≥ the
        // post-decay sample (~3000), and the post-decay sample should be
        // close to sustain_level * peak (= 0.7 * peak), noticeably lower
        // than the attack peak.
        let mut voice = inst.make_voice(73, 60, 127, 48_000).unwrap(); // sine for clean envelope
        let mut buf = vec![0.0f32; 4096];
        voice.render(&mut buf);
        // Find the peak around sample 240 (end of attack).
        let attack_peak = buf[230..260].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        // Sample at ~3500 (well past decay) should be at sustain.
        let sustain_peak = buf[3400..3500]
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(attack_peak > 0.05, "attack peak too quiet: {attack_peak}",);
        assert!(
            sustain_peak < attack_peak,
            "sustain ({sustain_peak}) should be quieter than attack peak ({attack_peak})",
        );
        // Sustain should be roughly 70 % of attack peak.
        let ratio = sustain_peak / attack_peak;
        assert!(
            (0.5..=0.85).contains(&ratio),
            "sustain/attack ratio {ratio} outside expected ADSR shape",
        );
    }

    #[test]
    fn pitch_bend_changes_voice_frequency() {
        // Sample the phase increment indirectly: render two equal-length
        // buffers, one at centre bend, one with +200 cents. The bent
        // voice completes its first cycle in ≈ 89 % of the centre's
        // sample count (2^(-200/1200) ≈ 0.89).
        let inst = ToneInstrument::new();
        let mut a = inst.make_voice(73, 69, 127, 48_000).unwrap(); // A4 sine
        let mut b = inst.make_voice(73, 69, 127, 48_000).unwrap();
        b.set_pitch_bend_cents(200); // +2 semitones
        let mut buf_a = vec![0.0f32; 1024];
        let mut buf_b = vec![0.0f32; 1024];
        a.render(&mut buf_a);
        b.render(&mut buf_b);
        // Count zero crossings — `b` should have more.
        let cross_a = buf_a.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let cross_b = buf_b.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            cross_b > cross_a,
            "+2 semitones should give more zero crossings: a={cross_a}, b={cross_b}",
        );
    }

    #[test]
    fn pressure_increases_amplitude() {
        let inst = ToneInstrument::new();
        let mut a = inst.make_voice(73, 69, 100, 48_000).unwrap();
        let mut b = inst.make_voice(73, 69, 100, 48_000).unwrap();
        b.set_pressure(1.0);
        // Render past attack (5 ms = 240 samples).
        let mut buf_a = vec![0.0f32; 4096];
        let mut buf_b = vec![0.0f32; 4096];
        a.render(&mut buf_a);
        b.render(&mut buf_b);
        let peak_a = buf_a[1000..2000]
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        let peak_b = buf_b[1000..2000]
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(
            peak_b > peak_a * 1.2,
            "pressure should boost amplitude: a={peak_a}, b={peak_b}",
        );
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
