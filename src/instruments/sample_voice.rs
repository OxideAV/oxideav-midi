//! Generic sample-playback voice shared between the SFZ and DLS
//! instrument adapters.
//!
//! Both formats reduce to the same shape at note-on time: pick a sample
//! buffer (already decoded to f32 mono frames), pick a playback rate
//! (derived from the requested MIDI key vs the sample's native key plus
//! coarse/fine tuning), pick a loop window (start/end frames + a
//! [`SampleLoopMode`]), pick a DAHDSR amplitude envelope, and (for SFZ)
//! a vibrato LFO. The SF2 path has its own [`Sf2Voice`](super::sf2::Sf2Voice)
//! because it carries extra state (mod-env routings into pitch and
//! filter cutoff, native stereo zones, sm24 24-bit samples) that the
//! shared voice would have to optionalise; rather than bloat one struct
//! with mostly-unused state, we keep two voice types side-by-side.
//!
//! The voice runs on **mono** sample data — both SFZ and DLS samples
//! that are conceptually mono are decoded to f32 mono before the voice
//! sees them. Stereo SFZ/DLS samples are deferred (round 2 will fan the
//! decoder out to two channels and mark the voice as stereo).
//!
//! Round-1 articulation coverage:
//! - **DAHDSR amplitude envelope** (always wired). Same shape as the
//!   `Sf2Voice` / `ToneVoice`: linear attack from 0→1, hold at peak,
//!   exponential-ish decay to sustain, exponential release on note-off.
//! - **Vibrato LFO** (SFZ `lfo01_freq` / `lfo01_pitch` opcodes). A
//!   simple sine LFO routed into pitch as a cents offset; rate is in Hz
//!   and depth is in cents. Defaults to 0/0 so DLS / unconfigured SFZ
//!   regions emit no vibrato.
//! - **Pitch bend** via [`Voice::set_pitch_bend_cents`]. Layered on top
//!   of the LFO depth without modifying the base playback rate.
//!
//! Pitch envelopes (`pitcheg_*`) and full DLS art1/art2 connection
//! evaluation are deferred to round 2 — see
//! [`SampleLoopMode`] and [`SamplePlayerConfig`] doc-comments for the
//! exact data the round-1 voice consumes.

use std::sync::Arc;

use super::Voice;

/// Loop semantics for the sample-playback voice. Mirrors the SFZ
/// `loop_mode` opcode + the DLS `WLOOP_TYPE_*` flags so both formats
/// can share one render path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleLoopMode {
    /// Play once from the start to the end and stop. Note-off triggers
    /// the release envelope but does not interrupt the playback head.
    NoLoop,
    /// Play once and **ignore** note-off — used for percussion. The
    /// release envelope only fires when the sample reaches its end.
    OneShot,
    /// Loop continuously between `loop_start` and `loop_end`, even
    /// after note-off; the release envelope fades the looped audio.
    LoopContinuous,
    /// Loop while the note is held. On note-off the playback head
    /// breaks out of the loop and continues to the sample end (i.e.
    /// runs the post-loop tail from `loop_end..end`).
    LoopSustain,
}

/// DAHDSR amplitude envelope, in seconds (with `sustain_level` as a
/// linear `0.0..=1.0` fraction of peak). Each phase falls back to a
/// musical default when its field is `0.0` *and* the field was not
/// explicitly set; callers (SFZ / DLS adapters) populate the struct
/// with `..Default::default()` so unset fields read as the defaults
/// below.
#[derive(Clone, Copy, Debug)]
pub struct EnvelopeParams {
    /// Pre-attack delay, seconds. Default 0.
    pub delay_s: f32,
    /// Linear attack from 0→1, seconds. Default 0.005 (5 ms — enough
    /// to avoid a click on transient-light samples).
    pub attack_s: f32,
    /// Hold-at-peak, seconds. Default 0.
    pub hold_s: f32,
    /// Decay-to-sustain, seconds. Default 0.100 (100 ms).
    pub decay_s: f32,
    /// Linear sustain level, `0.0..=1.0`. Default 1.0 (no decay loss).
    pub sustain_level: f32,
    /// Release-on-note-off, seconds. Default 0.100.
    pub release_s: f32,
}

impl Default for EnvelopeParams {
    fn default() -> Self {
        Self {
            delay_s: 0.0,
            attack_s: 0.005,
            hold_s: 0.0,
            decay_s: 0.100,
            sustain_level: 1.0,
            release_s: 0.100,
        }
    }
}

/// Vibrato LFO parameters. Both `freq_hz` and `depth_cents` default to
/// 0 so an unconfigured region produces no vibrato.
#[derive(Clone, Copy, Debug, Default)]
pub struct VibratoParams {
    /// LFO rate in Hz. SFZ `lfo01_freq` defaults to 0 (no LFO).
    pub freq_hz: f32,
    /// Pitch modulation depth in cents — peak deviation from the
    /// nominal pitch. SFZ `lfo01_pitch` defaults to 0.
    pub depth_cents: f32,
    /// LFO start delay in seconds (`lfo01_delay`). The LFO output is
    /// gated to 0 until `delay_s` of audio has been rendered.
    pub delay_s: f32,
}

/// Configuration for one [`SamplePlayer`] voice. Owns its sample data
/// via an `Arc` so the bank can hand the same buffer to many concurrent
/// voices without copying.
#[derive(Clone, Debug)]
pub struct SamplePlayerConfig {
    /// Mono sample frames in normalised f32 (`-1.0..=1.0`). Long-lived
    /// — held in an `Arc` so multiple voices can share one buffer.
    pub samples: Arc<[f32]>,
    /// Native sample rate of `samples`, Hz. Combines with the output
    /// rate + pitch ratio to yield the per-output-sample phase increment.
    pub native_rate: u32,
    /// Loop start frame, inclusive (frames into `samples`).
    pub loop_start: u32,
    /// Loop end frame, **exclusive** (one past the last looped frame).
    pub loop_end: u32,
    /// Sample-end frame (one past the last frame to ever play).
    /// Defaults to `samples.len()` when no override is wanted.
    pub sample_end: u32,
    pub loop_mode: SampleLoopMode,
    /// Frequency multiplier — `2^((target_key - native_key)/12 +
    /// fine_cents/1200 + transpose/12)`. The voice multiplies this
    /// against the live pitch-bend / LFO modulation each sample.
    pub pitch_ratio: f64,
    /// Linear amplitude `0.0..=1.0` at peak (1.0 = sample's own peak,
    /// post-velocity-curve). Folded into the velocity-derived gain at
    /// construction.
    pub amplitude: f32,
    pub envelope: EnvelopeParams,
    pub vibrato: VibratoParams,
    /// Exclusive-class id (drum-kit style hi-hat open/closed cuts).
    /// Surfaced through [`Voice::exclusive_class`]; 0 = no class.
    pub exclusive_class: u16,
}

/// Sample-playback voice. Mono in, mono out — the mixer pans into
/// stereo. Round-1 covers the basics: DAHDSR + vibrato LFO + pitch
/// bend + four loop modes + exclusive-class. Filter/pitch envelopes
/// are deferred.
pub struct SamplePlayer {
    samples: Arc<[f32]>,
    end: u32,
    loop_start: u32,
    loop_end: u32,
    loop_mode: SampleLoopMode,
    /// Current playback position, fractional frames into `samples`.
    phase: f64,
    /// Base playback rate (frames of `samples` per output frame). Set
    /// at construction from `pitch_ratio * native_rate / output_rate`.
    base_phase_inc: f64,
    /// Live playback rate, including pitch-bend + vibrato LFO mod.
    phase_inc: f64,
    amplitude: f32,
    pressure_gain: f32,
    pitch_bend_cents: i32,
    /// Output sample counter — drives the envelope phases and the LFO.
    elapsed: u32,
    /// Sample at which `release()` fired, or `None` while held.
    release_pos: Option<u32>,
    /// Envelope level at the moment of release — release decays from
    /// here to 0 over `release_samples`.
    release_start_level: f32,
    /// DAHDSR phase boundaries in *output* frames.
    delay_samples: u32,
    attack_samples: u32,
    hold_samples: u32,
    decay_samples: u32,
    release_samples: u32,
    sustain_level: f32,
    /// Set once the envelope has fully released or the sample has run
    /// past its non-looping end.
    done: bool,
    /// Vibrato LFO parameters (rate, depth, start-delay).
    lfo_freq_hz: f32,
    lfo_depth_cents: f32,
    lfo_delay_samples: u32,
    /// Output sample rate, Hz. Stashed for the LFO phase math.
    output_rate: f32,
    exclusive_class: u16,
    /// `true` once we've broken out of a `LoopSustain` loop on note-off
    /// — once set, the playback head no longer wraps at `loop_end`.
    loop_broken: bool,
}

impl SamplePlayer {
    /// Build a voice from a config + the output sample rate the synth
    /// is rendering at.
    pub fn new(cfg: SamplePlayerConfig, output_rate: u32) -> Self {
        let sr = output_rate.max(1) as f32;
        let phase_inc = cfg.pitch_ratio * (cfg.native_rate as f64 / output_rate.max(1) as f64);
        // Cap end at the buffer length so a malformed config can't read
        // past the end of the sample data.
        let buf_len = cfg.samples.len() as u32;
        let end = cfg.sample_end.min(buf_len);
        let loop_end = cfg.loop_end.min(end);
        let loop_start = cfg.loop_start.min(loop_end);

        Self {
            samples: cfg.samples,
            end,
            loop_start,
            loop_end,
            loop_mode: cfg.loop_mode,
            phase: 0.0,
            base_phase_inc: phase_inc,
            phase_inc,
            amplitude: cfg.amplitude,
            pressure_gain: 1.0,
            pitch_bend_cents: 0,
            elapsed: 0,
            release_pos: None,
            release_start_level: 1.0,
            delay_samples: (sr * cfg.envelope.delay_s.max(0.0)) as u32,
            attack_samples: (sr * cfg.envelope.attack_s.max(0.0)).max(1.0) as u32,
            hold_samples: (sr * cfg.envelope.hold_s.max(0.0)) as u32,
            decay_samples: (sr * cfg.envelope.decay_s.max(0.0)).max(1.0) as u32,
            release_samples: (sr * cfg.envelope.release_s.max(0.0)).max(1.0) as u32,
            sustain_level: cfg.envelope.sustain_level.clamp(0.0, 1.0),
            done: false,
            lfo_freq_hz: cfg.vibrato.freq_hz.max(0.0),
            lfo_depth_cents: cfg.vibrato.depth_cents,
            lfo_delay_samples: (sr * cfg.vibrato.delay_s.max(0.0)) as u32,
            output_rate: sr,
            exclusive_class: cfg.exclusive_class,
            loop_broken: false,
        }
    }

    /// DAHDSR envelope value at `t` output frames. Same shape as the
    /// `Sf2Voice` envelope — kept in lockstep so the two voice types
    /// behave identically when the parameters match.
    fn envelope_at(&self, t: u32) -> f32 {
        if let Some(rel_at) = self.release_pos {
            let since = t.saturating_sub(rel_at);
            if since >= self.release_samples {
                return 0.0;
            }
            let x = since as f32 / self.release_samples.max(1) as f32;
            let curve = (1.0 - x) * (1.0 - x);
            return self.release_start_level * curve;
        }
        if t < self.delay_samples {
            return 0.0;
        }
        let t = t - self.delay_samples;
        if t < self.attack_samples {
            return t as f32 / self.attack_samples.max(1) as f32;
        }
        let t = t - self.attack_samples;
        if t < self.hold_samples {
            return 1.0;
        }
        let t = t - self.hold_samples;
        if t < self.decay_samples {
            let x = t as f32 / self.decay_samples.max(1) as f32;
            let drop = 1.0 - self.sustain_level;
            let curve = 1.0 - (1.0 - x) * (1.0 - x);
            return 1.0 - drop * curve;
        }
        self.sustain_level
    }

    /// Vibrato LFO output in cents at output frame `t`. Returns 0
    /// during the start delay and when no LFO is configured.
    fn lfo_cents_at(&self, t: u32) -> f32 {
        if self.lfo_depth_cents == 0.0 || self.lfo_freq_hz == 0.0 {
            return 0.0;
        }
        if t < self.lfo_delay_samples {
            return 0.0;
        }
        let t_active = (t - self.lfo_delay_samples) as f32 / self.output_rate.max(1.0);
        let phase = t_active * self.lfo_freq_hz * std::f32::consts::TAU;
        phase.sin() * self.lfo_depth_cents
    }

    /// Linear-interpolate one frame at fractional position `phase`.
    /// Out-of-bounds reads return 0.
    fn fetch(&self, phase: f64) -> f32 {
        let i = phase.floor() as i64;
        if i < 0 || (i as usize) + 1 >= self.samples.len() {
            return 0.0;
        }
        let a = self.samples[i as usize];
        let b = self.samples[i as usize + 1];
        let frac = (phase - i as f64) as f32;
        a + (b - a) * frac
    }
}

impl Voice for SamplePlayer {
    fn render(&mut self, out: &mut [f32]) -> usize {
        if self.done {
            return 0;
        }
        for (i, slot) in out.iter_mut().enumerate() {
            // Recompute pitch each sample if pitch-bend or LFO is non-zero.
            let lfo_cents = self.lfo_cents_at(self.elapsed);
            if self.pitch_bend_cents != 0 || lfo_cents != 0.0 {
                let total_cents = self.pitch_bend_cents as f32 + lfo_cents;
                let bend_ratio = (2.0f64).powf(total_cents as f64 / 1200.0);
                self.phase_inc = self.base_phase_inc * bend_ratio;
            }

            let env = self.envelope_at(self.elapsed);
            if self.release_pos.is_some() && env <= 0.0 {
                self.done = true;
                return i;
            }

            // Loop / end handling.
            if self.phase >= self.end as f64 {
                // Past the very end of the sample → no signal, voice done
                // (unless we're in a loop mode that wraps).
                let should_wrap = matches!(
                    self.loop_mode,
                    SampleLoopMode::LoopContinuous | SampleLoopMode::LoopSustain
                ) && !self.loop_broken
                    && self.loop_end > self.loop_start;
                if should_wrap {
                    let over = self.phase - self.loop_end as f64;
                    let loop_len = (self.loop_end as f64 - self.loop_start as f64).max(1.0);
                    let wrapped = over.rem_euclid(loop_len);
                    self.phase = self.loop_start as f64 + wrapped;
                } else {
                    self.done = true;
                    return i;
                }
            } else if matches!(
                self.loop_mode,
                SampleLoopMode::LoopContinuous | SampleLoopMode::LoopSustain
            ) && !self.loop_broken
                && self.loop_end > self.loop_start
                && self.phase >= self.loop_end as f64
            {
                let over = self.phase - self.loop_end as f64;
                let loop_len = (self.loop_end as f64 - self.loop_start as f64).max(1.0);
                let wrapped = over.rem_euclid(loop_len);
                self.phase = self.loop_start as f64 + wrapped;
            }

            let s = self.fetch(self.phase);
            *slot = s * env * self.amplitude * self.pressure_gain;
            self.phase += self.phase_inc;
            self.elapsed = self.elapsed.wrapping_add(1);
        }
        out.len()
    }

    fn release(&mut self) {
        // OneShot ignores note-off — the sample plays to its natural
        // end and only then runs the release envelope.
        if matches!(self.loop_mode, SampleLoopMode::OneShot) {
            return;
        }
        if self.release_pos.is_none() {
            self.release_start_level = self.envelope_at(self.elapsed).max(0.0);
            self.release_pos = Some(self.elapsed);
            // LoopSustain breaks out of the loop on note-off.
            if matches!(self.loop_mode, SampleLoopMode::LoopSustain) {
                self.loop_broken = true;
            }
        }
    }

    fn done(&self) -> bool {
        self.done
    }

    fn set_pitch_bend_cents(&mut self, cents: i32) {
        self.pitch_bend_cents = cents;
    }

    fn set_pressure(&mut self, pressure: f32) {
        let p = pressure.clamp(0.0, 1.0);
        self.pressure_gain = 1.0 + 0.5 * p;
    }

    fn exclusive_class(&self) -> u16 {
        self.exclusive_class
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Arc<[f32]> {
        // -0.5 → +0.5 ramp.
        (0..n)
            .map(|i| (i as f32 / n.saturating_sub(1).max(1) as f32) - 0.5)
            .collect::<Vec<f32>>()
            .into()
    }

    #[test]
    fn no_loop_voice_finishes_when_sample_ends() {
        let buf = ramp(64);
        let len = buf.len() as u32;
        let cfg = SamplePlayerConfig {
            samples: buf,
            native_rate: 44_100,
            loop_start: 0,
            loop_end: len,
            sample_end: len,
            loop_mode: SampleLoopMode::NoLoop,
            pitch_ratio: 1.0,
            amplitude: 1.0,
            envelope: EnvelopeParams::default(),
            vibrato: VibratoParams::default(),
            exclusive_class: 0,
        };
        let mut v = SamplePlayer::new(cfg, 44_100);
        let mut out = vec![0.0f32; 256];
        let n = v.render(&mut out);
        // Sample is shorter than 256 frames so render must report an
        // early stop.
        assert!(n < 256, "expected early stop, got {n}");
        assert!(v.done());
    }

    #[test]
    fn looping_voice_runs_indefinitely_until_release() {
        let buf = ramp(32);
        let len = buf.len() as u32;
        let cfg = SamplePlayerConfig {
            samples: buf,
            native_rate: 44_100,
            loop_start: 0,
            loop_end: len,
            sample_end: len,
            loop_mode: SampleLoopMode::LoopContinuous,
            pitch_ratio: 1.0,
            amplitude: 1.0,
            envelope: EnvelopeParams::default(),
            vibrato: VibratoParams::default(),
            exclusive_class: 0,
        };
        let mut v = SamplePlayer::new(cfg, 44_100);
        let mut out = vec![0.0f32; 1024];
        let n = v.render(&mut out);
        assert_eq!(n, 1024, "looping voice should render the full buffer");
        assert!(!v.done());
        // Release: voice must finish within the release window
        // (100 ms ≈ 4410 samples). Render in chunks until done.
        v.release();
        let mut total = 0;
        for _ in 0..16 {
            let n = v.render(&mut out);
            total += n;
            if v.done() {
                break;
            }
        }
        assert!(v.done(), "voice should finish post-release");
        assert!(total > 0);
    }

    #[test]
    fn one_shot_ignores_release() {
        let buf = ramp(2048);
        let len = buf.len() as u32;
        let cfg = SamplePlayerConfig {
            samples: buf,
            native_rate: 44_100,
            loop_start: 0,
            loop_end: len,
            sample_end: len,
            loop_mode: SampleLoopMode::OneShot,
            pitch_ratio: 1.0,
            amplitude: 1.0,
            envelope: EnvelopeParams::default(),
            vibrato: VibratoParams::default(),
            exclusive_class: 0,
        };
        let mut v = SamplePlayer::new(cfg, 44_100);
        let mut out = vec![0.0f32; 256];
        v.render(&mut out);
        v.release();
        // Voice should still produce audio (release was ignored).
        let n = v.render(&mut out);
        assert_eq!(n, 256, "OneShot must ignore note-off");
        assert!(!v.done());
    }

    #[test]
    fn pitch_bend_changes_phase_inc() {
        let buf = ramp(16_384);
        let len = buf.len() as u32;
        let cfg = SamplePlayerConfig {
            samples: buf,
            native_rate: 44_100,
            loop_start: 0,
            loop_end: len,
            sample_end: len,
            loop_mode: SampleLoopMode::NoLoop,
            pitch_ratio: 1.0,
            amplitude: 1.0,
            envelope: EnvelopeParams::default(),
            vibrato: VibratoParams::default(),
            exclusive_class: 0,
        };
        let mut a = SamplePlayer::new(cfg.clone(), 44_100);
        let mut b = SamplePlayer::new(cfg, 44_100);
        b.set_pitch_bend_cents(1200); // +1 octave → 2x rate.
        let mut buf_a = vec![0.0f32; 1024];
        let mut buf_b = vec![0.0f32; 1024];
        a.render(&mut buf_a);
        b.render(&mut buf_b);
        // After 1024 output samples, A has advanced 1024 frames and B
        // has advanced ~2048 frames. The B buffer should reach a value
        // closer to the end of the ramp than A.
        let last_a = buf_a[1000];
        let last_b = buf_b[1000];
        assert!(
            last_b > last_a,
            "+1 octave bend should advance further into the ramp: a={last_a} b={last_b}"
        );
    }

    #[test]
    fn vibrato_lfo_modulates_pitch() {
        // A long ramp; with vibrato, two voices started in lockstep
        // should produce a different sample-by-sample output.
        let buf = ramp(16_384);
        let len = buf.len() as u32;
        let cfg_no_lfo = SamplePlayerConfig {
            samples: buf.clone(),
            native_rate: 44_100,
            loop_start: 0,
            loop_end: len,
            sample_end: len,
            loop_mode: SampleLoopMode::NoLoop,
            pitch_ratio: 1.0,
            amplitude: 1.0,
            envelope: EnvelopeParams::default(),
            vibrato: VibratoParams::default(),
            exclusive_class: 0,
        };
        let mut cfg_lfo = cfg_no_lfo.clone();
        cfg_lfo.vibrato = VibratoParams {
            freq_hz: 5.0,
            depth_cents: 100.0,
            delay_s: 0.0,
        };
        let mut a = SamplePlayer::new(cfg_no_lfo, 44_100);
        let mut b = SamplePlayer::new(cfg_lfo, 44_100);
        let mut out_a = vec![0.0f32; 4096];
        let mut out_b = vec![0.0f32; 4096];
        a.render(&mut out_a);
        b.render(&mut out_b);
        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(diff > 0.001, "LFO should perturb output, got diff {diff}");
    }
}
