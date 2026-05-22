//! Generic sample-playback voice shared between the SFZ and DLS
//! instrument adapters.
//!
//! Both formats reduce to the same shape at note-on time: pick a sample
//! buffer (already decoded to f32 mono frames), pick a playback rate
//! (derived from the requested MIDI key vs the sample's native key plus
//! coarse/fine tuning), pick a loop window (start/end frames + a
//! [`SampleLoopMode`]), pick a DAHDSR amplitude envelope, and (for SFZ)
//! a vibrato LFO. The SF2 path has its own [`Sf2Voice`](super::sf2::Sf2Voice)
//! because it carries extra state (native stereo zones, sm24 24-bit
//! samples) that the shared voice would have to optionalise; rather
//! than bloat one struct with mostly-unused state, we keep two voice
//! types side-by-side.
//!
//! The voice runs on **mono** sample data — both SFZ and DLS samples
//! that are conceptually mono are decoded to f32 mono before the voice
//! sees them. Stereo SFZ/DLS samples are deferred (a later round will
//! fan the decoder out to two channels and mark the voice as stereo).
//!
//! Articulation coverage:
//! - **DAHDSR amplitude envelope (EG1)** (always wired). Same shape as
//!   the `Sf2Voice` / `ToneVoice`: linear attack from 0→1, hold at peak,
//!   `(1-x)^2`-shaped decay to sustain, `(1-x)^2`-shaped release on
//!   note-off.
//! - **Vibrato LFO** (SFZ `lfo01_freq` / `lfo01_pitch` opcodes). A
//!   simple sine LFO routed into pitch as a cents offset; rate is in Hz
//!   and depth is in cents. Defaults to 0/0 so DLS / unconfigured SFZ
//!   regions emit no vibrato.
//! - **Pitch bend** via [`Voice::set_pitch_bend_cents`]. Layered on top
//!   of the LFO depth without modifying the base playback rate.
//! - **Modulation envelope (EG2)** + **2-pole resonant low-pass
//!   filter** (round 91). [`ModEnvParams`] carries the same DAHDSR
//!   shape as [`EnvelopeParams`] but with a `0..=1` sustain level (per
//!   SF2 §8.1.3 `sustainModEnv` — 0 = peak, 1 = silence). The mod-env
//!   is routed into the filter cutoff (SF2 gen 11 `modEnvToFilterFc` /
//!   DLS2 `SRC_EG2 → DST_FILTER_CUTOFF`) as a cents offset added to the
//!   initial cutoff each output frame. The biquad coefficients are
//!   computed via the RBJ low-pass cookbook with the SF2 v2.04 §8.1.3
//!   cents-to-Hz conversion `fc_hz = 8.176 * 2^(cents/1200)`; Q (in
//!   centibels) maps via `q_lin = sqrt(0.5) * 10^(cb/200)` so 0 cB ≈
//!   Butterworth.
//!
//! Both EG2 + filter are off by default — a [`ModEnvParams`] with
//! `attack_s == 0 && sustain_level == 0 && to_filter_cents == 0` skips
//! the per-sample mod-env compute, and a [`FilterParams`] left at its
//! default (`cutoff_cents == 13500`, the SF2 "filter inert" sentinel,
//! plus `kind == TwoPoleLowPass`) skips the biquad entirely so legacy
//! SFZ / unconfigured DLS regions stay bit-identical to the round-80
//! output.
//!
//! Round 95 generalises the biquad to all six SFZ v1 `fil_type` shapes
//! via [`FilterType`]: one-pole / two-pole low-pass / high-pass /
//! band-pass / band-reject. SF2 + DLS keep the round-91 two-pole
//! low-pass shape; SFZ patches with explicit `fil_type=` opcodes
//! select the matching variant.

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

/// Modulation envelope (EG2) parameters + routing depths. Per SF2 v2.04
/// §8.1.3 generators 25–30 and DLS Level 2.2 §1.5 EG2 destinations
/// (`CONN_DST_EG2_*`).
///
/// Shape mirrors [`EnvelopeParams`] (linear attack 0→1, hold at peak,
/// `(1-x)^2`-shaped decay toward `sustain_level`, `(1-x)^2`-shaped
/// release-to-zero on note-off). The semantic difference from EG1: the
/// SF2 spec defines `sustainModEnv` as a *decrease* from full scale in
/// 0.1 % units, so `sustain_level` here is the post-decrease linear
/// fraction (1.0 = stay at peak, 0.0 = decay all the way to zero).
///
/// Defaults match the SF2 "mod-env-inert" state: zero delay/attack/hold/
/// decay/release with `sustain_level = 0.0` and `to_filter_cents = 0`.
/// In that state [`SamplePlayer::render`] skips the per-sample mod-env
/// compute entirely.
#[derive(Clone, Copy, Debug)]
pub struct ModEnvParams {
    /// Pre-attack delay, seconds (SF2 gen 25 `delayModEnv` /
    /// DLS `CONN_DST_EG2_DELAYTIME`).
    pub delay_s: f32,
    /// Linear attack from 0→1, seconds (gen 26 `attackModEnv`).
    pub attack_s: f32,
    /// Hold-at-peak, seconds (gen 27 `holdModEnv`).
    pub hold_s: f32,
    /// Decay-to-sustain, seconds (gen 28 `decayModEnv`).
    pub decay_s: f32,
    /// Linear sustain level, `0.0..=1.0` (gen 29 `sustainModEnv`,
    /// converted from "0.1 % decrease from peak" → linear `1 - x/1000`).
    pub sustain_level: f32,
    /// Release-to-zero, seconds (gen 30 `releaseModEnv`).
    pub release_s: f32,
    /// Modulation-envelope → filter cutoff depth, in cents (SF2 gen 11
    /// `modEnvToFilterFc` / DLS `SRC_EG2 → DST_FILTER_CUTOFF`). The
    /// live cutoff is `filter.cutoff_cents + mod_env_level *
    /// to_filter_cents`.
    pub to_filter_cents: i32,
}

impl Default for ModEnvParams {
    fn default() -> Self {
        // SF2 "all generators at default" — mod-env is inert.
        Self {
            delay_s: 0.0,
            attack_s: 0.0,
            hold_s: 0.0,
            decay_s: 0.0,
            sustain_level: 0.0,
            release_s: 0.0,
            to_filter_cents: 0,
        }
    }
}

impl ModEnvParams {
    /// `true` when the mod-env is effectively a constant 0 — no filter
    /// routing, no per-sample state needed. Lets [`SamplePlayer::new`]
    /// skip allocating the mod-env counters and the per-sample compute.
    pub fn is_inert(&self) -> bool {
        self.to_filter_cents == 0
    }
}

/// Filter response shape. The SF2 / DLS path uses [`FilterType::LowPass2P`]
/// exclusively (both formats hard-code a 2-pole resonant low-pass); SFZ
/// patches can request any of the six SFZ v1 `fil_type` values via
/// [`FilterParams::kind`].
///
/// Round 95 adds the SFZ shapes: one-pole and two-pole low-pass /
/// high-pass / band-pass / band-reject. Coefficient derivation is the
/// project's own clean-room RBJ-cookbook math (bilinear transform of the
/// canonical analog prototypes); the SFZ "Aria reference" pages
/// (`sfz-legacy.html` "Filter type" table and `sfz-opcodes-index.html`
/// `fil_type` entry) only document the response *shape* and slope
/// (6 dB/oct one-pole, 12 dB/oct two-pole), not the discretisation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterType {
    /// `lpf_1p` — one-pole low-pass, 6 dB/octave. Resonance opcodes are
    /// ignored (one-pole sections have no resonant peak).
    OnePoleLowPass,
    /// `hpf_1p` — one-pole high-pass, 6 dB/octave. Resonance ignored.
    OnePoleHighPass,
    /// `lpf_2p` — two-pole resonant low-pass, 12 dB/octave (SFZ default,
    /// and the only response the SF2 / DLS biquad uses).
    #[default]
    TwoPoleLowPass,
    /// `hpf_2p` — two-pole resonant high-pass, 12 dB/octave.
    TwoPoleHighPass,
    /// `bpf_2p` — two-pole resonant band-pass, 12 dB/octave.
    TwoPoleBandPass,
    /// `brf_2p` — two-pole resonant band-reject (notch), 12 dB/octave.
    TwoPoleBandReject,
}

impl FilterType {
    /// Parse a SFZ `fil_type` value. Unknown strings fall back to
    /// `TwoPoleLowPass` per the SFZ-legacy default (`fil_type=lpf_2p`).
    pub fn parse_sfz(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "lpf_1p" => FilterType::OnePoleLowPass,
            "hpf_1p" => FilterType::OnePoleHighPass,
            "lpf_2p" => FilterType::TwoPoleLowPass,
            "hpf_2p" => FilterType::TwoPoleHighPass,
            "bpf_2p" => FilterType::TwoPoleBandPass,
            "brf_2p" => FilterType::TwoPoleBandReject,
            _ => FilterType::TwoPoleLowPass,
        }
    }

    /// `true` for the two single-pole responses (which ignore `q_centibels`
    /// and use a simpler one-pole IIR rather than a biquad).
    pub fn is_one_pole(self) -> bool {
        matches!(
            self,
            FilterType::OnePoleLowPass | FilterType::OnePoleHighPass
        )
    }
}

/// 2-pole resonant low-pass filter parameters. Per SF2 v2.04 §8.1.3
/// generators 8 + 9 ("second order resonant pole pair") and DLS Level
/// 2.2 §1.5.2 `CONN_DST_FILTER_CUTOFF` / `CONN_DST_FILTER_Q`.
///
/// `cutoff_cents` is in **absolute cents** with the SF2 reference of
/// 8.176 Hz: `fc_hz = 8.176 * 2^(cents/1200)`. Useful range per §8.1.3
/// is `1500..=13500` (20 Hz → 20 kHz). The default `13500` is the SF2
/// "filter open" sentinel — [`SamplePlayer::new`] detects this and
/// skips the biquad entirely.
///
/// `q_centibels` is in centibels (0.1 dB units), useful range
/// `0..=960` per §8.1.3 gen 9. `0` = Butterworth (`Q ≈ 0.707`).
///
/// `kind` selects the response shape (round 95). SF2 / DLS construct
/// `FilterParams` without touching this field and therefore inherit the
/// default `TwoPoleLowPass`, which preserves round-91 behaviour exactly.
/// SFZ patches that author `fil_type=` get the matching variant.
#[derive(Clone, Copy, Debug)]
pub struct FilterParams {
    /// Initial cutoff in absolute cents re. 8.176 Hz. Default 13500
    /// (≈ 19914 Hz, effectively no filter).
    pub cutoff_cents: i32,
    /// Resonance in centibels (10 cB = 1 dB). Default 0 (Butterworth).
    /// Ignored for the two one-pole filter kinds.
    pub q_centibels: i32,
    /// Response shape. Defaults to two-pole low-pass — the SF2 / DLS
    /// behaviour from round 91. SFZ regions with `fil_type=` opcodes
    /// override this.
    pub kind: FilterType,
}

impl Default for FilterParams {
    fn default() -> Self {
        // SF2 §8.1.3 default: initialFilterFc = 13500 (filter open),
        // initialFilterQ = 0 (no resonance). SFZ default fil_type=lpf_2p
        // matches the SF2 hard-coded shape, so the default `kind` is the
        // two-pole low-pass for both formats.
        Self {
            cutoff_cents: 13_500,
            q_centibels: 0,
            kind: FilterType::TwoPoleLowPass,
        }
    }
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
    /// Modulation envelope (EG2) + filter-routing depth. Defaults to
    /// inert — `is_inert()` true → the player skips the per-sample
    /// mod-env compute.
    pub mod_env: ModEnvParams,
    /// 2-pole resonant low-pass filter. Defaults to "open" — the
    /// player skips the biquad entirely.
    pub filter: FilterParams,
    /// Exclusive-class id (drum-kit style hi-hat open/closed cuts).
    /// Surfaced through [`Voice::exclusive_class`]; 0 = no class.
    pub exclusive_class: u16,
}

/// Sample-playback voice. Mono in, mono out — the mixer pans into
/// stereo. Covers: DAHDSR amplitude envelope + DAHDSR modulation
/// envelope (EG2) → cutoff routing + 2-pole resonant low-pass filter +
/// vibrato LFO + pitch bend + four loop modes + exclusive-class.
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

    // ---- Modulation envelope (EG2) ----
    /// EG2 DAHDSR phase boundaries in output frames.
    mod_env_delay: u32,
    mod_env_attack: u32,
    mod_env_hold: u32,
    mod_env_decay: u32,
    mod_env_release: u32,
    mod_env_sustain_level: f32,
    /// EG2 level at the moment of release — separately captured from
    /// the amplitude envelope so the filter cutoff tails off from its
    /// actual mid-flight value rather than re-starting from peak.
    mod_env_release_start_level: f32,
    /// EG2 → cutoff routing depth in cents.
    mod_env_to_filter_cents: i32,

    // ---- 2-pole resonant low-pass filter ----
    /// Initial cutoff in absolute cents (SF2 reference 8.176 Hz). Live
    /// cutoff = `initial_filter_fc_cents + mod_env * to_filter_cents`.
    initial_filter_fc_cents: i32,
    /// Resonance, centibels (0 = Butterworth).
    initial_filter_q_cb: i32,
    /// Filter response shape (round 95). SF2 / DLS keep
    /// [`FilterType::TwoPoleLowPass`]; SFZ patches with `fil_type=`
    /// opcodes select one of the six SFZ v1 shapes.
    filter_kind: FilterType,
    /// Per-voice biquad. `None` when the filter is "open" (cutoff at
    /// the SF2 default 13500 cents *and* no mod-env routing pulls it
    /// into the audible range) — skips the per-sample compute entirely.
    filter: Option<BiquadState>,
}

/// Direct-form 1 biquad state for one channel. Coefficients are
/// recomputed lazily when the cutoff drifts more than ~50 cents.
/// Mirrors the [`super::sf2::Sf2Voice`]'s biquad math so a DLS bank
/// that authors the same filter generators as an equivalent SF2 bank
/// produces an identical sweep shape.
struct BiquadState {
    /// Most-recent live cutoff in cents — used as the gate for
    /// coefficient recomputation.
    last_cutoff_cents: i32,
    /// Live coefficients (a1, a2, b0, b1, b2). a0 normalised to 1.
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    /// Delay line (x[n-1], x[n-2], y[n-1], y[n-2]).
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiquadState {
    fn new() -> Self {
        Self {
            // Sentinel forces the first call to `update_filter_coeffs`
            // to actually compute.
            last_cutoff_cents: i32::MIN,
            a1: 0.0,
            a2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// One sample through the direct-form-1 biquad.
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
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

        // Filter: only allocate the biquad when the bank actually wants
        // a filter. SF2 §8.1.3 default `initialFilterFc = 13500` is the
        // "filter open" sentinel; we also skip when the mod-env can't
        // drag it below 13000 cents (≈18 kHz) because anything above
        // that is well outside the audible band on a typical 44.1 kHz
        // render. The "open" shortcut only applies to low-pass shapes —
        // a high-pass / band-pass / band-reject filter is *not* inert at
        // 13500 cents, so any explicit non-low-pass `fil_type` selection
        // forces the biquad on.
        let non_lowpass = !matches!(
            cfg.filter.kind,
            FilterType::TwoPoleLowPass | FilterType::OnePoleLowPass
        );
        let needs_filter = cfg.filter.cutoff_cents < 13_000
            || cfg.mod_env.to_filter_cents.abs() > 200
            || non_lowpass;
        let filter = if needs_filter {
            Some(BiquadState::new())
        } else {
            None
        };

        // Mod-env sustain is "fraction of full-scale" — clamped at
        // construction so the render loop doesn't have to.
        let mod_env_sustain_level = cfg.mod_env.sustain_level.clamp(0.0, 1.0);

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

            mod_env_delay: (sr * cfg.mod_env.delay_s.max(0.0)) as u32,
            mod_env_attack: (sr * cfg.mod_env.attack_s.max(0.0)).max(1.0) as u32,
            mod_env_hold: (sr * cfg.mod_env.hold_s.max(0.0)) as u32,
            mod_env_decay: (sr * cfg.mod_env.decay_s.max(0.0)).max(1.0) as u32,
            mod_env_release: (sr * cfg.mod_env.release_s.max(0.0)).max(1.0) as u32,
            mod_env_sustain_level,
            mod_env_release_start_level: 1.0,
            mod_env_to_filter_cents: cfg.mod_env.to_filter_cents,

            initial_filter_fc_cents: cfg.filter.cutoff_cents,
            initial_filter_q_cb: cfg.filter.q_centibels,
            filter_kind: cfg.filter.kind,
            filter,
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

    /// Modulation-envelope (EG2) value at output frame `t`, `0..=1`.
    /// Same DAHDSR shape as [`Self::envelope_at`] but with `0..=1`
    /// sustain semantics (per SF2 §8.1.3 sustainModEnv). Release uses
    /// the EG2's own captured start level so the filter sweep tails off
    /// from wherever the cutoff was at note-off, not from peak.
    fn mod_env_at(&self, t: u32) -> f32 {
        if let Some(rel_at) = self.release_pos {
            let since = t.saturating_sub(rel_at);
            if since >= self.mod_env_release {
                return 0.0;
            }
            let x = since as f32 / self.mod_env_release.max(1) as f32;
            let curve = (1.0 - x) * (1.0 - x);
            return self.mod_env_release_start_level * curve;
        }
        if t < self.mod_env_delay {
            return 0.0;
        }
        let t = t - self.mod_env_delay;
        if t < self.mod_env_attack {
            return t as f32 / self.mod_env_attack.max(1) as f32;
        }
        let t = t - self.mod_env_attack;
        if t < self.mod_env_hold {
            return 1.0;
        }
        let t = t - self.mod_env_hold;
        if t < self.mod_env_decay {
            let x = t as f32 / self.mod_env_decay.max(1) as f32;
            let drop = 1.0 - self.mod_env_sustain_level;
            let curve = 1.0 - (1.0 - x) * (1.0 - x);
            return 1.0 - drop * curve;
        }
        self.mod_env_sustain_level
    }

    /// Recompute biquad coefficients for the configured
    /// [`FilterType`] at the given absolute-cents cutoff. Direct-form 1,
    /// RBJ-cookbook math (clean-room derivation; the SF2 spec only
    /// specifies "second order resonant pole pair" per §8.1.3 + §9.7,
    /// and the SFZ-legacy reference's `fil_type` table likewise only
    /// names the response shape + 6/12 dB-per-octave slope without
    /// specifying the discrete realisation).
    ///
    /// `cutoff_cents` is clamped to the SF2 useful range
    /// `1500..=13500` (§8.1.3), then converted to Hz via
    /// `fc_hz = 8.176 * 2^(cents/1200)` and further clipped at
    /// `0.99 * Nyquist` so a hostile bank can't request a cutoff that
    /// the bi-linear transform can't represent.
    ///
    /// One-pole shapes (`OnePoleLowPass`, `OnePoleHighPass`) bypass the
    /// resonance term entirely (a one-pole section has no resonant
    /// peak) and write `b2 = a2 = 0` so the same `tick()` math still
    /// applies — keeping the per-sample inner loop branch-free.
    fn update_filter_coeffs(&mut self, cutoff_cents: i32) {
        let Some(filter) = self.filter.as_mut() else {
            return;
        };
        let cents = cutoff_cents.clamp(1_500, 13_500);
        let cutoff_hz = 8.176_f32 * (2.0_f32).powf(cents as f32 / 1200.0);
        let nyquist = self.output_rate * 0.5;
        let cutoff_hz = cutoff_hz.min(nyquist * 0.99).max(20.0);

        match self.filter_kind {
            FilterType::OnePoleLowPass => {
                // RBJ-style one-pole low-pass via bilinear of `1/(s+ω₀)`:
                //   y[n] = b0·x[n] + b1·x[n-1] - a1·y[n-1]
                // Normalised so DC gain = 1. `a2 = b2 = 0` so the
                // direct-form-1 biquad math reduces to the one-pole
                // recurrence on its own.
                let omega = 2.0 * std::f32::consts::PI * cutoff_hz / self.output_rate;
                // tan(ω/2) prewarp gives the canonical bilinear
                // low-pass coefficients.
                let k = (omega * 0.5).tan();
                let norm = 1.0 / (1.0 + k);
                filter.b0 = k * norm;
                filter.b1 = k * norm;
                filter.b2 = 0.0;
                filter.a1 = (k - 1.0) * norm;
                filter.a2 = 0.0;
            }
            FilterType::OnePoleHighPass => {
                // Bilinear of `s/(s+ω₀)` — DC gain = 0, Nyquist gain = 1.
                let omega = 2.0 * std::f32::consts::PI * cutoff_hz / self.output_rate;
                let k = (omega * 0.5).tan();
                let norm = 1.0 / (1.0 + k);
                filter.b0 = norm;
                filter.b1 = -norm;
                filter.b2 = 0.0;
                filter.a1 = (k - 1.0) * norm;
                filter.a2 = 0.0;
            }
            FilterType::TwoPoleLowPass
            | FilterType::TwoPoleHighPass
            | FilterType::TwoPoleBandPass
            | FilterType::TwoPoleBandReject => {
                // Q (cB) → linear Q. 0 cB = Butterworth (≈ 0.707). Clamp
                // at 16 to bound resonance — a runaway Q produces NaN
                // coefficients.
                let q_lin = (10.0_f32).powf(self.initial_filter_q_cb as f32 / 200.0)
                    * std::f32::consts::FRAC_1_SQRT_2;
                let q = q_lin.clamp(0.1, 16.0);
                let omega = 2.0 * std::f32::consts::PI * cutoff_hz / self.output_rate;
                let sin_w = omega.sin();
                let cos_w = omega.cos();
                let alpha = sin_w / (2.0 * q);
                let a0 = 1.0 + alpha;
                let inv_a0 = 1.0 / a0;
                // Numerators differ per response (RBJ cookbook); the
                // denominators are shared across all four 2-pole shapes.
                let (b0, b1, b2) = match self.filter_kind {
                    FilterType::TwoPoleLowPass => {
                        let v = (1.0 - cos_w) * 0.5;
                        (v, 1.0 - cos_w, v)
                    }
                    FilterType::TwoPoleHighPass => {
                        let v = (1.0 + cos_w) * 0.5;
                        (v, -(1.0 + cos_w), v)
                    }
                    FilterType::TwoPoleBandPass => {
                        // "Constant skirt gain, peak gain = Q" form.
                        (sin_w * 0.5, 0.0, -sin_w * 0.5)
                    }
                    FilterType::TwoPoleBandReject => (1.0, -2.0 * cos_w, 1.0),
                    // Unreachable — outer match already covered the
                    // one-pole arms.
                    FilterType::OnePoleLowPass | FilterType::OnePoleHighPass => (1.0, 0.0, 0.0),
                };
                filter.b0 = b0 * inv_a0;
                filter.b1 = b1 * inv_a0;
                filter.b2 = b2 * inv_a0;
                filter.a1 = (-2.0 * cos_w) * inv_a0;
                filter.a2 = (1.0 - alpha) * inv_a0;
            }
        }
        filter.last_cutoff_cents = cutoff_cents;
    }

    /// Live filter cutoff at output frame `t`, in absolute cents.
    /// Adds the mod-env contribution to the initial cutoff.
    fn live_cutoff_cents(&self, t: u32) -> i32 {
        if self.mod_env_to_filter_cents == 0 {
            return self.initial_filter_fc_cents;
        }
        let lvl = self.mod_env_at(t);
        self.initial_filter_fc_cents + (lvl * self.mod_env_to_filter_cents as f32) as i32
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

            // Filter cutoff modulation. Only recompute coefficients
            // when the cutoff drifts more than ~50 cents from the last
            // computed value — a perceptually-imperceptible gate that
            // keeps the inner loop multiplication-only on the common
            // path (mod-env is constant in delay/hold/sustain phases).
            if self.filter.is_some() {
                let target = self.live_cutoff_cents(self.elapsed);
                let last = self
                    .filter
                    .as_ref()
                    .map(|f| f.last_cutoff_cents)
                    .unwrap_or(i32::MIN);
                if target.saturating_sub(last).saturating_abs() > 50 {
                    self.update_filter_coeffs(target);
                }
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

            let mut s = self.fetch(self.phase);
            if let Some(filter) = self.filter.as_mut() {
                s = filter.tick(s);
            }
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
            // Capture the EG2 level too so the filter cutoff tails
            // off from its mid-flight value, not from peak.
            self.mod_env_release_start_level = self.mod_env_at(self.elapsed).max(0.0);
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
            mod_env: ModEnvParams::default(),
            filter: FilterParams::default(),
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
            mod_env: ModEnvParams::default(),
            filter: FilterParams::default(),
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
            mod_env: ModEnvParams::default(),
            filter: FilterParams::default(),
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
            mod_env: ModEnvParams::default(),
            filter: FilterParams::default(),
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
            mod_env: ModEnvParams::default(),
            filter: FilterParams::default(),
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

    // ---------------- Round 91 EG2 + filter tests ----------------

    /// Square-wave-shaped buffer, useful for filter tests: a long input
    /// with strong high-frequency content. Alternates +/-0.5 every
    /// `period` samples so a low-pass filter has plenty of harmonics to
    /// chew on.
    fn square(period: usize, frames: usize) -> Arc<[f32]> {
        (0..frames)
            .map(|i| if (i / period) & 1 == 0 { 0.5 } else { -0.5 })
            .collect::<Vec<f32>>()
            .into()
    }

    /// Long sine at a known frequency. Used to validate that the
    /// low-pass filter actually attenuates frequencies above its
    /// cutoff. `freq_hz` × `frames / native_rate` should be a multiple
    /// of 1 so the buffer is integer-periodic (avoids edge-frame
    /// ringing).
    fn sine(freq_hz: f32, native_rate: u32, frames: usize) -> Arc<[f32]> {
        let step = std::f32::consts::TAU * freq_hz / native_rate as f32;
        (0..frames)
            .map(|i| (i as f32 * step).sin() * 0.5)
            .collect::<Vec<f32>>()
            .into()
    }

    fn rms(buf: &[f32]) -> f32 {
        let n = buf.len().max(1) as f32;
        let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
        (sum_sq / n).sqrt()
    }

    fn cfg_with(samples: Arc<[f32]>, native_rate: u32) -> SamplePlayerConfig {
        let len = samples.len() as u32;
        SamplePlayerConfig {
            samples,
            native_rate,
            loop_start: 0,
            loop_end: len,
            sample_end: len,
            loop_mode: SampleLoopMode::NoLoop,
            pitch_ratio: 1.0,
            amplitude: 1.0,
            // Long attack so the amplitude envelope doesn't shape the
            // output during the filter measurement window — keep the
            // signal at near-peak the whole time.
            envelope: EnvelopeParams {
                attack_s: 0.001,
                hold_s: 1.0,
                decay_s: 0.001,
                sustain_level: 1.0,
                release_s: 0.1,
                ..Default::default()
            },
            vibrato: VibratoParams::default(),
            mod_env: ModEnvParams::default(),
            filter: FilterParams::default(),
            exclusive_class: 0,
        }
    }

    #[test]
    fn default_filter_is_inert_no_biquad_allocated() {
        // SF2 §8.1.3 "filter open" defaults must skip the biquad
        // entirely so legacy SFZ / unconfigured DLS banks render
        // bit-identically to the round-80 pre-filter path.
        let cfg = cfg_with(square(8, 4096), 44_100);
        let v = SamplePlayer::new(cfg, 44_100);
        assert!(
            v.filter.is_none(),
            "default FilterParams + inert ModEnvParams should not allocate biquad"
        );
    }

    #[test]
    fn low_cutoff_filter_attenuates_high_frequencies() {
        // A 5 kHz sine through a low-pass at 500 Hz should be
        // dramatically attenuated; through an open filter (default),
        // it should pass essentially unchanged. We measure RMS on the
        // tail of the buffer so the filter's transient response has
        // had time to settle.
        let buf = sine(5_000.0, 44_100, 8_192);
        // Open-filter baseline.
        let cfg_open = cfg_with(buf.clone(), 44_100);
        let mut v_open = SamplePlayer::new(cfg_open, 44_100);
        let mut out_open = vec![0.0f32; 8_192];
        v_open.render(&mut out_open);

        // 500 Hz cutoff: cents = 1200 * log2(500 / 8.176) ≈ 7138.
        let mut cfg_lp = cfg_with(buf, 44_100);
        cfg_lp.filter = FilterParams {
            cutoff_cents: 7_138,
            q_centibels: 0,
            ..Default::default()
        };
        let mut v_lp = SamplePlayer::new(cfg_lp, 44_100);
        assert!(
            v_lp.filter.is_some(),
            "low-cutoff config must allocate biquad"
        );
        let mut out_lp = vec![0.0f32; 8_192];
        v_lp.render(&mut out_lp);

        // Measure RMS over the latter half so the filter transient
        // has settled. A 2-pole low-pass at 500 Hz against a 5 kHz
        // sine should attenuate by roughly 40 dB (12 dB/oct over
        // ~3.3 octaves), so RMS ratio < 0.05 is a safe assertion.
        let rms_open = rms(&out_open[4_096..]);
        let rms_lp = rms(&out_lp[4_096..]);
        let ratio = rms_lp / rms_open.max(1e-6);
        assert!(
            ratio < 0.1,
            "low-pass at 500 Hz should attenuate 5 kHz by >20 dB; got ratio {ratio} (open={rms_open}, lp={rms_lp})"
        );
    }

    #[test]
    fn mod_env_dahdsr_shape_matches_spec() {
        // Build a voice with a known mod-env: 0 ms delay, 100 ms
        // attack, 0 ms hold, 100 ms decay to sustain 0.5. Sample at
        // a few known time points and assert the curve shape.
        let buf = sine(440.0, 44_100, 44_100);
        let mut cfg = cfg_with(buf, 44_100);
        cfg.mod_env = ModEnvParams {
            delay_s: 0.0,
            attack_s: 0.100,
            hold_s: 0.0,
            decay_s: 0.100,
            sustain_level: 0.5,
            release_s: 0.100,
            // Non-zero filter routing forces the SamplePlayer to
            // actually compute mod_env_at every sample.
            to_filter_cents: -3000,
        };
        // Low initial cutoff so the biquad is allocated.
        cfg.filter = FilterParams {
            cutoff_cents: 10_000,
            q_centibels: 0,
            ..Default::default()
        };
        let v = SamplePlayer::new(cfg, 44_100);

        // At t=0: pre-attack, level = 0.
        assert!(
            v.mod_env_at(0).abs() < 1e-3,
            "mod env at t=0 should be 0, got {}",
            v.mod_env_at(0)
        );
        // Mid-attack (50 ms in, half-way through 100 ms attack): ~0.5.
        let mid_attack = v.mod_env_at(44_100 / 20);
        assert!(
            (mid_attack - 0.5).abs() < 0.02,
            "mod env mid-attack should be ~0.5, got {mid_attack}"
        );
        // End of attack (100 ms): peak = 1.0.
        let peak = v.mod_env_at(44_100 / 10);
        assert!(
            (peak - 1.0).abs() < 0.02,
            "mod env at attack peak should be ~1.0, got {peak}"
        );
        // Late in sustain (500 ms): held at sustain_level = 0.5.
        let sus = v.mod_env_at(44_100 / 2);
        assert!(
            (sus - 0.5).abs() < 0.02,
            "mod env in sustain should be ~0.5, got {sus}"
        );
    }

    #[test]
    fn eg2_filter_sweep_changes_spectrum_over_note() {
        // Drive a 5 kHz sine through a voice with EG2 sweeping the
        // filter cutoff from ~closed → ~open over the note. The early
        // part of the rendered audio should be heavily attenuated;
        // the late part should be near pass-through. A constant-cutoff
        // baseline (no EG2 routing) should produce the same RMS over
        // both windows.
        //
        // Sine is chosen well above the starting cutoff (~250 Hz) so
        // the early window sits in the filter's stop-band; the EG2
        // attack runs slowly enough that the early-window measurement
        // catches the filter still nearly closed.
        let buf = sine(5_000.0, 44_100, 44_100);

        // Baseline: filter at 250 Hz, no EG2 routing.
        // cents = 1200 * log2(250 / 8.176) ≈ 5938.
        let base_cutoff = 5_938;
        let mut cfg_base = cfg_with(buf.clone(), 44_100);
        cfg_base.filter = FilterParams {
            cutoff_cents: base_cutoff,
            q_centibels: 0,
            ..Default::default()
        };
        let mut v_base = SamplePlayer::new(cfg_base, 44_100);
        let mut out_base = vec![0.0f32; 44_100];
        v_base.render(&mut out_base);

        // EG2 sweep: cutoff starts at 250 Hz, rises +6 octaves over a
        // 500 ms attack (7200 cents → cutoff lands at ~16 kHz, opening
        // the filter well above the 5 kHz probe). Sustain at peak
        // (1.0) so the filter stays open after the attack.
        let mut cfg_sweep = cfg_with(buf, 44_100);
        cfg_sweep.filter = FilterParams {
            cutoff_cents: base_cutoff,
            q_centibels: 0,
            ..Default::default()
        };
        cfg_sweep.mod_env = ModEnvParams {
            delay_s: 0.0,
            attack_s: 0.500,
            hold_s: 1.0,
            decay_s: 0.001,
            sustain_level: 1.0,
            release_s: 0.001,
            to_filter_cents: 7_200,
        };
        let mut v_sweep = SamplePlayer::new(cfg_sweep, 44_100);
        let mut out_sweep = vec![0.0f32; 44_100];
        v_sweep.render(&mut out_sweep);

        // Early window: 20–80 ms in. Mod-env is ~0.04..0.16 of attack
        // (linear ramp over 500 ms), so cutoff sits ~290..380 Hz.
        // 5 kHz through a 380 Hz lowpass is heavily attenuated.
        //
        // Late window: 850–950 ms in — well past the 500 ms attack,
        // mod-env saturated at sustain=1.0, cutoff at the +7200 cent
        // ceiling. 5 kHz now passes essentially unattenuated.
        let early_sweep = rms(&out_sweep[880..3_528]);
        let late_sweep = rms(&out_sweep[37_485..41_895]);
        let early_base = rms(&out_base[880..3_528]);
        let late_base = rms(&out_base[37_485..41_895]);

        // Sanity: the baseline (static cutoff) RMS in the two windows
        // should be roughly equal once the filter has settled.
        let base_ratio = (late_base / early_base.max(1e-6) - 1.0).abs();
        assert!(
            base_ratio < 0.3,
            "static-filter baseline should be roughly steady; got early={early_base}, late={late_base}"
        );

        // Filter-sweep test: late-window RMS must dominate early-
        // window RMS by a wide margin (the EG2 has pulled the cutoff
        // from below the probe frequency to above it).
        let sweep_gain = late_sweep / early_sweep.max(1e-6);
        assert!(
            sweep_gain > 5.0,
            "EG2 should sweep the cutoff open, raising late-window RMS over early-window RMS; got gain={sweep_gain} (early={early_sweep}, late={late_sweep})"
        );

        // Cross-check: the post-sweep filter must pass the 5 kHz
        // probe more freely than the static-cutoff baseline.
        assert!(
            late_sweep > late_base * 2.0,
            "post-sweep filter should be substantially wider than the static 250 Hz baseline; got sweep={late_sweep}, base={late_base}"
        );
    }

    #[test]
    fn high_q_filter_resonates_at_cutoff() {
        // A high-Q low-pass should produce a measurable peak at the
        // cutoff frequency. We probe with a sine sitting *at* the
        // cutoff: high-Q should make the output louder than the input,
        // low-Q (Butterworth) should attenuate it (-3 dB at fc).
        let buf = sine(1_000.0, 44_100, 16_384);

        // 1 kHz cutoff, Q = 0 (Butterworth).
        let mut cfg_q0 = cfg_with(buf.clone(), 44_100);
        cfg_q0.filter = FilterParams {
            cutoff_cents: 8_338,
            q_centibels: 0,
            ..Default::default()
        };
        let mut v_q0 = SamplePlayer::new(cfg_q0, 44_100);
        let mut out_q0 = vec![0.0f32; 16_384];
        v_q0.render(&mut out_q0);

        // 1 kHz cutoff, Q = 240 cB = 24 dB resonance peak.
        let mut cfg_q24 = cfg_with(buf, 44_100);
        cfg_q24.filter = FilterParams {
            cutoff_cents: 8_338,
            q_centibels: 240,
            ..Default::default()
        };
        let mut v_q24 = SamplePlayer::new(cfg_q24, 44_100);
        let mut out_q24 = vec![0.0f32; 16_384];
        v_q24.render(&mut out_q24);

        // Measure RMS on the tail so the resonant transient has
        // settled into steady state.
        let rms_q0 = rms(&out_q0[8_192..]);
        let rms_q24 = rms(&out_q24[8_192..]);
        assert!(
            rms_q24 > rms_q0 * 2.0,
            "Q=24 dB should boost the cutoff-frequency sine over Q=0; got q0={rms_q0}, q24={rms_q24}"
        );
    }

    #[test]
    fn release_captures_mod_env_level() {
        // After release, the EG2 should tail off from its mid-flight
        // value, not restart from peak. We test this by checking the
        // captured `mod_env_release_start_level` after `release()`
        // matches the pre-release `mod_env_at`.
        let buf = sine(440.0, 44_100, 44_100);
        let mut cfg = cfg_with(buf, 44_100);
        cfg.mod_env = ModEnvParams {
            attack_s: 0.100,
            decay_s: 0.100,
            sustain_level: 0.5,
            release_s: 0.100,
            to_filter_cents: -3000,
            ..Default::default()
        };
        cfg.filter = FilterParams {
            cutoff_cents: 10_000,
            q_centibels: 0,
            ..Default::default()
        };
        let mut v = SamplePlayer::new(cfg, 44_100);
        // Render 50 ms — mid-attack on the mod-env.
        let mut out = vec![0.0f32; 44_100 / 20];
        v.render(&mut out);
        let pre_release_level = v.mod_env_at(v.elapsed);
        v.release();
        // Captured level must match the pre-release reading.
        assert!(
            (v.mod_env_release_start_level - pre_release_level).abs() < 1e-4,
            "release should capture mid-flight EG2 level; pre={pre_release_level}, captured={}",
            v.mod_env_release_start_level
        );
        // And the captured level should be roughly 0.5 (mid-attack
        // on a 100 ms linear ramp at 50 ms in).
        assert!(
            (pre_release_level - 0.5).abs() < 0.05,
            "mid-attack EG2 should be ~0.5; got {pre_release_level}"
        );
    }

    // ---------------------------------------------------------------------
    // Round 95: filter-kind coverage on the SamplePlayer biquad.
    // ---------------------------------------------------------------------

    /// Build a sine at `freq_hz`, sampled at `sr`, as an `Arc<[f32]>`
    /// ready for [`SamplePlayerConfig`].
    fn sine_buf(freq_hz: f32, sr: u32, len: usize) -> Arc<[f32]> {
        let v: Vec<f32> = (0..len)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect();
        Arc::from(v.into_boxed_slice())
    }

    #[test]
    fn high_pass_attenuates_below_cutoff() {
        // 5 kHz sine through a 10 kHz two-pole high-pass: most of the
        // signal sits in the stop-band.
        let sr = 44_100;
        let buf = sine_buf(5_000.0, sr, 16_384);
        let mut cfg = cfg_with(buf, sr);
        // 10 kHz cutoff: cents = 1200 * log2(10000 / 8.176) ≈ 12331.
        cfg.filter = FilterParams {
            cutoff_cents: 12_331,
            q_centibels: 0,
            kind: FilterType::TwoPoleHighPass,
        };
        let mut v = SamplePlayer::new(cfg, sr);
        assert!(
            v.filter.is_some(),
            "non-low-pass must always allocate biquad"
        );
        let mut out = vec![0.0f32; 16_384];
        v.render(&mut out);
        let r = rms(&out[8_192..]);
        // Open-filter baseline (FilterType default = low-pass at the
        // 13500 sentinel → no biquad → unfiltered output).
        let buf2 = sine_buf(5_000.0, sr, 16_384);
        let cfg_open = cfg_with(buf2, sr);
        let mut v_open = SamplePlayer::new(cfg_open, sr);
        let mut out_open = vec![0.0f32; 16_384];
        v_open.render(&mut out_open);
        let r_open = rms(&out_open[8_192..]);
        assert!(
            r < r_open * 0.5,
            "10 kHz hpf_2p should attenuate 5 kHz; got r={r}, open={r_open}",
        );
    }

    #[test]
    fn band_pass_peaks_at_cutoff() {
        // Two-pole band-pass: a sine at the cutoff frequency should
        // pass with less loss than a sine an octave below.
        let sr = 44_100;
        let cutoff_cents = 8_338; // ≈ 1 kHz
                                  // Sine at the cutoff.
        let mut cfg_on = cfg_with(sine_buf(1_000.0, sr, 16_384), sr);
        cfg_on.filter = FilterParams {
            cutoff_cents,
            q_centibels: 60,
            kind: FilterType::TwoPoleBandPass,
        };
        let mut v_on = SamplePlayer::new(cfg_on, sr);
        let mut out_on = vec![0.0f32; 16_384];
        v_on.render(&mut out_on);
        // Sine an octave below the cutoff.
        let mut cfg_off = cfg_with(sine_buf(250.0, sr, 16_384), sr);
        cfg_off.filter = FilterParams {
            cutoff_cents,
            q_centibels: 60,
            kind: FilterType::TwoPoleBandPass,
        };
        let mut v_off = SamplePlayer::new(cfg_off, sr);
        let mut out_off = vec![0.0f32; 16_384];
        v_off.render(&mut out_off);
        let r_on = rms(&out_on[8_192..]);
        let r_off = rms(&out_off[8_192..]);
        assert!(
            r_on > r_off * 2.0,
            "bpf_2p should peak at cutoff; got at_cutoff={r_on}, two_octaves_off={r_off}",
        );
    }

    #[test]
    fn band_reject_kills_signal_at_cutoff() {
        // Two-pole band-reject (notch): a sine at the cutoff should be
        // heavily attenuated; an out-of-band sine should pass.
        let sr = 44_100;
        let cutoff_cents = 8_338; // ≈ 1 kHz
                                  // Sine at the cutoff (in the notch).
        let mut cfg_in = cfg_with(sine_buf(1_000.0, sr, 16_384), sr);
        cfg_in.filter = FilterParams {
            cutoff_cents,
            q_centibels: 240,
            kind: FilterType::TwoPoleBandReject,
        };
        let mut v_in = SamplePlayer::new(cfg_in, sr);
        let mut out_in = vec![0.0f32; 16_384];
        v_in.render(&mut out_in);
        // Sine well below the notch.
        let mut cfg_lo = cfg_with(sine_buf(100.0, sr, 16_384), sr);
        cfg_lo.filter = FilterParams {
            cutoff_cents,
            q_centibels: 240,
            kind: FilterType::TwoPoleBandReject,
        };
        let mut v_lo = SamplePlayer::new(cfg_lo, sr);
        let mut out_lo = vec![0.0f32; 16_384];
        v_lo.render(&mut out_lo);
        let r_in = rms(&out_in[8_192..]);
        let r_lo = rms(&out_lo[8_192..]);
        assert!(
            r_in < r_lo * 0.3,
            "brf_2p should null the cutoff sine; got at_cutoff={r_in}, off_band={r_lo}",
        );
    }

    #[test]
    fn one_pole_low_pass_attenuates_high_frequencies() {
        // 5 kHz sine through a 1-pole 500 Hz low-pass. Slope is 6 dB/oct
        // so over ~3.3 octaves we expect ~20 dB attenuation.
        let sr = 44_100;
        let buf = sine_buf(5_000.0, sr, 16_384);
        let mut cfg = cfg_with(buf, sr);
        cfg.filter = FilterParams {
            cutoff_cents: 7_138,
            q_centibels: 0,
            kind: FilterType::OnePoleLowPass,
        };
        let mut v = SamplePlayer::new(cfg, sr);
        let mut out = vec![0.0f32; 16_384];
        v.render(&mut out);
        let r = rms(&out[8_192..]);
        // Baseline: same sine, no filter.
        let buf2 = sine_buf(5_000.0, sr, 16_384);
        let mut v_open = SamplePlayer::new(cfg_with(buf2, sr), sr);
        let mut out_open = vec![0.0f32; 16_384];
        v_open.render(&mut out_open);
        let r_open = rms(&out_open[8_192..]);
        assert!(
            r < r_open * 0.3,
            "lpf_1p at 500 Hz should attenuate 5 kHz; got r={r}, open={r_open}",
        );
    }
}
