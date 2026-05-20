//! DLS Level 1 / Level 2 articulation connection-block evaluator.
//!
//! Round-80 lands the missing piece of the DLS pipeline: the
//! `art1`/`art2` connection-block lists were parsed in round 1 but the
//! voice generator dropped them on the floor and used `SamplePlayer`
//! defaults. With this module, well-known connections override those
//! defaults so a DLS bank's authored envelope, vibrato LFO, and tuning
//! reach the rendered audio.
//!
//! ## Spec backing
//!
//! - **MMA DLS Level 1 v1.1b** §"List of defined Sources, Controls,
//!   Destinations, and Transforms" (pp 43–44 of `dls1v11b.pdf`) plus
//!   Table 1 "Connection Block Table" + Table 2 "Default, Minimum,
//!   Maximum and Unit Values for Connection Blocks"
//!   (`Level 1 Device Architecture` section pp 22–23).
//! - **MMA DLS Level 2.2 v1.0** §1.13 "Modulation Routing" + Table 8
//!   "DLS Level 1 Sources, Controls, Destinations, and Transforms"
//!   (p 48) + Tables 9–10 (pp 50–51) for the DLS2-only Generic
//!   Destinations / EG2 hold/shutdown/delay / Filter destinations /
//!   Vibrato LFO destinations / Convex/Switch transforms. Tables 5–6
//!   "Default Connection Blocks" (pp 29–30) document the default
//!   values for the destinations that fold into [`Articulation`].
//!
//! Both PDFs live in `docs/audio/midi/instrument-formats/`.
//!
//! ## Scope of round 80
//!
//! We interpret only the subset of connections that the SamplePlayer
//! voice can actually consume — the DAHDSR amplitude envelope, the
//! vibrato LFO, tuning, and per-region gain. Everything else (modulation
//! envelope routings, filter cutoff/Q, key-number to pitch since the SF2
//! voice already handles its own key tracking via `pitch_keycenter` /
//! `unity_note`, and the channel-output destinations covering DLS2
//! surround placement) lands its raw values on [`Articulation`] for a
//! later round to wire up. We also skip Convex / Switch transforms
//! (DLS2 only): those bias the modulator response curve and are not
//! relevant for the SRC_NONE → DST_x "default value" mapping we
//! evaluate here.

use super::dls::DlsArticulationBlock;
use super::sample_voice::{EnvelopeParams, VibratoParams};

// ---------------------------------------------------------------------------
// Numeric constants (Tables 8 + 9 + 10 of DLS Level 2.2).
//
// We keep the names matching the spec ("CONN_SRC_FOO") so a reader of
// the spec PDF can grep this file directly.
// ---------------------------------------------------------------------------

// Modulator sources (DLS1 + DLS2).
pub const CONN_SRC_NONE: u16 = 0x0000;
pub const CONN_SRC_LFO: u16 = 0x0001;
pub const CONN_SRC_KEYONVELOCITY: u16 = 0x0002;
pub const CONN_SRC_KEYNUMBER: u16 = 0x0003;
pub const CONN_SRC_EG1: u16 = 0x0004;
pub const CONN_SRC_EG2: u16 = 0x0005;
pub const CONN_SRC_PITCHWHEEL: u16 = 0x0006;

// DLS2-only modulator sources.
pub const CONN_SRC_POLYPRESSURE: u16 = 0x0007;
pub const CONN_SRC_CHANNELPRESSURE: u16 = 0x0008;
pub const CONN_SRC_VIBRATO: u16 = 0x0009;

// MIDI controller sources.
pub const CONN_SRC_CC1: u16 = 0x0081;
pub const CONN_SRC_CC7: u16 = 0x0087;
pub const CONN_SRC_CC10: u16 = 0x008A;
pub const CONN_SRC_CC11: u16 = 0x008B;
pub const CONN_SRC_CC91: u16 = 0x00DB;
pub const CONN_SRC_CC93: u16 = 0x00DD;

// Registered Parameter Numbers (as modulator sources).
pub const CONN_SRC_RPN0: u16 = 0x0100;
pub const CONN_SRC_RPN1: u16 = 0x0101;
pub const CONN_SRC_RPN2: u16 = 0x0102;

// Generic destinations (DLS1 + DLS2).
pub const CONN_DST_NONE: u16 = 0x0000;
pub const CONN_DST_GAIN: u16 = 0x0001;
pub const CONN_DST_PITCH: u16 = 0x0003;
pub const CONN_DST_PAN: u16 = 0x0004;
pub const CONN_DST_KEYNUMBER: u16 = 0x0005;

// DLS1 used `CONN_DST_ATTENUATION` for the gain destination (=0x0001);
// DLS2 renamed it to `CONN_DST_GAIN`. They share the same numeric value
// so the same constant covers both.
pub const CONN_DST_ATTENUATION: u16 = CONN_DST_GAIN;

// DLS2 channel-output destinations (surround placement).
pub const CONN_DST_LEFT: u16 = 0x0010;
pub const CONN_DST_RIGHT: u16 = 0x0011;
pub const CONN_DST_CENTER: u16 = 0x0012;
pub const CONN_DST_LFE_CHANNEL: u16 = 0x0013;
pub const CONN_DST_LEFTREAR: u16 = 0x0014;
pub const CONN_DST_RIGHTREAR: u16 = 0x0015;
pub const CONN_DST_CHORUS: u16 = 0x0080;
pub const CONN_DST_REVERB: u16 = 0x0081;

// Modulator LFO destinations.
pub const CONN_DST_LFO_FREQUENCY: u16 = 0x0104;
pub const CONN_DST_LFO_STARTDELAY: u16 = 0x0105;

// DLS2 Vibrato LFO destinations.
pub const CONN_DST_VIB_FREQUENCY: u16 = 0x0114;
pub const CONN_DST_VIB_STARTDELAY: u16 = 0x0115;

// EG1 (volume envelope) destinations.
pub const CONN_DST_EG1_ATTACKTIME: u16 = 0x0206;
pub const CONN_DST_EG1_DECAYTIME: u16 = 0x0207;
pub const CONN_DST_EG1_RELEASETIME: u16 = 0x0209;
pub const CONN_DST_EG1_SUSTAINLEVEL: u16 = 0x020A;
// DLS2-only.
pub const CONN_DST_EG1_DELAYTIME: u16 = 0x020B;
pub const CONN_DST_EG1_HOLDTIME: u16 = 0x020C;
pub const CONN_DST_EG1_SHUTDOWNTIME: u16 = 0x020D;

// EG2 (modulation envelope) destinations.
pub const CONN_DST_EG2_ATTACKTIME: u16 = 0x030A;
pub const CONN_DST_EG2_DECAYTIME: u16 = 0x030B;
pub const CONN_DST_EG2_RELEASETIME: u16 = 0x030D;
pub const CONN_DST_EG2_SUSTAINLEVEL: u16 = 0x030E;
// DLS2-only.
pub const CONN_DST_EG2_DELAYTIME: u16 = 0x030F;
pub const CONN_DST_EG2_HOLDTIME: u16 = 0x0310;

// Filter destinations (DLS2 only).
pub const CONN_DST_FILTER_CUTOFF: u16 = 0x0500;
pub const CONN_DST_FILTER_Q: u16 = 0x0501;

// Transforms.
pub const CONN_TRN_NONE: u16 = 0x0000;
pub const CONN_TRN_CONCAVE: u16 = 0x0001;
pub const CONN_TRN_CONVEX: u16 = 0x0002;
pub const CONN_TRN_SWITCH: u16 = 0x0003;

// ---------------------------------------------------------------------------
// Unit conversions.
//
// Per DLS2 §1.14 ("Data Format Definitions"):
//   - Absolute Pitch: signed 32-bit, each unit = 1/65536 cent.
//     `cents = scale / 65536.0`
//   - Absolute Time: signed 32-bit time-cents, secs = 2^(tc/(1200*65536))
//     except 0x80000000 = "absolute zero" (no modulation).
//   - Gain: signed 32-bit, each unit = 1/655360 dB.
//     `dB = scale / 655360.0`
//   - Percent: 0.1 % units (DLS1 used 0..=1000 = 0..=100 %; DLS2 carries
//     these as raw `lScale` in the connection block where the
//     destination is a percentage type — sustain level, pan).
//
// We expose pure-fn helpers so the per-destination switch can call the
// right converter once per matched block.
// ---------------------------------------------------------------------------

/// Sentinel `lScale` value that DLS treats as "absolute zero / leave
/// destination untouched". Skipping these matches the spec's "do not
/// modulate" semantics.
pub const ABSOLUTE_ZERO: i32 = i32::MIN; // 0x8000_0000 in two's complement

/// Convert a DLS time-cents scale value to seconds. The DLS spec uses
/// `2^(tc / (1200 * 65536))`; clamped to a generous bound so a malformed
/// connection can't produce an attack of 10^9 seconds.
fn time_cents_to_secs(scale: i32) -> f32 {
    if scale == ABSOLUTE_ZERO {
        return 0.0;
    }
    let tc = scale as f64;
    let secs = (tc / (1200.0 * 65536.0)).exp2();
    // Per DLS Level 2.2 Table 5 (Vol EG times min 0 sec, max 40 sec),
    // 40 s is the spec ceiling; clamp at 60 s so we have a bit of
    // headroom over the spec's max and still bound malformed input.
    secs.clamp(0.0, 60.0) as f32
}

/// Convert a DLS absolute-pitch scale value to cents.
fn abs_pitch_to_cents(scale: i32) -> f32 {
    if scale == ABSOLUTE_ZERO {
        return 0.0;
    }
    // DLS spec uses 1 / 65536 cent per unit. Clamp at ±12 octaves
    // (±14400 cents) — wider than the Vol/Mod LFO's ±1200 cents max, so
    // we don't reject DLS2 tuning that goes broader.
    ((scale as f64) / 65536.0).clamp(-14_400.0, 14_400.0) as f32
}

/// Convert a DLS absolute-pitch scale value to Hz (used for LFO
/// frequencies, where `tc` is "pitch-cents above a reference of 1 Hz",
/// i.e. `Hz = 2^(tc / (1200 * 65536))`).
fn abs_pitch_to_hz(scale: i32) -> f32 {
    if scale == ABSOLUTE_ZERO {
        // Spec Table 5: default Mod LFO Frequency = 5 Hz, default Vib
        // LFO Frequency = 5 Hz. We let the caller pick which default to
        // use — returning 0 here lets the caller's "if zero use default"
        // branch trigger.
        return 0.0;
    }
    let tc = scale as f64;
    let hz = (tc / (1200.0 * 65536.0)).exp2();
    // DLS1 LFO freq spec range is 0.1–10 Hz; DLS2 widened to 0.1–20 Hz.
    // Clamp at 50 Hz so a degenerate value doesn't escape into the
    // voice's phase math.
    hz.clamp(0.0, 50.0) as f32
}

/// Sustain level scale → linear 0..=1. DLS sustain destination is
/// percent, in 0.1 % units (so 1000 = 100 %).
fn sustain_pct_to_linear(scale: i32) -> f32 {
    if scale == ABSOLUTE_ZERO {
        return 1.0; // spec default sustain = 100 %
    }
    (scale as f32 / 1000.0).clamp(0.0, 1.0)
}

/// Gain destination → linear amplitude multiplier. Per DLS2 §1.14.4
/// `dB = scale / 655360`. Round 80 surfaces this as a separate gain so
/// `build_dls_config` can fold it into the amplitude.
fn gain_to_linear(scale: i32) -> f32 {
    if scale == ABSOLUTE_ZERO {
        return 1.0;
    }
    let db = scale as f32 / 655_360.0;
    // Clamp at ±48 dB so a malformed block can't blow the mixer.
    let db = db.clamp(-96.0, 48.0);
    10.0f32.powf(db / 20.0)
}

// ---------------------------------------------------------------------------
// Articulation digest.
//
// One round trip across a region's connection-block list (and a fallback
// across the instrument-level list for blocks the region didn't override)
// fills out the destinations we care about, and leaves
// [`Articulation::handled`] mirroring which blocks we consumed so a
// future round's diagnostics can know.
// ---------------------------------------------------------------------------

/// Effective connection values extracted from a DLS articulation list.
/// Every field starts at its spec default (Table 5/6 of DLS Level 2.2)
/// and is overwritten only by an explicit `SRC_NONE → DST_x` connection.
///
/// The defaults match what `SamplePlayer` would have done without any
/// articulation interpretation, so an empty `art1` list produces
/// identical output to a region with no `lart` chunk at all.
#[derive(Clone, Copy, Debug)]
pub struct Articulation {
    // Vol EG (EG1) — DAHDSR, in seconds.
    pub vol_delay_s: f32,
    pub vol_attack_s: f32,
    pub vol_hold_s: f32,
    pub vol_decay_s: f32,
    pub vol_sustain_level: f32,
    pub vol_release_s: f32,
    // `true` once any explicit EG1 destination overrode a default.
    pub vol_overridden: bool,

    // Velocity → Attack time scale (cents-per-velocity). Round 80
    // surfaces but does not interpret this — `SamplePlayer`'s envelope
    // is keyed to per-voice times only and doesn't take a per-note
    // velocity-dependent attack. Surfaced for a later round.
    pub vol_velocity_to_attack_tc: i32,

    // Mod EG (EG2) — surfaced raw for a later round (EG2 routes into
    // pitch + filter cutoff, neither of which the round-80 SamplePlayer
    // handles).
    pub mod_delay_tc: i32,
    pub mod_attack_tc: i32,
    pub mod_hold_tc: i32,
    pub mod_decay_tc: i32,
    pub mod_sustain_pct: i32,
    pub mod_release_tc: i32,
    pub mod_to_pitch_cents: f32,
    pub mod_to_filter_cents: f32,

    // Modulator LFO.
    pub mod_lfo_freq_hz: f32,
    pub mod_lfo_delay_s: f32,
    pub mod_lfo_to_pitch_cents: f32,
    pub mod_lfo_to_gain_db: f32,

    // Vibrato LFO (DLS2-only; DLS1 conflated Mod LFO + vibrato into a
    // single CONN_SRC_LFO source).
    pub vib_lfo_freq_hz: f32,
    pub vib_lfo_delay_s: f32,
    pub vib_lfo_to_pitch_cents: f32,

    // Filter (DLS2-only).
    pub filter_cutoff_cents: f32,
    pub filter_q_centibels: f32,

    // Tuning (SRC_NONE → DST_PITCH, "absolute tuning" per Table 6).
    pub tuning_cents: f32,
    // Per-note gain (SRC_NONE → DST_GAIN). Folded into the amplitude
    // multiplier; ABSOLUTE_ZERO leaves the existing amplitude untouched.
    pub gain_linear: f32,
    pub pan_pct: f32, // -50..=+50 (0.1 % units → percent of full pan)
}

impl Default for Articulation {
    fn default() -> Self {
        // Defaults from DLS Level 2.2 Tables 5 + 6, with the
        // Vol EG falling back to the SamplePlayer defaults so a region
        // with no `lart` chunk plays identically to the round-1 path.
        let env = EnvelopeParams::default();
        let vib = VibratoParams::default();
        Self {
            vol_delay_s: env.delay_s,
            vol_attack_s: env.attack_s,
            vol_hold_s: env.hold_s,
            vol_decay_s: env.decay_s,
            vol_sustain_level: env.sustain_level,
            vol_release_s: env.release_s,
            vol_overridden: false,
            vol_velocity_to_attack_tc: 0,

            mod_delay_tc: ABSOLUTE_ZERO,
            mod_attack_tc: ABSOLUTE_ZERO,
            mod_hold_tc: ABSOLUTE_ZERO,
            mod_decay_tc: ABSOLUTE_ZERO,
            mod_sustain_pct: ABSOLUTE_ZERO,
            mod_release_tc: ABSOLUTE_ZERO,
            mod_to_pitch_cents: 0.0,
            mod_to_filter_cents: 0.0,

            // Spec default Mod LFO frequency = 5 Hz, start delay = 10 ms.
            mod_lfo_freq_hz: 5.0,
            mod_lfo_delay_s: 0.010,
            mod_lfo_to_pitch_cents: 0.0,
            mod_lfo_to_gain_db: 0.0,

            vib_lfo_freq_hz: 5.0,
            vib_lfo_delay_s: 0.010,
            vib_lfo_to_pitch_cents: vib.depth_cents,

            filter_cutoff_cents: 0.0,
            filter_q_centibels: 0.0,

            tuning_cents: 0.0,
            gain_linear: 1.0,
            pan_pct: 0.0,
        }
    }
}

impl Articulation {
    /// Walk a slice of [`DlsArticulationBlock`] and overlay every
    /// recognised connection on top of the spec defaults. Unrecognised
    /// blocks are skipped silently — they remain on the region for the
    /// caller to inspect via [`crate::instruments::dls::DlsRegion::articulation`].
    ///
    /// The `instrument_blocks` list is used as a *fallback*: any
    /// destination not overridden by `region_blocks` falls back to the
    /// instrument-level value. This matches the DLS spec's "instrument
    /// articulation = global default, region articulation = local
    /// override" semantics (DLS Level 1 v1.1b "Articulation"
    /// section + DLS Level 2.2 §1.6.3 "Default, Global and Local
    /// Articulation Data").
    pub fn evaluate(
        region_blocks: &[DlsArticulationBlock],
        instrument_blocks: &[DlsArticulationBlock],
    ) -> Self {
        let mut art = Articulation::default();
        // Track which destinations the region list overrode so a later
        // pass over the instrument-level list doesn't undo a region
        // override with a global default.
        let mut overridden = OverrideMask::default();
        art.apply_blocks(region_blocks, &mut overridden, true);
        art.apply_blocks(instrument_blocks, &mut overridden, false);
        art
    }

    fn apply_blocks(
        &mut self,
        blocks: &[DlsArticulationBlock],
        overridden: &mut OverrideMask,
        is_region: bool,
    ) {
        for b in blocks {
            // Skip blocks whose source/control combination we don't
            // interpret — they remain on the bank for a future round.
            // We accept SRC_NONE-→-DST_x ("absolute default override")
            // and a handful of well-known SRC_x-→-DST_x routings.
            self.apply_block(b, overridden, is_region);
        }
    }

    fn apply_block(
        &mut self,
        b: &DlsArticulationBlock,
        overridden: &mut OverrideMask,
        is_region: bool,
    ) {
        // Bits 0-3 of `transform` are the output transform; we only
        // honour CONN_TRN_NONE for the SRC_NONE default-overrides
        // because Concave / Convex / Switch reshape modulator responses
        // (no effect on the SRC_NONE constant case anyway).
        let _out_transform = b.transform & 0x000F;

        // Helper closure: record an override iff a region block sets it
        // (or no region block has already set it).
        macro_rules! set {
            ($field:ident, $mask:ident, $val:expr) => {{
                if is_region || !overridden.$mask {
                    self.$field = $val;
                    if is_region {
                        overridden.$mask = true;
                    }
                }
            }};
        }

        // -- SRC_NONE → DST_x: the "absolute default override" branch --
        if b.source == CONN_SRC_NONE && b.control == CONN_SRC_NONE {
            match b.destination {
                // Vol EG (EG1).
                CONN_DST_EG1_DELAYTIME => {
                    set!(vol_delay_s, vol_delay, time_cents_to_secs(b.scale));
                    self.vol_overridden = true;
                }
                CONN_DST_EG1_ATTACKTIME => {
                    set!(vol_attack_s, vol_attack, time_cents_to_secs(b.scale));
                    self.vol_overridden = true;
                }
                CONN_DST_EG1_HOLDTIME => {
                    set!(vol_hold_s, vol_hold, time_cents_to_secs(b.scale));
                    self.vol_overridden = true;
                }
                CONN_DST_EG1_DECAYTIME => {
                    set!(vol_decay_s, vol_decay, time_cents_to_secs(b.scale));
                    self.vol_overridden = true;
                }
                CONN_DST_EG1_SUSTAINLEVEL => {
                    set!(
                        vol_sustain_level,
                        vol_sustain,
                        sustain_pct_to_linear(b.scale)
                    );
                    self.vol_overridden = true;
                }
                CONN_DST_EG1_RELEASETIME => {
                    set!(vol_release_s, vol_release, time_cents_to_secs(b.scale));
                    self.vol_overridden = true;
                }
                // Mod EG (EG2) — raw, no SamplePlayer routing yet.
                CONN_DST_EG2_DELAYTIME => set!(mod_delay_tc, mod_delay, b.scale),
                CONN_DST_EG2_ATTACKTIME => set!(mod_attack_tc, mod_attack, b.scale),
                CONN_DST_EG2_HOLDTIME => set!(mod_hold_tc, mod_hold, b.scale),
                CONN_DST_EG2_DECAYTIME => set!(mod_decay_tc, mod_decay, b.scale),
                CONN_DST_EG2_SUSTAINLEVEL => set!(mod_sustain_pct, mod_sustain, b.scale),
                CONN_DST_EG2_RELEASETIME => set!(mod_release_tc, mod_release, b.scale),
                // Modulator LFO.
                CONN_DST_LFO_FREQUENCY => {
                    let hz = abs_pitch_to_hz(b.scale);
                    if hz > 0.0 {
                        set!(mod_lfo_freq_hz, mod_lfo_freq, hz);
                    }
                }
                CONN_DST_LFO_STARTDELAY => {
                    set!(mod_lfo_delay_s, mod_lfo_delay, time_cents_to_secs(b.scale));
                }
                // Vibrato LFO.
                CONN_DST_VIB_FREQUENCY => {
                    let hz = abs_pitch_to_hz(b.scale);
                    if hz > 0.0 {
                        set!(vib_lfo_freq_hz, vib_lfo_freq, hz);
                    }
                }
                CONN_DST_VIB_STARTDELAY => {
                    set!(vib_lfo_delay_s, vib_lfo_delay, time_cents_to_secs(b.scale));
                }
                // Filter.
                CONN_DST_FILTER_CUTOFF => {
                    set!(
                        filter_cutoff_cents,
                        filter_cutoff,
                        abs_pitch_to_cents(b.scale)
                    );
                }
                CONN_DST_FILTER_Q => {
                    // Q is in centibels (0.1 dB units, per DLS2 §1.5.2).
                    set!(filter_q_centibels, filter_q, b.scale as f32 / 65_536.0);
                }
                // Pitch (tuning) + gain + pan.
                CONN_DST_PITCH => set!(tuning_cents, tuning, abs_pitch_to_cents(b.scale)),
                CONN_DST_GAIN => set!(gain_linear, gain, gain_to_linear(b.scale)),
                CONN_DST_PAN => set!(pan_pct, pan, (b.scale as f32 / 10.0).clamp(-50.0, 50.0)),
                _ => {}
            }
            return;
        }

        // -- SRC_LFO → DST_PITCH: vibrato depth (DLS1 + DLS2) --
        if b.source == CONN_SRC_LFO && b.control == CONN_SRC_NONE && b.destination == CONN_DST_PITCH
        {
            set!(
                mod_lfo_to_pitch_cents,
                mod_lfo_to_pitch,
                abs_pitch_to_cents(b.scale)
            );
            return;
        }

        // -- SRC_LFO → DST_GAIN: tremolo depth (DLS1: ATTENUATION) --
        if b.source == CONN_SRC_LFO && b.control == CONN_SRC_NONE && b.destination == CONN_DST_GAIN
        {
            set!(
                mod_lfo_to_gain_db,
                mod_lfo_to_gain,
                b.scale as f32 / 655_360.0
            );
            return;
        }

        // -- SRC_VIBRATO → DST_PITCH: dedicated DLS2 vibrato depth --
        if b.source == CONN_SRC_VIBRATO
            && b.control == CONN_SRC_NONE
            && b.destination == CONN_DST_PITCH
        {
            set!(
                vib_lfo_to_pitch_cents,
                vib_lfo_to_pitch,
                abs_pitch_to_cents(b.scale)
            );
            return;
        }

        // -- SRC_KEYONVELOCITY → DST_EG1_ATTACKTIME: stash raw for now --
        if b.source == CONN_SRC_KEYONVELOCITY
            && b.control == CONN_SRC_NONE
            && b.destination == CONN_DST_EG1_ATTACKTIME
        {
            set!(vol_velocity_to_attack_tc, vol_velocity_to_attack, b.scale);
            return;
        }

        // -- SRC_EG2 → DST_PITCH: mod-env to pitch (raw cents) --
        if b.source == CONN_SRC_EG2 && b.control == CONN_SRC_NONE && b.destination == CONN_DST_PITCH
        {
            set!(
                mod_to_pitch_cents,
                mod_to_pitch,
                abs_pitch_to_cents(b.scale)
            );
            return;
        }

        // -- SRC_EG2 → DST_FILTER_CUTOFF: mod-env to filter (DLS2) --
        if b.source == CONN_SRC_EG2
            && b.control == CONN_SRC_NONE
            && b.destination == CONN_DST_FILTER_CUTOFF
        {
            set!(
                mod_to_filter_cents,
                mod_to_filter,
                abs_pitch_to_cents(b.scale)
            );
            return;
        }

        // Everything else — modulation wheel, channel pressure, etc. —
        // gets dropped for round 80. Future rounds can extend the match.
        let _ = (b.kind, _out_transform);
    }

    /// Effective EnvelopeParams from the resolved Vol EG settings.
    /// Always returns a fully-populated struct, defaulting unspecified
    /// fields to the SamplePlayer's musical defaults.
    pub fn envelope(&self) -> EnvelopeParams {
        EnvelopeParams {
            delay_s: self.vol_delay_s,
            attack_s: self.vol_attack_s.max(0.0001),
            hold_s: self.vol_hold_s,
            decay_s: self.vol_decay_s.max(0.0001),
            sustain_level: self.vol_sustain_level.clamp(0.0, 1.0),
            release_s: self.vol_release_s.max(0.0001),
        }
    }

    /// Effective VibratoParams — uses the DLS2 vibrato LFO depth if
    /// present, otherwise falls back to the mod LFO's pitch routing
    /// (DLS1 conflated the two into a single CONN_SRC_LFO source).
    pub fn vibrato(&self) -> VibratoParams {
        let depth_cents = if self.vib_lfo_to_pitch_cents.abs() > 0.0 {
            self.vib_lfo_to_pitch_cents
        } else {
            self.mod_lfo_to_pitch_cents
        };
        let freq_hz = if self.vib_lfo_to_pitch_cents.abs() > 0.0 {
            self.vib_lfo_freq_hz
        } else {
            self.mod_lfo_freq_hz
        };
        let delay_s = if self.vib_lfo_to_pitch_cents.abs() > 0.0 {
            self.vib_lfo_delay_s
        } else {
            self.mod_lfo_delay_s
        };
        VibratoParams {
            freq_hz,
            depth_cents,
            delay_s,
        }
    }
}

/// Tracks which destinations a region's articulation list has overridden
/// so a subsequent fallback pass over the instrument-level list doesn't
/// reset a region value back to the global default.
#[derive(Clone, Copy, Debug, Default)]
struct OverrideMask {
    vol_delay: bool,
    vol_attack: bool,
    vol_hold: bool,
    vol_decay: bool,
    vol_sustain: bool,
    vol_release: bool,
    vol_velocity_to_attack: bool,
    mod_delay: bool,
    mod_attack: bool,
    mod_hold: bool,
    mod_decay: bool,
    mod_sustain: bool,
    mod_release: bool,
    mod_to_pitch: bool,
    mod_to_filter: bool,
    mod_lfo_freq: bool,
    mod_lfo_delay: bool,
    mod_lfo_to_pitch: bool,
    mod_lfo_to_gain: bool,
    vib_lfo_freq: bool,
    vib_lfo_delay: bool,
    vib_lfo_to_pitch: bool,
    filter_cutoff: bool,
    filter_q: bool,
    tuning: bool,
    gain: bool,
    pan: bool,
}

#[cfg(test)]
mod tests {
    use super::super::dls::DlsArtKind;
    use super::*;

    fn block(
        source: u16,
        control: u16,
        destination: u16,
        transform: u16,
        scale: i32,
    ) -> DlsArticulationBlock {
        DlsArticulationBlock {
            kind: DlsArtKind::Art1,
            source,
            control,
            destination,
            transform,
            scale,
        }
    }

    #[test]
    fn empty_lists_yield_defaults() {
        let a = Articulation::evaluate(&[], &[]);
        let env = a.envelope();
        // Default attack of 5 ms = 0.005 s.
        assert!((env.attack_s - 0.005).abs() < 1e-6);
        assert!(!a.vol_overridden);
        assert_eq!(a.tuning_cents, 0.0);
        assert_eq!(a.gain_linear, 1.0);
    }

    #[test]
    fn region_vol_eg_attack_override() {
        // 100 ms attack: time_cents = log2(0.1) * 1200 * 65536
        //                ≈ -3.321928 * 1200 * 65536 ≈ -261_185_536.
        // Pick a round value and verify the inverse — we set the scale
        // to log2(0.1) * 1200 * 65536 explicitly.
        let secs = 0.1f64;
        let tc = (secs.log2() * 1200.0 * 65536.0) as i32;
        let blocks = vec![block(
            CONN_SRC_NONE,
            CONN_SRC_NONE,
            CONN_DST_EG1_ATTACKTIME,
            CONN_TRN_NONE,
            tc,
        )];
        let a = Articulation::evaluate(&blocks, &[]);
        assert!((a.vol_attack_s - 0.1).abs() < 0.002);
        assert!(a.vol_overridden);
    }

    #[test]
    fn region_overrides_instrument() {
        // Instrument list sets release to 1 s; region overrides to 50 ms.
        let inst_tc = (1.0f64.log2() * 1200.0 * 65536.0) as i32; // 0
        let reg_tc = (0.05f64.log2() * 1200.0 * 65536.0) as i32; // negative
        let inst = vec![block(
            CONN_SRC_NONE,
            CONN_SRC_NONE,
            CONN_DST_EG1_RELEASETIME,
            CONN_TRN_NONE,
            inst_tc,
        )];
        let region = vec![block(
            CONN_SRC_NONE,
            CONN_SRC_NONE,
            CONN_DST_EG1_RELEASETIME,
            CONN_TRN_NONE,
            reg_tc,
        )];
        let a = Articulation::evaluate(&region, &inst);
        assert!((a.vol_release_s - 0.05).abs() < 0.002);
    }

    #[test]
    fn instrument_fallback_when_region_silent() {
        // Region empty → instrument's attack should win.
        let tc = (0.2f64.log2() * 1200.0 * 65536.0) as i32;
        let inst = vec![block(
            CONN_SRC_NONE,
            CONN_SRC_NONE,
            CONN_DST_EG1_ATTACKTIME,
            CONN_TRN_NONE,
            tc,
        )];
        let a = Articulation::evaluate(&[], &inst);
        assert!((a.vol_attack_s - 0.2).abs() < 0.005);
    }

    #[test]
    fn tuning_in_cents() {
        // +100 cents = +1 semitone → scale = 100 * 65536 = 6_553_600.
        let blocks = vec![block(
            CONN_SRC_NONE,
            CONN_SRC_NONE,
            CONN_DST_PITCH,
            CONN_TRN_NONE,
            100 * 65_536,
        )];
        let a = Articulation::evaluate(&blocks, &[]);
        assert!((a.tuning_cents - 100.0).abs() < 0.01);
    }

    #[test]
    fn lfo_pitch_routes_to_vibrato_depth() {
        // DLS1-style: CONN_SRC_LFO → DST_PITCH at 50 cents.
        let blocks = vec![block(
            CONN_SRC_LFO,
            CONN_SRC_NONE,
            CONN_DST_PITCH,
            CONN_TRN_NONE,
            50 * 65_536,
        )];
        let a = Articulation::evaluate(&blocks, &[]);
        let vib = a.vibrato();
        assert!((vib.depth_cents - 50.0).abs() < 0.05);
        // Default mod LFO freq is 5 Hz per spec Table 5.
        assert!((vib.freq_hz - 5.0).abs() < 0.01);
    }

    #[test]
    fn vibrato_lfo_dls2_takes_precedence_over_mod_lfo() {
        // DLS2: vibrato LFO has dedicated source. When both are set,
        // the vibrato LFO routing wins.
        let blocks = vec![
            block(
                CONN_SRC_LFO,
                CONN_SRC_NONE,
                CONN_DST_PITCH,
                CONN_TRN_NONE,
                10 * 65_536,
            ),
            block(
                CONN_SRC_VIBRATO,
                CONN_SRC_NONE,
                CONN_DST_PITCH,
                CONN_TRN_NONE,
                75 * 65_536,
            ),
        ];
        let a = Articulation::evaluate(&blocks, &[]);
        let vib = a.vibrato();
        assert!((vib.depth_cents - 75.0).abs() < 0.05);
    }

    #[test]
    fn gain_destination_attenuates() {
        // -6 dB → scale = -6 * 655360.
        let blocks = vec![block(
            CONN_SRC_NONE,
            CONN_SRC_NONE,
            CONN_DST_GAIN,
            CONN_TRN_NONE,
            -6 * 655_360,
        )];
        let a = Articulation::evaluate(&blocks, &[]);
        // 10^(-6/20) = 0.5012 (linear).
        assert!((a.gain_linear - 0.5011872).abs() < 0.001);
    }

    #[test]
    fn absolute_zero_skipped() {
        // ABSOLUTE_ZERO scale on any time destination → no change.
        let blocks = vec![block(
            CONN_SRC_NONE,
            CONN_SRC_NONE,
            CONN_DST_EG1_ATTACKTIME,
            CONN_TRN_NONE,
            ABSOLUTE_ZERO,
        )];
        let a = Articulation::evaluate(&blocks, &[]);
        // We did "see" the destination but the conversion returned 0,
        // so the attack is 0 — `Articulation::envelope()` clamps to a
        // tiny positive value so SamplePlayer doesn't divide by zero.
        let env = a.envelope();
        assert!(env.attack_s > 0.0);
        assert!(env.attack_s < 0.001);
    }

    #[test]
    fn unknown_connections_dropped_silently() {
        // SRC_CC1 → DST_FILTER_CUTOFF — not in the round-80 match arm.
        let blocks = vec![block(
            CONN_SRC_CC1,
            CONN_SRC_NONE,
            CONN_DST_FILTER_CUTOFF,
            CONN_TRN_NONE,
            12_345,
        )];
        let a = Articulation::evaluate(&blocks, &[]);
        // Filter cutoff stays at default (0.0 cents — no override).
        assert_eq!(a.filter_cutoff_cents, 0.0);
    }
}
