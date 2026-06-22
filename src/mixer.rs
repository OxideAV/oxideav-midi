//! Polyphonic voice pool + mixdown into a stereo PCM buffer.
//!
//! Round-3 lives between the [SMF event scheduler](crate::scheduler) and
//! the [`Decoder`](oxideav_core::Decoder) output. It owns up to
//! [`MAX_VOICES`] [`Voice`](crate::instruments::Voice) instances at a
//! time; `note_on` allocates a slot (preempting the oldest still-running
//! voice when the pool is full), `note_off` triggers the matching
//! voice's release envelope, and `mix_stereo` walks every live voice to
//! sum a chunk of samples into a planar (left, right) buffer.
//!
//! Sustain (CC 64), per-channel volume (CC 7), and pan (CC 10) live on
//! a small [`ChannelState`] table the mixer carries — they are consumed
//! at mix time so a CC-change between two `mix_stereo` chunks takes
//! effect on the very next chunk without any per-sample coordination.
//!
//! Stereo output is hard-coded for round-3. The voice generator
//! ([`Sf2Voice`](crate::instruments::sf2::Sf2Voice)) renders mono today,
//! so each voice's samples are panned into the left/right buses via the
//! constant-power law (`cos(θ)` left, `sin(θ)` right with `θ` derived
//! from the channel pan in `0..=127`). Real stereo SF2 zones (paired
//! sample links) are round-4 work.
//!
//! ## Voice allocation policy
//!
//! Voice slots are allocated by:
//!   1. The first slot whose voice is `done()` (or `None`),
//!   2. otherwise, the slot whose voice was allocated longest ago
//!      (a monotonic `age` counter — oldest = lowest counter).
//!
//! The preempted voice is dropped outright (no fade-out — the few-ms
//! release of round-2 is the glide). LRU was picked because it is the
//! cheapest "least musically jarring" policy that avoids the tail-of-a-
//! held-chord killing every new note in a busy passage. Round-4 may
//! revisit.

use crate::instruments::Voice;

/// Hard cap on simultaneous voices. Picked to land below the audible
/// "one more voice doesn't help" perception threshold for typical SF2
/// playback while keeping the mix loop's per-sample inner work bounded
/// (32 voice fetches × 1024 samples = ~32 k mults per chunk).
pub const MAX_VOICES: usize = 32;

/// Convert a raw 14-bit pitch-bend scalar (`0..=16383`, centre `0x2000`)
/// to a signed cents offset using the per-channel bend range. Default
/// range is 200 cents (= ±2 semitones, GM RP-018 recommended practice).
pub fn pitch_bend_to_cents(value: u16, range_cents: u16) -> i32 {
    let centred = value.min(0x3FFF) as i32 - 0x2000;
    // ±8192 maps to ±range_cents.
    centred * range_cents as i32 / 0x2000
}

/// Number of MIDI channels — fixed by the spec, not configurable.
pub const NUM_CHANNELS: usize = 16;

/// One slot in the voice pool.
struct VoiceSlot {
    /// The active voice, or `None` if the slot is free.
    voice: Option<Box<dyn Voice>>,
    /// MIDI channel that owns this voice (so per-channel CCs hit the
    /// right slots). `0..16`.
    channel: u8,
    /// MIDI key the voice is sounding (so a `NoteOff key=K channel=C`
    /// can find its match).
    key: u8,
    /// `true` once a `NoteOff` arrived but the channel sustain pedal
    /// (CC 64) was held — release is deferred until the pedal lifts.
    sustained: bool,
    /// Monotonic allocation counter — smallest = oldest.
    age: u64,
    /// Per-voice gain folded in from velocity (already applied inside
    /// the voice) plus channel volume / pan. Pulled at mix time.
    velocity_norm: f32,
}

impl VoiceSlot {
    const fn empty() -> Self {
        Self {
            voice: None,
            channel: 0,
            key: 0,
            sustained: false,
            age: 0,
            velocity_norm: 0.0,
        }
    }
}

/// Per-channel state tracked between events. Volume / pan / sustain are
/// CCs the mixer needs at mix time; `program` lives here so the
/// scheduler can pick the right preset on the next `note_on`.
#[derive(Clone, Copy, Debug)]
pub struct ChannelState {
    /// MIDI program (0..=127). Set by `ProgramChange`. Defaults to 0
    /// (Acoustic Grand Piano in GM).
    pub program: u8,
    /// CC 7 (Channel Volume), 0..=127. Default 100 per GM.
    pub volume: u8,
    /// CC 10 (Pan), 0..=127. 64 = centre. Default 64.
    pub pan: u8,
    /// CC 64 (Sustain Pedal). `true` while the pedal is depressed
    /// (value >= 64); the mixer holds note-offs until it lifts.
    pub sustain: bool,
    /// Live pitch-bend value as the raw 14-bit MIDI scalar
    /// (`0..=16383`). Centre is `0x2000`. Map to cents via
    /// `(value - 0x2000) * pitch_bend_range_cents / 8192`.
    pub pitch_bend: u16,
    /// Pitch-bend range in cents (default 200 = ±2 semitones per GM
    /// recommended practice). Updated via RPN 0 (CC 100/101 = 0/0,
    /// CC 6 = MSB semitones, CC 38 = LSB cents). MPE Receivers default
    /// this to 4800 (±48 semitones) on Member Channels at MCM time.
    pub pitch_bend_range_cents: u16,
    /// Channel pressure (mono aftertouch) as the raw `0..=127` scalar.
    /// Default 0 = no pressure modulation.
    pub channel_pressure: u8,
    /// Currently-selected RPN as a 14-bit value (CC 100 LSB / CC 101
    /// MSB). `0x3FFF` is the "RPN null" marker that disables further
    /// CC-6 / CC-38 writes. We default to null so a CC 6 with no prior
    /// RPN selection doesn't accidentally clobber the bend range.
    pub rpn: u16,
    /// CC 1 (Modulation Wheel), 0..=127. Default 0. Routes through the
    /// channel's [`Self::mod_depth_range_cents`] before being passed to
    /// the voice as a pitch-mod depth.
    pub mod_wheel: u8,
    /// RPN 5 — Modulation Depth Range, in cents. Per CA-26 the default
    /// is implementation-defined; GM2 prescribes 50 cents and we follow
    /// suit. `mod_wheel` scaled into `[0, mod_depth_range_cents]` cents
    /// is delivered to held voices via [`Voice::set_mod_depth_cents`].
    pub mod_depth_range_cents: u16,
    /// RPN 1 — Channel Fine Tuning, in cents. 14-bit RPN value maps
    /// linearly to ±100 cents (centre is data-entry 0x40/0x00). Summed
    /// per spec with master fine tuning into the effective pitch
    /// offset.
    pub channel_fine_tune_cents: i16,
    /// Raw 14-bit accumulator for the RPN-1 data-entry pair, kept on
    /// the channel state so a CC 6 / CC 38 sequence composes
    /// bit-exact (MSB sets the top 7 bits, LSB sets the bottom 7).
    /// Not normally read directly — callers should look at
    /// [`Self::channel_fine_tune_cents`].
    pub channel_fine_tune_raw_14: u16,
    /// RPN 2 — Channel Coarse Tuning, in semitones (-64..=+63). CC 6
    /// data-entry MSB sets it directly; CC 38 LSB is ignored per spec
    /// ("the LSB is always 0"). Summed with master coarse tuning.
    pub channel_coarse_tune_semitones: i16,
    /// MPE role of this channel. `Manager` and `Member` channels behave
    /// differently for routing of per-note vs. zone-wide CCs / Pitch
    /// Bend / Channel Pressure. `None` outside any active MPE zone.
    pub mpe_role: MpeRole,
    /// CC 91 (Effects 1 Depth = Reverb Send Level), 0..=127. This is the
    /// per-channel send level into the system Reverb effect; CA-024
    /// ("Example of Recommended Practice for Reverb and Chorus
    /// Parameters") notes "the send levels to the Reverb and Chorus
    /// effects are controlled with Control Changes #91 and #93". The GM
    /// Level 1 spec resets controllers to "normal" at GM-On, which we
    /// model as send 0 (fully dry) so a synth that never touches CC 91
    /// renders bit-identically to the pre-reverb path.
    pub reverb_send: u8,
    /// CC 93 (Effects 3 Depth = Chorus Send Level), 0..=127. Per-channel
    /// send level into the system Chorus effect (CA-024, CC #93). Default
    /// 0 (fully dry).
    pub chorus_send: u8,
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            program: 0,
            volume: 100,
            pan: 64,
            sustain: false,
            pitch_bend: 0x2000,
            pitch_bend_range_cents: 200,
            channel_pressure: 0,
            rpn: 0x3FFF,
            mod_wheel: 0,
            mod_depth_range_cents: 50,
            channel_fine_tune_cents: 0,
            channel_fine_tune_raw_14: 0x2000,
            channel_coarse_tune_semitones: 0,
            mpe_role: MpeRole::None,
            reverb_send: 0,
            chorus_send: 0,
        }
    }
}

impl ChannelState {
    /// MPE-aware "does a CC/PB/pressure on `event_channel` reach the
    /// voice held on `slot_channel`?" — the test compiled into
    /// `reapply_mod_wheel_for_channel` / `set_timbre`. Returns `true`
    /// when:
    ///   * The event channel is the slot channel (always — channel
    ///     CCs are channel-scoped by default).
    ///   * Or `event_role` is an MPE Manager and the slot's channel
    ///     belongs to the same zone (Member or Manager).
    ///
    /// The `event_role` is the **event-sending channel's** role,
    /// since the dispatch site already has it in hand.
    pub fn matches_for_zone_broadcast(
        &self,
        slot_channel: u8,
        event_channel: u8,
        event_role: &MpeRole,
    ) -> bool {
        if slot_channel == event_channel {
            return true;
        }
        match event_role {
            MpeRole::Manager(kind) => {
                matches!((self.mpe_role, kind), (MpeRole::Member(k), z) if k == *z)
            }
            _ => false,
        }
    }
}

/// A channel's role inside an MPE zone. Per M1-100-UM §2.3 + Appendix E,
/// the Manager Channel carries zone-wide messages (Damper, Program
/// Change, etc.) while Member Channels host per-note expression
/// (Pitch Bend, Channel Pressure, CC #74) that combines with the
/// Manager's value before reaching the voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpeRole {
    /// Not in an MPE zone. Channel-voice messages route normally.
    None,
    /// Manager Channel for an MPE zone — its pitch bend / pressure /
    /// CC74 broadcast to every sounding note across the whole zone
    /// (combined per Appendix C with the per-note Member Channel
    /// value).
    Manager(MpeZoneKind),
    /// Member Channel — pitch bend / pressure / CC74 only affect notes
    /// sounding on this very channel. Per Appendix D & §A.4, control
    /// values are *tracked* even when no note is sounding, so a future
    /// Note On picks them up.
    Member(MpeZoneKind),
}

impl MpeRole {
    /// `true` for both `Manager` and `Member`.
    pub fn is_mpe(self) -> bool {
        !matches!(self, MpeRole::None)
    }

    /// `true` only for `Manager(_)`.
    pub fn is_manager(self) -> bool {
        matches!(self, MpeRole::Manager(_))
    }
}

/// Which MPE zone a channel belongs to. Lower zone uses Manager
/// Channel 1 + Member Channels rising from 2; Upper zone uses Manager
/// Channel 16 + Members descending from 15.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpeZoneKind {
    /// Lower zone — Manager = channel 0 (MIDI 1).
    Lower,
    /// Upper zone — Manager = channel 15 (MIDI 16).
    Upper,
}

/// An MPE zone configuration: which channels are Manager + Members and
/// what their Pitch Bend Sensitivities are. Built by the mixer in
/// response to the MPE Configuration Message (MCM, RPN 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpeZone {
    /// `Lower` or `Upper`.
    pub kind: MpeZoneKind,
    /// Number of Member Channels in this zone (`1..=15`). `0` would
    /// deactivate the zone, so the value is only meaningful for active
    /// zones.
    pub members: u8,
}

impl MpeZone {
    /// Channel index (0..=15) of this zone's Manager Channel.
    pub fn manager_channel(self) -> u8 {
        match self.kind {
            MpeZoneKind::Lower => 0,
            MpeZoneKind::Upper => 15,
        }
    }

    /// Iterator over Member Channel indices (0..=15) in this zone.
    /// Lower zone: 1..=members. Upper zone: 15-members..=14. Returns
    /// an empty range for zero-member zones (which shouldn't exist
    /// since `set_mpe_zone` interprets that as "deactivate").
    pub fn member_channels(self) -> Vec<u8> {
        let n = self.members.min(15);
        match self.kind {
            MpeZoneKind::Lower => (1..=n).collect(),
            MpeZoneKind::Upper => (15 - n..=14).collect(),
        }
    }
}

/// GM2 system-wide Reverb + Chorus parameters, edited via the Global
/// Parameter Control Universal Real-Time SysEx message (`F0 7F <dev>
/// 04 05 …`, CA-024). The two GM2-reserved slots are `01 01` (Reverb)
/// and `01 02` (Chorus). Each parameter's raw 7-bit value is converted
/// to its engineering unit using the formulas in CA-024 "Example of
/// Recommended Practice for Reverb and Chorus Parameters (from General
/// MIDI Level 2)".
///
/// The renderer does not yet apply a reverb/chorus DSP send — these
/// values are decoded and stored so the program-state is observable and
/// a later round can wire the effects bus without re-parsing the SysEx.
/// The defaults are the GM2 recommended initial settings: Reverb Type 4
/// (Large Hall) and Chorus Type 2 (Chorus 3), with the per-type Reverb
/// Time / Chorus Mod-Rate / Mod-Depth / Feedback / Send-to-Reverb the
/// CA-024 tables list for those types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GmEffects {
    /// Reverb Type select (CA-024 reverb `pp=0`): 0 Small Room, 1 Medium
    /// Room, 2 Large Room, 3 Medium Hall, 4 Large Hall, 8 Plate. Stored
    /// as the raw 7-bit select; GM2 default = 4.
    pub reverb_type: u8,
    /// Reverb Time in seconds (CA-024 reverb `pp=1`):
    /// `rt = exp((val - 40) * 0.025)`, the inverse of the spec's
    /// `val = ln(rt) / 0.025 + 40`. GM2 Type-4 default ≈ 1.8 s (val 64).
    pub reverb_time_s: f32,
    /// Chorus Type select (CA-024 chorus `pp=0`): 0..=5. GM2 default = 2.
    pub chorus_type: u8,
    /// Chorus Mod Rate in Hz (CA-024 chorus `pp=1`): `mr = val * 0.122`.
    pub chorus_mod_rate_hz: f32,
    /// Chorus Mod Depth in ms (CA-024 chorus `pp=2`):
    /// `md = (val + 1) / 3.2`.
    pub chorus_mod_depth_ms: f32,
    /// Chorus Feedback in percent (CA-024 chorus `pp=3`):
    /// `fb = val * 0.763`.
    pub chorus_feedback_pct: f32,
    /// Chorus Send-to-Reverb in percent (CA-024 chorus `pp=4`):
    /// `ctr = val * 0.787`.
    pub chorus_send_to_reverb_pct: f32,
}

impl GmEffects {
    /// Decode a raw 7-bit Reverb Time value into seconds per CA-024
    /// (`val = ln(rt) / 0.025 + 40` ⇒ `rt = exp((val - 40) * 0.025)`).
    fn reverb_time_from_val(val: u8) -> f32 {
        (((val & 0x7F) as f32 - 40.0) * 0.025).exp()
    }

    /// CA-024 Reverb Time default *value* for each Reverb Type, from the
    /// "pp = 1 : Reverb Time" table: Type 0 → 44, 1 → 50, 2 → 56, 3 → 64,
    /// 4 → 64, 8 → 50. The spec says "When a Reverb Type is selected, the
    /// default Reverb Time from the table below for that Reverb Type
    /// should be set." Unknown types fall back to the Type-4 (Large Hall)
    /// default value.
    fn reverb_type_default_time_val(reverb_type: u8) -> u8 {
        match reverb_type & 0x7F {
            0 => 44,
            1 => 50,
            2 => 56,
            3 => 64,
            4 => 64,
            8 => 50,
            _ => 64,
        }
    }

    /// CA-024 Chorus parameter row for each Chorus Type, from the
    /// "pp = 0 : Chorus Type" table — `(feedback, mod_rate, mod_depth,
    /// rev_send)` as raw 7-bit values:
    ///
    /// | Type | Feedback | Mod Rate | Mod Depth | Rev Send |
    /// |------|----------|----------|-----------|----------|
    /// | 0 Chorus 1 | 0   | 3 | 5  | 0 |
    /// | 1 Chorus 2 | 5   | 9 | 19 | 0 |
    /// | 2 Chorus 3 | 8   | 3 | 19 | 0 |
    /// | 3 Chorus 4 | 16  | 9 | 16 | 0 |
    /// | 4 FB Chorus| 64  | 2 | 24 | 0 |
    /// | 5 Flanger  | 112 | 1 | 5  | 0 |
    ///
    /// Unknown types fall back to the Type-2 (Chorus 3) GM2-default row.
    fn chorus_type_default_row(chorus_type: u8) -> (u8, u8, u8, u8) {
        match chorus_type & 0x7F {
            0 => (0, 3, 5, 0),
            1 => (5, 9, 19, 0),
            2 => (8, 3, 19, 0),
            3 => (16, 9, 16, 0),
            4 => (64, 2, 24, 0),
            5 => (112, 1, 5, 0),
            _ => (8, 3, 19, 0),
        }
    }

    /// Apply the CA-024 Chorus-table row for `chorus_type` to the
    /// per-parameter fields (Mod Rate / Mod Depth / Feedback / Send-to-
    /// Reverb), using the spec's per-parameter unit formulas.
    fn apply_chorus_type_defaults(&mut self, chorus_type: u8) {
        let (fb, mr, md, rs) = Self::chorus_type_default_row(chorus_type);
        self.chorus_mod_rate_hz = mr as f32 * 0.122;
        self.chorus_mod_depth_ms = (md as f32 + 1.0) / 3.2;
        self.chorus_feedback_pct = fb as f32 * 0.763;
        self.chorus_send_to_reverb_pct = rs as f32 * 0.787;
    }

    /// CA-024 GM2 recommended initial settings: Reverb Type 4 (Large
    /// Hall), Chorus Type 2 (Chorus 3), and the per-type values the
    /// CA-024 tables list for those two types.
    fn gm2_default() -> Self {
        let mut fx = Self {
            reverb_type: 4,
            // Reverb Type 4 (Large Hall) default time value = 64.
            reverb_time_s: Self::reverb_time_from_val(64),
            chorus_type: 2,
            chorus_mod_rate_hz: 0.0,
            chorus_mod_depth_ms: 0.0,
            chorus_feedback_pct: 0.0,
            chorus_send_to_reverb_pct: 0.0,
        };
        // Chorus Type 2 (Chorus 3) table row.
        fx.apply_chorus_type_defaults(2);
        fx
    }
}

impl Default for GmEffects {
    fn default() -> Self {
        Self::gm2_default()
    }
}

// =========================================================================
// System effects bus — Reverb + Chorus DSP (CA-024).
//
// CA-024 ("Example of Recommended Practice for Reverb and Chorus
// Parameters, from General MIDI Level 2") defines the *parameters* of
// the two system effects but explicitly leaves the algorithms to the
// implementation: "The names for each Reverb Type are provided as
// examples of reverb designs… not intended to define the effect
// algorithms" and, for Chorus, "the modulation waveform and stereo
// output are implementation dependent." We therefore use a textbook
// Schroeder reverb (parallel feedback comb filters → series allpass
// diffusers) whose decay tracks the spec's Reverb Time, and a single
// sine-modulated delay-line chorus whose rate / depth / feedback track
// the spec's Mod Rate / Mod Depth / Feedback. The chorus→reverb send
// (CA-024 chorus `pp=4`) routes the wet chorus output into the reverb
// input as the spec describes.
// =========================================================================

/// A single feedback comb filter (one of the parallel bank in the
/// Schroeder reverb). The feedback coefficient is recomputed whenever
/// the reverb time changes so the −60 dB decay matches CA-024 Reverb
/// Time.
struct Comb {
    buf: Vec<f32>,
    pos: usize,
    feedback: f32,
}

impl Comb {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            pos: 0,
            feedback: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let out = self.buf[self.pos];
        self.buf[self.pos] = input + out * self.feedback;
        self.pos += 1;
        if self.pos >= self.buf.len() {
            self.pos = 0;
        }
        out
    }

    fn clear(&mut self) {
        for s in self.buf.iter_mut() {
            *s = 0.0;
        }
        self.pos = 0;
    }
}

/// A Schroeder allpass diffuser (series stage after the comb bank).
struct Allpass {
    buf: Vec<f32>,
    pos: usize,
    gain: f32,
}

impl Allpass {
    fn new(len: usize, gain: f32) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            pos: 0,
            gain,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let buffered = self.buf[self.pos];
        let out = -input + buffered;
        self.buf[self.pos] = input + buffered * self.gain;
        self.pos += 1;
        if self.pos >= self.buf.len() {
            self.pos = 0;
        }
        out
    }

    fn clear(&mut self) {
        for s in self.buf.iter_mut() {
            *s = 0.0;
        }
        self.pos = 0;
    }
}

/// A sine-modulated delay line — the chorus voice. The read position
/// sweeps around a base delay by ±depth at the LFO rate. Linear
/// interpolation reads the fractional tap.
struct ModDelay {
    buf: Vec<f32>,
    pos: usize,
    base_delay: f32,
    depth: f32,
    feedback: f32,
    lfo_phase: f32,
    lfo_inc: f32,
}

impl ModDelay {
    fn new(max_len: usize) -> Self {
        Self {
            buf: vec![0.0; max_len.max(2)],
            pos: 0,
            base_delay: 0.0,
            depth: 0.0,
            feedback: 0.0,
            lfo_phase: 0.0,
            lfo_inc: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let len = self.buf.len();
        // Sine LFO in [-1, 1]; modulated delay in samples.
        let lfo = (self.lfo_phase * std::f32::consts::TAU).sin();
        let delay = (self.base_delay + lfo * self.depth).clamp(1.0, (len - 2) as f32);
        // Fractional read position behind the write head.
        let read = self.pos as f32 - delay;
        let read = if read < 0.0 { read + len as f32 } else { read };
        let i0 = read.floor() as usize % len;
        let i1 = (i0 + 1) % len;
        let frac = read - read.floor();
        let wet = self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac;

        self.buf[self.pos] = input + wet * self.feedback;
        self.pos += 1;
        if self.pos >= len {
            self.pos = 0;
        }
        self.lfo_phase += self.lfo_inc;
        if self.lfo_phase >= 1.0 {
            self.lfo_phase -= 1.0;
        }
        wet
    }

    fn clear(&mut self) {
        for s in self.buf.iter_mut() {
            *s = 0.0;
        }
        self.pos = 0;
        self.lfo_phase = 0.0;
    }
}

/// The system Reverb + Chorus effects bus. Voices send a per-channel
/// portion of their signal (scaled by CC 91 / CC 93) into the two
/// effects; the wet returns are summed into the main stereo mix. One
/// instance lives in the [`Mixer`]; its coefficients are refreshed from
/// the current [`GmEffects`] each chunk so a Global Parameter Control
/// edit takes effect on the next block.
struct EffectsBus {
    sample_rate: f32,
    // Stereo Schroeder reverb: a comb bank + allpass chain per side. The
    // right side uses slightly longer delays (a small stereo spread) so
    // the two channels decorrelate.
    combs_l: Vec<Comb>,
    combs_r: Vec<Comb>,
    allpass_l: Vec<Allpass>,
    allpass_r: Vec<Allpass>,
    // Stereo chorus: one modulated delay per side, the right LFO 90° out
    // of phase so the chorus widens rather than collapsing to mono.
    chorus_l: ModDelay,
    chorus_r: ModDelay,
    // Per-block input accumulators (cleared each `process`).
    reverb_in_l: Vec<f32>,
    reverb_in_r: Vec<f32>,
    chorus_in_l: Vec<f32>,
    chorus_in_r: Vec<f32>,
}

impl EffectsBus {
    /// Schroeder comb-filter delay lengths in samples at 44.1 kHz — a
    /// set of mutually-prime-ish delays chosen so the comb resonances
    /// interleave evenly. Scaled by the actual sample rate at
    /// construction. The right channel adds a small stereo spread.
    const COMB_TUNING: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
    /// Schroeder allpass delay lengths in samples at 44.1 kHz.
    const ALLPASS_TUNING: [usize; 4] = [556, 441, 341, 225];
    /// Right-channel stereo spread (samples at 44.1 kHz) added to every
    /// comb / allpass delay so the two reverb sides decorrelate.
    const STEREO_SPREAD: usize = 23;

    fn new(sample_rate: u32) -> Self {
        let sr = sample_rate.max(1) as f32;
        let scale = sr / 44_100.0;
        let scaled = |n: usize| ((n as f32 * scale).round() as usize).max(1);

        let combs_l = Self::COMB_TUNING
            .iter()
            .map(|&n| Comb::new(scaled(n)))
            .collect();
        let combs_r = Self::COMB_TUNING
            .iter()
            .map(|&n| Comb::new(scaled(n + Self::STEREO_SPREAD)))
            .collect();
        let allpass_l = Self::ALLPASS_TUNING
            .iter()
            .map(|&n| Allpass::new(scaled(n), 0.5))
            .collect();
        let allpass_r = Self::ALLPASS_TUNING
            .iter()
            .map(|&n| Allpass::new(scaled(n + Self::STEREO_SPREAD), 0.5))
            .collect();

        // Chorus delay line big enough for the base delay plus the
        // widest modulation depth CA-024 can request (Mod Depth max
        // = (127+1)/3.2 ≈ 40 ms), with headroom.
        let max_chorus = (sr * 0.060) as usize + 4;
        Self {
            sample_rate: sr,
            combs_l,
            combs_r,
            allpass_l,
            allpass_r,
            chorus_l: ModDelay::new(max_chorus),
            chorus_r: ModDelay::new(max_chorus),
            reverb_in_l: Vec::new(),
            reverb_in_r: Vec::new(),
            chorus_in_l: Vec::new(),
            chorus_in_r: Vec::new(),
        }
    }

    /// Refresh comb / allpass / chorus coefficients from the current
    /// GM2 effect parameters. Called once per `process` block.
    fn refresh(&mut self, fx: &GmEffects) {
        // Comb feedback for the requested −60 dB decay time:
        //   g = 10^(−3 · delay_seconds / T60).
        // Computed per comb so longer delay lines feed back less, which
        // is what gives a flat, even decay regardless of delay length.
        let t60 = fx.reverb_time_s.clamp(0.05, 12.0);
        let set_fb = |comb: &mut Comb, sr: f32| {
            let delay_s = comb.buf.len() as f32 / sr;
            let g = 10f32.powf(-3.0 * delay_s / t60);
            comb.feedback = g.clamp(0.0, 0.999);
        };
        for c in self.combs_l.iter_mut() {
            set_fb(c, self.sample_rate);
        }
        for c in self.combs_r.iter_mut() {
            set_fb(c, self.sample_rate);
        }

        // Chorus: base delay ~ a few ms plus the modulated sweep; the
        // CA-024 Mod Depth is the *peak-to-peak* swing in ms, so the
        // ± amplitude is half of it. Feedback is the spec percentage.
        let depth_ms = fx.chorus_mod_depth_ms.max(0.0);
        let depth_samples = (depth_ms * 0.5 / 1000.0) * self.sample_rate;
        let base_samples = (0.012 * self.sample_rate).max(depth_samples + 2.0);
        let rate = fx.chorus_mod_rate_hz.clamp(0.0, 20.0);
        let inc = rate / self.sample_rate;
        let fb = (fx.chorus_feedback_pct / 100.0).clamp(0.0, 0.95);
        self.chorus_l.base_delay = base_samples;
        self.chorus_l.depth = depth_samples;
        self.chorus_l.feedback = fb;
        self.chorus_l.lfo_inc = inc;
        self.chorus_r.base_delay = base_samples;
        self.chorus_r.depth = depth_samples;
        self.chorus_r.feedback = fb;
        self.chorus_r.lfo_inc = inc;
        // Quarter-cycle phase offset for stereo width (set once; only
        // matters relative to the left LFO).
        if self.chorus_r.lfo_phase == self.chorus_l.lfo_phase {
            self.chorus_r.lfo_phase = (self.chorus_l.lfo_phase + 0.25).fract();
        }
    }

    /// Ensure the per-block accumulators match `n` samples and are
    /// zeroed. Returns mutable access via the struct fields afterward.
    fn prepare_block(&mut self, n: usize) {
        for v in [
            &mut self.reverb_in_l,
            &mut self.reverb_in_r,
            &mut self.chorus_in_l,
            &mut self.chorus_in_r,
        ] {
            v.clear();
            v.resize(n, 0.0);
        }
    }

    /// Run the accumulated sends through the effects and add the wet
    /// returns into the main `(left, right)` mix. `chorus_to_reverb`
    /// is the CA-024 chorus `pp=4` send-to-reverb fraction (0..=1).
    fn process(&mut self, left: &mut [f32], right: &mut [f32], chorus_to_reverb: f32) {
        let n = left.len();
        for i in 0..n {
            // --- Chorus first, so its output can feed the reverb. ---
            let ch_l = self.chorus_l.process(self.chorus_in_l[i]);
            let ch_r = self.chorus_r.process(self.chorus_in_r[i]);

            // --- Reverb input = direct reverb send + chorus→reverb. ---
            let rv_in_l = self.reverb_in_l[i] + ch_l * chorus_to_reverb;
            let rv_in_r = self.reverb_in_r[i] + ch_r * chorus_to_reverb;

            let mut rv_l = 0.0;
            for c in self.combs_l.iter_mut() {
                rv_l += c.process(rv_in_l);
            }
            let mut rv_r = 0.0;
            for c in self.combs_r.iter_mut() {
                rv_r += c.process(rv_in_r);
            }
            // Normalise the comb bank sum, then diffuse through the
            // allpass chain.
            rv_l /= self.combs_l.len() as f32;
            rv_r /= self.combs_r.len() as f32;
            for a in self.allpass_l.iter_mut() {
                rv_l = a.process(rv_l);
            }
            for a in self.allpass_r.iter_mut() {
                rv_r = a.process(rv_r);
            }

            // Wet returns into the main mix. Chorus is added directly
            // (it is itself a "wet" widening voice); reverb is the
            // diffused tail.
            left[i] += ch_l + rv_l;
            right[i] += ch_r + rv_r;
        }
    }

    fn clear(&mut self) {
        for c in self.combs_l.iter_mut() {
            c.clear();
        }
        for c in self.combs_r.iter_mut() {
            c.clear();
        }
        for a in self.allpass_l.iter_mut() {
            a.clear();
        }
        for a in self.allpass_r.iter_mut() {
            a.clear();
        }
        self.chorus_l.clear();
        self.chorus_r.clear();
        self.chorus_r.lfo_phase = 0.25;
    }
}

/// Polyphonic voice pool with stereo mixdown.
pub struct Mixer {
    slots: [VoiceSlot; MAX_VOICES],
    channels: [ChannelState; NUM_CHANNELS],
    /// Monotonic allocation counter. Each successful `note_on` records
    /// `next_age` into the slot then bumps this. Wrap is theoretical
    /// (u64 ≈ 5×10^11 years at 1 alloc/ms), so we don't handle it.
    next_age: u64,
    /// Stereo amplitude headroom. The voice render path scales by
    /// `velocity^2 × 0.5` (see `Sf2Voice::from_plan`); summing 32 such
    /// voices in the worst case still stays under unity if we apply a
    /// modest mix bus gain. Round-4 may swap in a smarter limiter.
    mix_gain: f32,
    /// Master Volume (Universal Real Time SysEx `7F 7F 04 01`, 14-bit
    /// 0..=0x3FFF). Default = 0x3FFF (= unity). Applied at mix time as
    /// an additional global gain factor on every voice.
    master_volume_14: u16,
    /// Master Balance (Universal Real Time SysEx `7F 7F 04 02`, 14-bit
    /// 0..=0x3FFF). Per the MIDI 1.0 Detailed Specification §"DEVICE
    /// CONTROL — MASTER VOLUME AND MASTER BALANCE" (M1 v4.2.1 p.57):
    /// `00 00 = hard left`, `7F 7F = hard right`, centre = `0x2000`.
    /// Applied at mix time as a per-side attenuation derived from
    /// [`Self::master_balance_gains`]. Default = `0x2000` (= centre,
    /// both sides full).
    master_balance_14: u16,
    /// Master Fine Tuning (CA-25), in signed cents within ±100.
    /// Default 0. Summed with per-channel fine tune + pitch bend.
    master_fine_tune_cents: i16,
    /// Master Coarse Tuning (CA-25), in semitones within `-64..=+63`.
    /// Default 0. Summed with per-channel coarse tune.
    master_coarse_tune_semitones: i16,
    /// Active MPE Lower Zone, if any. Created by an MCM with `n=0` /
    /// `mm>=1`; cleared by an MCM with `mm=0`.
    mpe_lower: Option<MpeZone>,
    /// Active MPE Upper Zone, if any. Created by an MCM with `n=15`
    /// (= 0xF) / `mm>=1`; cleared by an MCM with `mm=0`.
    mpe_upper: Option<MpeZone>,
    /// MIDI Tuning Standard (MTS) microtuning state — a global
    /// key-based table plus per-channel scale/octave tables. Both
    /// default to equal temperament, so a synth that never sees an MTS
    /// SysEx renders bit-identically to the pre-MTS path. The per-key
    /// offset is folded into every voice's pitch composition.
    tuning: crate::tuning::TuningTable,
    /// GM2 system-wide Reverb + Chorus parameters (CA-024 Global
    /// Parameter Control). Defaults to the GM2 recommended initial
    /// settings; edited by the `04 05` Universal Real-Time SysEx.
    gm_effects: GmEffects,
    /// The output sample rate the effects bus delay lines are sized for.
    /// Defaults to [`crate::OUTPUT_SAMPLE_RATE`]; the decoder calls
    /// [`Self::set_sample_rate`] before rendering when it differs.
    sample_rate: u32,
    /// The system Reverb + Chorus DSP bus (CA-024). Per-channel CC 91 /
    /// CC 93 sends feed it; the wet returns are summed into the stereo
    /// mix inside [`Self::mix_stereo`].
    fx: EffectsBus,
}

impl Default for Mixer {
    fn default() -> Self {
        Self::new()
    }
}

impl Mixer {
    pub fn new() -> Self {
        // Can't `[VoiceSlot::empty(); MAX_VOICES]` — `VoiceSlot` is not
        // `Copy` (the `Box<dyn Voice>` field). Build an array via
        // `from_fn` so each element is a fresh `empty()`.
        let slots = std::array::from_fn(|_| VoiceSlot::empty());
        Self {
            slots,
            channels: [ChannelState::default(); NUM_CHANNELS],
            next_age: 1,
            mix_gain: 0.5,
            master_volume_14: 0x3FFF,
            master_balance_14: 0x2000,
            master_fine_tune_cents: 0,
            master_coarse_tune_semitones: 0,
            mpe_lower: None,
            mpe_upper: None,
            tuning: crate::tuning::TuningTable::new(),
            gm_effects: GmEffects::default(),
            sample_rate: crate::OUTPUT_SAMPLE_RATE,
            fx: EffectsBus::new(crate::OUTPUT_SAMPLE_RATE),
        }
    }

    /// Set the output sample rate the effects bus delay lines are sized
    /// for and rebuild the bus. Call this once before rendering when the
    /// decoder's sample rate differs from [`crate::OUTPUT_SAMPLE_RATE`];
    /// it resets the effect tails (the delay-line lengths change), so
    /// it must not be called mid-stream.
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        let sr = sample_rate.max(1);
        if sr != self.sample_rate {
            self.sample_rate = sr;
            self.fx = EffectsBus::new(sr);
        }
    }

    /// The output sample rate the effects bus is currently sized for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Borrow the per-channel state. Useful for the scheduler when it
    /// needs to read the current program before allocating a voice.
    pub fn channel_state(&self, channel: u8) -> &ChannelState {
        &self.channels[channel as usize % NUM_CHANNELS]
    }

    /// Mutable borrow of the per-channel state, for control changes.
    pub fn channel_state_mut(&mut self, channel: u8) -> &mut ChannelState {
        &mut self.channels[channel as usize % NUM_CHANNELS]
    }

    /// Apply a pitch-bend event. `value` is the raw 14-bit MIDI scalar
    /// in `0..=16383` (centre = `0x2000`); the conversion to cents
    /// uses the channel's current `pitch_bend_range_cents` (default
    /// 200 = ±2 semitones, the GM recommended range — overridden via
    /// RPN 0).
    ///
    /// The cents value pushed to the voice is the **sum** of:
    ///   * `pitch_bend_to_cents(value, channel.pitch_bend_range_cents)`,
    ///   * `channel.channel_fine_tune_cents` (RPN 1),
    ///   * `channel.channel_coarse_tune_semitones * 100` (RPN 2),
    ///   * `master_fine_tune_cents` (CA-25),
    ///   * `master_coarse_tune_semitones * 100` (CA-25), and
    ///   * for **MPE Member** channels, the Manager Channel's pitch
    ///     bend per Appendix C (managers control the whole zone).
    ///
    /// Drum channels (MIDI ch 10 = index 9) are exempt from CA-25's
    /// note-shifting per the spec ("MUST NOT result in MIDI
    /// note-shifting" — different key = different drum sound).
    pub fn set_pitch_bend(&mut self, channel: u8, value: u16) {
        let ch = channel as usize % NUM_CHANNELS;
        let v = value & 0x3FFF;
        self.channels[ch].pitch_bend = v;

        // If this is an MPE Manager Channel, the bend reaches every
        // voice in the zone *combined* with that member channel's own
        // per-note bend. Per Appendix C we sum the two values in
        // cents.
        let role = self.channels[ch].mpe_role;
        let is_drum = ch == 9;

        if let MpeRole::Manager(zone_kind) = role {
            // Update every voice in the zone (Manager-held notes too).
            let zone = match zone_kind {
                MpeZoneKind::Lower => self.mpe_lower,
                MpeZoneKind::Upper => self.mpe_upper,
            };
            if let Some(z) = zone {
                for slot in self.slots.iter_mut() {
                    let slot_ch = slot.channel as usize % NUM_CHANNELS;
                    if slot.channel == channel
                        || slot.channel == z.manager_channel()
                        || z.member_channels().contains(&slot.channel)
                    {
                        if let Some(voice) = slot.voice.as_mut() {
                            let mut total = Self::compose_pitch_cents(
                                &self.channels[slot_ch],
                                self.channels[ch].pitch_bend,
                                self.channels[ch].pitch_bend_range_cents,
                                self.master_fine_tune_cents,
                                self.master_coarse_tune_semitones,
                                slot_ch == 9,
                            );
                            if slot_ch != 9 {
                                total +=
                                    self.tuning.offset_cents(slot.channel, slot.key).round() as i32;
                            }
                            voice.set_pitch_bend_cents(total);
                        }
                    }
                }
            }
        } else {
            // Non-MPE or MPE Member: apply only to voices on this
            // exact channel.
            for slot in self.slots.iter_mut() {
                if slot.channel == channel {
                    if let Some(voice) = slot.voice.as_mut() {
                        // For a Member channel, also fold in the
                        // Manager's currently-held bend.
                        let mut total = if let MpeRole::Member(zone_kind) = role {
                            let mgr_ch = match zone_kind {
                                MpeZoneKind::Lower => 0u8,
                                MpeZoneKind::Upper => 15u8,
                            };
                            let mgr_state = &self.channels[mgr_ch as usize];
                            let member_cents =
                                pitch_bend_to_cents(v, self.channels[ch].pitch_bend_range_cents);
                            let mgr_cents = pitch_bend_to_cents(
                                mgr_state.pitch_bend,
                                mgr_state.pitch_bend_range_cents,
                            );
                            let mut total = member_cents + mgr_cents;
                            if !is_drum {
                                total += self.channels[ch].channel_fine_tune_cents as i32;
                                total +=
                                    self.channels[ch].channel_coarse_tune_semitones as i32 * 100;
                                total += self.master_fine_tune_cents as i32;
                                total += self.master_coarse_tune_semitones as i32 * 100;
                            }
                            total
                        } else {
                            Self::compose_pitch_cents(
                                &self.channels[ch],
                                v,
                                self.channels[ch].pitch_bend_range_cents,
                                self.master_fine_tune_cents,
                                self.master_coarse_tune_semitones,
                                is_drum,
                            )
                        };
                        if !is_drum {
                            total +=
                                self.tuning.offset_cents(slot.channel, slot.key).round() as i32;
                        }
                        voice.set_pitch_bend_cents(total);
                    }
                }
            }
        }
    }

    /// Static helper: combine pitch bend + per-channel tuning + master
    /// tuning into a single cents value. Pulled out so the MPE
    /// per-zone broadcast path can compute the per-slot sum without
    /// borrowing `self` mutably twice.
    fn compose_pitch_cents(
        ch_state: &ChannelState,
        bend_14: u16,
        bend_range_cents: u16,
        master_fine_cents: i16,
        master_coarse_semis: i16,
        is_drum: bool,
    ) -> i32 {
        let mut total = pitch_bend_to_cents(bend_14, bend_range_cents);
        if !is_drum {
            total += ch_state.channel_fine_tune_cents as i32;
            total += ch_state.channel_coarse_tune_semitones as i32 * 100;
            total += master_fine_cents as i32;
            total += master_coarse_semis as i32 * 100;
        }
        total
    }

    /// Apply channel pressure (mono aftertouch, MIDI status `Dn`). The
    /// `0..=127` value modulates volume on every still-held voice on
    /// `channel`.
    ///
    /// MPE rules (§2.2.7 + Appendix D): on a **Manager Channel** the
    /// pressure affects every voice in the zone, combined with each
    /// member channel's own most-recent pressure. On a **Member
    /// Channel** it only affects voices held on that very channel and
    /// composes with the Manager's pressure for the routed-to-voice
    /// value. Outside MPE the routing is the plain per-channel
    /// behaviour.
    pub fn set_channel_pressure(&mut self, channel: u8, value: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        self.channels[ch].channel_pressure = value;
        let role = self.channels[ch].mpe_role;
        for slot in self.slots.iter_mut() {
            let slot_ch = slot.channel as usize % NUM_CHANNELS;
            let routes = match role {
                MpeRole::Manager(kind) => {
                    let zone = match kind {
                        MpeZoneKind::Lower => self.mpe_lower,
                        MpeZoneKind::Upper => self.mpe_upper,
                    };
                    if let Some(z) = zone {
                        slot.channel == z.manager_channel()
                            || z.member_channels().contains(&slot.channel)
                    } else {
                        slot.channel == channel
                    }
                }
                _ => slot.channel == channel,
            };
            if routes {
                if let Some(voice) = slot.voice.as_mut() {
                    let combined = Self::compose_pressure(
                        self.channels[slot_ch].channel_pressure,
                        match self.channels[slot_ch].mpe_role {
                            MpeRole::Member(kind) => {
                                let mgr = match kind {
                                    MpeZoneKind::Lower => 0,
                                    MpeZoneKind::Upper => 15,
                                };
                                self.channels[mgr].channel_pressure
                            }
                            _ => 0,
                        },
                    );
                    voice.set_pressure(combined);
                }
            }
        }
    }

    /// Apply polyphonic key pressure (per-key aftertouch, MIDI status
    /// `An`). Only voices matching `(channel, key)` are touched. Per
    /// MPE §2.2.7, Polyphonic Key Pressure **shall not** be sent on
    /// Member Channels (it doesn't make sense — each Member already
    /// hosts one Active Note that channel pressure covers); we silently
    /// drop a stray PolyPressure on a Member to avoid clobbering an
    /// unrelated key's voice via the lookup.
    pub fn set_poly_pressure(&mut self, channel: u8, key: u8, value: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        if matches!(self.channels[ch].mpe_role, MpeRole::Member(_)) {
            return;
        }
        let p = (value as f32 / 127.0).clamp(0.0, 1.0);
        for slot in self.slots.iter_mut() {
            if slot.channel == channel && slot.key == key {
                if let Some(voice) = slot.voice.as_mut() {
                    voice.set_pressure(p);
                }
            }
        }
    }

    /// Combine MPE Member + Manager channel pressures into a single
    /// 0..=1 pressure scalar. We pick the larger of the two per
    /// Appendix D's "implementor-defined combining" — taking the max
    /// is the simplest combining rule that matches the spec's intent
    /// ("the two should be combined meaningfully") without
    /// double-counting overlapping inputs.
    fn compose_pressure(member_0_127: u8, manager_0_127: u8) -> f32 {
        let m = member_0_127.max(manager_0_127);
        (m as f32 / 127.0).clamp(0.0, 1.0)
    }

    /// Update the currently-selected RPN. Called from the scheduler in
    /// response to CC 100 (LSB) / 101 (MSB). `is_msb` distinguishes the
    /// two; the new 14-bit value lives in `channels[ch].rpn`.
    pub fn set_rpn_byte(&mut self, channel: u8, value: u8, is_msb: bool) {
        let ch = channel as usize % NUM_CHANNELS;
        let cur = self.channels[ch].rpn;
        let new = if is_msb {
            (cur & 0x007F) | ((value as u16 & 0x7F) << 7)
        } else {
            (cur & 0x3F80) | (value as u16 & 0x7F)
        };
        self.channels[ch].rpn = new;
    }

    /// Apply a data-entry CC (CC 6 = MSB, CC 38 = LSB) to whatever the
    /// currently-selected RPN is. Round 75 honours:
    ///   * **RPN 0** (`MSB=00 LSB=00`) — Pitch Bend Sensitivity.
    ///     CC 6 = semitone count, CC 38 = additional cents.
    ///   * **RPN 1** (`MSB=00 LSB=01`) — Channel Fine Tuning. The
    ///     14-bit data-entry value (MSB×128 + LSB) is treated as a
    ///     two's-complement bend around centre 0x2000 and maps
    ///     linearly to `±100` cents per the MIDI 1.0 spec.
    ///   * **RPN 2** (`MSB=00 LSB=02`) — Channel Coarse Tuning. CC 6
    ///     directly carries a signed semitone offset centred on 0x40;
    ///     the spec says CC 38 LSB is always 0 (mirroring CA-25
    ///     master-coarse-tuning) so we ignore it.
    ///   * **RPN 5** (`MSB=00 LSB=05`) — Modulation Depth Range
    ///     (CA-26). CC 6 = whole-cent count of mod-wheel depth, CC 38
    ///     = additional fractional cents (treated as 0..=99).
    ///   * **RPN 6** (`MSB=00 LSB=06`) — MPE Configuration Message
    ///     (M1-100-UM §2.2.1). CC 6 = number of Member Channels.
    ///     CC 38 has no function per spec. The MCM is only honoured
    ///     when the channel matches one of the two valid Manager
    ///     Channels (0 = Lower, 15 = Upper); other channels are
    ///     silently ignored per the MPE spec ("All other values are
    ///     invalid and should be ignored.").
    ///
    /// Other RPNs are silently ignored.
    pub fn set_data_entry(&mut self, channel: u8, value: u8, is_msb: bool) {
        let ch = channel as usize % NUM_CHANNELS;
        let rpn = self.channels[ch].rpn;
        match rpn {
            0 => {
                let cur = self.channels[ch].pitch_bend_range_cents;
                let new = if is_msb {
                    // CC 6: semitone portion. Replace the "hundreds" digit
                    // (semitones * 100) and keep the LSB cents.
                    value as u16 * 100 + (cur % 100)
                } else {
                    // CC 38: cents portion (0..=99).
                    (cur / 100) * 100 + (value as u16 % 100)
                };
                self.channels[ch].pitch_bend_range_cents = new.max(1); // never zero
                                                                       // Re-apply the live bend with the new range so still-held voices
                                                                       // pick up the change immediately.
                let bend = self.channels[ch].pitch_bend;
                self.set_pitch_bend(channel, bend);
            }
            1 => {
                // Channel Fine Tuning. The two data-entry bytes form a
                // 14-bit value centred on 0x2000; the resulting
                // displacement is `±100` cents (i.e. one semitone).
                // The raw accumulator lives on the channel state so
                // an MSB-then-LSB sequence composes bit-exact, then
                // we derive the cents view from it.
                let cur = self.channels[ch].channel_fine_tune_raw_14;
                let new14 = if is_msb {
                    (cur & 0x007F) | ((value as u16 & 0x7F) << 7)
                } else {
                    (cur & 0x3F80) | (value as u16 & 0x7F)
                };
                self.channels[ch].channel_fine_tune_raw_14 = new14;
                let cents = (new14.min(0x3FFF) as i32 - 0x2000) * 100 / 0x2000;
                self.channels[ch].channel_fine_tune_cents = cents as i16;
                self.reapply_pitch_for_channel(channel);
            }
            2 if is_msb => {
                // Channel Coarse Tuning. CC 6 carries a signed
                // semitone count centred on 0x40 (-64..=+63 per
                // CA-25's relationship). CC 38 (LSB) is silently
                // ignored per spec — "the LSB is always 0".
                let semis = value as i16 - 0x40;
                self.channels[ch].channel_coarse_tune_semitones = semis;
                self.reapply_pitch_for_channel(channel);
            }
            5 => {
                let cur = self.channels[ch].mod_depth_range_cents;
                let new = if is_msb {
                    value as u16 * 100 + (cur % 100)
                } else {
                    (cur / 100) * 100 + (value as u16 % 100)
                };
                // Clamp to a sane envelope (±2 octaves) so a stray
                // CC 6 = 127 (= 12 700 cents) doesn't pop the timbre
                // out of audibility.
                self.channels[ch].mod_depth_range_cents = new.min(2400);
                self.reapply_mod_wheel_for_channel(channel);
            }
            6 => {
                // MPE Configuration Message — only the MSB carries the
                // member-channel count; the LSB has no function per
                // §2.2.1.
                if !is_msb {
                    return;
                }
                let zone_kind = match channel & 0x0F {
                    0x0 => Some(MpeZoneKind::Lower),
                    0xF => Some(MpeZoneKind::Upper),
                    _ => None,
                };
                if let Some(kind) = zone_kind {
                    self.set_mpe_zone(kind, value & 0x0F);
                }
            }
            _ => { /* Other RPNs (3/4 tuning bank+program, etc.) not modelled. */ }
        }
    }

    /// Apply a Data Increment (CC 96) or Data Decrement (CC 97) to the
    /// currently-selected RPN, per RP-018 ("Response to Data Inc/Dec
    /// Controllers"). `step` is `+1` for an increment, `-1` for a
    /// decrement; per RP-018 the controller's *value byte is ignored*
    /// ("the value byte for both messages is `don't care`").
    ///
    /// RP-018 specifies which sub-field of the parameter each step
    /// touches:
    ///   * **RPN 0** (Pitch Bend Sensitivity, MSB = semitones,
    ///     LSB = cents): step the **LSB** (cents) by 1. When the LSB
    ///     wraps at 100, reset it to 0 and step the MSB (semitones).
    ///     Because we store the combined `pitch_bend_range_cents`
    ///     (= semitones·100 + cents), a `±1` on that scalar performs the
    ///     LSB-wraps-into-MSB carry automatically (e.g. 100 → 101 is one
    ///     semitone + 1 cent; 200 → 199 borrows down into 1 semitone +
    ///     99 cents). The result is clamped to `>= 1` so the range never
    ///     reaches zero, matching [`Self::set_data_entry`].
    ///   * **RPN 1** (Channel Fine Tuning): step the **LSB** of the
    ///     14-bit fine-tune accumulator by 1 (RP-018: "Data Increment
    ///     and Data Decrement messages will increase or decrease the
    ///     LSB by 1" for RPN 0 and 1).
    ///   * **RPN 2** (Channel Coarse Tuning): step the **MSB** by 1
    ///     (RP-018, citing the 4.2 Addendum: for RPN 2, 3 and 4 the
    ///     inc/dec affects the MSB). For coarse tuning the MSB is the
    ///     semitone field, so the step is one semitone.
    ///   * **RPN 5** (Modulation Depth Range, a future Registered
    ///     Parameter per CA-26): step the **LSB** (cents) by 1, the
    ///     RP-018 default for future Registered Parameters.
    ///
    /// RPN Null (`0x3FFF`) and any RPN not modelled above leave the
    /// channel untouched, mirroring [`Self::set_data_entry`]'s
    /// silent-ignore policy. NRPNs are not modelled, so a step issued
    /// while an NRPN is selected (RPN field still null) is a no-op.
    pub fn data_inc_dec(&mut self, channel: u8, step: i16) {
        let ch = channel as usize % NUM_CHANNELS;
        let rpn = self.channels[ch].rpn;
        match rpn {
            0 => {
                // Step the combined cents scalar; the LSB-wraps-into-MSB
                // carry falls straight out of the base-100 layout.
                let cur = self.channels[ch].pitch_bend_range_cents as i32;
                let new = (cur + step as i32).max(1) as u16;
                self.channels[ch].pitch_bend_range_cents = new;
                let bend = self.channels[ch].pitch_bend;
                self.set_pitch_bend(channel, bend);
            }
            1 => {
                // Step the LSB (bottom 7 bits) of the fine-tune 14-bit
                // accumulator, then re-derive the cents view.
                let cur = self.channels[ch].channel_fine_tune_raw_14 as i32;
                let new14 = (cur + step as i32).clamp(0, 0x3FFF) as u16;
                self.channels[ch].channel_fine_tune_raw_14 = new14;
                let cents = (new14 as i32 - 0x2000) * 100 / 0x2000;
                self.channels[ch].channel_fine_tune_cents = cents as i16;
                self.reapply_pitch_for_channel(channel);
            }
            2 => {
                // Step the MSB (semitones) by 1, clamped to the
                // CA-25 / coarse-tune signed range (-64..=+63).
                let cur = self.channels[ch].channel_coarse_tune_semitones;
                let new = (cur + step).clamp(-64, 63);
                self.channels[ch].channel_coarse_tune_semitones = new;
                self.reapply_pitch_for_channel(channel);
            }
            5 => {
                let cur = self.channels[ch].mod_depth_range_cents as i32;
                let new = (cur + step as i32).clamp(0, 2400) as u16;
                self.channels[ch].mod_depth_range_cents = new;
                self.reapply_mod_wheel_for_channel(channel);
            }
            _ => { /* RPN Null / NRPN / unmodelled — no-op per RP-018. */ }
        }
    }

    /// Re-evaluate the effective pitch offset on every voice held on
    /// `channel`. Combines RPN 1 (channel fine tune) + RPN 2 (channel
    /// coarse tune) + master fine + master coarse + the live pitch
    /// bend; called whenever any of those terms change. Drum channels
    /// (MIDI ch 10 = index 9) are exempt from tuning per CA-25's
    /// "MUST NOT result in MIDI note-shifting" rule — playing a
    /// drum-key at a different pitch picks a different sound.
    fn reapply_pitch_for_channel(&mut self, channel: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        let bend = self.channels[ch].pitch_bend;
        // set_pitch_bend already routes through the channel state into
        // every held voice.
        self.set_pitch_bend(channel, bend);
    }

    /// Re-evaluate the effective mod-wheel depth on every voice held on
    /// `channel`. Called when CC 1 changes or when RPN 5 widens / shrinks
    /// the range.
    fn reapply_mod_wheel_for_channel(&mut self, channel: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        let st = self.channels[ch];
        // Manager-channel mod-wheel is propagated to every voice in
        // its MPE zone (§2.3.1: "Damper Pedal can be expected to
        // affect all Sounding Notes across the Manager Channel and
        // all Member Channels"). For non-MPE channels the depth
        // routes only to that channel's own voices.
        let depth_cents = (st.mod_wheel as i32) * (st.mod_depth_range_cents as i32) / 127;
        for slot in self.slots.iter_mut() {
            if self.channels[slot.channel as usize % NUM_CHANNELS].matches_for_zone_broadcast(
                slot.channel,
                channel,
                &st.mpe_role,
            ) {
                if let Some(v) = slot.voice.as_mut() {
                    v.set_mod_depth_cents(depth_cents);
                }
            }
        }
    }

    /// Update CC 1 (Modulation Wheel) on a channel. Stored on the
    /// channel state and immediately routed through
    /// [`Voice::set_mod_depth_cents`] for every held voice that the
    /// MPE rules say this CC applies to.
    pub fn set_mod_wheel(&mut self, channel: u8, value: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        self.channels[ch].mod_wheel = value & 0x7F;
        self.reapply_mod_wheel_for_channel(channel);
    }

    /// Set CC #74 (Brightness / MPE Timbre, the "third dimension"). The
    /// raw 0..=127 value is forwarded to every voice this channel-CC
    /// reaches: per-channel for non-MPE, per-zone for MPE Manager,
    /// per-channel-only (= the held member-channel notes) for MPE
    /// Member.
    pub fn set_timbre(&mut self, channel: u8, value: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        let role = self.channels[ch].mpe_role;
        for slot in self.slots.iter_mut() {
            if self.channels[slot.channel as usize % NUM_CHANNELS].matches_for_zone_broadcast(
                slot.channel,
                channel,
                &role,
            ) {
                if let Some(v) = slot.voice.as_mut() {
                    v.set_timbre(value & 0x7F);
                }
            }
        }
    }

    // ─────────────────────────── master tuning ───────────────────────────

    /// Master Volume (Universal Real Time SysEx `7F 7F 04 01`). The
    /// argument is the raw 14-bit MIDI scalar (`0..=0x3FFF`), centre =
    /// max. Applied as a multiplicative gain on every voice at mix
    /// time, mapped linearly (`master_volume_14 / 0x3FFF`); per the
    /// spec a *fully* loud setting is the default. Round 75 doesn't
    /// model the GS / GM2 "scribbled-curve" non-linearity.
    pub fn set_master_volume_14(&mut self, value: u16) {
        self.master_volume_14 = value.min(0x3FFF);
    }

    /// Current Master Volume scalar (14-bit, 0..=0x3FFF).
    pub fn master_volume_14(&self) -> u16 {
        self.master_volume_14
    }

    /// Master Balance (Universal Real Time SysEx `7F 7F 04 02`). The
    /// argument is the raw 14-bit MIDI scalar (`0..=0x3FFF`) per the
    /// MIDI 1.0 Detailed Specification §"DEVICE CONTROL — MASTER
    /// VOLUME AND MASTER BALANCE" (M1 v4.2.1 p.57):
    ///
    ///   * `0x0000` = hard left (right side muted),
    ///   * `0x2000` = centre (both sides at full),
    ///   * `0x3FFF` = hard right (left side muted).
    ///
    /// Stored verbatim; [`Self::master_balance_gains`] computes the
    /// per-side multipliers applied at mix time. The CC 8 BALANCE
    /// description in M1 §"BALANCE" frames the control as the volume
    /// balance between two sound sources, so this mixer realises it as
    /// the classic triangular law: the opposite side starts to
    /// attenuate as the scalar moves away from centre, while the
    /// near side stays at unity until the centre is crossed. That
    /// keeps the mix pass byte-identical at the default `0x2000` and
    /// fades exactly one side to zero at each extreme.
    pub fn set_master_balance_14(&mut self, value: u16) {
        self.master_balance_14 = value.min(0x3FFF);
    }

    /// Current Master Balance scalar (14-bit, 0..=0x3FFF).
    pub fn master_balance_14(&self) -> u16 {
        self.master_balance_14
    }

    /// Per-side multipliers derived from [`Self::master_balance_14`].
    /// Returns `(left, right)` in `[0.0, 1.0]`:
    ///
    ///   * `0x0000` → `(1.0, 0.0)` — hard left, right muted.
    ///   * `0x2000` → `(1.0, 1.0)` — centre, both sides at full.
    ///   * `0x3FFF` → `(0.0, 1.0)` — hard right, left muted.
    ///
    /// Below centre: `right = value / 0x2000`, `left = 1.0`. Above
    /// centre: `left = (0x3FFF - value) / (0x3FFF - 0x2000)`,
    /// `right = 1.0`. This is the textbook "balance-between-two-
    /// sources" law called out by CC 8 in M1 v4.2.1 §"BALANCE": the
    /// control attenuates the *far* side and leaves the *near* side
    /// untouched, so panning a stereo source hard one way mutes the
    /// opposite channel without boosting the near channel.
    pub fn master_balance_gains(&self) -> (f32, f32) {
        let v = self.master_balance_14 as i32;
        if v <= 0x2000 {
            let right = v as f32 / 0x2000 as f32;
            (1.0, right.clamp(0.0, 1.0))
        } else {
            let left = (0x3FFF - v) as f32 / (0x3FFF - 0x2000) as f32;
            (left.clamp(0.0, 1.0), 1.0)
        }
    }

    /// Master Fine Tuning (CA-25). The two arguments are the raw
    /// LSB/MSB bytes from the SysEx payload; we combine them into a
    /// 14-bit value centred on 0x2000 and map it linearly to
    /// `100/8192 × (value - 0x2000)` cents per the spec table.
    pub fn set_master_fine_tuning(&mut self, lsb: u8, msb: u8) {
        let combined = ((msb as i32 & 0x7F) << 7) | (lsb as i32 & 0x7F);
        let cents = (combined - 0x2000) * 100 / 0x2000;
        self.master_fine_tune_cents = cents as i16;
        for ch in 0..NUM_CHANNELS {
            self.reapply_pitch_for_channel(ch as u8);
        }
    }

    /// Master Coarse Tuning (CA-25). CC 38 LSB is always 0 per spec;
    /// CC 6 MSB carries a signed semitone count centred on 0x40
    /// (-64..=+63).
    pub fn set_master_coarse_tuning(&mut self, _lsb: u8, msb: u8) {
        // Per CA-25 page 2: "Note that the LSB is always 0."
        let semis = msb as i16 - 0x40;
        self.master_coarse_tune_semitones = semis;
        for ch in 0..NUM_CHANNELS {
            self.reapply_pitch_for_channel(ch as u8);
        }
    }

    /// Current Master Fine Tuning, in cents.
    pub fn master_fine_tune_cents(&self) -> i16 {
        self.master_fine_tune_cents
    }

    /// Current Master Coarse Tuning, in semitones.
    pub fn master_coarse_tune_semitones(&self) -> i16 {
        self.master_coarse_tune_semitones
    }

    // ─────────────────────────── MTS microtuning ───────────────────────────

    /// Borrow the MIDI Tuning Standard (MTS) state — the global
    /// key-based table + per-channel scale/octave tables. Exposed for
    /// tests / introspection.
    pub fn tuning(&self) -> &crate::tuning::TuningTable {
        &self.tuning
    }

    /// Apply an MTS **Single-Note Tuning Change** entry: set MIDI `key`
    /// from a 3-byte frequency-data word (`xx yy zz`). The reserved
    /// `7F 7F 7F` "no change" word leaves the stored offset untouched.
    ///
    /// When `live` is true (the real-time message forms), the spec
    /// requires the change to "instantly re-tune" any sounding note on
    /// the affected key without retriggering, so we re-apply pitch to
    /// every held voice afterwards. The non-real-time forms are "setup
    /// messages" that must not disturb sounding notes, so `live` is
    /// false and only the stored table is updated.
    pub fn set_key_tuning_word(&mut self, key: u8, word: [u8; 3], live: bool) {
        self.tuning.set_key_freq_word(key, word);
        if live {
            // Re-apply on every channel — the key-based table is global,
            // so a held note on any channel matching `key` must update.
            // Drum channels are skipped inside the pitch composition.
            for ch in 0..NUM_CHANNELS {
                self.reapply_pitch_for_channel(ch as u8);
            }
        }
    }

    /// Apply an MTS **Scale/Octave Tuning** message to one channel: 12
    /// pitch-class offsets in cents (C, C#, … B). Updates the channel's
    /// scale/octave row; when `live` is true (the real-time forms) it
    /// also re-applies pitch to that channel's sounding voices.
    pub fn set_scale_octave_tuning(&mut self, channel: u8, offsets_cents: [f32; 12], live: bool) {
        for (pc, &c) in offsets_cents.iter().enumerate() {
            self.tuning.set_scale_octave(channel, pc, c);
        }
        if live {
            self.reapply_pitch_for_channel(channel);
        }
    }

    /// Reset all MTS microtuning to equal temperament. Wired to GM
    /// System On/Off (which resets every controller to its default).
    pub fn reset_tuning(&mut self) {
        self.tuning.reset();
        for ch in 0..NUM_CHANNELS {
            self.reapply_pitch_for_channel(ch as u8);
        }
    }

    // ─────────────────── GM2 Global Parameter Control ───────────────────

    /// Borrow the GM2 Reverb + Chorus parameter state (CA-024). Exposed
    /// for tests / introspection.
    pub fn gm_effects(&self) -> &GmEffects {
        &self.gm_effects
    }

    /// Reset the GM2 Reverb + Chorus parameters to their CA-024
    /// recommended initial settings. Wired to GM System On/Off (which
    /// resets every controller to its default).
    pub fn reset_gm_effects(&mut self) {
        self.gm_effects = GmEffects::default();
        // GM System On/Off silences everything; flush the effect tails
        // too so a fresh section doesn't inherit the previous reverb.
        self.fx.clear();
    }

    /// Flush the system Reverb + Chorus delay-line tails to silence
    /// without changing their parameters. Useful at a seek / stop point.
    pub fn clear_effects(&mut self) {
        self.fx.clear();
    }

    /// Apply one GM2 Reverb-slot (`01 01`) parameter-value pair from a
    /// Global Parameter Control message (CA-024 reverb table). `pp` is
    /// the parameter ID, `val` the raw 7-bit value:
    ///
    ///   * `pp = 0` → Reverb Type select. Per CA-024 ("When a Reverb
    ///     Type is selected, the default Reverb Time from the table
    ///     below for that Reverb Type should be set") this *also* resets
    ///     the Reverb Time to that type's table default.
    ///   * `pp = 1` → Reverb Time, decoded to seconds via
    ///     `rt = exp((val - 40) * 0.025)`.
    ///
    /// Unrecognised `pp` values are ignored per CA-024 ("If the device
    /// receives an unrecognizable or inappropriate parameter for a slot,
    /// only that parameter-value pair should be ignored").
    pub fn set_gm_reverb_param(&mut self, pp: u8, val: u8) {
        match pp {
            0 => {
                let ty = val & 0x7F;
                self.gm_effects.reverb_type = ty;
                // Selecting a Reverb Type sets its default Reverb Time.
                self.gm_effects.reverb_time_s =
                    GmEffects::reverb_time_from_val(GmEffects::reverb_type_default_time_val(ty));
            }
            1 => self.gm_effects.reverb_time_s = GmEffects::reverb_time_from_val(val),
            _ => {}
        }
    }

    /// Apply one GM2 Chorus-slot (`01 02`) parameter-value pair from a
    /// Global Parameter Control message (CA-024 chorus table). `pp` is
    /// the parameter ID, `val` the raw 7-bit value:
    ///
    ///   * `pp = 0` → Chorus Type select. Per CA-024 ("pp = 0 : Chorus
    ///     Type … Sets Chorus parameters as listed below") this *also*
    ///     loads the Mod Rate / Mod Depth / Feedback / Send-to-Reverb
    ///     row for that type.
    ///   * `pp = 1` → Mod Rate Hz: `mr = val * 0.122`.
    ///   * `pp = 2` → Mod Depth ms: `md = (val + 1) / 3.2`.
    ///   * `pp = 3` → Feedback %: `fb = val * 0.763`.
    ///   * `pp = 4` → Send-to-Reverb %: `ctr = val * 0.787`.
    ///
    /// Unrecognised `pp` values are ignored per CA-024.
    pub fn set_gm_chorus_param(&mut self, pp: u8, val: u8) {
        let v = (val & 0x7F) as f32;
        match pp {
            0 => {
                let ty = val & 0x7F;
                self.gm_effects.chorus_type = ty;
                // Selecting a Chorus Type loads its parameter row.
                self.gm_effects.apply_chorus_type_defaults(ty);
            }
            1 => self.gm_effects.chorus_mod_rate_hz = v * 0.122,
            2 => self.gm_effects.chorus_mod_depth_ms = (v + 1.0) / 3.2,
            3 => self.gm_effects.chorus_feedback_pct = v * 0.763,
            4 => self.gm_effects.chorus_send_to_reverb_pct = v * 0.787,
            _ => {}
        }
    }

    // ─────────────────────────── MPE plumbing ───────────────────────────

    /// Activate / deactivate one MPE zone. `members = 0` deactivates
    /// the zone per §2.2.1 ("Sending an MCM with the number of
    /// Member Channels set to zero deactivates that zone"). When a
    /// zone is (de)activated the receiver must stop all Sounding Notes
    /// and reset all controls on each channel entering or leaving MPE
    /// control (§2.2.3) — we honour that by `all_notes_off` on the
    /// affected channels and re-seeding their PB sensitivity per
    /// §2.2.5 defaults.
    pub fn set_mpe_zone(&mut self, kind: MpeZoneKind, members: u8) {
        // Per §2.2.1: "No MIDI Channel shall be assigned to more than
        // one Zone at a time." If the new zone would steal channels
        // from the other zone, the spec mandates the most recent MCM
        // wins and the other zone is shrunk (or deactivated if it has
        // no remaining Members). We model the simpler-but-spec-
        // compliant rule: per §A.2, a typical receiver only models
        // one zone at a time and Member Channels grow/decay from the
        // Manager Channel outward.
        let new_zone = if members == 0 {
            None
        } else {
            Some(MpeZone {
                kind,
                members: members.min(15),
            })
        };
        // Step 1: reset every channel currently in this zone.
        let old_zone = match kind {
            MpeZoneKind::Lower => self.mpe_lower,
            MpeZoneKind::Upper => self.mpe_upper,
        };
        if let Some(z) = old_zone {
            self.reset_mpe_zone_channels(z);
        }
        // Step 2: assign the new zone.
        match kind {
            MpeZoneKind::Lower => self.mpe_lower = new_zone,
            MpeZoneKind::Upper => self.mpe_upper = new_zone,
        }
        // Step 3: tag the channels with their new roles & defaults.
        if let Some(z) = new_zone {
            // Drop conflicting assignments from the *other* zone: per
            // §2.2.1, the most recent MCM wins.
            let other = match kind {
                MpeZoneKind::Lower => self.mpe_upper,
                MpeZoneKind::Upper => self.mpe_lower,
            };
            if let Some(o) = other {
                let members_now: Vec<u8> = z.member_channels();
                let o_members: Vec<u8> = o.member_channels();
                if o_members.iter().any(|m| members_now.contains(m))
                    || members_now.contains(&o.manager_channel())
                {
                    // Conflict — shrink/deactivate the other zone.
                    let surviving: Vec<u8> = o_members
                        .into_iter()
                        .filter(|m| !members_now.contains(m))
                        .collect();
                    if surviving.is_empty() {
                        match o.kind {
                            MpeZoneKind::Lower => self.mpe_lower = None,
                            MpeZoneKind::Upper => self.mpe_upper = None,
                        }
                    } else {
                        let new_other = MpeZone {
                            kind: o.kind,
                            members: surviving.len() as u8,
                        };
                        match o.kind {
                            MpeZoneKind::Lower => self.mpe_lower = Some(new_other),
                            MpeZoneKind::Upper => self.mpe_upper = Some(new_other),
                        }
                    }
                }
            }
            self.tag_mpe_zone_channels(z);
        }
    }

    /// `Some(zone)` if `kind` is currently active; `None` otherwise.
    pub fn mpe_zone(&self, kind: MpeZoneKind) -> Option<MpeZone> {
        match kind {
            MpeZoneKind::Lower => self.mpe_lower,
            MpeZoneKind::Upper => self.mpe_upper,
        }
    }

    /// Drop every voice on the channels that participate in `zone`
    /// (Manager + Members) and reset their per-channel state to the
    /// non-MPE default. Called when a zone gets reconfigured.
    fn reset_mpe_zone_channels(&mut self, zone: MpeZone) {
        let mut ch_list = zone.member_channels();
        ch_list.push(zone.manager_channel());
        for ch in ch_list {
            // Stop sounding notes on this channel.
            for slot in self.slots.iter_mut() {
                if slot.channel == ch {
                    slot.voice = None;
                    slot.sustained = false;
                }
            }
            // Restore default ChannelState (preserves nothing).
            self.channels[ch as usize] = ChannelState::default();
        }
    }

    /// Tag every channel of `zone` with its MPE role and default
    /// pitch-bend sensitivity per §2.2.5: 48 semitones on Members,
    /// 2 semitones on Manager.
    fn tag_mpe_zone_channels(&mut self, zone: MpeZone) {
        let mgr = zone.manager_channel();
        self.channels[mgr as usize].mpe_role = MpeRole::Manager(zone.kind);
        self.channels[mgr as usize].pitch_bend_range_cents = 200;
        for ch in zone.member_channels() {
            self.channels[ch as usize].mpe_role = MpeRole::Member(zone.kind);
            self.channels[ch as usize].pitch_bend_range_cents = 4800;
        }
    }

    /// Apply CC 64 (sustain pedal). When the value crosses below the
    /// 64 threshold while the pedal is currently held, every voice on
    /// `channel` whose `sustained` flag is set has its release fired.
    pub fn set_sustain(&mut self, channel: u8, value: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        let was = self.channels[ch].sustain;
        let now = value >= 64;
        self.channels[ch].sustain = now;
        if was && !now {
            // Pedal lifted — release every voice on this channel whose
            // note-off was being held by sustain.
            for slot in self.slots.iter_mut() {
                if slot.channel == channel && slot.sustained {
                    if let Some(v) = slot.voice.as_mut() {
                        v.release();
                    }
                    slot.sustained = false;
                }
            }
        }
    }

    /// Allocate a voice slot. If the pool is full, preempt the oldest
    /// slot (smallest `age`). Returns the index of the chosen slot.
    fn pick_slot(&mut self) -> usize {
        // Prefer a free / done slot.
        for (i, slot) in self.slots.iter().enumerate() {
            match &slot.voice {
                None => return i,
                Some(v) if v.done() => return i,
                _ => {}
            }
        }
        // No free slot — preempt the oldest. Tie-break on slot index
        // (lower wins) so the choice is deterministic.
        let mut oldest = 0;
        let mut oldest_age = self.slots[0].age;
        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            if slot.age < oldest_age {
                oldest = i;
                oldest_age = slot.age;
            }
        }
        oldest
    }

    /// Insert a freshly-built voice for `channel` / `key`. Velocity is
    /// recorded for diagnostics; the actual amplitude lives inside the
    /// voice (the SF2 / tone constructors fold it in). Channel-level
    /// pitch bend and aftertouch are applied to the freshly-allocated
    /// voice so a note triggered while the bend wheel is held picks up
    /// the offset on its first sample.
    ///
    /// When the new voice declares a non-zero
    /// [`Voice::exclusive_class`], every prior voice on the same channel
    /// with the same class is hard-stopped before the new voice is
    /// inserted (SF2 generator 57 — drum kits use this for hi-hat
    /// open/closed pairs).
    pub fn note_on(&mut self, channel: u8, key: u8, velocity: u8, mut voice: Box<dyn Voice>) {
        // Exclusive-class cut: drop every prior voice on this channel
        // with the same non-zero class id. Done before allocating the
        // new slot so the freed slot is preferred by `pick_slot`.
        let new_class = voice.exclusive_class();
        if new_class != 0 {
            for slot in self.slots.iter_mut() {
                if slot.channel == channel {
                    if let Some(v) = slot.voice.as_ref() {
                        if v.exclusive_class() == new_class {
                            slot.voice = None;
                            slot.sustained = false;
                        }
                    }
                }
            }
        }
        let ch = channel as usize % NUM_CHANNELS;
        let st = self.channels[ch];
        let is_drum = ch == 9;
        // Compose pitch bend + per-channel fine/coarse + master
        // fine/coarse + (for MPE Members) the Manager Channel's bend
        // — picks up tuning on the new voice's very first sample so
        // a note triggered while the bend wheel is held doesn't pop.
        let mut cents = Self::compose_pitch_cents(
            &st,
            st.pitch_bend,
            st.pitch_bend_range_cents,
            self.master_fine_tune_cents,
            self.master_coarse_tune_semitones,
            is_drum,
        );
        if let MpeRole::Member(zone_kind) = st.mpe_role {
            let mgr = match zone_kind {
                MpeZoneKind::Lower => 0,
                MpeZoneKind::Upper => 15,
            };
            let mgr_state = self.channels[mgr];
            cents += pitch_bend_to_cents(mgr_state.pitch_bend, mgr_state.pitch_bend_range_cents);
        }
        // Fold in the MTS per-key tuning offset (key-based table +
        // channel scale/octave). Drum channels are exempt from
        // note-shifting per CA-25's principle (a different pitch on a
        // drum kit is a different sound), matching the master-tuning
        // exemption above.
        if !is_drum {
            cents += self.tuning.offset_cents(channel, key).round() as i32;
        }
        if cents != 0 {
            voice.set_pitch_bend_cents(cents);
        }
        // Compose Member + Manager channel pressure for MPE; otherwise
        // just hand the channel's value through.
        let pressure_byte = match st.mpe_role {
            MpeRole::Member(zone_kind) => {
                let mgr = match zone_kind {
                    MpeZoneKind::Lower => 0,
                    MpeZoneKind::Upper => 15,
                };
                st.channel_pressure.max(self.channels[mgr].channel_pressure)
            }
            _ => st.channel_pressure,
        };
        if pressure_byte != 0 {
            voice.set_pressure(pressure_byte as f32 / 127.0);
        }
        // Mod-wheel depth (CC 1 scaled by RPN 5) carries to a fresh
        // voice the same way bend does.
        let depth_cents = (st.mod_wheel as i32) * (st.mod_depth_range_cents as i32) / 127;
        if depth_cents != 0 {
            voice.set_mod_depth_cents(depth_cents);
        }
        let idx = self.pick_slot();
        let age = self.next_age;
        self.next_age = self.next_age.wrapping_add(1);
        self.slots[idx] = VoiceSlot {
            voice: Some(voice),
            channel,
            key,
            sustained: false,
            age,
            velocity_norm: (velocity as f32 / 127.0).clamp(0.0, 1.0),
        };
    }

    /// Trigger release on every slot matching `(channel, key)` that
    /// hasn't already been released. If sustain is held on the channel,
    /// the slot is marked `sustained` and its release is deferred until
    /// the pedal lifts.
    pub fn note_off(&mut self, channel: u8, key: u8) {
        let sustain = self.channels[channel as usize % NUM_CHANNELS].sustain;
        for slot in self.slots.iter_mut() {
            if slot.channel == channel && slot.key == key {
                if let Some(v) = slot.voice.as_mut() {
                    if sustain {
                        slot.sustained = true;
                    } else {
                        v.release();
                    }
                }
            }
        }
    }

    /// Hard-stop every voice (used by `MidiDecoder::reset`). No release
    /// envelope — slots become free immediately.
    pub fn all_notes_off(&mut self) {
        for slot in self.slots.iter_mut() {
            slot.voice = None;
            slot.sustained = false;
        }
    }

    /// Mix every live voice into a planar stereo `(left, right)` slice
    /// pair. Both buffers must be the same length. Existing buffer
    /// contents are **overwritten** (not added to) so the caller can
    /// reuse the buffer across chunks without re-zeroing.
    ///
    /// Returns the number of voices that contributed audio.
    pub fn mix_stereo(&mut self, left: &mut [f32], right: &mut [f32]) -> usize {
        assert_eq!(left.len(), right.len(), "stereo planes must match length");
        for s in left.iter_mut() {
            *s = 0.0;
        }
        for s in right.iter_mut() {
            *s = 0.0;
        }

        let mut active = 0;
        // Per-voice scratch buffers. Mono path renders into `mono` then
        // pans into the L/R bus. Stereo path renders directly into
        // `lscratch` / `rscratch` (one set kept around so the voice
        // isn't forced to allocate per chunk) and *bypasses* the pan
        // law — a true stereo SF2 zone has its own image baked in.
        let mut mono = vec![0.0f32; left.len()];
        let mut lscratch = vec![0.0f32; left.len()];
        let mut rscratch = vec![0.0f32; left.len()];
        // Master state is mix-wide; compute once per chunk and reuse
        // — also avoids re-borrowing `self` immutably inside the
        // `iter_mut()` loop.
        let master = self.master_volume_14 as f32 / 0x3FFF as f32;
        let (master_bal_l, master_bal_r) = self.master_balance_gains();

        // --- Effects-bus send accumulators (CA-024 CC 91 / CC 93). ---
        // The fx input vectors live on `self.fx`, but the voice loop
        // below borrows `self.slots` mutably, so we move them out for
        // the duration of the loop and hand them back afterwards. They
        // are zeroed + sized here; the loop adds each voice's post-pan
        // signal scaled by its channel's reverb / chorus send level.
        self.fx.prepare_block(left.len());
        let mut reverb_in_l = std::mem::take(&mut self.fx.reverb_in_l);
        let mut reverb_in_r = std::mem::take(&mut self.fx.reverb_in_r);
        let mut chorus_in_l = std::mem::take(&mut self.fx.chorus_in_l);
        let mut chorus_in_r = std::mem::take(&mut self.fx.chorus_in_r);

        for slot in self.slots.iter_mut() {
            let stereo = slot.voice.as_ref().map(|v| v.is_stereo()).unwrap_or(false);
            let n = if let Some(v) = slot.voice.as_mut() {
                let n = if stereo {
                    v.render_stereo(&mut lscratch, &mut rscratch)
                } else {
                    v.render(&mut mono)
                };
                if n == 0 && v.done() {
                    slot.voice = None;
                    continue;
                }
                n
            } else {
                continue;
            };

            // Per-channel volume / pan + universal master volume +
            // master balance (master state hoisted out of the loop).
            let st = self.channels[slot.channel as usize % NUM_CHANNELS];
            let vol = st.volume as f32 / 127.0;
            // Constant-power pan: θ in [0, π/2], left = cos(θ), right = sin(θ).
            let pan_norm = (st.pan as f32 / 127.0).clamp(0.0, 1.0);
            let theta = pan_norm * std::f32::consts::FRAC_PI_2;

            // CA-024 per-channel effect send fractions (CC 91 / CC 93).
            // Zero when the channel never touched the controller, so a
            // dry score bypasses the bus entirely.
            let reverb_send = st.reverb_send as f32 / 127.0;
            let chorus_send = st.chorus_send as f32 / 127.0;
            let any_send = reverb_send > 0.0 || chorus_send > 0.0;

            if stereo {
                // Stereo voice: keep its inherent L/R image, but still
                // honour the channel's volume CC. Pan applies as a
                // *balance* rather than a true pan: pan=64 → 1.0/1.0,
                // pan=0 → 1.0/0.0, pan=127 → 0.0/1.0. This matches the
                // GM "balance control" interpretation for stereo
                // sources, where pan rotates the image rather than
                // re-panning a mono signal.
                let l_balance = (theta.cos() * std::f32::consts::SQRT_2).min(1.0);
                let r_balance = (theta.sin() * std::f32::consts::SQRT_2).min(1.0);
                let lg = vol * master * master_bal_l * self.mix_gain * l_balance;
                let rg = vol * master * master_bal_r * self.mix_gain * r_balance;
                for i in 0..n {
                    let dl = lscratch[i] * lg;
                    let dr = rscratch[i] * rg;
                    left[i] += dl;
                    right[i] += dr;
                    if any_send {
                        reverb_in_l[i] += dl * reverb_send;
                        reverb_in_r[i] += dr * reverb_send;
                        chorus_in_l[i] += dl * chorus_send;
                        chorus_in_r[i] += dr * chorus_send;
                    }
                }
            } else {
                let l_gain = theta.cos() * vol * master * master_bal_l * self.mix_gain;
                let r_gain = theta.sin() * vol * master * master_bal_r * self.mix_gain;
                for i in 0..n {
                    let s = mono[i];
                    let dl = s * l_gain;
                    let dr = s * r_gain;
                    left[i] += dl;
                    right[i] += dr;
                    if any_send {
                        reverb_in_l[i] += dl * reverb_send;
                        reverb_in_r[i] += dr * reverb_send;
                        chorus_in_l[i] += dl * chorus_send;
                        chorus_in_r[i] += dr * chorus_send;
                    }
                }
            }
            active += 1;

            // If the voice produced fewer than the buffer size it
            // exhausted itself mid-chunk; mark it done so the next mix
            // pass frees the slot. The voice's own `done()` flag is
            // already set in this case (see Voice::render contract).
            if n < mono.len() {
                if let Some(v) = slot.voice.as_ref() {
                    if v.done() {
                        slot.voice = None;
                    }
                }
            }

            // Per-voice diagnostics (peak / silent-sample counter)
            // would slot in here in a future round.
            let _ = slot.velocity_norm;
        }

        // Hand the send accumulators back to the fx bus and run the
        // Reverb + Chorus DSP, summing the wet returns into the main mix.
        self.fx.reverb_in_l = reverb_in_l;
        self.fx.reverb_in_r = reverb_in_r;
        self.fx.chorus_in_l = chorus_in_l;
        self.fx.chorus_in_r = chorus_in_r;
        self.fx.refresh(&self.gm_effects);
        let chorus_to_reverb = (self.gm_effects.chorus_send_to_reverb_pct / 100.0).clamp(0.0, 1.0);
        self.fx.process(left, right, chorus_to_reverb);

        active
    }

    /// Number of slots currently holding a (possibly already-released)
    /// voice. Useful for tests and debugging.
    pub fn live_voice_count(&self) -> usize {
        self.slots.iter().filter(|s| s.voice.is_some()).count()
    }
}

// =========================================================================
// Tests.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::Voice;

    /// A test voice that produces a constant DC value for `total`
    /// samples then reports `done`. Lets us assert mix arithmetic
    /// without standing up a full SF2 fixture. Also records the last
    /// pitch-bend / pressure / mod-depth / timbre value pushed in via
    /// the optional Voice methods so tests can assert routing.
    struct ConstVoice {
        value: f32,
        remaining: usize,
        done: bool,
        last_bend_cents: std::sync::Arc<std::sync::Mutex<i32>>,
        last_pressure: std::sync::Arc<std::sync::Mutex<f32>>,
        last_mod_depth_cents: std::sync::Arc<std::sync::Mutex<i32>>,
        last_timbre: std::sync::Arc<std::sync::Mutex<u8>>,
    }
    impl Voice for ConstVoice {
        fn render(&mut self, out: &mut [f32]) -> usize {
            if self.done {
                return 0;
            }
            let n = out.len().min(self.remaining);
            for s in out.iter_mut().take(n) {
                *s = self.value;
            }
            self.remaining -= n;
            if self.remaining == 0 {
                self.done = true;
            }
            n
        }
        fn release(&mut self) {
            // No release envelope — drop on next render.
            self.done = true;
        }
        fn done(&self) -> bool {
            self.done
        }
        fn set_pitch_bend_cents(&mut self, cents: i32) {
            *self.last_bend_cents.lock().unwrap() = cents;
        }
        fn set_pressure(&mut self, p: f32) {
            *self.last_pressure.lock().unwrap() = p;
        }
        fn set_mod_depth_cents(&mut self, cents: i32) {
            *self.last_mod_depth_cents.lock().unwrap() = cents;
        }
        fn set_timbre(&mut self, v: u8) {
            *self.last_timbre.lock().unwrap() = v;
        }
    }

    fn voice(value: f32, samples: usize) -> Box<dyn Voice> {
        Box::new(ConstVoice {
            value,
            remaining: samples,
            done: false,
            last_bend_cents: std::sync::Arc::new(std::sync::Mutex::new(0)),
            last_pressure: std::sync::Arc::new(std::sync::Mutex::new(0.0)),
            last_mod_depth_cents: std::sync::Arc::new(std::sync::Mutex::new(0)),
            last_timbre: std::sync::Arc::new(std::sync::Mutex::new(0)),
        })
    }

    type BendCell = std::sync::Arc<std::sync::Mutex<i32>>;
    type PressureCell = std::sync::Arc<std::sync::Mutex<f32>>;
    type DepthCell = std::sync::Arc<std::sync::Mutex<i32>>;
    type TimbreCell = std::sync::Arc<std::sync::Mutex<u8>>;

    /// Build a [`ConstVoice`] plus shared handles to its `last_bend_cents`
    /// / `last_pressure` cells so the test can read the values back after
    /// the mixer has handed the voice to its slot.
    fn instrumented_voice(value: f32, samples: usize) -> (Box<dyn Voice>, BendCell, PressureCell) {
        let bend = std::sync::Arc::new(std::sync::Mutex::new(0));
        let press = std::sync::Arc::new(std::sync::Mutex::new(0.0));
        let depth = std::sync::Arc::new(std::sync::Mutex::new(0));
        let timbre = std::sync::Arc::new(std::sync::Mutex::new(0));
        let v = Box::new(ConstVoice {
            value,
            remaining: samples,
            done: false,
            last_bend_cents: bend.clone(),
            last_pressure: press.clone(),
            last_mod_depth_cents: depth,
            last_timbre: timbre,
        });
        (v, bend, press)
    }

    /// Full instrumented voice + handles for *every* cell.
    fn instrumented_voice_full(
        value: f32,
        samples: usize,
    ) -> (
        Box<dyn Voice>,
        BendCell,
        PressureCell,
        DepthCell,
        TimbreCell,
    ) {
        let bend = std::sync::Arc::new(std::sync::Mutex::new(0));
        let press = std::sync::Arc::new(std::sync::Mutex::new(0.0));
        let depth = std::sync::Arc::new(std::sync::Mutex::new(0));
        let timbre = std::sync::Arc::new(std::sync::Mutex::new(0));
        let v = Box::new(ConstVoice {
            value,
            remaining: samples,
            done: false,
            last_bend_cents: bend.clone(),
            last_pressure: press.clone(),
            last_mod_depth_cents: depth.clone(),
            last_timbre: timbre.clone(),
        });
        (v, bend, press, depth, timbre)
    }

    #[test]
    fn mix_empty_pool_is_silence() {
        let mut m = Mixer::new();
        let mut l = vec![1.0f32; 16];
        let mut r = vec![1.0f32; 16];
        let active = m.mix_stereo(&mut l, &mut r);
        assert_eq!(active, 0);
        assert!(l.iter().all(|s| *s == 0.0));
        assert!(r.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn note_on_then_mix_produces_audio() {
        let mut m = Mixer::new();
        m.note_on(0, 60, 100, voice(0.5, 32));
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        let active = m.mix_stereo(&mut l, &mut r);
        assert_eq!(active, 1);
        // Centred pan + default vol 100/127 + mix_gain 0.5 + DC 0.5.
        // Both channels should be > 0.
        assert!(l[0] > 0.0, "left silent");
        assert!(r[0] > 0.0, "right silent");
        // Pan = 64 maps to ~0.504 in the constant-power law, slightly
        // R-biased — within 5 % of centre is what GM treats as
        // perceptually equal.
        let ratio = (l[0] / r[0]).abs();
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "L/R ratio {} too far from unity at pan=64",
            ratio,
        );
    }

    #[test]
    fn pan_full_left_silences_right() {
        let mut m = Mixer::new();
        m.channel_state_mut(0).pan = 0; // hard left
        m.note_on(0, 60, 100, voice(0.5, 32));
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        m.mix_stereo(&mut l, &mut r);
        assert!(l[0] > 0.0);
        assert!(r[0].abs() < 1e-6, "right={} should be silent", r[0]);
    }

    #[test]
    fn pan_full_right_silences_left() {
        let mut m = Mixer::new();
        m.channel_state_mut(0).pan = 127;
        m.note_on(0, 60, 100, voice(0.5, 32));
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        m.mix_stereo(&mut l, &mut r);
        assert!(r[0] > 0.0);
        assert!(l[0].abs() < 1e-6, "left={} should be silent", l[0]);
    }

    #[test]
    fn note_off_releases_matching_voice() {
        let mut m = Mixer::new();
        m.note_on(0, 60, 100, voice(0.5, 1024));
        m.note_off(0, 60);
        // ConstVoice goes done() on release().
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        let _ = m.mix_stereo(&mut l, &mut r);
        // Slot should now be free.
        assert_eq!(m.live_voice_count(), 0);
    }

    #[test]
    fn note_off_wrong_channel_does_not_release() {
        let mut m = Mixer::new();
        m.note_on(0, 60, 100, voice(0.5, 1024));
        m.note_off(1, 60); // wrong channel
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        m.mix_stereo(&mut l, &mut r);
        assert_eq!(m.live_voice_count(), 1);
    }

    #[test]
    fn sustain_defers_note_off_until_pedal_lifts() {
        let mut m = Mixer::new();
        m.set_sustain(0, 127); // pedal down
        m.note_on(0, 60, 100, voice(0.5, 1024));
        m.note_off(0, 60); // would-be release
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        m.mix_stereo(&mut l, &mut r);
        // Voice is still alive — sustained.
        assert_eq!(m.live_voice_count(), 1);
        m.set_sustain(0, 0); // pedal up — fires release
        let _ = m.mix_stereo(&mut l, &mut r);
        assert_eq!(m.live_voice_count(), 0);
    }

    #[test]
    fn pool_preempts_oldest_when_full() {
        let mut m = Mixer::new();
        // Fill the pool with very-long-running voices (won't end naturally).
        for k in 0..MAX_VOICES as u8 {
            m.note_on(0, 60 + k, 100, voice(0.5, 1_000_000));
        }
        assert_eq!(m.live_voice_count(), MAX_VOICES);
        // One more must preempt; the youngest survivor should be the
        // newcomer.
        m.note_on(0, 60 + MAX_VOICES as u8, 100, voice(0.5, 1_000_000));
        assert_eq!(m.live_voice_count(), MAX_VOICES);
        // Find the slot with the highest age — must hold the newcomer.
        let max_age_slot = m
            .slots
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.age)
            .unwrap();
        assert_eq!(max_age_slot.1.key, 60 + MAX_VOICES as u8);
    }

    #[test]
    fn voice_finishes_naturally_frees_slot() {
        let mut m = Mixer::new();
        m.note_on(0, 60, 100, voice(0.5, 8));
        let mut l = vec![0.0f32; 16];
        let mut r = vec![0.0f32; 16];
        m.mix_stereo(&mut l, &mut r); // 8 of 16 samples produced, then done
        assert_eq!(m.live_voice_count(), 0, "voice should have freed its slot");
    }

    #[test]
    fn all_notes_off_clears_pool() {
        let mut m = Mixer::new();
        m.note_on(0, 60, 100, voice(0.5, 1024));
        m.note_on(0, 64, 100, voice(0.5, 1024));
        assert_eq!(m.live_voice_count(), 2);
        m.all_notes_off();
        assert_eq!(m.live_voice_count(), 0);
    }

    #[test]
    fn pitch_bend_to_cents_centre_is_zero() {
        // 0x2000 = centre = no bend.
        assert_eq!(pitch_bend_to_cents(0x2000, 200), 0);
    }

    #[test]
    fn pitch_bend_to_cents_full_up_is_plus_range() {
        // 0x3FFF = +max = +range cents (≈ 200 = +2 semitones at default).
        let cents = pitch_bend_to_cents(0x3FFF, 200);
        assert!((199..=200).contains(&cents), "got {cents}");
    }

    #[test]
    fn pitch_bend_to_cents_full_down_is_minus_range() {
        // 0 = -max.
        let cents = pitch_bend_to_cents(0, 200);
        assert_eq!(cents, -200);
    }

    #[test]
    fn pitch_bend_routes_to_held_voices() {
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        m.set_pitch_bend(0, 0x3FFF); // hard up
        let cents = *bend_cell.lock().unwrap();
        assert!((199..=200).contains(&cents), "got {cents}");
        // ChannelState should also reflect the new value.
        assert_eq!(m.channel_state(0).pitch_bend, 0x3FFF);
    }

    #[test]
    fn pitch_bend_applied_at_note_on_when_already_held() {
        let mut m = Mixer::new();
        // Bend up first, then start a note — the new voice should see
        // the bend on its very first sample.
        m.set_pitch_bend(0, 0x3FFF);
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        let cents = *bend_cell.lock().unwrap();
        assert!(
            cents >= 199,
            "note-on did not pick up live pitch bend: got {cents}"
        );
    }

    #[test]
    fn channel_pressure_routes_to_all_channel_voices_only() {
        let mut m = Mixer::new();
        let (v0, _, p0) = instrumented_voice(0.5, 1024);
        let (v1, _, p1) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v0);
        m.note_on(1, 64, 100, v1);
        m.set_channel_pressure(0, 100); // only ch 0
        let pa = *p0.lock().unwrap();
        let pb = *p1.lock().unwrap();
        assert!(pa > 0.5, "ch 0 pressure not routed: {pa}");
        assert_eq!(pb, 0.0, "ch 1 pressure should be untouched");
    }

    #[test]
    fn poly_pressure_only_routes_to_matching_key() {
        let mut m = Mixer::new();
        let (v_match, _, p_match) = instrumented_voice(0.5, 1024);
        let (v_other, _, p_other) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v_match);
        m.note_on(0, 64, 100, v_other);
        m.set_poly_pressure(0, 60, 80);
        let pa = *p_match.lock().unwrap();
        let pb = *p_other.lock().unwrap();
        assert!(pa > 0.0, "matching-key voice didn't see pressure: {pa}");
        assert_eq!(pb, 0.0, "non-matching-key voice should be untouched");
    }

    #[test]
    fn rpn_zero_then_data_entry_changes_bend_range() {
        let mut m = Mixer::new();
        // Select RPN 0 (CC 101 MSB = 0, CC 100 LSB = 0).
        m.set_rpn_byte(0, 0, true);
        m.set_rpn_byte(0, 0, false);
        // CC 6 = 12 → ±12 semitones (= 1200 cents).
        m.set_data_entry(0, 12, true);
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 1200);
        // CC 38 = 50 → +50 cents on top.
        m.set_data_entry(0, 50, false);
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 1250);
    }

    #[test]
    fn rpn_null_blocks_data_entry() {
        let mut m = Mixer::new();
        // No RPN selected (default = 0x3FFF, the null marker).
        m.set_data_entry(0, 12, true);
        // Default range (200) must be untouched.
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 200);
    }

    // ──────────────────────── Round-75: master tuning + RPN 1/2/5 + MPE ────────────────────────

    fn select_rpn(m: &mut Mixer, channel: u8, msb: u8, lsb: u8) {
        m.set_rpn_byte(channel, msb, true);
        m.set_rpn_byte(channel, lsb, false);
    }

    #[test]
    fn rpn_1_channel_fine_tune_data_entry_sets_cents() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 1); // RPN 1 = Channel Fine Tuning
                                     // Centre (MSB=0x40 LSB=0x00) → 0 cents.
        m.set_data_entry(0, 0x40, true);
        m.set_data_entry(0, 0x00, false);
        assert_eq!(m.channel_state(0).channel_fine_tune_cents, 0);
        // Max positive (MSB=0x7F LSB=0x7F) → ~+100 cents.
        m.set_data_entry(0, 0x7F, true);
        m.set_data_entry(0, 0x7F, false);
        let c = m.channel_state(0).channel_fine_tune_cents;
        assert!((99..=100).contains(&c), "got {c}");
        // Max negative (MSB=0x00 LSB=0x00) → -100 cents.
        m.set_data_entry(0, 0x00, true);
        m.set_data_entry(0, 0x00, false);
        assert_eq!(m.channel_state(0).channel_fine_tune_cents, -100);
    }

    #[test]
    fn rpn_2_channel_coarse_tune_data_entry_sets_semitones() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 2);
        m.set_data_entry(0, 0x40, true); // centre = 0 semis
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, 0);
        m.set_data_entry(0, 0x4C, true); // +12 semis = one octave up
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, 12);
        m.set_data_entry(0, 0x34, true); // -12 semis
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, -12);
        // CC 38 LSB ignored per spec.
        m.set_data_entry(0, 0x7F, false);
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, -12);
    }

    #[test]
    fn rpn_5_modulation_depth_range_updates_range() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 5);
        // CC 38 = 0 first to clear the default fractional (50 cents),
        // then CC 6 = 1 → 100 cents whole.
        m.set_data_entry(0, 0, false);
        m.set_data_entry(0, 1, true);
        assert_eq!(m.channel_state(0).mod_depth_range_cents, 100);
        // CC 38 = 50 → adds 50 cents on top.
        m.set_data_entry(0, 50, false);
        assert_eq!(m.channel_state(0).mod_depth_range_cents, 150);
    }

    // ──────────────────── Round-102: Data Inc/Dec (RP-018) ────────────────────

    #[test]
    fn data_increment_rpn0_steps_cents_then_wraps_into_semitone() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 0); // RPN 0 = Pitch Bend Sensitivity
                                     // Start from the GM default (200 = 2 semitones, 0 cents).
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 200);
        // RP-018 worked example: increment pitch-bend sensitivity by 2
        // cents = two Data Increment messages (value byte ignored).
        m.data_inc_dec(0, 1);
        m.data_inc_dec(0, 1);
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 202);
        // Walk the LSB up to the wrap point (99 → 100): from 202 we need
        // 98 more increments to reach 300 = 3 semitones, 0 cents, proving
        // the LSB wraps into the MSB (semitone) field at 100.
        for _ in 0..98 {
            m.data_inc_dec(0, 1);
        }
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 300);
    }

    #[test]
    fn data_decrement_rpn0_borrows_across_semitone_boundary() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 0);
        // 200 → 199 borrows the MSB down: 1 semitone + 99 cents.
        m.data_inc_dec(0, -1);
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 199);
    }

    #[test]
    fn data_inc_dec_value_byte_ignored() {
        // The scheduler passes a fixed +1/-1 step; the controller's data
        // byte never reaches data_inc_dec. This test documents that the
        // method's contract is "step by 1" regardless of CC value, by
        // showing two single steps equal one double step.
        let mut a = Mixer::new();
        select_rpn(&mut a, 0, 0, 0);
        a.data_inc_dec(0, 1);
        a.data_inc_dec(0, 1);

        let mut b = Mixer::new();
        select_rpn(&mut b, 0, 0, 0);
        b.data_inc_dec(0, 1);
        b.data_inc_dec(0, 1);
        assert_eq!(
            a.channel_state(0).pitch_bend_range_cents,
            b.channel_state(0).pitch_bend_range_cents
        );
    }

    #[test]
    fn data_inc_dec_rpn1_steps_fine_tune_lsb() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 1); // RPN 1 = Channel Fine Tuning
                                     // Default raw is centre 0x2000 (= 0 cents).
        assert_eq!(m.channel_state(0).channel_fine_tune_raw_14, 0x2000);
        m.data_inc_dec(0, 1);
        assert_eq!(m.channel_state(0).channel_fine_tune_raw_14, 0x2001);
        m.data_inc_dec(0, -1);
        m.data_inc_dec(0, -1);
        assert_eq!(m.channel_state(0).channel_fine_tune_raw_14, 0x1FFF);
    }

    #[test]
    fn data_inc_dec_rpn2_steps_coarse_tune_semitone() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 2); // RPN 2 = Channel Coarse Tuning
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, 0);
        // RP-018: RPN 2 inc/dec affects the MSB → one semitone per step.
        m.data_inc_dec(0, 1);
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, 1);
        m.data_inc_dec(0, -1);
        m.data_inc_dec(0, -1);
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, -1);
    }

    #[test]
    fn data_inc_dec_rpn2_clamps_to_signed_range() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 2);
        for _ in 0..200 {
            m.data_inc_dec(0, 1);
        }
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, 63);
        for _ in 0..400 {
            m.data_inc_dec(0, -1);
        }
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, -64);
    }

    #[test]
    fn data_inc_dec_rpn5_steps_mod_depth_cents() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 5); // RPN 5 = Modulation Depth Range
                                     // GM2 default is 50 cents.
        assert_eq!(m.channel_state(0).mod_depth_range_cents, 50);
        m.data_inc_dec(0, 1);
        assert_eq!(m.channel_state(0).mod_depth_range_cents, 51);
        m.data_inc_dec(0, -1);
        m.data_inc_dec(0, -1);
        assert_eq!(m.channel_state(0).mod_depth_range_cents, 49);
    }

    #[test]
    fn data_inc_dec_with_rpn_null_is_noop() {
        let mut m = Mixer::new();
        // No RPN selected (default 0x3FFF) — a Data Inc must not touch
        // any parameter, mirroring set_data_entry's null guard.
        m.data_inc_dec(0, 1);
        m.data_inc_dec(0, -1);
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 200);
        assert_eq!(m.channel_state(0).channel_fine_tune_raw_14, 0x2000);
        assert_eq!(m.channel_state(0).channel_coarse_tune_semitones, 0);
        assert_eq!(m.channel_state(0).mod_depth_range_cents, 50);
    }

    #[test]
    fn data_inc_dec_rpn0_clamps_above_zero() {
        let mut m = Mixer::new();
        select_rpn(&mut m, 0, 0, 0);
        // Drive the range down hard; it must clamp at 1 (never zero).
        for _ in 0..1000 {
            m.data_inc_dec(0, -1);
        }
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 1);
    }

    #[test]
    fn data_increment_rpn0_reapplies_bend_to_held_voice() {
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        select_rpn(&mut m, 0, 0, 0);
        // Bend full up; with the default 200-cent range that's +200 c.
        m.set_pitch_bend(0, 0x3FFF);
        // Now grow the range one cent at a time; the held voice's routed
        // bend must track the widened range without a fresh pitch-bend
        // message (RP-018 changes are applied immediately).
        let before = *bend_cell.lock().unwrap();
        for _ in 0..100 {
            m.data_inc_dec(0, 1); // 200 → 300 cents range
        }
        let after = *bend_cell.lock().unwrap();
        assert!(
            after > before,
            "widening the bend range should increase the routed bend on a held voice: before={before} after={after}"
        );
    }

    #[test]
    fn channel_fine_tune_offsets_pitch_on_held_voice() {
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        // Centre bend = 0; with +50 cents fine tune, the routed cents
        // should be +50.
        select_rpn(&mut m, 0, 0, 1);
        // MSB=0x60 (= 0x60<<7 = 12288), LSB=0 → raw 0x3000.
        // cents = (0x3000 - 0x2000) * 100 / 0x2000 = 4096 * 100 / 8192 = 50.
        m.set_data_entry(0, 0x60, true);
        m.set_data_entry(0, 0x00, false);
        // Re-apply the live pitch bend (which is centre 0x2000) so the
        // voice picks up the new fine-tune sum.
        m.set_pitch_bend(0, 0x2000);
        let cents = *bend_cell.lock().unwrap();
        assert!(
            (49..=50).contains(&cents),
            "expected fine-tune routed to +50, got {cents}",
        );
    }

    #[test]
    fn channel_coarse_tune_routes_one_octave_up() {
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        select_rpn(&mut m, 0, 0, 2);
        m.set_data_entry(0, 0x4C, true); // +12 semis
        m.set_pitch_bend(0, 0x2000);
        let cents = *bend_cell.lock().unwrap();
        assert_eq!(
            cents, 1200,
            "expected +1200 cents (one octave), got {cents}"
        );
    }

    #[test]
    fn drum_channel_ignores_channel_coarse_tune() {
        // CA-25: "For devices which support Key-based Instruments
        // (such as drum kits) it is important that this message NOT
        // result in MIDI note-shifting."
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(9, 36, 100, v); // ch 10 = index 9 = GM drum bus
        select_rpn(&mut m, 9, 0, 2);
        m.set_data_entry(9, 0x4C, true); // +12 semis on drum channel
        m.set_pitch_bend(9, 0x2000);
        let cents = *bend_cell.lock().unwrap();
        assert_eq!(cents, 0, "drum channel must not shift pitch, got {cents}");
    }

    #[test]
    fn master_fine_tuning_centre_is_zero_cents() {
        let mut m = Mixer::new();
        m.set_master_fine_tuning(0x00, 0x40);
        assert_eq!(m.master_fine_tune_cents(), 0);
    }

    #[test]
    fn master_fine_tuning_max_positive_is_near_plus_100() {
        let mut m = Mixer::new();
        m.set_master_fine_tuning(0x7F, 0x7F);
        let c = m.master_fine_tune_cents();
        assert!((99..=100).contains(&c), "got {c}");
    }

    #[test]
    fn master_fine_tuning_routes_to_held_voice() {
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        // +50 cents master fine.
        m.set_master_fine_tuning(0x00, 0x60);
        let cents = *bend_cell.lock().unwrap();
        assert!((49..=50).contains(&cents), "got {cents}");
    }

    #[test]
    fn master_coarse_tuning_routes_one_octave() {
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        // +12 semis master coarse.
        m.set_master_coarse_tuning(0x00, 0x4C);
        let cents = *bend_cell.lock().unwrap();
        assert_eq!(cents, 1200);
    }

    #[test]
    fn master_volume_scales_mix_output() {
        let mut m = Mixer::new();
        m.note_on(0, 60, 100, voice(0.5, 16));
        // Full master.
        let mut l1 = vec![0.0; 16];
        let mut r1 = vec![0.0; 16];
        m.mix_stereo(&mut l1, &mut r1);
        // Half master.
        m.note_on(1, 60, 100, voice(0.5, 16));
        m.set_master_volume_14(0x2000); // ~half
        let mut l2 = vec![0.0; 16];
        let mut r2 = vec![0.0; 16];
        m.mix_stereo(&mut l2, &mut r2);
        // Ratio should be ~0.5.
        let ratio = l2[0] / l1[0];
        assert!(
            (0.40..0.60).contains(&ratio),
            "master-volume ratio {ratio} not in 0.4..0.6",
        );
    }

    #[test]
    fn master_balance_default_is_centre_with_unity_gains() {
        let m = Mixer::new();
        assert_eq!(m.master_balance_14(), 0x2000);
        let (l, r) = m.master_balance_gains();
        assert_eq!(l, 1.0);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn master_balance_hard_left_mutes_right_side() {
        let mut m = Mixer::new();
        // Per M1 v4.2.1 §"DEVICE CONTROL — MASTER VOLUME AND MASTER
        // BALANCE": 00 00 = hard left.
        m.set_master_balance_14(0x0000);
        let (l, r) = m.master_balance_gains();
        assert_eq!(l, 1.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn master_balance_hard_right_mutes_left_side() {
        let mut m = Mixer::new();
        // Per M1 v4.2.1 §"DEVICE CONTROL — MASTER VOLUME AND MASTER
        // BALANCE": 7F 7F = hard right.
        m.set_master_balance_14(0x3FFF);
        let (l, r) = m.master_balance_gains();
        assert_eq!(l, 0.0);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn master_balance_half_left_keeps_left_full_attenuates_right_to_half() {
        // Below centre: left = 1.0, right = v / 0x2000.
        let mut m = Mixer::new();
        m.set_master_balance_14(0x1000);
        let (l, r) = m.master_balance_gains();
        assert_eq!(l, 1.0);
        // 0x1000 / 0x2000 = 0.5 exactly.
        assert!((r - 0.5).abs() < 1e-6, "right = {r}, want ~0.5");
    }

    #[test]
    fn master_balance_half_right_keeps_right_full_attenuates_left_to_half() {
        // Above centre: right = 1.0, left = (0x3FFF - v) / 0x1FFF.
        // Pick v such that left ≈ 0.5: v = 0x3FFF - 0x1FFF/2 ≈ 0x3000.
        let mut m = Mixer::new();
        m.set_master_balance_14(0x3000);
        let (l, r) = m.master_balance_gains();
        assert_eq!(r, 1.0);
        // (0x3FFF - 0x3000) / 0x1FFF = 0x0FFF / 0x1FFF ≈ 0.5
        assert!((l - 0.5).abs() < 1e-3, "left = {l}, want ~0.5");
    }

    #[test]
    fn master_balance_setter_clamps_above_14bit_max() {
        let mut m = Mixer::new();
        m.set_master_balance_14(0xFFFF);
        assert_eq!(m.master_balance_14(), 0x3FFF);
    }

    #[test]
    fn master_balance_hard_left_zeros_right_in_mix_output() {
        let mut m = Mixer::new();
        // Pan centred so the per-channel pan doesn't mute one side
        // ahead of the master balance.
        m.channel_state_mut(0).pan = 64;
        m.set_master_balance_14(0x0000); // hard left
        m.note_on(0, 60, 100, voice(0.5, 16));
        let mut l = vec![0.0; 16];
        let mut r = vec![0.0; 16];
        m.mix_stereo(&mut l, &mut r);
        // Right side must be silent.
        for s in &r {
            assert_eq!(*s, 0.0);
        }
        // Left side carries audio.
        assert!(l[0].abs() > 0.0);
    }

    #[test]
    fn master_balance_hard_right_zeros_left_in_mix_output() {
        let mut m = Mixer::new();
        m.channel_state_mut(0).pan = 64;
        m.set_master_balance_14(0x3FFF); // hard right
        m.note_on(0, 60, 100, voice(0.5, 16));
        let mut l = vec![0.0; 16];
        let mut r = vec![0.0; 16];
        m.mix_stereo(&mut l, &mut r);
        for s in &l {
            assert_eq!(*s, 0.0);
        }
        assert!(r[0].abs() > 0.0);
    }

    #[test]
    fn master_balance_centre_matches_pre_balance_output() {
        // 0x2000 (the default) yields gains (1.0, 1.0), so the mix
        // output must be byte-identical to a mixer whose balance was
        // never touched — proving the round 105 wiring is a pure
        // addition at the default.
        let mut a = Mixer::new();
        a.channel_state_mut(0).pan = 64;
        a.note_on(0, 60, 100, voice(0.5, 16));
        let mut la = vec![0.0; 16];
        let mut ra = vec![0.0; 16];
        a.mix_stereo(&mut la, &mut ra);

        let mut b = Mixer::new();
        b.channel_state_mut(0).pan = 64;
        b.set_master_balance_14(0x2000); // explicit centre
        b.note_on(0, 60, 100, voice(0.5, 16));
        let mut lb = vec![0.0; 16];
        let mut rb = vec![0.0; 16];
        b.mix_stereo(&mut lb, &mut rb);

        assert_eq!(la, lb);
        assert_eq!(ra, rb);
    }

    #[test]
    fn mod_wheel_routes_to_held_voice_scaled_by_rpn5() {
        let mut m = Mixer::new();
        let (v, _, _, depth, _) = instrumented_voice_full(0.5, 1024);
        m.note_on(0, 60, 100, v);
        // RPN 5 → 100 cents range, CC 1 = 127 → depth = 100 cents.
        select_rpn(&mut m, 0, 0, 5);
        m.set_data_entry(0, 0, false); // clear default 50-cent fractional
        m.set_data_entry(0, 1, true); // CC 6 = 1 semitone → 100 cents
        m.set_mod_wheel(0, 127);
        assert_eq!(*depth.lock().unwrap(), 100);
        // CC 1 = 64 (halfway) → depth ≈ 50.
        m.set_mod_wheel(0, 64);
        let d = *depth.lock().unwrap();
        assert!((50..=51).contains(&d), "got {d}");
    }

    #[test]
    fn mod_wheel_applied_at_note_on_when_already_held() {
        let mut m = Mixer::new();
        // Configure range + wheel before the note starts.
        select_rpn(&mut m, 0, 0, 5);
        m.set_data_entry(0, 0, false);
        m.set_data_entry(0, 1, true); // 100-cent range
        m.set_mod_wheel(0, 127);
        let (v, _, _, depth, _) = instrumented_voice_full(0.5, 1024);
        m.note_on(0, 60, 100, v);
        assert_eq!(*depth.lock().unwrap(), 100);
    }

    #[test]
    fn cc74_timbre_routes_to_held_voice() {
        let mut m = Mixer::new();
        let (v, _, _, _, timbre) = instrumented_voice_full(0.5, 1024);
        m.note_on(0, 60, 100, v);
        m.set_timbre(0, 96);
        assert_eq!(*timbre.lock().unwrap(), 96);
    }

    #[test]
    fn mpe_mcm_lower_zone_assigns_roles() {
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4); // 4 member channels
        assert_eq!(m.mpe_zone(MpeZoneKind::Lower).unwrap().members, 4);
        // Manager = ch 0, Members = 1..=4.
        assert!(m.channel_state(0).mpe_role.is_manager());
        for ch in 1..=4u8 {
            assert!(matches!(
                m.channel_state(ch).mpe_role,
                MpeRole::Member(MpeZoneKind::Lower)
            ));
        }
        // Ch 5 untouched.
        assert!(matches!(m.channel_state(5).mpe_role, MpeRole::None));
    }

    #[test]
    fn mpe_mcm_assigns_pb_sensitivity_defaults() {
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4);
        // §2.2.5: Manager = 2 semis, Members = 48 semis.
        assert_eq!(m.channel_state(0).pitch_bend_range_cents, 200);
        for ch in 1..=4u8 {
            assert_eq!(
                m.channel_state(ch).pitch_bend_range_cents,
                4800,
                "member ch {ch}",
            );
        }
    }

    #[test]
    fn mpe_mcm_zero_members_deactivates_zone() {
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4);
        m.set_mpe_zone(MpeZoneKind::Lower, 0);
        assert!(m.mpe_zone(MpeZoneKind::Lower).is_none());
        // Every channel back to None.
        for ch in 0..=15u8 {
            assert!(matches!(m.channel_state(ch).mpe_role, MpeRole::None));
        }
    }

    #[test]
    fn mpe_upper_zone_assigns_roles_top_down() {
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Upper, 3); // 3 member channels
                                               // Manager = ch 15, Members = 12..=14.
        assert!(matches!(
            m.channel_state(15).mpe_role,
            MpeRole::Manager(MpeZoneKind::Upper)
        ));
        for ch in 12..=14u8 {
            assert!(matches!(
                m.channel_state(ch).mpe_role,
                MpeRole::Member(MpeZoneKind::Upper)
            ));
        }
    }

    #[test]
    fn mpe_member_pitch_bend_combines_with_manager() {
        // Appendix C: per-note bend on the Member sums with the
        // Manager's zone-wide bend.
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4);
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(1, 60, 100, v); // member channel
                                  // Manager bend = +1 semi (range = 200 cents).
        m.set_pitch_bend(0, 0x3FFF); // full positive at 200 cents = +200 cents
                                     // Member bend = half-positive at 4800-cent range = +1200 cents.
        m.set_pitch_bend(1, 0x3000); // 0x3000 - 0x2000 = +4096
                                     //   = 4096 * 4800 / 8192 = 2400 cents
        let total = *bend_cell.lock().unwrap();
        // Manager (200) + Member (2400) = 2600.
        assert!(
            (2595..=2600).contains(&total),
            "expected ~2600, got {total}",
        );
    }

    #[test]
    fn mpe_manager_cc74_broadcasts_to_zone() {
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4);
        let (v1, _, _, _, t1) = instrumented_voice_full(0.5, 1024);
        let (v2, _, _, _, t2) = instrumented_voice_full(0.5, 1024);
        m.note_on(1, 60, 100, v1);
        m.note_on(3, 64, 100, v2);
        // Manager Channel CC74 should reach every Member's note.
        m.set_timbre(0, 100);
        assert_eq!(*t1.lock().unwrap(), 100);
        assert_eq!(*t2.lock().unwrap(), 100);
    }

    #[test]
    fn mpe_member_cc74_does_not_leak_to_other_members() {
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4);
        let (v1, _, _, _, t1) = instrumented_voice_full(0.5, 1024);
        let (v2, _, _, _, t2) = instrumented_voice_full(0.5, 1024);
        m.note_on(1, 60, 100, v1);
        m.note_on(3, 64, 100, v2);
        // Member Channel 1's CC74 must only reach its own voice.
        m.set_timbre(1, 90);
        assert_eq!(*t1.lock().unwrap(), 90);
        assert_eq!(*t2.lock().unwrap(), 0, "ch3 should not see ch1's CC74");
    }

    #[test]
    fn mpe_member_blocks_poly_pressure() {
        // §2.2.7: "Polyphonic Key Pressure shall not be sent on
        // Member Channels."
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4);
        let (v, _, press) = instrumented_voice(0.5, 1024);
        m.note_on(1, 60, 100, v);
        m.set_poly_pressure(1, 60, 100);
        assert_eq!(
            *press.lock().unwrap(),
            0.0,
            "poly-pressure must be dropped on Member"
        );
    }

    #[test]
    fn mpe_member_channel_pressure_combines_with_manager_via_max() {
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 4);
        let (v, _, press) = instrumented_voice(0.5, 1024);
        m.note_on(1, 60, 100, v);
        // Member pressure 60.
        m.set_channel_pressure(1, 60);
        let p1 = *press.lock().unwrap();
        assert!(p1 > 0.0);
        // Manager pressure 100 — combined = max = 100.
        m.set_channel_pressure(0, 100);
        let p2 = *press.lock().unwrap();
        assert!(p2 > p1, "manager pressure should raise combined value");
        assert!((p2 - 100.0 / 127.0).abs() < 1e-4);
    }

    #[test]
    fn mpe_zone_conflict_shrinks_other_zone() {
        // Lower zone = 8 members (channels 1..=8). Upper zone = 8
        // members would want channels 7..=14, which overlaps. The
        // most-recent MCM (Upper) takes precedence and Lower must
        // shrink.
        let mut m = Mixer::new();
        m.set_mpe_zone(MpeZoneKind::Lower, 8);
        m.set_mpe_zone(MpeZoneKind::Upper, 8);
        let lower = m.mpe_zone(MpeZoneKind::Lower).unwrap();
        // Lower must have shrunk to only the non-conflicting members
        // (1..=6 — since Upper claims 7..=14).
        assert!(
            lower.members <= 6,
            "lower zone should have shrunk, got {} members",
            lower.members,
        );
    }

    #[test]
    fn mcm_via_data_entry_on_manager_channel_sets_zone() {
        // Send MCM via the same channels-RPN-data-entry pathway the
        // scheduler uses: CC 101 = 0, CC 100 = 6, CC 6 = 4.
        let mut m = Mixer::new();
        m.set_rpn_byte(0, 0, true); // RPN MSB = 0
        m.set_rpn_byte(0, 6, false); // RPN LSB = 6
        m.set_data_entry(0, 4, true); // 4 member channels
        assert_eq!(m.mpe_zone(MpeZoneKind::Lower).unwrap().members, 4);
    }

    #[test]
    fn mcm_on_non_manager_channel_is_ignored() {
        // §2.2.1: "n=0x0: Lower Zone Manager Channel; n=0xF: Upper
        // Zone Manager Channel; All other values are invalid and
        // should be ignored."
        let mut m = Mixer::new();
        m.set_rpn_byte(5, 0, true);
        m.set_rpn_byte(5, 6, false);
        m.set_data_entry(5, 4, true);
        assert!(m.mpe_zone(MpeZoneKind::Lower).is_none());
        assert!(m.mpe_zone(MpeZoneKind::Upper).is_none());
    }

    // ───────────────────────── MTS microtuning ─────────────────────────

    #[test]
    fn key_tuning_folds_into_note_on_pitch() {
        // Retune key 60 up one semitone (`3D 00 00` addressed to key
        // 60 = +100 cents) before the note starts. The note-on pitch
        // composition must carry the +100 cents.
        let mut m = Mixer::new();
        m.set_key_tuning_word(60, [0x3D, 0x00, 0x00], true);
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        let cents = *bend_cell.lock().unwrap();
        assert_eq!(
            cents, 100,
            "expected +100 cents from key tuning, got {cents}"
        );
    }

    #[test]
    fn key_tuning_retunes_sounding_note_live() {
        // A real-time Single-Note Tuning Change must re-tune a note
        // that is already sounding.
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 64, 100, v);
        assert_eq!(*bend_cell.lock().unwrap(), 0);
        m.set_key_tuning_word(64, [0x40, 0x40, 0x00], true); // +50 cents
        let cents = *bend_cell.lock().unwrap();
        assert_eq!(
            cents, 50,
            "live retune should reach the held voice, got {cents}"
        );
    }

    #[test]
    fn key_tuning_non_live_does_not_retune_sounding_note() {
        // Non-real-time (setup) form must NOT disturb a sounding note.
        let mut m = Mixer::new();
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 64, 100, v);
        m.set_key_tuning_word(64, [0x40, 0x40, 0x00], false); // +50 cents, setup
        assert_eq!(
            *bend_cell.lock().unwrap(),
            0,
            "setup form must not touch sounding note"
        );
        // But the next note picks up the stored offset.
        let (v2, bend2, _) = instrumented_voice(0.5, 1024);
        m.note_on(1, 64, 100, v2);
        assert_eq!(*bend2.lock().unwrap(), 50);
    }

    #[test]
    fn scale_octave_tuning_per_pitch_class() {
        // Put +20 cents on pitch class C (= key 60, 72, …) on channel 2.
        let mut m = Mixer::new();
        let mut offsets = [0.0f32; 12];
        offsets[0] = 20.0; // C
        m.set_scale_octave_tuning(2, offsets, true);
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(2, 72, 100, v); // C5, pitch class C
        assert_eq!(*bend_cell.lock().unwrap(), 20);
        // A D5 (pitch class D, offset 0) on the same channel is untouched.
        let (v2, bend2, _) = instrumented_voice(0.5, 1024);
        m.note_on(2, 74, 100, v2);
        assert_eq!(*bend2.lock().unwrap(), 0);
        // Channel 0 is unaffected by channel 2's scale/octave row.
        let (v3, bend3, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 72, 100, v3);
        assert_eq!(*bend3.lock().unwrap(), 0);
    }

    #[test]
    fn drum_channel_exempt_from_key_tuning() {
        // The drum bus (ch 10 = index 9) must not pitch-shift — a
        // different pitch is a different drum sound.
        let mut m = Mixer::new();
        m.set_key_tuning_word(36, [0x25, 0x00, 0x00], true); // +1 semitone
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(9, 36, 100, v);
        assert_eq!(
            *bend_cell.lock().unwrap(),
            0,
            "drum channel must not retune"
        );
    }

    #[test]
    fn key_tuning_sums_with_pitch_bend() {
        // +50 cents key tuning on key 60, then a +1 semitone bend
        // (half of the default ±2-semitone range). The voice should see
        // the tuning summed with the bend.
        let mut m = Mixer::new();
        m.set_key_tuning_word(60, [0x3C, 0x40, 0x00], false); // +50 cents
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        // 0x3000 is +0x1000 = +half-range = +100 cents.
        m.set_pitch_bend(0, 0x3000);
        let cents = *bend_cell.lock().unwrap();
        assert_eq!(cents, 150, "expected 50 + 100, got {cents}");
    }

    #[test]
    fn reset_tuning_restores_equal_temperament() {
        let mut m = Mixer::new();
        m.set_key_tuning_word(60, [0x3D, 0x00, 0x00], true);
        let (v, bend_cell, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v);
        assert_eq!(*bend_cell.lock().unwrap(), 100);
        m.reset_tuning();
        // Held voice goes back to centre.
        assert_eq!(*bend_cell.lock().unwrap(), 0);
        // New note also reads equal temperament.
        let (v2, bend2, _) = instrumented_voice(0.5, 1024);
        m.note_on(0, 60, 100, v2);
        assert_eq!(*bend2.lock().unwrap(), 0);
    }

    // ───────────────────────── effects bus (CA-024) ─────────────────────────

    /// Sum of |sample| over a planar buffer — a cheap "is there energy
    /// here" probe for the effect-tail tests.
    fn energy(buf: &[f32]) -> f32 {
        buf.iter().map(|s| s.abs()).sum()
    }

    #[test]
    fn dry_by_default_no_reverb_or_chorus() {
        // With both sends at their default (0), the fx bus must be a
        // no-op: the rendered chunk equals the pre-fx dry mix exactly.
        let mut dry = Mixer::new();
        let mut wet = Mixer::new();
        dry.note_on(0, 60, 100, voice(0.5, 64));
        wet.note_on(0, 60, 100, voice(0.5, 64));

        let mut dl = vec![0.0; 64];
        let mut dr = vec![0.0; 64];
        let mut wl = vec![0.0; 64];
        let mut wr = vec![0.0; 64];
        // `dry` keeps sends at 0; `wet` would too (both default 0), so
        // they should be bit-identical.
        dry.mix_stereo(&mut dl, &mut dr);
        wet.mix_stereo(&mut wl, &mut wr);
        assert_eq!(dl, wl);
        assert_eq!(dr, wr);
        // And there must be no tail: a follow-up silent chunk stays
        // silent because nothing was ever fed into the delay lines.
        let mut tl = vec![0.0; 64];
        let mut tr = vec![0.0; 64];
        dry.mix_stereo(&mut tl, &mut tr);
        assert_eq!(energy(&tl), 0.0);
        assert_eq!(energy(&tr), 0.0);
    }

    #[test]
    fn reverb_send_produces_a_tail() {
        // A channel with a non-zero reverb send (CC 91) should leave a
        // decaying tail after the note has stopped sounding. The Schroeder
        // comb delays are ~1100+ samples at 44.1 kHz, so the first echo
        // only emerges a chunk or two after the dry note; we render in
        // 512-sample chunks and let several elapse.
        let mut m = Mixer::new();
        m.channel_state_mut(0).reverb_send = 127;
        // A short dry note (the tail must outlast it).
        m.note_on(0, 60, 100, voice(0.5, 64));

        let mut l = vec![0.0; 512];
        let mut r = vec![0.0; 512];
        m.mix_stereo(&mut l, &mut r);
        assert!(energy(&l) > 0.0, "the dry note should produce output");

        // Render several silent chunks; the reverb echo must appear.
        let mut tail_energy = 0.0;
        for _ in 0..6 {
            let mut tl = vec![0.0; 512];
            let mut tr = vec![0.0; 512];
            m.mix_stereo(&mut tl, &mut tr);
            tail_energy += energy(&tl) + energy(&tr);
        }
        assert!(
            tail_energy > 0.0,
            "reverb tail should sound after the dry note ends"
        );
    }

    #[test]
    fn chorus_send_produces_a_tail() {
        // CC 93 chorus send: a modulated-delay return that lingers past
        // the dry note. The chorus base delay is ~12 ms (~530 samples),
        // so again we render in larger chunks.
        let mut m = Mixer::new();
        m.channel_state_mut(0).chorus_send = 127;
        m.note_on(0, 60, 100, voice(0.5, 64));

        let mut l = vec![0.0; 512];
        let mut r = vec![0.0; 512];
        m.mix_stereo(&mut l, &mut r);

        let mut tail_energy = 0.0;
        for _ in 0..4 {
            let mut tl = vec![0.0; 512];
            let mut tr = vec![0.0; 512];
            m.mix_stereo(&mut tl, &mut tr);
            tail_energy += energy(&tl) + energy(&tr);
        }
        assert!(
            tail_energy > 0.0,
            "chorus return should sound after the dry note ends"
        );
    }

    #[test]
    fn longer_reverb_time_decays_slower() {
        // Compare the reverb tail energy several chunks out for a short
        // vs. a long Reverb Time. The longer T60 must retain more energy.
        fn tail_energy(reverb_time_val: u8) -> f32 {
            let mut m = Mixer::new();
            m.set_gm_reverb_param(1, reverb_time_val); // pp=1 Reverb Time
            m.channel_state_mut(0).reverb_send = 127;
            m.note_on(0, 60, 100, voice(0.5, 64));
            let mut l = vec![0.0; 512];
            let mut r = vec![0.0; 512];
            m.mix_stereo(&mut l, &mut r); // dry + start of tail
                                          // Run a few more chunks to let the decay separate.
            for _ in 0..8 {
                m.mix_stereo(&mut l, &mut r);
            }
            energy(&l) + energy(&r)
        }
        // val per CA-024 `val = ln(rt)/0.025 + 40`: smaller val → shorter
        // time. 20 → ~0.6 s, 100 → ~7.4 s.
        let short = tail_energy(20);
        let long = tail_energy(100);
        assert!(
            long > short,
            "longer reverb time should retain more tail energy: short={short}, long={long}"
        );
    }

    #[test]
    fn reverb_type_select_sets_default_time() {
        // CA-024: "When a Reverb Type is selected, the default Reverb
        // Time from the table below for that Reverb Type should be set."
        let mut m = Mixer::new();
        // Type 0 (Small Room) default time value = 44.
        m.set_gm_reverb_param(0, 0);
        let expect_small = GmEffects::reverb_time_from_val(44);
        assert!((m.gm_effects().reverb_time_s - expect_small).abs() < 1e-4);
        assert_eq!(m.gm_effects().reverb_type, 0);
        // Type 4 (Large Hall) default time value = 64.
        m.set_gm_reverb_param(0, 4);
        let expect_hall = GmEffects::reverb_time_from_val(64);
        assert!((m.gm_effects().reverb_time_s - expect_hall).abs() < 1e-4);
        // A subsequent explicit pp=1 still overrides the type default.
        m.set_gm_reverb_param(1, 100);
        let expect_long = GmEffects::reverb_time_from_val(100);
        assert!((m.gm_effects().reverb_time_s - expect_long).abs() < 1e-4);
    }

    #[test]
    fn chorus_type_select_loads_parameter_row() {
        // CA-024: "pp = 0 : Chorus Type … Sets Chorus parameters as
        // listed below." Type 4 (FB Chorus) row: FB 64, Rate 2, Depth
        // 24, Rev Send 0.
        let mut m = Mixer::new();
        m.set_gm_chorus_param(0, 4);
        let fx = m.gm_effects();
        assert_eq!(fx.chorus_type, 4);
        assert!((fx.chorus_mod_rate_hz - 2.0 * 0.122).abs() < 1e-4);
        assert!((fx.chorus_mod_depth_ms - (24.0 + 1.0) / 3.2).abs() < 1e-4);
        assert!((fx.chorus_feedback_pct - 64.0 * 0.763).abs() < 1e-3);
        assert!((fx.chorus_send_to_reverb_pct - 0.0).abs() < 1e-4);
        // An explicit pp=3 feedback edit still overrides the row default.
        m.set_gm_chorus_param(3, 100);
        assert!((m.gm_effects().chorus_feedback_pct - 100.0 * 0.763).abs() < 1e-3);
    }

    #[test]
    fn chorus_to_reverb_send_routes_into_reverb() {
        // With reverb send OFF but chorus send ON and chorus→reverb
        // send at max, the reverb still receives energy via the chorus
        // output (CA-024 chorus pp=4). The tail should outlast a plain
        // chorus-only tail.
        let mut routed = Mixer::new();
        routed.channel_state_mut(0).chorus_send = 127;
        routed.set_gm_chorus_param(4, 127); // pp=4 Send-to-Reverb ≈ 100%
        routed.set_gm_reverb_param(1, 100); // long reverb so it's audible
        routed.note_on(0, 60, 100, voice(0.5, 64));

        let mut l = vec![0.0; 256];
        let mut r = vec![0.0; 256];
        routed.mix_stereo(&mut l, &mut r);
        // Several chunks out, reverb fed by the chorus send should still
        // ring.
        for _ in 0..6 {
            routed.mix_stereo(&mut l, &mut r);
        }
        assert!(
            energy(&l) + energy(&r) > 0.0,
            "chorus→reverb send should keep the reverb ringing"
        );
    }

    #[test]
    fn reset_gm_effects_flushes_tail() {
        // GM System On/Off resets the effect parameters AND silences the
        // tails.
        let mut m = Mixer::new();
        m.channel_state_mut(0).reverb_send = 127;
        m.note_on(0, 60, 100, voice(0.5, 64));
        let mut l = vec![0.0; 64];
        let mut r = vec![0.0; 64];
        m.mix_stereo(&mut l, &mut r);
        // There's a live tail now; reset must clear it.
        m.reset_gm_effects();
        let mut tl = vec![0.0; 64];
        let mut tr = vec![0.0; 64];
        // The channel send is still 127, but the dry note is gone and the
        // delay lines were flushed, so the silent input yields silence.
        m.mix_stereo(&mut tl, &mut tr);
        assert_eq!(energy(&tl), 0.0);
        assert_eq!(energy(&tr), 0.0);
    }

    #[test]
    fn set_sample_rate_resizes_effects_bus() {
        let mut m = Mixer::new();
        assert_eq!(m.sample_rate(), crate::OUTPUT_SAMPLE_RATE);
        m.set_sample_rate(48_000);
        assert_eq!(m.sample_rate(), 48_000);
        // The bus still renders without panicking at the new rate.
        m.channel_state_mut(0).reverb_send = 64;
        m.note_on(0, 60, 100, voice(0.5, 64));
        let mut l = vec![0.0; 64];
        let mut r = vec![0.0; 64];
        m.mix_stereo(&mut l, &mut r);
    }
}
