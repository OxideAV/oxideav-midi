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
    /// CC 6 = MSB semitones, CC 38 = LSB cents).
    pub pitch_bend_range_cents: u16,
    /// Channel pressure (mono aftertouch) as the raw `0..=127` scalar.
    /// Default 0 = no pressure modulation.
    pub channel_pressure: u8,
    /// Currently-selected RPN as a 14-bit value (CC 100 LSB / CC 101
    /// MSB). `0x3FFF` is the "RPN null" marker that disables further
    /// CC-6 / CC-38 writes. We default to null so a CC 6 with no prior
    /// RPN selection doesn't accidentally clobber the bend range.
    pub rpn: u16,
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
        }
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
        }
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
    /// RPN 0). Every still-held voice on `channel` is updated at once.
    pub fn set_pitch_bend(&mut self, channel: u8, value: u16) {
        let ch = channel as usize % NUM_CHANNELS;
        let v = value & 0x3FFF;
        self.channels[ch].pitch_bend = v;
        let cents = pitch_bend_to_cents(v, self.channels[ch].pitch_bend_range_cents);
        for slot in self.slots.iter_mut() {
            if slot.channel == channel {
                if let Some(voice) = slot.voice.as_mut() {
                    voice.set_pitch_bend_cents(cents);
                }
            }
        }
    }

    /// Apply channel pressure (mono aftertouch, MIDI status `Dn`). The
    /// `0..=127` value modulates volume on every still-held voice on
    /// `channel`.
    pub fn set_channel_pressure(&mut self, channel: u8, value: u8) {
        let ch = channel as usize % NUM_CHANNELS;
        self.channels[ch].channel_pressure = value;
        let p = (value as f32 / 127.0).clamp(0.0, 1.0);
        for slot in self.slots.iter_mut() {
            if slot.channel == channel {
                if let Some(voice) = slot.voice.as_mut() {
                    voice.set_pressure(p);
                }
            }
        }
    }

    /// Apply polyphonic key pressure (per-key aftertouch, MIDI status
    /// `An`). Only voices matching `(channel, key)` are touched.
    pub fn set_poly_pressure(&mut self, channel: u8, key: u8, value: u8) {
        let p = (value as f32 / 127.0).clamp(0.0, 1.0);
        for slot in self.slots.iter_mut() {
            if slot.channel == channel && slot.key == key {
                if let Some(voice) = slot.voice.as_mut() {
                    voice.set_pressure(p);
                }
            }
        }
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
    /// currently-selected RPN is. Round-4 only honours RPN 0
    /// (pitch-bend range): CC 6 = semitone count, CC 38 = additional
    /// cents. Other RPNs are silently ignored.
    pub fn set_data_entry(&mut self, channel: u8, value: u8, is_msb: bool) {
        let ch = channel as usize % NUM_CHANNELS;
        if self.channels[ch].rpn != 0 {
            // Only RPN 0 (pitch-bend range) matters in round 4.
            return;
        }
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
        let st = self.channels[channel as usize % NUM_CHANNELS];
        let cents = pitch_bend_to_cents(st.pitch_bend, st.pitch_bend_range_cents);
        if cents != 0 {
            voice.set_pitch_bend_cents(cents);
        }
        if st.channel_pressure != 0 {
            voice.set_pressure(st.channel_pressure as f32 / 127.0);
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

            // Per-channel volume / pan. Both folded once per chunk.
            let st = self.channels[slot.channel as usize % NUM_CHANNELS];
            let vol = st.volume as f32 / 127.0;
            // Constant-power pan: θ in [0, π/2], left = cos(θ), right = sin(θ).
            let pan_norm = (st.pan as f32 / 127.0).clamp(0.0, 1.0);
            let theta = pan_norm * std::f32::consts::FRAC_PI_2;

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
                let lg = vol * self.mix_gain * l_balance;
                let rg = vol * self.mix_gain * r_balance;
                for i in 0..n {
                    left[i] += lscratch[i] * lg;
                    right[i] += rscratch[i] * rg;
                }
            } else {
                let l_gain = theta.cos() * vol * self.mix_gain;
                let r_gain = theta.sin() * vol * self.mix_gain;
                for i in 0..n {
                    let s = mono[i];
                    left[i] += s * l_gain;
                    right[i] += s * r_gain;
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
    /// pitch-bend / pressure value pushed in via the optional Voice
    /// methods so tests can assert routing.
    struct ConstVoice {
        value: f32,
        remaining: usize,
        done: bool,
        last_bend_cents: std::sync::Arc<std::sync::Mutex<i32>>,
        last_pressure: std::sync::Arc<std::sync::Mutex<f32>>,
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
    }

    fn voice(value: f32, samples: usize) -> Box<dyn Voice> {
        Box::new(ConstVoice {
            value,
            remaining: samples,
            done: false,
            last_bend_cents: std::sync::Arc::new(std::sync::Mutex::new(0)),
            last_pressure: std::sync::Arc::new(std::sync::Mutex::new(0.0)),
        })
    }

    type BendCell = std::sync::Arc<std::sync::Mutex<i32>>;
    type PressureCell = std::sync::Arc<std::sync::Mutex<f32>>;

    /// Build a [`ConstVoice`] plus shared handles to its `last_bend_cents`
    /// / `last_pressure` cells so the test can read the values back after
    /// the mixer has handed the voice to its slot.
    fn instrumented_voice(value: f32, samples: usize) -> (Box<dyn Voice>, BendCell, PressureCell) {
        let bend = std::sync::Arc::new(std::sync::Mutex::new(0));
        let press = std::sync::Arc::new(std::sync::Mutex::new(0.0));
        let v = Box::new(ConstVoice {
            value,
            remaining: samples,
            done: false,
            last_bend_cents: bend.clone(),
            last_pressure: press.clone(),
        });
        (v, bend, press)
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
}
