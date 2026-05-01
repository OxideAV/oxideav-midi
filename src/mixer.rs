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
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            program: 0,
            volume: 100,
            pan: 64,
            sustain: false,
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
    /// voice (the SF2 / tone constructors fold it in).
    pub fn note_on(&mut self, channel: u8, key: u8, velocity: u8, voice: Box<dyn Voice>) {
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
        // Per-voice scratch buffer to render into before panning. We
        // could re-render in-place but that would force the voice to
        // know about the stereo bus; keeping the voice mono is the
        // round-2/3 contract.
        let mut mono = vec![0.0f32; left.len()];
        for slot in self.slots.iter_mut() {
            let n = if let Some(v) = slot.voice.as_mut() {
                let n = v.render(&mut mono);
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
            let l_gain = theta.cos() * vol * self.mix_gain;
            let r_gain = theta.sin() * vol * self.mix_gain;

            for i in 0..n {
                let s = mono[i];
                left[i] += s * l_gain;
                right[i] += s * r_gain;
            }
            active += 1;

            // If the voice produced fewer than `mono.len()` samples it
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
    /// without standing up a full SF2 fixture.
    struct ConstVoice {
        value: f32,
        remaining: usize,
        done: bool,
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
    }

    fn voice(value: f32, samples: usize) -> Box<dyn Voice> {
        Box::new(ConstVoice {
            value,
            remaining: samples,
            done: false,
        })
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
}
