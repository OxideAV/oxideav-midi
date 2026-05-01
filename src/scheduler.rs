//! SMF event scheduler — converts a parsed [`SmfFile`](crate::smf::SmfFile)
//! into an absolute-tick, time-ordered event stream and walks it
//! sample-by-sample against the audio output rate.
//!
//! Round-3 implementation. Owns:
//!
//! - The merged event list, sorted by absolute tick. Stable sort so two
//!   events on the same tick fire in original (track-order, then
//!   in-track) order — matters for "program change followed by note on
//!   on tick 0".
//! - The current tempo (microseconds per quarter), updated by every
//!   [`MetaEvent::Tempo`](crate::smf::MetaEvent::Tempo) we walk past.
//!   Default is 500_000 µs/qn (= 120 BPM) per the SMF spec.
//! - The output sample rate.
//! - A monotonic playhead in **fractional samples**. Per-tick conversion
//!   uses an `f64` accumulator so a tempo change at any moment doesn't
//!   drift the per-sample alignment.
//!
//! The scheduler does *not* render audio — it dispatches into a
//! [`Mixer`](crate::mixer::Mixer) (`note_on` / `note_off` / control
//! changes) and stops when every track has run past its
//! End-of-Track. Audio synthesis happens inside the mixer, frame by
//! frame, between scheduler ticks.
//!
//! ## Tempo → samples-per-tick math
//!
//! For a `TicksPerQuarter(division)` SMF with tempo `usec_per_quarter`
//! and output rate `sr`:
//!
//! ```text
//!   seconds_per_tick = (usec_per_quarter / 1_000_000) / division
//!   samples_per_tick = seconds_per_tick * sr
//!                    = (usec_per_quarter * sr) / (1_000_000 * division)
//! ```
//!
//! For SMPTE divisions (`Smpte { fps, ticks_per_frame }`) the conversion
//! is independent of tempo:
//!
//! ```text
//!   samples_per_tick = sr / (fps * ticks_per_frame)
//! ```
//!
//! Both quantities are recomputed on every tempo change (which is a
//! no-op for SMPTE) so the scheduler never trusts a stale rate when a
//! tempo meta event arrives mid-track.

use crate::instruments::Instrument;
use crate::mixer::{Mixer, NUM_CHANNELS};
use crate::smf::{ChannelBody, ChannelMessage, Division, Event, MetaEvent, SmfFile};

/// Default tempo when no `FF 51 03` is seen yet — 120 BPM = 500 000 µs
/// per quarter note. The SMF spec mandates this default.
pub const DEFAULT_TEMPO_USEC_PER_QUARTER: u32 = 500_000;

/// One event in the merged time line. We keep the source `track` index
/// for stable-sort ties and `order` for intra-track ordering, both
/// secondary keys behind `tick`.
#[derive(Clone, Debug)]
struct AbsEvent {
    /// Absolute tick from the start of the file (sum of deltas in the
    /// originating track).
    tick: u64,
    /// Originating track index — used as a tie-breaker when two events
    /// fall on the same absolute tick.
    track: u16,
    /// Within-track order — tie-breaker after `track`.
    order: u32,
    event: Event,
}

/// SMF event scheduler. Owns the merged event list + current tempo +
/// fractional sample playhead; dispatches against a [`Mixer`].
pub struct Scheduler {
    events: Vec<AbsEvent>,
    /// Index of the next event to fire.
    cursor: usize,
    division: Division,
    /// Current tempo in microseconds per quarter note. Updated on every
    /// `MetaEvent::Tempo`.
    tempo_us_per_quarter: u32,
    /// Output sample rate, hertz.
    sample_rate: u32,
    /// Fractional sample playhead. `events[cursor].tick * samples_per_tick`
    /// is the absolute sample at which that event fires.
    samples_elapsed: f64,
    /// Cached `samples_per_tick` derived from `tempo_us_per_quarter` /
    /// `division` / `sample_rate`. Recomputed on every tempo change
    /// (which is a no-op for SMPTE divisions).
    samples_per_tick: f64,
    /// Accumulated samples that have already been "consumed" by the
    /// caller via `step`. Used to compute when the next event fires.
    sample_clock: f64,
}

impl Scheduler {
    /// Build a scheduler from a parsed SMF file. The merged event list
    /// is sorted on construction, so `step` is hot-path-cheap.
    pub fn new(smf: &SmfFile, sample_rate: u32) -> Self {
        let mut events: Vec<AbsEvent> = Vec::new();
        // Format-2 files are independent sequences — the spec says they
        // should be played one after the other, not merged. Round-3
        // treats them like format-1 (concurrent) so the test fixtures
        // and most ".mid" files in the wild work; round-4 may revisit.
        for (ti, track) in smf.tracks.iter().enumerate() {
            let mut tick: u64 = 0;
            for (oi, te) in track.events.iter().enumerate() {
                tick = tick.saturating_add(te.delta as u64);
                events.push(AbsEvent {
                    tick,
                    track: ti as u16,
                    order: oi as u32,
                    event: te.kind.clone(),
                });
            }
        }
        // Stable sort by (tick, track, order) — the secondary keys keep
        // chord notes that share a tick in their original on-disk order.
        events.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then_with(|| a.track.cmp(&b.track))
                .then_with(|| a.order.cmp(&b.order))
        });

        let mut s = Self {
            events,
            cursor: 0,
            division: smf.header.division,
            tempo_us_per_quarter: DEFAULT_TEMPO_USEC_PER_QUARTER,
            sample_rate: sample_rate.max(1),
            samples_elapsed: 0.0,
            samples_per_tick: 0.0,
            sample_clock: 0.0,
        };
        s.recompute_samples_per_tick();
        s
    }

    /// Sample rate the scheduler converts ticks against.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// `true` once every event has been dispatched.
    pub fn is_done(&self) -> bool {
        self.cursor >= self.events.len()
    }

    /// Total scheduled-event count. Useful for diagnostics.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Absolute sample at which the next event fires (relative to the
    /// scheduler's own clock origin). `None` once exhausted.
    pub fn next_event_sample(&self) -> Option<f64> {
        self.events
            .get(self.cursor)
            .map(|e| e.tick as f64 * self.samples_per_tick)
    }

    /// Advance the scheduler by `samples` audio frames. Every event
    /// whose absolute sample falls within
    /// `[sample_clock, sample_clock + samples)` is dispatched into
    /// `mixer` (and into `instrument` for `note_on`s).
    ///
    /// Returns `true` if the scheduler is now done (all events
    /// dispatched).
    pub fn step(&mut self, samples: usize, mixer: &mut Mixer, instrument: &dyn Instrument) -> bool {
        let next_clock = self.sample_clock + samples as f64;
        while self.cursor < self.events.len() {
            // Always recompute the firing sample against the *current*
            // samples_per_tick — a tempo change moves the firing time
            // of all subsequent events. We do this incrementally: the
            // base is `samples_elapsed` (samples spent up to the last
            // dispatched event's tick), plus the tick delta from that
            // event to the next, scaled by the current rate.
            let evt_tick = self.events[self.cursor].tick;
            let last_tick = if self.cursor == 0 {
                0
            } else {
                self.events[self.cursor - 1].tick
            };
            let dt_ticks = evt_tick.saturating_sub(last_tick);
            let evt_sample = self.samples_elapsed + dt_ticks as f64 * self.samples_per_tick;
            if evt_sample >= next_clock {
                break;
            }
            // Time has come — dispatch.
            self.samples_elapsed = evt_sample;
            // Clone the event so we don't hold a borrow across `dispatch`.
            let event = self.events[self.cursor].event.clone();
            self.dispatch(&event, mixer, instrument);
            self.cursor += 1;
        }
        self.sample_clock = next_clock;
        self.is_done()
    }

    /// Dispatch one event. Does not advance the cursor or the sample
    /// clock — the caller (`step`) handles both.
    fn dispatch(&mut self, evt: &Event, mixer: &mut Mixer, instrument: &dyn Instrument) {
        match evt {
            Event::Channel(ChannelMessage { channel, body }) => {
                self.dispatch_channel(*channel, *body, mixer, instrument);
            }
            Event::Meta(MetaEvent::Tempo(us)) => {
                if *us != 0 {
                    self.tempo_us_per_quarter = *us;
                    self.recompute_samples_per_tick();
                }
            }
            Event::Meta(MetaEvent::EndOfTrack) => {
                // Per-track end. The scheduler stops when *all* tracks
                // have run past their EOT, which the cursor handles
                // naturally — there's nothing extra to do here.
            }
            // Other meta events (text, time-signature, key-signature,
            // SMPTE offset, sequencer-specific, …) carry no playback
            // semantics for round-3.
            Event::Meta(_) => {}
            // Sysex events are ignored too — GM-on resets and the like
            // are common, but round-3 has no synth-engine state to
            // reset besides what `Mixer::all_notes_off` already covers.
            Event::Sysex { .. } => {}
        }
    }

    fn dispatch_channel(
        &self,
        channel: u8,
        body: ChannelBody,
        mixer: &mut Mixer,
        instrument: &dyn Instrument,
    ) {
        let ch = channel as usize % NUM_CHANNELS;
        match body {
            ChannelBody::NoteOn { key, velocity: 0 } => {
                // Velocity-0 NoteOn is conventionally a NoteOff (the
                // running-status optimisation).
                mixer.note_off(channel, key);
            }
            ChannelBody::NoteOn { key, velocity } => {
                let program = mixer.channel_state(channel).program;
                // Channel 10 (index 9) is the GM percussion bus — the
                // "key" is the drum kit slot, not a pitch. We pass it
                // through as program=0 and let the SF2 lookup find the
                // right sample by the key range. (Real GM banks
                // expose the drum kit as bank=128 program=0; round-3
                // doesn't yet ask for bank 128.)
                let _drum = ch == 9;
                if let Ok(voice) = instrument.make_voice(program, key, velocity, self.sample_rate) {
                    mixer.note_on(channel, key, velocity, voice);
                }
                // If the instrument refused (no preset, etc.) we drop
                // the note silently — callers expecting strict failure
                // can wire a logger here in a later round.
            }
            ChannelBody::NoteOff { key, velocity: _ } => {
                mixer.note_off(channel, key);
            }
            ChannelBody::ProgramChange { program } => {
                mixer.channel_state_mut(channel).program = program;
            }
            ChannelBody::ControlChange { controller, value } => match controller {
                6 => mixer.set_data_entry(channel, value, true), // RPN data MSB
                7 => mixer.channel_state_mut(channel).volume = value,
                10 => mixer.channel_state_mut(channel).pan = value,
                38 => mixer.set_data_entry(channel, value, false), // RPN data LSB
                64 => mixer.set_sustain(channel, value),
                100 => mixer.set_rpn_byte(channel, value, false), // RPN LSB
                101 => mixer.set_rpn_byte(channel, value, true),  // RPN MSB
                120 | 123 => {
                    // CC 120 = All Sound Off, CC 123 = All Notes Off.
                    mixer.all_notes_off();
                }
                _ => { /* other CCs not modelled in round 4 */ }
            },
            ChannelBody::PolyAftertouch { key, pressure } => {
                mixer.set_poly_pressure(channel, key, pressure);
            }
            ChannelBody::ChannelAftertouch { pressure } => {
                mixer.set_channel_pressure(channel, pressure);
            }
            ChannelBody::PitchBend { value } => {
                mixer.set_pitch_bend(channel, value);
            }
        }
    }

    /// (Re)compute `samples_per_tick` from the current tempo +
    /// division + sample_rate. Called on construction and on every
    /// tempo meta-event.
    fn recompute_samples_per_tick(&mut self) {
        let sr = self.sample_rate as f64;
        self.samples_per_tick = match self.division {
            Division::TicksPerQuarter(div) => {
                // (usec_per_quarter * sr) / (1_000_000 * division)
                self.tempo_us_per_quarter as f64 * sr / (1_000_000.0 * div.max(1) as f64)
            }
            Division::Smpte {
                frames_per_second,
                ticks_per_frame,
            } => {
                let denom = (frames_per_second.max(1) as f64) * (ticks_per_frame.max(1) as f64);
                sr / denom
            }
        };
    }

    /// Total scheduled length, in samples, at the current tempo. (If a
    /// tempo change later in the file shortens / lengthens the tail,
    /// this estimate becomes stale — but it's only used as a
    /// conservative "render until both this and the mixer are quiet"
    /// stop bound.)
    pub fn estimated_total_samples(&self) -> u64 {
        if let Some(last) = self.events.last() {
            (last.tick as f64 * self.samples_per_tick).ceil() as u64
        } else {
            0
        }
    }
}

// =========================================================================
// Tests.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::tone::ToneInstrument;
    use crate::smf::parse;

    /// Minimal helper: build a one-track SMF blob with the given event
    /// bytes after `MThd`/`MTrk`. The caller must include the EOT.
    fn smf_with_events(division: u16, events: &[u8]) -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"MThd");
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&0u16.to_be_bytes()); // format 0
        blob.extend_from_slice(&1u16.to_be_bytes()); // ntrks
        blob.extend_from_slice(&division.to_be_bytes());
        blob.extend_from_slice(b"MTrk");
        blob.extend_from_slice(&(events.len() as u32).to_be_bytes());
        blob.extend_from_slice(events);
        blob
    }

    #[test]
    fn samples_per_tick_default_tempo_120bpm_44100() {
        let smf = parse(&smf_with_events(480, &[0x00, 0xFF, 0x2F, 0x00])).unwrap();
        let s = Scheduler::new(&smf, 44_100);
        // 120 BPM = 500_000 us/qn. seconds_per_tick = 500_000e-6 / 480
        // = 1.04167 ms. samples_per_tick = 0.00104167 * 44_100 = 45.9375.
        assert!(
            (s.samples_per_tick - 45.9375).abs() < 1e-6,
            "got {}",
            s.samples_per_tick,
        );
    }

    #[test]
    fn samples_per_tick_tempo_change_60bpm() {
        // Tempo 1_000_000 us/qn = 60 BPM. seconds_per_tick =
        // 1_000_000e-6 / 480 = 2.0833 ms; samples_per_tick = 91.875.
        let mut s = Scheduler::new(
            &parse(&smf_with_events(480, &[0x00, 0xFF, 0x2F, 0x00])).unwrap(),
            44_100,
        );
        s.tempo_us_per_quarter = 1_000_000;
        s.recompute_samples_per_tick();
        assert!((s.samples_per_tick - 91.875).abs() < 1e-6);
    }

    #[test]
    fn samples_per_tick_smpte_division() {
        // -25 fps × 40 ticks/frame = 1000 ticks/sec. 48000/1000 = 48.
        let div = u16::from_be_bytes([0xE7, 0x28]);
        let smf = parse(&smf_with_events(div, &[0x00, 0xFF, 0x2F, 0x00])).unwrap();
        let s = Scheduler::new(&smf, 48_000);
        assert!((s.samples_per_tick - 48.0).abs() < 1e-9);
    }

    #[test]
    fn merges_multi_track_in_tick_order() {
        // Track 1: tick 0 → tempo, tick 100 → EOT.
        // Track 2: tick 50 → note on chan 0 key 60, tick 100 → EOT.
        let mut blob = Vec::new();
        blob.extend_from_slice(b"MThd");
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&1u16.to_be_bytes()); // format 1
        blob.extend_from_slice(&2u16.to_be_bytes()); // 2 tracks
        blob.extend_from_slice(&480u16.to_be_bytes());
        // Track 1
        let t1: &[u8] = &[
            0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20, // tempo at tick 0
            0x64, 0xFF, 0x2F, 0x00, // EOT at tick 100
        ];
        blob.extend_from_slice(b"MTrk");
        blob.extend_from_slice(&(t1.len() as u32).to_be_bytes());
        blob.extend_from_slice(t1);
        // Track 2
        let t2: &[u8] = &[
            0x32, 0x90, 0x3C, 0x64, // note on at tick 50
            0x32, 0xFF, 0x2F, 0x00, // EOT at tick 100
        ];
        blob.extend_from_slice(b"MTrk");
        blob.extend_from_slice(&(t2.len() as u32).to_be_bytes());
        blob.extend_from_slice(t2);

        let smf = parse(&blob).unwrap();
        let s = Scheduler::new(&smf, 44_100);
        // Order should be: tempo (tick 0, track 0), note (tick 50, track 1),
        // EOT-1 (tick 100, track 0), EOT-2 (tick 100, track 1).
        assert_eq!(s.events.len(), 4);
        assert_eq!(s.events[0].tick, 0);
        assert_eq!(s.events[1].tick, 50);
        assert_eq!(s.events[2].tick, 100);
        assert_eq!(s.events[3].tick, 100);
        // Stable sort preserves track order at tick 100.
        assert_eq!(s.events[2].track, 0);
        assert_eq!(s.events[3].track, 1);
    }

    #[test]
    fn step_dispatches_events_within_window() {
        // Note on at tick 0, note off at tick 480 (= half a second at
        // 120 BPM). At 44100 Hz that's sample 22050.
        let events = &[
            0x00, 0x90, 0x3C, 0x64, // note on key 60 vel 100 at tick 0
            0xC0, 0x00, // wait 96 ticks (one VLQ byte 0x60), then...
            // Actually, simplest: write delta=480 as `83 60` (two-byte VLQ).
            // We'll use the standard pattern that read_vlq decodes:
            // 480 = 0x1E0 → high bits set on first byte: 0x83 0x60 isn't right.
            // 480 = 256+224 = 0x01E0; 7-bit chunks: 0x03 then 0x60 → 0x83, 0x60.
            // Wait: VLQ: split into 7-bit groups MSB-first, set bit 7
            // on every byte except the last.
            // 480 = 0b001_1110000 → groups (3, 96) → bytes [0x83, 0x60].
            0x80, 0x3C, 0x40, // note off (running status broken, must re-state)
            0x00, 0xFF, 0x2F, 0x00,
        ];
        // The above slice is malformed because we're typing it manually.
        // Use a builder instead.
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        ev.extend_from_slice(&[0x83, 0x60]); // VLQ 480
        ev.extend_from_slice(&[0x80, 0x3C, 0x40]); // note off
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]); // EOT
        let _ = events; // silence the unused-var lint from the planning slice above.

        let smf = parse(&smf_with_events(480, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();

        // Step a small chunk that includes only the note-on (sample 0).
        let done = s.step(1024, &mut mixer, &inst);
        assert!(!done);
        assert_eq!(mixer.live_voice_count(), 1);

        // Step forward to past sample 22050 (the note-off).
        // 22050 / 1024 ≈ 22 chunks.
        for _ in 0..30 {
            s.step(1024, &mut mixer, &inst);
        }
        // Note-off should have fired; voice should be either released
        // or done. ToneVoice doesn't go done() instantly on release —
        // it has a 100 ms tail. So we just check the scheduler is done.
        assert!(s.is_done());
    }

    #[test]
    fn channel_state_program_change_takes_effect() {
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xC0, 0x18]); // program change ch0 → 24
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(4096, &mut mixer, &inst);
        assert_eq!(mixer.channel_state(0).program, 24);
    }

    #[test]
    fn cc_volume_and_pan_propagate_to_mixer() {
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xB0, 7, 50]); // CC volume = 50
        ev.extend_from_slice(&[0x00, 0xB0, 10, 0]); // CC pan = 0 (hard left)
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(4096, &mut mixer, &inst);
        assert_eq!(mixer.channel_state(0).volume, 50);
        assert_eq!(mixer.channel_state(0).pan, 0);
    }

    #[test]
    fn pitch_bend_event_propagates_to_mixer() {
        let mut ev = Vec::new();
        // Note on, then pitch bend, then EOT.
        ev.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
                                                         // Pitch bend max-up: status E0, lsb 0x7F, msb 0x7F.
        ev.extend_from_slice(&[0x10, 0xE0, 0x7F, 0x7F]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.channel_state(0).pitch_bend, 0x3FFF);
    }

    #[test]
    fn channel_pressure_event_propagates() {
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        ev.extend_from_slice(&[0x10, 0xD0, 0x40]); // channel pressure
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.channel_state(0).channel_pressure, 0x40);
    }

    #[test]
    fn rpn_zero_data_entry_changes_pitch_bend_range() {
        let mut ev = Vec::new();
        // CC 101 = 0 (RPN MSB), CC 100 = 0 (RPN LSB) → select RPN 0.
        ev.extend_from_slice(&[0x00, 0xB0, 101, 0]);
        ev.extend_from_slice(&[0x00, 0xB0, 100, 0]);
        // CC 6 = 12 → ±12 semitones.
        ev.extend_from_slice(&[0x00, 0xB0, 6, 12]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(4096, &mut mixer, &inst);
        assert_eq!(mixer.channel_state(0).pitch_bend_range_cents, 1200);
    }

    #[test]
    fn poly_aftertouch_event_propagates() {
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // Poly aftertouch: status An, key, pressure.
        ev.extend_from_slice(&[0x10, 0xA0, 0x3C, 0x50]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        // Just run the dispatcher — assertion is "doesn't crash and
        // voice still alive". Detailed routing is covered in mixer tests.
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.live_voice_count(), 1);
    }

    #[test]
    fn tempo_change_recomputes_samples_per_tick() {
        // Tempo meta: FF 51 03 03 D0 90  ← 0x03D090 = 250_000 us/qn (= 240 BPM).
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(480, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        // Initial (default 500 000 us/qn): 45.9375.
        assert!((s.samples_per_tick - 45.9375).abs() < 1e-6);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(2048, &mut mixer, &inst);
        // After tempo change to 250 000 us/qn:
        //   250_000 * 44_100 / (1_000_000 * 480) = 22.96875.
        assert!(
            (s.samples_per_tick - 22.96875).abs() < 1e-6,
            "got {}",
            s.samples_per_tick,
        );
    }
}
