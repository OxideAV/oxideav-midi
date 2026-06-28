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
            // Sysex events: route the Universal Non-Real-Time / Real-Time
            // payloads we recognise (GM On/Off, Master Volume, Master
            // Fine/Coarse Tuning per CA-25). Everything else is silently
            // ignored — manufacturer-specific blobs carry no playback
            // semantics for round 75's renderer.
            Event::Sysex { escape, data } => {
                if !escape {
                    dispatch_universal_sysex(data, mixer);
                }
            }
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
                1 => mixer.set_mod_wheel(channel, value), // CC 1 — Modulation Wheel
                6 => mixer.set_data_entry(channel, value, true), // RPN data MSB
                7 => mixer.channel_state_mut(channel).volume = value,
                10 => mixer.channel_state_mut(channel).pan = value,
                11 => mixer.channel_state_mut(channel).expression = value, // CC 11 — Expression

                38 => mixer.set_data_entry(channel, value, false), // RPN data LSB
                64 => mixer.set_sustain(channel, value),
                74 => mixer.set_timbre(channel, value), // MPE "third dimension" (CC #74)
                91 => mixer.channel_state_mut(channel).reverb_send = value, // CC 91 — Reverb Send (CA-024)
                93 => mixer.channel_state_mut(channel).chorus_send = value, // CC 93 — Chorus Send (CA-024)
                96 => mixer.data_inc_dec(channel, 1), // Data Increment (RP-018; value ignored)
                97 => mixer.data_inc_dec(channel, -1), // Data Decrement (RP-018; value ignored)
                100 => mixer.set_rpn_byte(channel, value, false), // RPN LSB
                101 => mixer.set_rpn_byte(channel, value, true), // RPN MSB
                120 | 123 => {
                    // CC 120 = All Sound Off, CC 123 = All Notes Off.
                    mixer.all_notes_off();
                }
                121 => mixer.reset_all_controllers(channel), // CC 121 — Reset All Controllers (RP-015)
                _ => { /* other CCs not modelled */ }
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

/// Walk one decoded SMF [`Event::Sysex`] payload and dispatch any
/// **Universal** SysEx message we model. The `data` slice is exactly
/// what [`crate::smf`] stored — the bytes *after* the leading `F0`
/// (the leading byte itself isn't in the slice). The trailing `F7`
/// may or may not be present per spec; we accept both shapes.
///
/// Recognised payloads (per
/// `docs/audio/midi/midi-1.0/Universal-System-Exclusive-Messages.pdf`):
///
/// * **Non-Real-Time `7E`** — sub-ID #1:
///   * `09` General MIDI — sub-ID #2 `01` (GM1 On) / `02` (GM Off) /
///     `03` (GM2 On). All three currently map to
///     [`Mixer::all_notes_off`] + reset of master volume / fine /
///     coarse / mod-wheel / RPN state to defaults. GM2 On additionally
///     bumps the default mod-depth-range to 50 cents (already the
///     default).
/// * **Real-Time `7F`** — sub-ID #1 `04` (Device Control):
///   * `01` Master Volume (`F0 7F <dev> 04 01 lsb msb F7`). 14-bit
///     value applied via [`Mixer::set_master_volume_14`].
///   * `02` Master Balance (`F0 7F <dev> 04 02 lsb msb F7`).
///     14-bit value (`00 00` = hard left, `7F 7F` = hard right,
///     centre = `0x2000`) applied via
///     [`Mixer::set_master_balance_14`].
///   * `03` Master Fine Tuning (CA-25) — applied via
///     [`Mixer::set_master_fine_tuning`].
///   * `04` Master Coarse Tuning (CA-25) — applied via
///     [`Mixer::set_master_coarse_tuning`].
///   * `05` Global Parameter Control (CA-024) — the GM2 Reverb (slot
///     `0101`) / Chorus (slot `0102`) parameter edits, applied via
///     [`Mixer::set_gm_reverb_param`] / [`Mixer::set_gm_chorus_param`].
///
/// All other Universal SysEx messages (sample dumps, file refs, MTC
/// cueing, MMC, …) are silently ignored — they carry no semantics for
/// the round-3 renderer.
pub fn dispatch_universal_sysex(data: &[u8], mixer: &mut crate::mixer::Mixer) {
    if data.len() < 3 {
        return;
    }
    // Strip the optional trailing F7 so the payload offsets line up
    // either way.
    let payload = if *data.last().unwrap() == 0xF7 {
        &data[..data.len() - 1]
    } else {
        data
    };
    match payload[0] {
        // 7E = Universal Non-Real-Time
        0x7E => dispatch_universal_non_real_time(payload, mixer),
        // 7F = Universal Real-Time
        0x7F => dispatch_universal_real_time(payload, mixer),
        _ => { /* manufacturer-specific — not modelled */ }
    }
}

/// `F0 7E <dev> <sub1> <sub2> ... F7` — the data slice passed here is
/// `7E <dev> <sub1> <sub2> ...` (trailing F7 already stripped).
fn dispatch_universal_non_real_time(payload: &[u8], mixer: &mut crate::mixer::Mixer) {
    if payload.len() < 4 {
        return;
    }
    let sub_id1 = payload[2];
    let sub_id2 = payload[3];
    if sub_id1 == 0x09 && matches!(sub_id2, 0x01..=0x03) {
        // General MIDI System On (01) / Off (02) / GM2 On (03).
        // Reset playback state to GM defaults. Per the MIDI 1.0
        // spec + GM2 R/P, the receiver "resets all controllers
        // and turns off all notes" on GM-on.
        mixer.all_notes_off();
        mixer.set_master_volume_14(0x3FFF);
        mixer.set_master_balance_14(0x2000); // centre
        mixer.set_master_fine_tuning(0, 0x40); // centre
        mixer.set_master_coarse_tuning(0, 0x40); // centre
        mixer.reset_tuning(); // back to equal temperament
        mixer.reset_gm_effects(); // GM2 reverb/chorus defaults (CA-024)
    } else if sub_id1 == 0x08 {
        // MIDI Tuning Standard. The non-real-time area carries the
        // single-note tuning bank form (07) and the non-real-time
        // scale/octave forms (08 / 09). Per the spec the non-real-time
        // forms are "setup messages" that should not retune sounding
        // notes; we update the table but skip the live re-apply.
        dispatch_mts(payload, mixer, false);
    }
}

/// `F0 7F <dev> <sub1> <sub2> ... F7` — the data slice passed here is
/// `7F <dev> <sub1> <sub2> ...` (trailing F7 already stripped).
fn dispatch_universal_real_time(payload: &[u8], mixer: &mut crate::mixer::Mixer) {
    if payload.len() < 4 {
        return;
    }
    let sub_id1 = payload[2];
    if sub_id1 == 0x08 {
        // MIDI Tuning Standard, real-time forms (single-note 02 / 07,
        // scale/octave 08 / 09). These MUST update sounding notes.
        dispatch_mts(payload, mixer, true);
        return;
    }
    let sub_id2 = payload[3];
    if sub_id1 != 0x04 || payload.len() < 6 {
        return;
    }
    if sub_id2 == 0x05 {
        // Global Parameter Control (CA-024).
        dispatch_global_parameter_control(payload, mixer);
        return;
    }
    // Device Control.
    match sub_id2 {
        0x01 => {
            // Master Volume: `04 01 lsb msb`.
            let lsb = payload[4] & 0x7F;
            let msb = payload[5] & 0x7F;
            let combined = ((msb as u16) << 7) | (lsb as u16);
            mixer.set_master_volume_14(combined);
        }
        0x02 => {
            // Master Balance (M1 v4.2.1 §"DEVICE CONTROL — MASTER
            // VOLUME AND MASTER BALANCE", p.57): `04 02 lsb msb`.
            // `00 00 = hard left`, `7F 7F = hard right`, centre =
            // `0x2000`. Identical 14-bit layout to Master Volume.
            let lsb = payload[4] & 0x7F;
            let msb = payload[5] & 0x7F;
            let combined = ((msb as u16) << 7) | (lsb as u16);
            mixer.set_master_balance_14(combined);
        }
        0x03 => {
            // Master Fine Tuning (CA-25): `04 03 lsb msb`.
            mixer.set_master_fine_tuning(payload[4], payload[5]);
        }
        0x04 => {
            // Master Coarse Tuning (CA-25): `04 04 00 msb`.
            mixer.set_master_coarse_tuning(payload[4], payload[5]);
        }
        _ => {}
    }
}

/// Parse a **Global Parameter Control** Universal Real-Time SysEx body
/// (CA-024) and route the GM2 Reverb / Chorus parameter edits into the
/// mixer. `payload` is `7F <dev> 04 05 …` (trailing F7 already
/// stripped).
///
/// Wire format (CA-024 "Details"):
///
/// ```text
/// F0 7F <dev> 04 05 sw pw vw [[sh sl] ×sw] [pp×pw vv×vw] … F7
///   sw  Slot Path Length   — number of 2-byte slot-path entries.
///   pw  Parameter ID Width — bytes per <pp> field.
///   vw  Value Width        — bytes per <vv> field.
///   sh sl  Slot Number MSB / LSB (per slot-path entry).
///   pp  Parameter ID, MSB first  (pw bytes).
///   vv  Parameter Value, LSB first (vw bytes).
/// ```
///
/// GM2 reserves Slot Path Length = 1 with Slot MSB = 1 ("all messages
/// with Slot Path Length = 1 and Slot Path Number MSB = 1 shall be
/// reserved for use only by GM2"). The Slot LSB then selects the effect:
/// `01` = Reverb (slot `0101`), `02` = Chorus (slot `0102`). We model
/// only those two GM2 effect slots; any other slot path is ignored.
///
/// Multiple parameter-value pairs may follow; we apply each in turn,
/// silently skipping a pair whose parameter the slot doesn't recognise
/// (per CA-024: "only that parameter-value pair should be ignored").
fn dispatch_global_parameter_control(payload: &[u8], mixer: &mut crate::mixer::Mixer) {
    // payload: [0]=7F [1]=dev [2]=04 [3]=05 [4]=sw [5]=pw [6]=vw …
    if payload.len() < 7 {
        return;
    }
    let sw = payload[4] as usize; // slot-path entry count
    let pw = payload[5] as usize; // parameter-id byte width
    let vw = payload[6] as usize; // value byte width
                                  // A zero parameter or value width can't address anything; a GM2 edit
                                  // always carries at least one byte of each.
    if pw == 0 || vw == 0 {
        return;
    }
    let mut p = 7;
    // Read the slot path (sw entries × 2 bytes each).
    let slot_path_end = p + sw * 2;
    if slot_path_end > payload.len() {
        return;
    }
    // GM2 reverb/chorus use exactly one slot-path entry: MSB=1, the LSB
    // selects the effect. Anything else is outside the GM2 slots we
    // model, so we ignore the whole message.
    if sw != 1 {
        return;
    }
    let slot_msb = payload[p];
    let slot_lsb = payload[p + 1];
    p = slot_path_end;
    if slot_msb != 1 {
        return;
    }
    // Walk the parameter-value pair list to EOX.
    while p + pw + vw <= payload.len() {
        // Parameter ID, MSB first → fold the pw bytes into one integer.
        let mut pp: u32 = 0;
        for &b in &payload[p..p + pw] {
            pp = (pp << 7) | (b & 0x7F) as u32;
        }
        p += pw;
        // Value, LSB first → fold the vw bytes (LSB at lowest index).
        let mut vv: u32 = 0;
        for (i, &b) in payload[p..p + vw].iter().enumerate() {
            vv |= ((b & 0x7F) as u32) << (7 * i);
        }
        p += vw;
        // Only the low byte of the parameter id / value is meaningful for
        // the GM2 reverb / chorus parameters (all single-byte).
        let pp8 = (pp & 0x7F) as u8;
        let vv8 = (vv & 0x7F) as u8;
        match slot_lsb {
            0x01 => mixer.set_gm_reverb_param(pp8, vv8),
            0x02 => mixer.set_gm_chorus_param(pp8, vv8),
            _ => {}
        }
    }
}

/// Parse a MIDI Tuning Standard (MTS) Universal SysEx body and route it
/// into the mixer's microtuning state. `payload` is `7E/7F <dev> 08
/// <sub2> ...` (trailing F7 already stripped); `live` is true for the
/// real-time message forms (which must retune sounding notes) and false
/// for the non-real-time "setup" forms.
///
/// Recognised sub-ID#2 values (per
/// `docs/audio/midi/extensions/MIDI-Tuning-Updated-Specification.pdf`,
/// incorporating CA-020 / CA-021):
///
/// * `02` Single-Note Tuning Change: `08 02 tt ll [kk xx yy zz]…`.
/// * `07` Single-Note Tuning Change (bank): `08 07 bb tt ll
///   [kk xx yy zz]…` (the extra `bb` bank byte is parsed but not
///   used — we model one current tuning program).
/// * `08` Scale/Octave Tuning 1-byte: `08 08 ff gg hh [ss×12]`.
/// * `09` Scale/Octave Tuning 2-byte: `08 09 ff gg hh [ss tt ×12]`.
///
/// The `04`/`05`/`06` *dump* replies (key-based / scale-octave dumps an
/// instrument transmits) are not consumed by the renderer — they carry
/// a 16-byte tuning name + a checksum and are inbound-to-an-instrument
/// in intent, so we ignore them here.
fn dispatch_mts(payload: &[u8], mixer: &mut crate::mixer::Mixer, live: bool) {
    // payload[0] = 7E/7F, payload[1] = device id, payload[2] = 08.
    if payload.len() < 4 {
        return;
    }
    let sub2 = payload[3];
    match sub2 {
        // Single-Note Tuning Change: `08 02 tt ll [kk xx yy zz]…`.
        0x02 => {
            // tt @4, ll @5, then ll × 4 bytes.
            if payload.len() < 6 {
                return;
            }
            let count = payload[5] as usize;
            let mut p = 6;
            for _ in 0..count {
                if p + 4 > payload.len() {
                    break;
                }
                let key = payload[p] & 0x7F;
                let word = [payload[p + 1], payload[p + 2], payload[p + 3]];
                mixer.set_key_tuning_word(key, word, live);
                p += 4;
            }
        }
        // Single-Note Tuning Change (bank): `08 07 bb tt ll [kk…]…`.
        0x07 => {
            // bb @4, tt @5, ll @6, then ll × 4 bytes.
            if payload.len() < 7 {
                return;
            }
            let count = payload[6] as usize;
            let mut p = 7;
            for _ in 0..count {
                if p + 4 > payload.len() {
                    break;
                }
                let key = payload[p] & 0x7F;
                let word = [payload[p + 1], payload[p + 2], payload[p + 3]];
                mixer.set_key_tuning_word(key, word, live);
                p += 4;
            }
        }
        // Scale/Octave Tuning, 1-byte form: `08 08 ff gg hh [ss×12]`.
        0x08 => {
            if payload.len() < 7 + 12 {
                return;
            }
            let mask = crate::tuning::scale_octave_channel_mask(payload[4], payload[5], payload[6]);
            let mut offsets = [0.0f32; 12];
            for (pc, off) in offsets.iter_mut().enumerate() {
                *off = crate::tuning::scale_octave_1byte_to_cents(payload[7 + pc]);
            }
            apply_scale_octave_to_mask(mixer, mask, offsets, live);
        }
        // Scale/Octave Tuning, 2-byte form: `08 09 ff gg hh [ss tt ×12]`.
        0x09 => {
            if payload.len() < 7 + 24 {
                return;
            }
            let mask = crate::tuning::scale_octave_channel_mask(payload[4], payload[5], payload[6]);
            let mut offsets = [0.0f32; 12];
            for (pc, off) in offsets.iter_mut().enumerate() {
                let msb = payload[7 + pc * 2];
                let lsb = payload[7 + pc * 2 + 1];
                *off = crate::tuning::scale_octave_2byte_to_cents(msb, lsb);
            }
            apply_scale_octave_to_mask(mixer, mask, offsets, live);
        }
        // Dump replies (04/05/06) + dump requests (00/01/03) carry no
        // semantics for the renderer.
        _ => {}
    }
}

/// Apply a 12-pitch-class scale/octave offset table to every MIDI
/// channel selected in `mask` (bit `c` set ⇒ channel index `c`).
fn apply_scale_octave_to_mask(
    mixer: &mut crate::mixer::Mixer,
    mask: u16,
    offsets: [f32; 12],
    live: bool,
) {
    for ch in 0..16u8 {
        if mask & (1 << ch) != 0 {
            mixer.set_scale_octave_tuning(ch, offsets, live);
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
    fn data_increment_decrement_route_to_mixer() {
        let mut ev = Vec::new();
        // Select RPN 0 (Pitch Bend Sensitivity).
        ev.extend_from_slice(&[0x00, 0xB0, 101, 0]);
        ev.extend_from_slice(&[0x00, 0xB0, 100, 0]);
        // RP-018: two Data Increment (CC 96) messages = +2 cents on the
        // default 200-cent range. The data byte is ignored, so we pass a
        // deliberately nonsense value (0x7F) to prove it doesn't matter.
        ev.extend_from_slice(&[0x00, 0xB0, 96, 0x7F]);
        ev.extend_from_slice(&[0x00, 0xB0, 96, 0x00]);
        // One Data Decrement (CC 97) → back to +1 cent (= 201).
        ev.extend_from_slice(&[0x00, 0xB0, 97, 0x55]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(4096, &mut mixer, &inst);
        assert_eq!(mixer.channel_state(0).pitch_bend_range_cents, 201);
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
    fn cc1_mod_wheel_propagates_via_scheduler() {
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xB0, 1, 100]); // CC 1 = 100
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(4096, &mut mixer, &inst);
        assert_eq!(mixer.channel_state(0).mod_wheel, 100);
    }

    #[test]
    fn cc74_timbre_does_not_panic_via_scheduler() {
        // CC 74 has no per-channel field on ChannelState (it's a
        // pure routed message); just check the dispatcher routes
        // it without panicking.
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        ev.extend_from_slice(&[0x10, 0xB0, 74, 96]); // CC 74 = 96
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.live_voice_count(), 1);
    }

    #[test]
    fn universal_master_volume_sysex_routes_to_mixer() {
        // F0 [len=7] 7F 7F 04 01 lsb=0x00 msb=0x40 F7  → 14-bit = 0x2000 (~half).
        // The SMF Sysex payload includes the trailing F7 + everything
        // *after* the F0, so its length is 7 bytes.
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x00, 0x40, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.master_volume_14(), 0x2000);
    }

    #[test]
    fn universal_master_balance_sysex_routes_centre() {
        // F0 [len=7] 7F 7F 04 02 lsb=0x00 msb=0x40 F7  → 14-bit = 0x2000 (centre).
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x02, 0x00, 0x40, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.master_balance_14(), 0x2000);
    }

    #[test]
    fn universal_master_balance_sysex_routes_hard_left() {
        // F0 [len=7] 7F 7F 04 02 00 00 F7  → hard left per M1 v4.2.1 §57.
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x02, 0x00, 0x00, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.master_balance_14(), 0x0000);
        let (l, r) = mixer.master_balance_gains();
        assert_eq!(l, 1.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn universal_master_balance_sysex_routes_hard_right() {
        // F0 [len=7] 7F 7F 04 02 7F 7F F7  → hard right per M1 v4.2.1 §57.
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x02, 0x7F, 0x7F, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.master_balance_14(), 0x3FFF);
        let (l, r) = mixer.master_balance_gains();
        assert_eq!(l, 0.0);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn universal_master_fine_tuning_sysex_routes() {
        // F0 [len=7] 7F 7F 04 03 lsb=0x00 msb=0x60 F7  → +50 cents.
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x03, 0x00, 0x60, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        let c = mixer.master_fine_tune_cents();
        assert!((49..=50).contains(&c), "got {c}");
    }

    #[test]
    fn universal_master_coarse_tuning_sysex_routes() {
        // F0 [len=7] 7F 7F 04 04 00 0x4C F7  → +12 semis.
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x04, 0x00, 0x4C, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.master_coarse_tune_semitones(), 12);
    }

    /// Build the SMF-escaped SysEx bytes for a one-track blob: a
    /// leading `F0`, a VLQ length, the body, and a trailing `F7`. The
    /// body does NOT include the leading F0 (the SMF stores the bytes
    /// after it).
    fn sysex_event(delta: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![delta, 0xF0, (body.len() + 1) as u8];
        v.extend_from_slice(body);
        v.push(0xF7);
        v
    }

    #[test]
    fn gm2_reverb_chorus_defaults() {
        // No SysEx: the mixer should boot to the CA-024 GM2 recommended
        // initial settings (Reverb Type 4, Chorus Type 2).
        let mixer = Mixer::new();
        let fx = mixer.gm_effects();
        assert_eq!(fx.reverb_type, 4);
        assert_eq!(fx.chorus_type, 2);
        // Reverb Type-4 default time = val 64 → exp((64-40)*0.025) ≈ 1.822 s.
        assert!((fx.reverb_time_s - ((24.0f32) * 0.025).exp()).abs() < 1e-4);
    }

    #[test]
    fn global_parameter_control_reverb_type_and_time() {
        // CA-024 Slot 0101 (Reverb), one entry slot path (sw=1), pw=1,
        // vw=1. Set Reverb Type (pp=0) = 2 and Reverb Time (pp=1) val=80.
        //   F0 7F 7F 04 05 01 01 01 01 01 00 02 01 50 F7
        //          dev  04 05 sw pw vw sh sl pp vv pp vv
        let body = [
            0x7F, 0x7F, 0x04, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x02, 0x01, 0x50,
        ];
        let mut ev = sysex_event(0x00, &body);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        let fx = mixer.gm_effects();
        assert_eq!(fx.reverb_type, 2);
        // val 0x50 = 80 → rt = exp((80-40)*0.025) = exp(1.0).
        assert!(
            (fx.reverb_time_s - 1.0f32.exp()).abs() < 1e-4,
            "{}",
            fx.reverb_time_s
        );
    }

    #[test]
    fn global_parameter_control_chorus_params() {
        // Slot 0102 (Chorus): set every documented parameter at once.
        //   pp=0 type=3, pp=1 rate=10, pp=2 depth=15, pp=3 fb=64, pp=4 send=50.
        let body = [
            0x7F, 0x7F, 0x04, 0x05, 0x01, 0x01, 0x01, // sw pw vw
            0x01, 0x02, // slot 0102 = Chorus
            0x00, 0x03, // type = 3
            0x01, 0x0A, // mod rate val=10
            0x02, 0x0F, // mod depth val=15
            0x03, 0x40, // feedback val=64
            0x04, 0x32, // send-to-reverb val=50
        ];
        let mut ev = sysex_event(0x00, &body);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        let fx = mixer.gm_effects();
        assert_eq!(fx.chorus_type, 3);
        assert!((fx.chorus_mod_rate_hz - 10.0 * 0.122).abs() < 1e-5);
        assert!((fx.chorus_mod_depth_ms - (15.0 + 1.0) / 3.2).abs() < 1e-5);
        assert!((fx.chorus_feedback_pct - 64.0 * 0.763).abs() < 1e-4);
        assert!((fx.chorus_send_to_reverb_pct - 50.0 * 0.787).abs() < 1e-4);
    }

    #[test]
    fn global_parameter_control_non_gm2_slot_ignored() {
        // Slot MSB != 1 is NOT a GM2 slot → the whole message must be
        // ignored, leaving the reverb type at its default (4).
        let body = [
            0x7F, 0x7F, 0x04, 0x05, 0x01, 0x01, 0x01, 0x02, 0x01, // slot 0201
            0x00, 0x00, // would set reverb type 0 if it weren't ignored
        ];
        let mut ev = sysex_event(0x00, &body);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.gm_effects().reverb_type, 4);
    }

    #[test]
    fn global_parameter_control_unknown_param_ignored() {
        // A recognised Reverb slot but an out-of-range parameter id
        // (pp=9): per CA-024 only that pair is ignored, the rest apply.
        // Here we send pp=9 (ignored) then pp=0 type=1 (applies).
        let body = [
            0x7F, 0x7F, 0x04, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, // slot 0101
            0x09, 0x7F, // unknown param → ignored
            0x00, 0x01, // type = 1 → applies
        ];
        let mut ev = sysex_event(0x00, &body);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.gm_effects().reverb_type, 1);
    }

    #[test]
    fn gm_on_resets_gm2_effects() {
        // Push the reverb type off-default, then GM1 System On → the
        // GM2 effect parameters reset to their CA-024 defaults.
        let mut ev = Vec::new();
        let gpc = [
            0x7F, 0x7F, 0x04, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
        ];
        ev.extend_from_slice(&sysex_event(0x00, &gpc));
        // GM1 System On.
        ev.extend_from_slice(&[0x10, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        assert_eq!(mixer.gm_effects().reverb_type, 4);
        assert_eq!(mixer.gm_effects().chorus_type, 2);
    }

    #[test]
    fn universal_gm_on_sysex_resets_state() {
        // First push master state away from defaults, then send GM
        // System On — state should be reset.
        let mut ev = Vec::new();
        // Master Volume = 0x1000 (len=7).
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x00, 0x20, 0xF7]);
        // Master Balance = hard left (len=7) — to prove the GM-on
        // reset also returns balance to centre.
        ev.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x02, 0x00, 0x00, 0xF7]);
        // GM1 System On: F0 [len=5] 7E 7F 09 01 F7.
        ev.extend_from_slice(&[0x10, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        // Master volume back to full per GM-on reset.
        assert_eq!(mixer.master_volume_14(), 0x3FFF);
        assert_eq!(mixer.master_balance_14(), 0x2000);
        assert_eq!(mixer.master_fine_tune_cents(), 0);
        assert_eq!(mixer.master_coarse_tune_semitones(), 0);
    }

    #[test]
    fn mcm_via_smf_data_entry_assigns_mpe_zone() {
        // Lower MCM: B0 65 00 B0 64 06 B0 06 04  (n=0 mm=4).
        let mut ev = Vec::new();
        ev.extend_from_slice(&[0x00, 0xB0, 0x65, 0x00]);
        ev.extend_from_slice(&[0x00, 0xB0, 0x64, 0x06]);
        ev.extend_from_slice(&[0x00, 0xB0, 0x06, 0x04]);
        ev.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let smf = parse(&smf_with_events(96, &ev)).unwrap();
        let mut s = Scheduler::new(&smf, 44_100);
        let mut mixer = Mixer::new();
        let inst = ToneInstrument::new();
        s.step(8192, &mut mixer, &inst);
        let zone = mixer
            .mpe_zone(crate::mixer::MpeZoneKind::Lower)
            .expect("MCM must have created the lower zone");
        assert_eq!(zone.members, 4);
        assert!(mixer.channel_state(0).mpe_role.is_manager());
    }

    // ───────────────────────── MTS via SysEx ─────────────────────────

    #[test]
    fn mts_single_note_real_time_routes_to_mixer() {
        // F0 7F <dev=7F> 08 02 tt=00 ll=01 kk=60 xx=3D yy=00 zz=00 F7.
        // `3D 00 00` addressed to key 60 = +100 cents. The slice passed
        // to dispatch_universal_sysex omits the leading F0.
        let data = [
            0x7F, 0x7F, 0x08, 0x02, 0x00, 0x01, 0x3C, 0x3D, 0x00, 0x00, 0xF7,
        ];
        // (kk = 0x3C = key 60.)
        let mut mixer = Mixer::new();
        dispatch_universal_sysex(&data, &mut mixer);
        let off = mixer.tuning().offset_cents(0, 60);
        assert!((off - 100.0).abs() < 1e-3, "off {off}");
    }

    #[test]
    fn mts_single_note_bank_form_routes() {
        // F0 7F <dev> 08 07 bb=00 tt=00 ll=01 kk=64 word=41 00 00 F7.
        // `41 00 00` on key 64 = +1 semitone = +100 cents.
        let data = [
            0x7F, 0x7F, 0x08, 0x07, 0x00, 0x00, 0x01, 0x40, 0x41, 0x00, 0x00, 0xF7,
        ];
        let mut mixer = Mixer::new();
        dispatch_universal_sysex(&data, &mut mixer);
        let off = mixer.tuning().offset_cents(0, 64);
        assert!((off - 100.0).abs() < 1e-3, "off {off}");
    }

    #[test]
    fn mts_scale_octave_1byte_routes_to_selected_channel() {
        // F0 7F <dev> 08 08 ff gg hh [12 × ss] F7.
        // hh bit0 ⇒ channel 1 (index 0). 12 offsets: C = 7F (+63 c),
        // rest = 40 (0 c).
        let mut data = vec![0x7F, 0x7F, 0x08, 0x08, 0x00, 0x00, 0x01];
        data.push(0x7F); // C = +63 cents
        data.resize(data.len() + 11, 0x40); // C#..B = 0 cents
        data.push(0xF7);
        let mut mixer = Mixer::new();
        dispatch_universal_sysex(&data, &mut mixer);
        assert!((mixer.tuning().offset_cents(0, 60) - 63.0).abs() < 1e-3);
        // C# (key 61) on channel 0 is the 0-cent class.
        assert!(mixer.tuning().offset_cents(0, 61).abs() < 1e-3);
        // Channel 1 was not selected by the mask.
        assert!(mixer.tuning().offset_cents(1, 60).abs() < 1e-3);
    }

    #[test]
    fn mts_scale_octave_2byte_routes() {
        // F0 7F <dev> 08 09 ff gg hh [12 × ss tt] F7. Select channel 1
        // (hh bit0). C = 0x7F 0x7F (≈ +100 c), rest = 0x40 0x00 (0 c).
        let mut data = vec![0x7F, 0x7F, 0x08, 0x09, 0x00, 0x00, 0x01];
        data.extend_from_slice(&[0x7F, 0x7F]); // C ≈ +100 cents
        for _ in 0..11 {
            data.extend_from_slice(&[0x40, 0x00]); // 0 cents
        }
        data.push(0xF7);
        let mut mixer = Mixer::new();
        dispatch_universal_sysex(&data, &mut mixer);
        let c = mixer.tuning().offset_cents(0, 60);
        assert!((c - 99.988).abs() < 0.05, "C offset {c}");
    }

    #[test]
    fn mts_non_real_time_single_note_does_not_retune_sounding_note() {
        // The non-real-time (7E) single-note bank form is a setup
        // message: it updates the table but must not disturb a held
        // note. We can only observe the table here (no live voice),
        // so assert the offset landed in the table.
        let data = [
            0x7E, 0x7F, 0x08, 0x07, 0x00, 0x00, 0x01, 0x40, 0x41, 0x00, 0x00, 0xF7,
        ];
        let mut mixer = Mixer::new();
        dispatch_universal_sysex(&data, &mut mixer);
        let off = mixer.tuning().offset_cents(0, 64);
        assert!((off - 100.0).abs() < 1e-3, "off {off}");
    }

    #[test]
    fn mts_multi_change_message_sets_several_keys() {
        // ll = 2: two [kk xx yy zz] entries in one message.
        let data = [
            0x7F, 0x7F, 0x08, 0x02, 0x00, 0x02, // header + count
            0x3C, 0x3D, 0x00, 0x00, // key 60 → +100 c
            0x40, 0x40, 0x40, 0x00, // key 64 → +50 c
            0xF7,
        ];
        let mut mixer = Mixer::new();
        dispatch_universal_sysex(&data, &mut mixer);
        assert!((mixer.tuning().offset_cents(0, 60) - 100.0).abs() < 1e-3);
        assert!((mixer.tuning().offset_cents(0, 64) - 50.0).abs() < 1e-3);
    }

    #[test]
    fn gm_system_on_resets_mts_tuning() {
        let mut mixer = Mixer::new();
        // First retune key 60.
        dispatch_universal_sysex(
            &[
                0x7F, 0x7F, 0x08, 0x02, 0x00, 0x01, 0x3C, 0x3D, 0x00, 0x00, 0xF7,
            ],
            &mut mixer,
        );
        assert!(mixer.tuning().offset_cents(0, 60) > 50.0);
        // GM1 System On: F0 7E 7F 09 01 F7.
        dispatch_universal_sysex(&[0x7E, 0x7F, 0x09, 0x01, 0xF7], &mut mixer);
        assert!(mixer.tuning().offset_cents(0, 60).abs() < 1e-3);
    }

    #[test]
    fn mts_truncated_message_does_not_panic() {
        // A single-note message that promises ll=4 entries but only
        // carries one must not read past the buffer.
        let data = [0x7F, 0x7F, 0x08, 0x02, 0x00, 0x04, 0x3C, 0x3D, 0x00, 0x00];
        let mut mixer = Mixer::new();
        dispatch_universal_sysex(&data, &mut mixer);
        assert!((mixer.tuning().offset_cents(0, 60) - 100.0).abs() < 1e-3);
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
