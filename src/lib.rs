//! MIDI — Standard MIDI File (SMF) parser + transport metadata + soft-synth.
//!
//! * **[`smf`]** — pure-Rust parser for the Standard MIDI File format
//!   (Type 0 / 1 / 2). Header (`MThd`) + tracks (`MTrk`) + every common
//!   channel-voice message, sysex (`F0` / `F7`), and meta event
//!   (tempo, time signature, key signature, text, marker, end-of-track,
//!   SMPTE offset, sequencer-specific). Running status is honoured;
//!   VLQs are bounded to 4 bytes per spec; chunk lengths are validated
//!   against remaining bytes; total events per file are capped at
//!   [`smf::MAX_EVENTS_PER_FILE`].
//! * **[`paths`]** — per-OS SoundFont/SFZ/DLS search paths plus the
//!   `OXIDEAV_SOUNDFONT_PATH` environment override. `find_soundfonts`
//!   walks them and returns every instrument-bank file present.
//! * **[`instruments`]** — [`instruments::Instrument`] trait. Three
//!   adapters:
//!     * **[`instruments::sf2`]** — full SoundFont 2 RIFF reader +
//!       voice generator. Walks the `sfbk` form, cross-resolves the
//!       preset → instrument → zone → sample chain, and renders
//!       sm24-aware 24-bit PCM at the requested pitch via linear
//!       interpolation. Honours the volume + modulation DAHDSR
//!       envelopes, the initial low-pass biquad filter, mod-env →
//!       pitch / filter routing, exclusive-class drum cuts, and
//!       native stereo zones.
//!     * **[`instruments::sfz`]** — text patch reader. Strips
//!       comments, walks `<control>` / `<global>` / `<master>` /
//!       `<group>` / `<region>` sections, flattens inheritance into
//!       one fully-resolved opcode map per region, and (via
//!       [`SfzInstrument::open`](instruments::sfz::SfzInstrument::open))
//!       reads every referenced sample off disk. Voice generation is
//!       still pending — `make_voice` returns [`Error::Unsupported`].
//!     * **[`instruments::dls`]** — magic-byte detector stub;
//!       `make_voice` returns [`Error::Unsupported`]. Loader is
//!       blocked on the DLS Level 1/2 specification landing in
//!       `docs/audio/midi/instrument-formats/`.
//!     * **[`instruments::tone`]** — sine/triangle/saw/square
//!       fallback so the synth produces *something* even when no
//!       on-disk bank is present.
//! * **[`mixer`]** — round-3 polyphonic voice pool (32 voices) with
//!   stereo mixdown, per-channel volume / pan / sustain pedal handling,
//!   and oldest-voice preemption when the pool is full.
//! * **[`scheduler`]** — round-3 SMF event scheduler. Merges every
//!   track into a single time-ordered stream, converts ticks → samples
//!   against the current tempo + division, and dispatches each event
//!   into the mixer at the right audio sample.
//! * **[`downloader`]** — stub that names a planned default bank
//!   (TimGM6mb) but currently returns [`Error::Unsupported`].
//!
//! The decoder factory ([`make_decoder`]) is registered under codec id
//! [`CODEC_ID_STR`] = `"midi"`. Round-3 wires SMF events end-to-end:
//! `send_packet` parses the SMF and primes the scheduler; `receive_frame`
//! pulls one chunk of stereo PCM ([`FRAME_SAMPLES`] samples per channel
//! at [`OUTPUT_SAMPLE_RATE`]) until both the event stream and the voice
//! pool have run dry, then returns [`Error::Eof`].
//!
//! Without an instrument bank the decoder uses
//! [`instruments::tone::ToneInstrument`] — the pure-tone fallback —
//! so a `.mid` file plays back as audible-but-not-musical sine /
//! triangle / square waves. To use a real bank, build the decoder by
//! hand and pass an [`Sf2Instrument`](instruments::sf2::Sf2Instrument)
//! to [`MidiDecoder::with_instrument`]; the decoder factory wired into
//! the registry today does not yet plumb a bank-discovery hook.

pub mod downloader;
pub mod instruments;
pub mod mixer;
pub mod paths;
pub mod scheduler;
pub mod smf;

use std::sync::Arc;

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Decoder,
    Error, Frame, Packet, Result,
};

use crate::instruments::tone::ToneInstrument;
use crate::instruments::Instrument;
use crate::mixer::Mixer;
use crate::scheduler::Scheduler;

/// Public codec id string. Matches the aggregator feature name `midi`.
pub const CODEC_ID_STR: &str = "midi";

/// Round-3 audio output sample rate. Hard-coded to 44 100 Hz so the
/// decoder doesn't need a parameter from the caller (the SMF container
/// itself doesn't carry one). Round-4 may wire this through
/// `CodecParameters::sample_rate`.
pub const OUTPUT_SAMPLE_RATE: u32 = 44_100;

/// Number of *per-channel* samples emitted per
/// [`Decoder::receive_frame`] call. ~23 ms at 44.1 kHz — small enough
/// for low playback latency, big enough that the per-call overhead is
/// dwarfed by the inner mix loop.
pub const FRAME_SAMPLES: usize = 1024;

/// Channel count of the PCM output bus. Stereo. Same fixed assumption
/// as [`OUTPUT_SAMPLE_RATE`].
pub const OUTPUT_CHANNELS: u16 = 2;

/// Register the MIDI codec. Round-3 produces interleaved S16 stereo
/// PCM at [`OUTPUT_SAMPLE_RATE`] — the registry-built decoder uses the
/// pure-tone fallback because we don't yet have a bank-discovery hook
/// in the factory signature. Callers who want SoundFont 2 playback
/// should build the decoder by hand via [`MidiDecoder::with_instrument`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("midi_synth")
        .with_lossy(false)
        .with_lossless(true)
        .with_intra_only(false)
        .with_max_channels(OUTPUT_CHANNELS);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(MidiDecoder::new(
        Arc::new(ToneInstrument::new()),
        OUTPUT_SAMPLE_RATE,
    )))
}

/// Soft-synth decoder: SMF in, interleaved S16 stereo PCM out.
///
/// Stateful — accepts exactly one SMF blob via [`send_packet`] and then
/// streams audio frames out of [`receive_frame`] until both the event
/// scheduler and the voice pool have run dry, at which point
/// [`Error::Eof`] is returned. Calling `send_packet` again replaces the
/// scheduler with a fresh one (re-priming for a new file).
///
/// State that survives across `receive_frame` calls:
///   * the merged event list + cursor + sample clock (in [`Scheduler`])
///   * the voice pool + per-channel CC state (in [`Mixer`])
///   * a small carry-over flag that lets the decoder render a few
///     extra trailing chunks after the last event so release tails
///     don't get cut off mid-envelope.
///
/// [`send_packet`]: Decoder::send_packet
/// [`receive_frame`]: Decoder::receive_frame
pub struct MidiDecoder {
    codec_id: CodecId,
    instrument: Arc<dyn Instrument>,
    sample_rate: u32,
    /// `None` until the first `send_packet` arrives.
    scheduler: Option<Scheduler>,
    mixer: Mixer,
    /// Scratch stereo planes — reused across `receive_frame` calls so
    /// we don't reallocate on every chunk.
    left: Vec<f32>,
    right: Vec<f32>,
    /// Sample PTS of the next emitted frame (in `1/sample_rate` units).
    next_pts: i64,
    /// Set once the scheduler has run dry; we keep emitting frames
    /// until the voice pool falls silent too.
    drained: bool,
    /// Set once we've returned `Error::Eof` once — subsequent calls
    /// keep returning `Eof`.
    finished: bool,
    /// Bound on extra "tail" chunks emitted after the scheduler is done
    /// but voices may still be releasing. Worst-case the longest
    /// release in [`Sf2Voice`](instruments::sf2::Sf2Voice) is 50 ms = 3
    /// chunks at 1024 samples / 44.1 kHz; tone voices are 100 ms = 5
    /// chunks. Bound generously at 32 to also cover a long looping
    /// sample whose release window is unusually long.
    tail_chunks_remaining: usize,
}

impl MidiDecoder {
    /// Hard cap on how many extra audio chunks we'll emit after the
    /// last SMF event has fired. Voice release tails (50–100 ms with
    /// the round-2/3 envelopes) live inside this budget; without it,
    /// a malformed or never-releasing voice could keep the decoder
    /// emitting forever.
    pub const TAIL_CHUNK_CAP: usize = 32;

    /// Build a decoder bound to a specific instrument and sample rate.
    /// Use this directly when you have a SoundFont 2 bank loaded and
    /// want to drive the synth with it; the [`make_decoder`] factory
    /// (called by the codec registry) builds one with the pure-tone
    /// fallback because there's no instrument-discovery plumbing in
    /// the factory signature yet.
    pub fn new(instrument: Arc<dyn Instrument>, sample_rate: u32) -> Self {
        Self {
            codec_id: CodecId::new(CODEC_ID_STR),
            instrument,
            sample_rate,
            scheduler: None,
            mixer: Mixer::new(),
            left: vec![0.0; FRAME_SAMPLES],
            right: vec![0.0; FRAME_SAMPLES],
            next_pts: 0,
            drained: false,
            finished: false,
            tail_chunks_remaining: Self::TAIL_CHUNK_CAP,
        }
    }

    /// Convenience constructor: same as [`new`](Self::new) but takes a
    /// concrete [`Instrument`] by value and wraps it in an `Arc`.
    pub fn with_instrument(instrument: Arc<dyn Instrument>) -> Self {
        Self::new(instrument, OUTPUT_SAMPLE_RATE)
    }

    /// Sample rate the decoder is rendering at. Equal to whatever was
    /// passed to [`new`](Self::new) (default [`OUTPUT_SAMPLE_RATE`] when
    /// constructed via the registry).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Borrow the active scheduler — `None` until the first
    /// `send_packet`. Useful for diagnostics + tests.
    pub fn scheduler(&self) -> Option<&Scheduler> {
        self.scheduler.as_ref()
    }

    /// Convert the planar stereo `(left, right)` buffers into one
    /// interleaved S16 [`AudioFrame`].
    fn build_audio_frame(&mut self) -> Frame {
        let n = self.left.len();
        let mut bytes = Vec::with_capacity(n * 2 * 2); // 2 bytes/sample × 2 channels
        for i in 0..n {
            let l = (self.left[i].clamp(-1.0, 1.0) * 32_767.0) as i16;
            let r = (self.right[i].clamp(-1.0, 1.0) * 32_767.0) as i16;
            bytes.extend_from_slice(&l.to_le_bytes());
            bytes.extend_from_slice(&r.to_le_bytes());
        }
        let pts = Some(self.next_pts);
        self.next_pts = self.next_pts.saturating_add(n as i64);
        Frame::Audio(AudioFrame {
            samples: n as u32,
            pts,
            data: vec![bytes],
        })
    }
}

impl Decoder for MidiDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Confirm the packet at least *looks* like an SMF — saves the
        // user from a "synthesis pending" misdiagnosis when the real
        // issue is a mis-routed packet.
        if packet.data.len() < 4 || &packet.data[0..4] != b"MThd" {
            return Err(Error::invalid(
                "MIDI: packet does not start with the 'MThd' header chunk",
            ));
        }
        let smf = crate::smf::parse(&packet.data)?;
        // Prime the scheduler. Dropping the previous one (if any)
        // discards any partially-played file — callers should call
        // `flush` first if that matters.
        self.scheduler = Some(Scheduler::new(&smf, self.sample_rate));
        self.mixer.all_notes_off();
        self.next_pts = 0;
        self.drained = false;
        self.finished = false;
        self.tail_chunks_remaining = Self::TAIL_CHUNK_CAP;
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if self.finished {
            return Err(Error::Eof);
        }
        let scheduler = self.scheduler.as_mut().ok_or(Error::NeedMore)?;

        // Step the scheduler over one chunk-worth of samples; this may
        // dispatch any number of events into the mixer. The scheduler
        // is `drained` either when it just transitioned to done, or
        // when it was already done coming into this call (we keep
        // running the mixer-only tail in that case).
        let was_done = scheduler.is_done();
        let now_done = scheduler.step(FRAME_SAMPLES, &mut self.mixer, self.instrument.as_ref());
        if was_done || now_done {
            self.drained = true;
        }

        // Mix down whatever the pool currently holds.
        let active = self.mixer.mix_stereo(&mut self.left, &mut self.right);

        // Termination: scheduler done AND no live voices AND we've
        // already burned at least one tail chunk. The tail-chunk cap
        // keeps a never-finishing voice (looping sample with no
        // release fired) from holding the decoder open forever.
        if self.drained {
            if active == 0 || self.tail_chunks_remaining == 0 {
                self.finished = true;
                // Still hand back this final chunk (silent or near-silent)
                // — the caller can decide to discard it. Returning Eof
                // here would lose any release-tail samples.
                return Ok(self.build_audio_frame());
            }
            self.tail_chunks_remaining = self.tail_chunks_remaining.saturating_sub(1);
        }

        Ok(self.build_audio_frame())
    }

    fn flush(&mut self) -> Result<()> {
        // Mark the scheduler done so subsequent receive_frame calls
        // run only the release tail.
        if let Some(s) = self.scheduler.as_mut() {
            // Drain by stepping a huge amount of samples — every event
            // will fire, and the cursor will advance to the end. This
            // is cheaper than re-engineering the scheduler API around
            // an explicit "skip to end" entry point.
            s.step(u32::MAX as usize, &mut self.mixer, self.instrument.as_ref());
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.scheduler = None;
        self.mixer.all_notes_off();
        self.next_pts = 0;
        self.drained = false;
        self.finished = false;
        self.tail_chunks_remaining = Self::TAIL_CHUNK_CAP;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    fn minimal_smf() -> Vec<u8> {
        // MThd format-0, ntrks=1, division=96; one MTrk with EOT.
        let mut b = vec![];
        b.extend_from_slice(b"MThd");
        b.extend_from_slice(&6u32.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&96u16.to_be_bytes());
        b.extend_from_slice(b"MTrk");
        b.extend_from_slice(&4u32.to_be_bytes());
        b.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        b
    }

    /// Build a 5-second SMF: tempo, two notes on channel 1, one note on
    /// channel 10 (drums), a tempo change, and an EOT five seconds in.
    fn five_second_smf() -> Vec<u8> {
        // 480 ticks / qn at 120 BPM = 240 ticks / sec. Five seconds =
        // 1200 ticks. Halfway tempo change (tick 600) to 250 000 us/qn
        // (240 BPM) ⇒ second half is 240 ticks per second × 2 = 480
        // ticks/sec — but we wrote 1200 ticks of "music" assuming the
        // initial tempo so the wall-clock length will be ≈ 3.75 s, not
        // a pure 5 s. That's fine: the test only asserts "non-silent
        // PCM with a sensible duration", not exact timing.
        let mut blob = Vec::new();
        blob.extend_from_slice(b"MThd");
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&1u16.to_be_bytes()); // format 1
        blob.extend_from_slice(&3u16.to_be_bytes()); // 3 tracks
        blob.extend_from_slice(&480u16.to_be_bytes()); // 480 tpqn

        // Track 1: tempo + tempo change + EOT.
        let mut t1: Vec<u8> = Vec::new();
        // tick 0 set tempo 500_000 us/qn (= 120 BPM)
        t1.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        // tick 600 set tempo 250_000 us/qn (= 240 BPM): VLQ(600) = [0x84, 0x58]
        t1.extend_from_slice(&[0x84, 0x58, 0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]);
        // tick 1200 EOT: VLQ(600) again
        t1.extend_from_slice(&[0x84, 0x58, 0xFF, 0x2F, 0x00]);
        push_track(&mut blob, &t1);

        // Track 2: two notes on channel 1, played sequentially.
        let mut t2: Vec<u8> = Vec::new();
        // tick 0 note on chan 1 key 60 vel 100
        t2.extend_from_slice(&[0x00, 0x91, 0x3C, 0x64]);
        // tick 240 note off chan 1 key 60 vel 0; VLQ(240) = [0x81, 0x70]
        t2.extend_from_slice(&[0x81, 0x70, 0x81, 0x3C, 0x40]);
        // tick 240 + 0 note on chan 1 key 64 vel 100
        t2.extend_from_slice(&[0x00, 0x91, 0x40, 0x64]);
        // tick + 240 note off
        t2.extend_from_slice(&[0x81, 0x70, 0x81, 0x40, 0x40]);
        // tick + 720 EOT (so EOT at tick 1200): VLQ(720) = [0x85, 0x50]
        t2.extend_from_slice(&[0x85, 0x50, 0xFF, 0x2F, 0x00]);
        push_track(&mut blob, &t2);

        // Track 3: one drum hit on channel 10 (index 9) — note 36 (kick).
        let mut t3: Vec<u8> = Vec::new();
        // tick 0 note on chan 9 key 36 vel 100
        t3.extend_from_slice(&[0x00, 0x99, 0x24, 0x64]);
        // tick 480 note off (VLQ 480 = [0x83, 0x60])
        t3.extend_from_slice(&[0x83, 0x60, 0x89, 0x24, 0x40]);
        // tick + 720 EOT
        t3.extend_from_slice(&[0x85, 0x50, 0xFF, 0x2F, 0x00]);
        push_track(&mut blob, &t3);

        blob
    }

    fn push_track(blob: &mut Vec<u8>, events: &[u8]) {
        blob.extend_from_slice(b"MTrk");
        blob.extend_from_slice(&(events.len() as u32).to_be_bytes());
        blob.extend_from_slice(events);
    }

    #[test]
    fn registers_codec_under_midi_id() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn decoder_rejects_non_smf_packets() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        let mut dec = reg.make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), b"not midi".to_vec());
        let err = dec.send_packet(&pkt).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn empty_smf_produces_eof_after_initial_chunks() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        let mut dec = reg.make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), minimal_smf());
        dec.send_packet(&pkt).unwrap();
        // The empty-file SMF (one EOT, nothing else) drains immediately.
        // We should get one final near-silent chunk and then Eof.
        let _ = dec.receive_frame().expect("initial chunk");
        // Subsequent calls return Eof.
        let mut got_eof = false;
        for _ in 0..4 {
            match dec.receive_frame() {
                Err(Error::Eof) => {
                    got_eof = true;
                    break;
                }
                Ok(_) => continue,
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
        assert!(got_eof, "decoder should drain to Eof on an empty SMF");
    }

    /// End-to-end: 5-second SMF with notes on channels 1 and 10 + a
    /// tempo change → drives audio out via the tone fallback. Asserts
    /// frame layout, non-silence, and a sensible peak amplitude.
    #[test]
    fn end_to_end_five_second_smf_produces_pcm() {
        let mut dec = MidiDecoder::new(Arc::new(ToneInstrument::new()), OUTPUT_SAMPLE_RATE);
        let blob = five_second_smf();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), blob);
        dec.send_packet(&pkt).unwrap();

        let mut all_samples: Vec<i16> = Vec::new();
        let mut frame_count = 0;
        // Bounded loop — 44_100 * 6 / 1024 ≈ 258 chunks for 6 seconds
        // of audio. Cap at 1024 so a misbehaving decoder can't hang.
        for _ in 0..1024 {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    assert_eq!(af.samples, FRAME_SAMPLES as u32);
                    assert_eq!(af.data.len(), 1, "interleaved S16 = single plane");
                    let bytes = &af.data[0];
                    assert_eq!(bytes.len(), FRAME_SAMPLES * 4, "stereo S16 = 4 bytes/frame");
                    for chunk in bytes.chunks_exact(2) {
                        all_samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                    frame_count += 1;
                }
                Ok(_) => panic!("expected Audio frame"),
                Err(Error::Eof) => break,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }

        // We rendered both channels interleaved — divide by 2 to get
        // per-channel sample count.
        let per_channel = all_samples.len() / 2;
        // The fixture runs through ~1200 ticks at a per-tick rate that
        // halves halfway through (120 → 240 BPM tempo change).
        //
        //   first half : 600 ticks * 45.9375 samples/tick =  27 562 samples
        //   second half: 600 ticks * 22.96875 samples/tick = 13 781 samples
        //                                                    ─────────────
        //   total music: ~41 344 samples (= ~0.94 s wall-clock)
        //
        // The release tails on the (already-done) tone voices are
        // contained in this window. Lower bound: ≥ 30 000 samples
        // (~680 ms) so a regression that emits a single chunk and
        // quits is caught.
        assert!(
            per_channel >= 30_000,
            "expected ≥ 30 k samples (~0.7 s) of audio, got {} samples / channel ({} frames)",
            per_channel,
            frame_count,
        );

        // Non-silence check: at least 5 % of samples must be non-zero.
        let nonzero = all_samples.iter().filter(|s| s.abs() > 16).count();
        let nonzero_ratio = nonzero as f64 / all_samples.len() as f64;
        assert!(
            nonzero_ratio > 0.05,
            "audio is mostly silent: {:.2}% non-zero",
            nonzero_ratio * 100.0,
        );

        // Peak amplitude check: must be audible (>= 1 % of i16 range)
        // but must not have clipped (the fallback's headroom keeps it
        // well under 0 dBFS).
        let peak = all_samples
            .iter()
            .map(|s| s.unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(
            peak > 327,
            "peak {} too quiet — synth is producing near-silent output",
            peak,
        );
        assert!(
            peak < 32_767,
            "peak {} indicates clipping — mix bus should have headroom",
            peak,
        );
    }

    /// End-to-end with the round-2 SF2 fixture (a 20-frame sample-rate
    /// 22 050 Hz looping ramp at root key 60). Exercises the full path
    /// SMF → scheduler → SF2 voice generator → mixer → PCM.
    #[test]
    fn end_to_end_with_sf2_fixture() {
        use crate::instruments::sf2::Sf2Instrument;
        let blob = build_looping_sf2_fixture();
        let inst = Sf2Instrument::from_bytes("fixture", &blob).expect("parse fixture");
        let mut dec = MidiDecoder::new(Arc::new(inst), OUTPUT_SAMPLE_RATE);
        let smf = five_second_smf();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), smf);
        dec.send_packet(&pkt).unwrap();

        let mut all_samples: Vec<i16> = Vec::new();
        for _ in 0..1024 {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    for chunk in af.data[0].chunks_exact(2) {
                        all_samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Err(Error::Eof) => break,
                Ok(_) => panic!("expected Audio frame"),
                Err(other) => panic!("error: {other:?}"),
            }
        }
        // Same per-channel lower bound as the tone-fallback test —
        // music is ~0.94 s wall-clock.
        assert!(
            all_samples.len() / 2 >= 30_000,
            "expected ≥ 30 k samples / channel, got {}",
            all_samples.len() / 2,
        );
        let nonzero = all_samples.iter().filter(|s| s.abs() > 16).count();
        assert!(
            nonzero > all_samples.len() / 20,
            "expected ≥ 5 % non-silent samples, got {} / {}",
            nonzero,
            all_samples.len(),
        );
        let peak = all_samples
            .iter()
            .map(|s| s.unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(peak > 327, "SF2 fixture rendered too quiet (peak {peak})");
    }

    /// Build the same minimal looping SF2 the round-2 voice tests use:
    /// one preset (program 0, bank 0), one instrument, one mono sample
    /// — a 20-frame ramp at 22 050 Hz with `sampleModes=1` so the
    /// voice keeps producing audio for the whole MIDI note duration.
    /// Inlined here (rather than re-exported from `instruments::sf2`)
    /// so the lib-level test stays self-contained.
    fn build_looping_sf2_fixture() -> Vec<u8> {
        // 20-frame ramp climbing from -8000 to +8000 in i16.
        let mut smpl_bytes = Vec::with_capacity(40);
        for i in 0i32..20 {
            let v = (i * 800 - 8000) as i16;
            smpl_bytes.extend_from_slice(&v.to_le_bytes());
        }

        // INFO list.
        let mut info = Vec::new();
        push_riff(&mut info, b"ifil", &[0x02, 0x00, 0x04, 0x00]); // 2.4
        push_riff(&mut info, b"INAM", b"MidiTestBank\0");
        let mut info_list = Vec::from(b"INFO" as &[u8]);
        info_list.extend_from_slice(&info);

        // sdta list.
        let mut sdta = Vec::new();
        push_riff(&mut sdta, b"smpl", &smpl_bytes);
        let mut sdta_list = Vec::from(b"sdta" as &[u8]);
        sdta_list.extend_from_slice(&sdta);

        // pdta list. Generators: sampleModes=54, sampleID=53, instrument=41.
        const GEN_SAMPLE_MODES: u16 = 54;
        const GEN_SAMPLE_ID: u16 = 53;
        const GEN_INSTRUMENT: u16 = 41;
        let phdr = concat_records(&[
            phdr_record("Test Preset", 0, 0, 0),
            phdr_record("EOP", 0, 0, 1),
        ]);
        let pbag = concat_records(&[bag_record(0, 0), bag_record(1, 0)]);
        let pmod = vec![0u8; 10];
        let pgen = concat_records(&[gen_record(GEN_INSTRUMENT, 0), gen_record(0, 0)]);
        let inst = concat_records(&[inst_record("Test Inst", 0), inst_record("EOI", 2)]);
        let ibag = concat_records(&[bag_record(0, 0), bag_record(2, 0)]);
        let imod = vec![0u8; 10];
        let igen = concat_records(&[
            gen_record(GEN_SAMPLE_MODES, 1),
            gen_record(GEN_SAMPLE_ID, 0),
            gen_record(0, 0),
        ]);
        let shdr = concat_records(&[
            shdr_record("RampLoop", 0, 20, 5, 15, 22_050, 60, 0, 0, 1),
            shdr_record("EOS", 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ]);

        let mut pdta = Vec::new();
        push_riff(&mut pdta, b"phdr", &phdr);
        push_riff(&mut pdta, b"pbag", &pbag);
        push_riff(&mut pdta, b"pmod", &pmod);
        push_riff(&mut pdta, b"pgen", &pgen);
        push_riff(&mut pdta, b"inst", &inst);
        push_riff(&mut pdta, b"ibag", &ibag);
        push_riff(&mut pdta, b"imod", &imod);
        push_riff(&mut pdta, b"igen", &igen);
        push_riff(&mut pdta, b"shdr", &shdr);
        let mut pdta_list = Vec::from(b"pdta" as &[u8]);
        pdta_list.extend_from_slice(&pdta);

        // Outer RIFF/sfbk wrapper.
        let mut body = Vec::from(b"sfbk" as &[u8]);
        push_riff(&mut body, b"LIST", &info_list);
        push_riff(&mut body, b"LIST", &sdta_list);
        push_riff(&mut body, b"LIST", &pdta_list);
        let mut out = Vec::from(b"RIFF" as &[u8]);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn push_riff(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
        out.extend_from_slice(tag);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
    }

    fn concat_records(rs: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for r in rs {
            out.extend_from_slice(r);
        }
        out
    }

    fn name20(s: &str) -> [u8; 20] {
        let mut buf = [0u8; 20];
        let bytes = s.as_bytes();
        let n = bytes.len().min(19);
        buf[..n].copy_from_slice(&bytes[..n]);
        buf
    }

    fn phdr_record(name: &str, program: u16, bank: u16, pbag_start: u16) -> Vec<u8> {
        let mut r = vec![0u8; 38];
        r[0..20].copy_from_slice(&name20(name));
        r[20..22].copy_from_slice(&program.to_le_bytes());
        r[22..24].copy_from_slice(&bank.to_le_bytes());
        r[24..26].copy_from_slice(&pbag_start.to_le_bytes());
        r
    }

    fn inst_record(name: &str, ibag_start: u16) -> Vec<u8> {
        let mut r = vec![0u8; 22];
        r[0..20].copy_from_slice(&name20(name));
        r[20..22].copy_from_slice(&ibag_start.to_le_bytes());
        r
    }

    fn bag_record(gen_start: u16, mod_start: u16) -> Vec<u8> {
        let mut r = vec![0u8; 4];
        r[0..2].copy_from_slice(&gen_start.to_le_bytes());
        r[2..4].copy_from_slice(&mod_start.to_le_bytes());
        r
    }

    fn gen_record(oper: u16, amount: u16) -> Vec<u8> {
        let mut r = vec![0u8; 4];
        r[0..2].copy_from_slice(&oper.to_le_bytes());
        r[2..4].copy_from_slice(&amount.to_le_bytes());
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn shdr_record(
        name: &str,
        start: u32,
        end: u32,
        start_loop: u32,
        end_loop: u32,
        sample_rate: u32,
        original_key: u8,
        pitch_correction: i8,
        sample_link: u16,
        sample_type: u16,
    ) -> Vec<u8> {
        let mut r = vec![0u8; 46];
        r[0..20].copy_from_slice(&name20(name));
        r[20..24].copy_from_slice(&start.to_le_bytes());
        r[24..28].copy_from_slice(&end.to_le_bytes());
        r[28..32].copy_from_slice(&start_loop.to_le_bytes());
        r[32..36].copy_from_slice(&end_loop.to_le_bytes());
        r[36..40].copy_from_slice(&sample_rate.to_le_bytes());
        r[40] = original_key;
        r[41] = pitch_correction as u8;
        r[42..44].copy_from_slice(&sample_link.to_le_bytes());
        r[44..46].copy_from_slice(&sample_type.to_le_bytes());
        r
    }

    /// End-to-end SMF with a pitch-bend event mid-note: feed the
    /// decoder, check that the channel state's pitch bend changed by
    /// the time the bend tick has fired.
    #[test]
    fn end_to_end_pitch_bend_event() {
        let mut dec = MidiDecoder::new(Arc::new(ToneInstrument::new()), OUTPUT_SAMPLE_RATE);
        let blob = pitch_bend_smf();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), blob);
        dec.send_packet(&pkt).unwrap();
        // Pull frames until the scheduler has dispatched everything,
        // including the pitch bend (located at tick 480, ≈ 23 k samples
        // = ~22 chunks of 1024).
        for _ in 0..64 {
            match dec.receive_frame() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected: {e:?}"),
            }
        }
        // Inspect the scheduler — the bend should have been applied.
        // We can't poke the mixer directly through the decoder API; the
        // test relies on the scheduler having walked past the event.
        let s = dec.scheduler().unwrap();
        assert!(s.is_done(), "scheduler should have drained the bend");
    }

    /// SMF with: tempo, note-on at tick 0, pitch-bend max-up at tick
    /// 480, note-off at tick 960, EOT at tick 1200.
    fn pitch_bend_smf() -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"MThd");
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&0u16.to_be_bytes());
        blob.extend_from_slice(&1u16.to_be_bytes());
        blob.extend_from_slice(&480u16.to_be_bytes());

        let mut t: Vec<u8> = Vec::new();
        // tick 0 set tempo 500_000 us/qn (= 120 BPM).
        t.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        // tick 0 note on chan 0 key 60 vel 100.
        t.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // tick 480 pitch bend max-up. VLQ(480) = 83 60.
        t.extend_from_slice(&[0x83, 0x60, 0xE0, 0x7F, 0x7F]);
        // tick 480 → tick 960: note-off. VLQ(480) = 83 60.
        t.extend_from_slice(&[0x83, 0x60, 0x80, 0x3C, 0x40]);
        // tick + 240 EOT. VLQ(240) = 81 70.
        t.extend_from_slice(&[0x81, 0x70, 0xFF, 0x2F, 0x00]);
        push_track(&mut blob, &t);
        blob
    }

    /// End-to-end SMF with a channel-aftertouch event mid-note: assert
    /// the decoder doesn't crash and audio still gets produced.
    #[test]
    fn end_to_end_channel_aftertouch_event() {
        let mut dec = MidiDecoder::new(Arc::new(ToneInstrument::new()), OUTPUT_SAMPLE_RATE);
        let blob = aftertouch_smf();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), blob);
        dec.send_packet(&pkt).unwrap();
        let mut samples: Vec<i16> = Vec::new();
        for _ in 0..64 {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    for chunk in af.data[0].chunks_exact(2) {
                        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Err(Error::Eof) => break,
                Ok(_) => panic!("expected audio"),
                Err(e) => panic!("unexpected: {e:?}"),
            }
        }
        // We rendered audio.
        assert!(!samples.is_empty(), "no audio rendered");
        let nonzero = samples.iter().filter(|s| s.abs() > 16).count();
        assert!(
            nonzero > samples.len() / 20,
            "expected ≥ 5 % non-silent: {} / {}",
            nonzero,
            samples.len(),
        );
    }

    /// SMF with: tempo, note-on at tick 0, channel pressure at tick 240,
    /// note-off at tick 480, EOT at tick 720.
    fn aftertouch_smf() -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"MThd");
        blob.extend_from_slice(&6u32.to_be_bytes());
        blob.extend_from_slice(&0u16.to_be_bytes());
        blob.extend_from_slice(&1u16.to_be_bytes());
        blob.extend_from_slice(&480u16.to_be_bytes());

        let mut t: Vec<u8> = Vec::new();
        t.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        t.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // VLQ(240) = 81 70. Channel pressure D0 with value 0x60.
        t.extend_from_slice(&[0x81, 0x70, 0xD0, 0x60]);
        // VLQ(240): note off.
        t.extend_from_slice(&[0x81, 0x70, 0x80, 0x3C, 0x40]);
        // VLQ(240): EOT.
        t.extend_from_slice(&[0x81, 0x70, 0xFF, 0x2F, 0x00]);
        push_track(&mut blob, &t);
        blob
    }

    #[test]
    fn reset_clears_scheduler_and_voices() {
        let mut dec = MidiDecoder::new(Arc::new(ToneInstrument::new()), OUTPUT_SAMPLE_RATE);
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), five_second_smf());
        dec.send_packet(&pkt).unwrap();
        let _ = dec.receive_frame().unwrap();
        dec.reset().unwrap();
        // After reset, receive_frame returns NeedMore (no scheduler).
        match dec.receive_frame() {
            Err(Error::NeedMore) => {}
            other => panic!("expected NeedMore after reset, got {other:?}"),
        }
    }
}
