//! End-to-end MIDI Clip File playback: a `.midi2` byte stream carrying
//! MIDI 2.0 Protocol content goes through `MidiDecoder::send_packet`
//! and renders to PCM via the M2-104 Appendix-D Default Translation
//! into the existing scheduler + 32-voice mixer path.

use std::sync::Arc;

use oxideav_core::{Decoder, Error, Frame, Packet, TimeBase};
use oxideav_midi::clip::{self, ClipEvent, ClipFile};
use oxideav_midi::instruments::tone::ToneInstrument;
use oxideav_midi::ump::flex::FlexDataMessage;
use oxideav_midi::ump::message::{Midi2ChannelVoice, UmpMessage};
use oxideav_midi::MidiDecoder;

/// One-second clip at 120 BPM / 480 TPQ, played entirely with MIDI 2.0
/// Channel Voice messages (16-bit velocities) so playback exercises
/// the 2.0 → 1.0 translation, plus a Flex Data Set Tempo.
fn one_second_midi2_clip() -> Vec<u8> {
    let note = |tick: u64, on: bool, note: u8| ClipEvent {
        tick,
        message: UmpMessage::Midi2 {
            group: 0,
            msg: if on {
                Midi2ChannelVoice::NoteOn {
                    channel: 0,
                    note,
                    attribute_type: 0,
                    velocity: 0xC000,
                    attribute: 0,
                }
            } else {
                Midi2ChannelVoice::NoteOff {
                    channel: 0,
                    note,
                    attribute_type: 0,
                    velocity: 0x8000,
                    attribute: 0,
                }
            },
        },
    };
    let clip = ClipFile {
        ticks_per_quarter_note: 480,
        profiles: vec![],
        config: vec![ClipEvent {
            tick: 0,
            message: UmpMessage::Flex(FlexDataMessage::SetTempo {
                group: 0,
                ten_ns_per_quarter_note: 50_000_000, // 120 BPM
            }),
        }],
        start_tick: 0,
        sequence: vec![
            note(0, true, 60),
            note(480, false, 60),
            note(480, true, 64),
            note(960, false, 64),
        ],
        end_tick: 960,
    };
    clip::write(&clip).expect("clip serialises")
}

#[test]
fn midi2_clip_renders_nonsilent_pcm() {
    let mut dec = MidiDecoder::new(Arc::new(ToneInstrument::new()), 44_100);
    let pkt = Packet::new(0, TimeBase::new(1, 44_100), one_second_midi2_clip());
    dec.send_packet(&pkt).expect("SMF2CLIP accepted");

    let mut total_samples = 0u64;
    let mut peak = 0i16;
    for _ in 0..400 {
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                total_samples += u64::from(a.samples);
                for pair in a.data[0].chunks_exact(2) {
                    peak = peak.max(i16::from_le_bytes([pair[0], pair[1]]).saturating_abs());
                }
            }
            Ok(other) => panic!("unexpected frame {other:?}"),
            Err(Error::Eof) => break,
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }
    // Two beats at 120 BPM = 1 s ≈ 44 100 samples (plus release tail).
    assert!(
        total_samples >= 40_000,
        "expected about a second of audio, got {total_samples} samples"
    );
    assert!(
        peak > 1_000,
        "MIDI 2.0 clip content should be audible, peak was {peak}"
    );
}

#[test]
fn clip_without_notes_drains_to_eof() {
    let clip = ClipFile {
        ticks_per_quarter_note: 96,
        profiles: vec![],
        config: vec![],
        start_tick: 0,
        sequence: vec![],
        end_tick: 0,
    };
    let mut dec = MidiDecoder::new(Arc::new(ToneInstrument::new()), 44_100);
    let pkt = Packet::new(
        0,
        TimeBase::new(1, 44_100),
        clip::write(&clip).expect("clip serialises"),
    );
    dec.send_packet(&pkt).expect("SMF2CLIP accepted");
    let mut got_eof = false;
    for _ in 0..40 {
        match dec.receive_frame() {
            Ok(_) => continue,
            Err(Error::Eof) => {
                got_eof = true;
                break;
            }
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }
    assert!(got_eof, "empty clip should drain to Eof");
}
