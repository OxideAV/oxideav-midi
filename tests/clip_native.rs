//! Native MIDI Clip File playback (`Scheduler::from_clip`) measured
//! against the Appendix-D translated path (`ClipFile::to_smf` +
//! `Scheduler::new`) on rendered PCM.
//!
//! * A MIDI 1.0-in-UMP clip — and a MIDI 2.0 clip whose values all sit
//!   on the 7/14-bit grid — render **bit-identically** either way.
//! * A MIDI 2.0 clip carrying resolution the downscale folds away
//!   (16-bit velocity low bits, a sub-14-bit pitch bend, a Per-Note
//!   Pitch Bend) renders identically to its on-grid twin through the
//!   translated path, and differently — audibly finer — natively.

use std::sync::Arc;

use oxideav_core::{Decoder, Error, Frame, Packet, TimeBase};
use oxideav_midi::clip::{self, ClipEvent, ClipFile};
use oxideav_midi::instruments::tone::ToneInstrument;
use oxideav_midi::mixer::Mixer;
use oxideav_midi::scheduler::Scheduler;
use oxideav_midi::ump::flex::FlexDataMessage;
use oxideav_midi::ump::message::{Midi1ChannelVoice, Midi2ChannelVoice, UmpMessage};
use oxideav_midi::ump::scaling::{scale_7_to_16, scale_7_to_32};
use oxideav_midi::ump::sysex7_packets;
use oxideav_midi::MidiDecoder;

const RATE: u32 = 44_100;
const BLOCK: usize = 1024;
const BLOCKS: usize = 48; // ≈ 1.1 s

fn m1(tick: u64, msg: Midi1ChannelVoice) -> ClipEvent {
    ClipEvent {
        tick,
        message: UmpMessage::Midi1 { group: 0, msg },
    }
}

fn m2(tick: u64, msg: Midi2ChannelVoice) -> ClipEvent {
    ClipEvent {
        tick,
        message: UmpMessage::Midi2 { group: 0, msg },
    }
}

fn tempo(tick: u64, ten_ns: u32) -> ClipEvent {
    ClipEvent {
        tick,
        message: UmpMessage::Flex(FlexDataMessage::SetTempo {
            group: 0,
            ten_ns_per_quarter_note: ten_ns,
        }),
    }
}

fn sysex(tick: u64, payload: &[u8]) -> Vec<ClipEvent> {
    sysex7_packets(0, payload)
        .expect("sysex packets")
        .into_iter()
        .map(|p| ClipEvent {
            tick,
            message: UmpMessage::decode(&p).expect("decodes"),
        })
        .collect()
}

fn clip_of(sequence: Vec<ClipEvent>) -> ClipFile {
    let end_tick = sequence.last().map_or(0, |e| e.tick);
    ClipFile {
        ticks_per_quarter_note: 480,
        profiles: vec![],
        config: vec![tempo(0, 50_000_000)], // 120 BPM
        start_tick: 0,
        sequence,
        end_tick,
    }
}

fn render(mut sched: Scheduler) -> Vec<f32> {
    let inst = ToneInstrument::new();
    let mut mixer = Mixer::new();
    let mut out = Vec::with_capacity(BLOCK * BLOCKS);
    let mut l = vec![0.0f32; BLOCK];
    let mut r = vec![0.0f32; BLOCK];
    for _ in 0..BLOCKS {
        sched.step(BLOCK, &mut mixer, &inst);
        mixer.mix_stereo(&mut l, &mut r);
        out.extend_from_slice(&l);
    }
    out
}

fn native(clip: &ClipFile) -> Vec<f32> {
    render(Scheduler::from_clip(clip, RATE))
}

fn translated(clip: &ClipFile) -> Vec<f32> {
    render(Scheduler::new(&clip.to_smf().expect("to_smf"), RATE))
}

fn peak(a: &[f32]) -> f32 {
    a.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}

/// A busy MIDI 1.0-in-UMP clip: notes on two channels, CC 7 / CC 10,
/// a program change, a 14-bit pitch bend, a mid-clip tempo change and
/// a Universal Real-Time Master Volume SysEx.
fn midi1_clip() -> ClipFile {
    use Midi1ChannelVoice as M;
    let mut seq = vec![
        m1(
            0,
            M::ProgramChange {
                channel: 1,
                program: 30,
            },
        ),
        m1(
            0,
            M::ControlChange {
                channel: 0,
                index: 7,
                data: 100,
            },
        ),
        m1(
            0,
            M::ControlChange {
                channel: 1,
                index: 10,
                data: 20,
            },
        ),
        m1(
            0,
            M::NoteOn {
                channel: 0,
                note: 60,
                velocity: 100,
            },
        ),
        m1(
            120,
            M::NoteOn {
                channel: 1,
                note: 67,
                velocity: 80,
            },
        ),
        m1(
            240,
            M::PitchBend {
                channel: 0,
                lsb: 0,
                msb: 0x50,
            },
        ),
    ];
    seq.extend(sysex(300, &[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x60]));
    seq.push(tempo(360, 25_000_000)); // 240 BPM from here
    seq.extend([
        m1(
            480,
            M::NoteOff {
                channel: 0,
                note: 60,
                velocity: 0,
            },
        ),
        m1(
            600,
            M::NoteOff {
                channel: 1,
                note: 67,
                velocity: 64,
            },
        ),
        m1(
            600,
            M::NoteOn {
                channel: 0,
                note: 64,
                velocity: 0x7F,
            },
        ),
        m1(
            900,
            M::NoteOff {
                channel: 0,
                note: 64,
                velocity: 0,
            },
        ),
    ]);
    clip_of(seq)
}

/// A MIDI 2.0 clip whose every value sits on the 7/14-bit grid
/// (`grid = true`), or the same clip with sub-grid resolution
/// (`grid = false`): 16-bit velocity low bits, a 32-bit pitch bend
/// between two 14-bit steps, a 32-bit CC 7 between two 7-bit steps.
fn midi2_clip(grid: bool) -> ClipFile {
    use Midi2ChannelVoice as M;
    // The off-grid values stay below the next §D.1.4 truncation
    // boundary, so the downscale folds them onto the same 7/14-bit
    // values as the on-grid clip.
    let vel: u16 = if grid {
        scale_7_to_16(100)
    } else {
        scale_7_to_16(100) + 0x80
    };
    let bend: u32 = if grid {
        0x8000_0000
    } else {
        0x8000_0000 + (1 << 17)
    };
    let vol: u32 = if grid {
        scale_7_to_32(100)
    } else {
        scale_7_to_32(100) + (1 << 20)
    };
    let seq = vec![
        m2(
            0,
            M::ProgramChange {
                channel: 1,
                bank_valid: true,
                program: 30,
                bank_msb: 0x79,
                bank_lsb: 0,
            },
        ),
        m2(
            0,
            M::ControlChange {
                channel: 0,
                index: 7,
                data: vol,
            },
        ),
        m2(
            0,
            M::RegisteredController {
                channel: 0,
                bank: 0,
                index: 0,
                data: 12 << 25, // ±12 semitones
            },
        ),
        m2(
            0,
            M::NoteOn {
                channel: 0,
                note: 60,
                attribute_type: 0,
                velocity: vel,
                attribute: 0,
            },
        ),
        m2(
            120,
            M::NoteOn {
                channel: 1,
                note: 67,
                attribute_type: 0,
                velocity: 0xFFFF,
                attribute: 0,
            },
        ),
        m2(
            240,
            M::PitchBend {
                channel: 0,
                data: bend,
            },
        ),
        m2(
            300,
            M::ChannelPressure {
                channel: 1,
                data: 64 << 25,
            },
        ),
        m2(
            480,
            M::NoteOff {
                channel: 0,
                note: 60,
                attribute_type: 0,
                velocity: 0x8000,
                attribute: 0,
            },
        ),
        m2(
            600,
            M::NoteOff {
                channel: 1,
                note: 67,
                attribute_type: 0,
                velocity: 0x8000,
                attribute: 0,
            },
        ),
    ];
    clip_of(seq)
}

#[test]
fn midi1_clip_renders_bit_identically_native_and_translated() {
    let clip = midi1_clip();
    let n = native(&clip);
    let t = translated(&clip);
    assert!(peak(&n) > 0.05, "render must be audible");
    assert_eq!(n, t);
}

#[test]
fn midi2_clip_on_the_grid_renders_bit_identically_native_and_translated() {
    let clip = midi2_clip(true);
    let n = native(&clip);
    let t = translated(&clip);
    assert!(peak(&n) > 0.05, "render must be audible");
    assert_eq!(n, t);
}

#[test]
fn midi2_clip_off_the_grid_is_folded_by_translation_but_kept_natively() {
    let on = midi2_clip(true);
    let off = midi2_clip(false);
    // The Appendix-D downscale cannot tell the two clips apart …
    assert_eq!(translated(&on), translated(&off));
    // … the native path can, and it is a refinement, not a re-write:
    // the sub-grid clip stays within 1 % of its on-grid twin.
    let n_on = native(&on);
    let n_off = native(&off);
    assert_ne!(n_on, n_off);
    let d = n_on
        .iter()
        .zip(&n_off)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(d > 0.0 && d < 0.02 * peak(&n_on), "delta {d}");
}

#[test]
fn per_note_pitch_bend_is_dropped_by_translation_but_rendered_natively() {
    use Midi2ChannelVoice as M;
    let plain = clip_of(vec![
        m2(
            0,
            M::NoteOn {
                channel: 0,
                note: 60,
                attribute_type: 0,
                velocity: scale_7_to_16(100),
                attribute: 0,
            },
        ),
        m2(
            960,
            M::NoteOff {
                channel: 0,
                note: 60,
                attribute_type: 0,
                velocity: 0x8000,
                attribute: 0,
            },
        ),
    ]);
    let mut bent = plain.clone();
    bent.sequence.insert(
        1,
        m2(
            240,
            M::PerNotePitchBend {
                channel: 0,
                note: 60,
                data: 0xC000_0000,
            },
        ),
    );
    // §D.2.8: no MIDI 1.0 counterpart → the translated renders match.
    assert_eq!(translated(&plain), translated(&bent));
    assert_eq!(native(&plain), translated(&plain));
    assert_ne!(native(&plain), native(&bent));
}

#[test]
fn decoder_plays_a_clip_natively_and_matches_the_smf_of_a_midi1_clip() {
    let clip = midi1_clip();
    let clip_bytes = clip::write(&clip).expect("clip serialises");
    let smf_bytes = clip
        .to_smf()
        .expect("to_smf")
        .to_bytes()
        .expect("smf bytes");
    let run = |bytes: Vec<u8>| {
        let mut dec = MidiDecoder::new(Arc::new(ToneInstrument::new()), RATE);
        dec.send_packet(&Packet::new(0, TimeBase::new(1, i64::from(RATE)), bytes))
            .expect("accepted");
        let mut pcm = Vec::new();
        for _ in 0..400 {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => pcm.extend_from_slice(&a.data[0]),
                Ok(other) => panic!("unexpected frame {other:?}"),
                Err(Error::Eof) => break,
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
        pcm
    };
    let from_clip = run(clip_bytes);
    let from_smf = run(smf_bytes);
    assert!(from_clip.len() > 40_000);
    assert_eq!(from_clip, from_smf);
}
