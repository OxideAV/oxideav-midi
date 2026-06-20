//! Round-353 integration tests for the Universal MIDI Packet (UMP)
//! subsystem (`oxideav_midi::ump`).
//!
//! These exercise the public UMP surface end-to-end: walking a flat
//! `&[u32]` word buffer of mixed-size packets into typed messages,
//! translating between the MIDI 1.0 and MIDI 2.0 Protocols under the
//! spec Appendix D Default Translation Mode, and re-encoding back to
//! packets. They complement the in-module unit tests by checking the
//! cross-module contract (packet ⇄ message ⇄ translation ⇄ packet).

use oxideav_midi::ump::{
    MessageType, Midi1ChannelVoice, Midi2ChannelVoice, Ump, UmpMessage, UmpStream, UtilityMessage,
};

/// A heterogeneous stream — JR Timestamp (1 word), MIDI 1.0 Note On
/// (1 word), MIDI 2.0 Note On (2 words), System Timing Clock (1 word) —
/// walks into exactly four typed messages with the right sizes.
#[test]
fn mixed_stream_decodes_to_typed_messages() {
    let words = [
        0x0002_1000, // MT0 Utility: JR Timestamp = 0x1000
        0x2491_3C64, // MT2 MIDI 1.0 Note On ch1 note 0x3C vel 0x64
        0x4190_3C00, // MT4 MIDI 2.0 Note On ch0 note 0x3C word0
        0xC000_0000, // ... word1: velocity 0xC000
        0x10F8_0000, // MT1 System Timing Clock
    ];

    let decoded: Vec<UmpMessage> = UmpStream::new(&words)
        .map(|r| UmpMessage::decode(&r.expect("packet")).expect("decode"))
        .collect();

    assert_eq!(decoded.len(), 4);
    assert!(matches!(
        decoded[0],
        UmpMessage::Utility(UtilityMessage::JrTimestamp {
            sender_clock_timestamp: 0x1000
        })
    ));
    assert!(matches!(
        decoded[1],
        UmpMessage::Midi1 {
            group: 4,
            msg: Midi1ChannelVoice::NoteOn {
                channel: 1,
                note: 0x3C,
                velocity: 0x64,
            },
        }
    ));
    assert!(matches!(
        decoded[2],
        UmpMessage::Midi2 {
            group: 1,
            msg: Midi2ChannelVoice::NoteOn {
                channel: 0,
                note: 0x3C,
                velocity: 0xC000,
                ..
            },
        }
    ));
    assert!(matches!(decoded[3], UmpMessage::System(_)));
}

/// A MIDI 1.0 packet, decoded, translated to MIDI 2.0, re-encoded, then
/// translated back, must reproduce the original MIDI 1.0 message for the
/// losslessly-translatable Control Change family.
#[test]
fn midi1_cc_round_trip_through_midi2_and_back() {
    // MT2 group 0, CC ch2 index 74 (brightness) value 0x55.
    let orig = Ump::from_words(&[0x20B2_4A55]).unwrap();
    assert_eq!(orig.group(), Some(0));
    let m1 = Midi1ChannelVoice::decode(&orig).unwrap();

    let m2 = m1.to_midi2().expect("CC 74 translates");
    // Re-encode the MIDI 2.0 form to a packet and decode it back to prove
    // the encode/decode pair is faithful.
    let m2_packet = m2.encode(0);
    assert_eq!(m2_packet.message_type(), MessageType::Midi2ChannelVoice);
    let m2_again = Midi2ChannelVoice::decode(&m2_packet).unwrap();
    assert_eq!(m2_again, m2);

    let back = m2_again.to_midi1().expect("back to MIDI 1.0");
    assert_eq!(back, m1);
    assert_eq!(back.encode(0), orig);
}

/// MIDI 2.0 -> MIDI 1.0 -> MIDI 2.0 of a high-resolution Note On loses
/// the sub-7-bit velocity precision (expected, §D.1.1) but preserves the
/// note, channel, and the velocity within MIDI 1.0 quantisation.
#[test]
fn midi2_note_on_to_midi1_quantises_velocity() {
    let m2 = Midi2ChannelVoice::NoteOn {
        channel: 0,
        note: 60,
        attribute_type: 0,
        velocity: 0xFFFF, // max -> 0x7F
        attribute: 0,
    };
    let m1 = m2.to_midi1().unwrap();
    assert_eq!(
        m1,
        Midi1ChannelVoice::NoteOn {
            channel: 0,
            note: 60,
            velocity: 0x7F,
        }
    );
}

/// The self-delimiting stream reader yields a truncation error for a
/// trailing partial packet without panicking, and stops afterward.
#[test]
fn truncated_trailing_packet_surfaces_error() {
    // valid 1-word MIDI 1.0, then a 2-word MT begun with one word only.
    let words = [0x2090_407F, 0x4090_4000];
    let results: Vec<_> = UmpStream::new(&words).collect();
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok());
    assert!(results[1].is_err());
}
