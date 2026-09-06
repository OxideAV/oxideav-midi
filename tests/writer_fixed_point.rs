//! Writer fixed points: everything the readers parse, the writers
//! emit — and a write → parse → write cycle is a fixed point.
//!
//! * `.midi2` MIDI Clip File (M2-116): a clip carrying every Universal
//!   MIDI Packet Message Type (Utility JR messages, System, MIDI 1.0
//!   and 2.0 Channel Voice, SysEx7 / SysEx8 / Mixed Data Set runs, every
//!   Flex Data family incl. 32-packet text, every UMP Stream message
//!   incl. multi-packet names, and Reserved Message Types), Set Profile
//!   On profiles, configuration events, and a gap beyond the 20-bit
//!   Delta Clockstamp — parses back equal and re-writes byte-identical.
//! * SMF (`.mid`): every meta kind, `F0` / `F7` sysex, every channel
//!   voice message on every channel, TPQN and SMPTE divisions, formats
//!   0 / 1 / 2, multi-byte deltas — through both the explicit-status and
//!   running-status writers.
//! * The in-tree fixture corpus, and the SMF ⇄ clip concordance (lossy
//!   once, then stable).

use oxideav_midi::clip::{self, ClipEvent, ClipFile};
use oxideav_midi::smf::{
    self, ChannelBody, ChannelMessage, Division, Event, MetaEvent, SmfFile, SmfFormat, SmfHeader,
    Track, TrackEvent,
};
use oxideav_midi::ump::data::{sysex8_packets, Data128Message, DataFormat};
use oxideav_midi::ump::flex::{
    flex_text_packets, ChordAlteration, ChordName, FlexAddress, FlexDataMessage,
};
use oxideav_midi::ump::message::{Midi1ChannelVoice, Midi2ChannelVoice, UmpMessage};
use oxideav_midi::ump::stream::{
    endpoint_name_packets, function_block_name_packets, product_instance_id_packets,
    UmpStreamMessage,
};
use oxideav_midi::ump::{sysex7_packets, Ump};

fn from_words(words: &[u32]) -> UmpMessage {
    UmpMessage::decode(&Ump::from_words(words).expect("packet")).expect("decodes")
}

fn from_packets(packets: Vec<Ump>) -> Vec<UmpMessage> {
    packets
        .iter()
        .map(|p| UmpMessage::decode(p).expect("decodes"))
        .collect()
}

/// Every UMP Message Type the reader can hand back, in one sequence.
fn every_message_type() -> Vec<UmpMessage> {
    let mut v: Vec<UmpMessage> = Vec::new();
    // MT 0x0 Utility: NOOP is consumed by the reader (§3.2.2), so only
    // the JR messages ride in the sequence.
    v.push(from_words(&[0x0010_1234])); // JR Clock
    v.push(from_words(&[0x0020_5678])); // JR Timestamp
                                        // MT 0x1 System Common / Real Time.
    for w in [
        0x11F1_2300u32, // MTC Quarter Frame
        0x11F2_0F7F,    // Song Position Pointer
        0x11F3_0500,    // Song Select
        0x11F6_0000,    // Tune Request
        0x11F8_0000,    // Timing Clock
        0x11FA_0000,    // Start
        0x11FB_0000,    // Continue
        0x11FC_0000,    // Stop
        0x11FE_0000,    // Active Sensing
        0x11FF_0000,    // Reset
    ] {
        v.push(from_words(&[w]));
    }
    // MT 0x2 MIDI 1.0 Channel Voice, every opcode.
    for (op, d1, d2) in [
        (0x8u32, 0x3C, 0x40),
        (0x9, 0x3C, 0x7F),
        (0xA, 0x3C, 0x11),
        (0xB, 0x07, 0x64),
        (0xC, 0x1E, 0x00),
        (0xD, 0x22, 0x00),
        (0xE, 0x00, 0x40),
    ] {
        v.push(from_words(&[0x2200_0000
            | (op << 20)
            | (3 << 16)
            | (d1 << 8)
            | d2]));
    }
    // MT 0x3 SysEx7: a complete packet and a Start / Continue / End run.
    v.extend(from_packets(
        sysex7_packets(1, &[0x7E, 0x7F, 0x09]).unwrap(),
    ));
    v.extend(from_packets(
        sysex7_packets(1, &(0u8..14).collect::<Vec<_>>()).unwrap(),
    ));
    // MT 0x4 MIDI 2.0 Channel Voice, every opcode.
    for w in [
        [0x4500_3C01u32, 0x1234_5678], // Registered Per-Note Controller
        [0x4510_3C02, 0x2345_6789],    // Assignable Per-Note Controller
        [0x4520_0000, 0x1800_0000],    // Registered Controller
        [0x4530_0102, 0x0000_0001],    // Assignable Controller
        [0x4540_0001, 0xFFFF_FFFF],    // Relative Registered
        [0x4550_0102, 0x0000_0002],    // Relative Assignable
        [0x4560_3C00, 0xC000_0000],    // Per-Note Pitch Bend
        [0x4580_3C00, 0x8000_0000],    // Note Off
        [0x4590_3C03, 0xC800_7900],    // Note On, Pitch 7.9
        [0x45A0_3C00, 0x8000_0000],    // Poly Pressure
        [0x45B0_0700, 0xC800_0000],    // Control Change
        [0x45C0_0001, 0x1E00_7900],    // Program Change, bank valid
        [0x45D0_0000, 0x4000_0000],    // Channel Pressure
        [0x45E0_0000, 0x8001_0000],    // Pitch Bend
        [0x45F0_3C03, 0x0000_0000],    // Per-Note Management D+S
    ] {
        v.push(from_words(&w));
    }
    // MT 0x5 Data 128: SysEx8 complete + a run, abort, MDS header +
    // payload.
    v.extend(from_packets(sysex8_packets(2, 0x11, &[0x80, 0x01, 0xFF])));
    v.extend(from_packets(sysex8_packets(
        2,
        0x12,
        &(0u8..40).collect::<Vec<_>>(),
    )));
    v.push(UmpMessage::Data128(Data128Message::Sysex8Abort {
        group: 2,
        stream_id: 0x12,
    }));
    v.push(UmpMessage::Data128(Data128Message::MixedDataSetHeader {
        group: 2,
        mds_id: 0x3,
        valid_bytes_in_chunk: 14,
        num_chunks: 2,
        chunk_num: 1,
        manufacturer_id: 0x0123,
        device_id: 0x0456,
        sub_id_1: 0x0007,
        sub_id_2: 0x0008,
    }));
    v.push(UmpMessage::Data128(Data128Message::MixedDataSetPayload {
        group: 2,
        mds_id: 0x3,
        data: [0x5A; 14],
    }));
    // MT 0xD Flex Data: every family.
    v.push(UmpMessage::Flex(FlexDataMessage::SetTempo {
        group: 3,
        ten_ns_per_quarter_note: 50_000_000,
    }));
    v.push(UmpMessage::Flex(FlexDataMessage::SetTimeSignature {
        group: 3,
        numerator: 7,
        denominator: 3,
        number_of_32nd_notes: 8,
    }));
    v.push(UmpMessage::Flex(FlexDataMessage::SetMetronome {
        group: 3,
        clocks_per_primary_click: 24,
        bar_accents: [1, 2, 3],
        subdivision_clicks: [4, 5],
    }));
    v.push(UmpMessage::Flex(FlexDataMessage::SetKeySignature {
        group: 3,
        address: FlexAddress::Channel(9),
        sharps_flats: -3,
        tonic: 0x2,
    }));
    let alt = |t: u8, d: u8| ChordAlteration {
        alteration_type: t,
        degree: d,
    };
    v.push(UmpMessage::Flex(FlexDataMessage::SetChordName {
        group: 3,
        address: FlexAddress::Group,
        chord: ChordName {
            tonic_sharps_flats: 1,
            tonic: 0x4,
            chord_type: 0x11,
            alterations: [alt(1, 5), alt(2, 9), alt(3, 11), alt(4, 13)],
            bass_sharps_flats: -1,
            bass_note: 0x3,
            bass_chord_type: 0x02,
            bass_alterations: [alt(1, 3), alt(2, 7)],
        },
    }));
    // Text: a short metadata run and a full 32-packet lyric.
    v.extend(from_packets(
        flex_text_packets(3, FlexAddress::Group, 0x01, 0x03, "Clip name").unwrap(),
    ));
    let long: String = "lyric-".repeat(64);
    v.extend(from_packets(
        flex_text_packets(3, FlexAddress::Channel(4), 0x02, 0x01, &long[..12 * 32]).unwrap(),
    ));
    // MT 0xF UMP Stream: every message (Start / End of Clip are the
    // framing and are not sequence content).
    v.push(UmpMessage::Stream(UmpStreamMessage::EndpointDiscovery {
        ump_version_major: 1,
        ump_version_minor: 1,
        filter: 0x1F,
    }));
    v.push(UmpMessage::Stream(
        UmpStreamMessage::EndpointInfoNotification {
            ump_version_major: 1,
            ump_version_minor: 1,
            static_function_blocks: true,
            num_function_blocks: 3,
            midi2_capable: true,
            midi1_capable: true,
            rx_jr: false,
            tx_jr: true,
        },
    ));
    v.push(UmpMessage::Stream(
        UmpStreamMessage::DeviceIdentityNotification {
            device_manufacturer: [0x00, 0x21, 0x09],
            device_family: 0x1234,
            device_family_model: 0x2345,
            software_revision: [1, 2, 3, 4],
        },
    ));
    v.extend(from_packets(
        endpoint_name_packets(&"E".repeat(98)).unwrap(),
    ));
    v.extend(from_packets(
        product_instance_id_packets(&"P".repeat(42)).unwrap(),
    ));
    v.push(UmpMessage::Stream(
        UmpStreamMessage::StreamConfigurationRequest {
            protocol: 2,
            rx_jr: true,
            tx_jr: false,
        },
    ));
    v.push(UmpMessage::Stream(
        UmpStreamMessage::StreamConfigurationNotification {
            protocol: 2,
            rx_jr: true,
            tx_jr: true,
        },
    ));
    v.push(UmpMessage::Stream(
        UmpStreamMessage::FunctionBlockDiscovery {
            block: 0xFF,
            filter: 0x3,
        },
    ));
    v.push(UmpMessage::Stream(
        UmpStreamMessage::FunctionBlockInfoNotification {
            active: true,
            block: 1,
            ui_hint: 3,
            midi1_port: 1,
            direction: 3,
            first_group: 2,
            groups_spanned: 4,
            ci_version: 0x02,
            max_sysex8_streams: 5,
        },
    ));
    v.extend(from_packets(
        function_block_name_packets(1, &"F".repeat(91)).unwrap(),
    ));
    // Reserved Message Types (§2.1.4 Table 4 sizes): 0x6 / 0x7 (1
    // word), 0x8 / 0x9 / 0xA (2), 0xB / 0xC (3), 0xE (4).
    v.push(from_words(&[0x6123_4567]));
    v.push(from_words(&[0x7000_0001]));
    v.push(from_words(&[0x8ABC_DEF0, 0x0123_4567]));
    v.push(from_words(&[0x9000_0000, 0xFFFF_FFFF]));
    v.push(from_words(&[0xA111_1111, 0x2222_2222]));
    v.push(from_words(&[0xB000_0001, 0x0000_0002, 0x0000_0003]));
    v.push(from_words(&[0xC000_0001, 0x0000_0002, 0x0000_0003]));
    v.push(from_words(&[
        0xE000_0001,
        0x0000_0002,
        0x0000_0003,
        0x0000_0004,
    ]));
    v
}

fn exhaustive_clip() -> ClipFile {
    let messages = every_message_type();
    // Sequence ticks start at the Start of Clip (tick 10, after the
    // configuration events).
    let mut tick = 10u64;
    let mut sequence = Vec::with_capacity(messages.len());
    for (i, message) in messages.into_iter().enumerate() {
        // Mostly small gaps, a few simultaneous events, and one jump
        // beyond the 20-bit Delta Clockstamp field (§3.2.2 restart).
        tick += match i % 7 {
            0 => 0,
            3 => 0x0010_0000 + 5,
            _ => (i as u64 * 37) % 500,
        };
        sequence.push(ClipEvent { tick, message });
    }
    let end_tick = tick + 0x0020_0000; // two restarts before End of Clip
    ClipFile {
        ticks_per_quarter_note: 960,
        profiles: vec![
            vec![
                0x7E, 0x7F, 0x0D, 0x22, 0x02, 1, 2, 3, 4, 5, 6, 7, 8, 0x7E, 0, 1, 1, 1, 0, 0,
            ],
            vec![
                0x7E, 0x7E, 0x0D, 0x22, 0x01, 1, 2, 3, 4, 5, 6, 7, 8, 0x7E, 0, 2, 1, 1,
            ],
        ],
        config: vec![
            ClipEvent {
                tick: 0,
                message: UmpMessage::Flex(FlexDataMessage::SetTempo {
                    group: 0,
                    ten_ns_per_quarter_note: 60_000_000,
                }),
            },
            ClipEvent {
                tick: 10,
                message: UmpMessage::Midi2 {
                    group: 0,
                    msg: Midi2ChannelVoice::ProgramChange {
                        channel: 0,
                        bank_valid: true,
                        program: 5,
                        bank_msb: 0x79,
                        bank_lsb: 0,
                    },
                },
            },
        ],
        start_tick: 10,
        sequence,
        end_tick,
    }
}

#[test]
fn clip_writer_is_a_fixed_point_over_every_message_type() {
    let clip = exhaustive_clip();
    assert!(clip.sequence.len() > 90, "{}", clip.sequence.len());
    let bytes = clip::write(&clip).expect("writes");
    let back = clip::parse(&bytes).expect("parses");
    assert_eq!(back, clip, "parse(write(clip)) == clip");
    let again = clip::write(&back).expect("re-writes");
    assert_eq!(again, bytes, "write is idempotent");
    // The §3.2.2 restart is really in the stream: a DCS of 0xFFFFF
    // followed by a NOOP.
    let words: Vec<u32> = bytes[8..]
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let restarts = words
        .windows(2)
        .filter(|w| w[0] == 0x004F_FFFF && w[1] == 0x0000_0000)
        .count();
    assert!(restarts >= 3, "{restarts} DCS+NOOP restarts");
}

#[test]
fn clip_writer_rejects_what_the_reader_cannot_produce() {
    let mut clip = exhaustive_clip();
    // Out-of-order ticks.
    clip.sequence[5].tick = 0;
    assert!(clip::write(&clip).is_err());
    let mut clip = exhaustive_clip();
    clip.end_tick = 0;
    assert!(clip::write(&clip).is_err());
    let mut clip = exhaustive_clip();
    clip.start_tick = 0; // before the configuration event at tick 10
    assert!(clip::write(&clip).is_err());
}

/// Every SMF construct on one track.
fn exhaustive_track() -> Track {
    let mut events: Vec<TrackEvent> = Vec::new();
    let mut push = |delta: u32, kind: Event| events.push(TrackEvent { delta, kind });
    push(0, Event::Meta(MetaEvent::SequenceNumber(0xBEEF)));
    for kind in 0x01..=0x0F {
        push(
            1,
            Event::Meta(MetaEvent::Text {
                kind,
                text: format!("text kind {kind:#04x}").into_bytes(),
            }),
        );
    }
    push(0, Event::Meta(MetaEvent::ChannelPrefix(0x0F)));
    push(0, Event::Meta(MetaEvent::Port(0x7F)));
    push(0, Event::Meta(MetaEvent::Tempo(0x0F_FFFF)));
    push(
        0,
        Event::Meta(MetaEvent::SmpteOffset {
            hours: 0x61,
            minutes: 59,
            seconds: 59,
            frames: 29,
            subframes: 99,
        }),
    );
    push(
        0,
        Event::Meta(MetaEvent::TimeSignature {
            numerator: 7,
            denominator_pow2: 3,
            clocks_per_click: 36,
            notated_32nd_per_quarter: 8,
        }),
    );
    push(
        0,
        Event::Meta(MetaEvent::KeySignature {
            sharps_flats: -7,
            mode: 1,
        }),
    );
    push(
        0,
        Event::Meta(MetaEvent::SequencerSpecific(vec![
            0x00, 0x21, 0x09, 1, 2, 3,
        ])),
    );
    push(
        0,
        Event::Meta(MetaEvent::Unknown {
            type_byte: 0x60,
            data: vec![9, 8, 7],
        }),
    );
    push(
        0,
        Event::Meta(MetaEvent::Unknown {
            type_byte: 0x7E,
            data: vec![],
        }),
    );
    push(
        5,
        Event::Sysex {
            escape: false,
            data: vec![0x7E, 0x7F, 0x09, 0x01, 0xF7],
        },
    );
    push(
        5,
        Event::Sysex {
            escape: true,
            data: vec![0xF3, 0x01],
        },
    );
    push(
        0,
        Event::Sysex {
            escape: false,
            data: vec![],
        },
    );
    for channel in 0..16u8 {
        let bodies = [
            ChannelBody::NoteOn {
                key: channel * 8,
                velocity: 127 - channel,
            },
            ChannelBody::PolyAftertouch {
                key: channel * 8,
                pressure: 100,
            },
            ChannelBody::ControlChange {
                controller: 7,
                value: 100,
            },
            ChannelBody::ProgramChange { program: channel },
            ChannelBody::ChannelAftertouch { pressure: 64 },
            ChannelBody::PitchBend {
                value: 0x2000 + u16::from(channel),
            },
            ChannelBody::NoteOff {
                key: channel * 8,
                velocity: 0x40,
            },
            ChannelBody::NoteOn {
                key: channel * 8,
                velocity: 0,
            },
        ];
        for (i, body) in bodies.into_iter().enumerate() {
            let delta = match i {
                0 => 0x0FFF_FFFF, // maximal 4-byte VLQ
                1 => 0x4000,
                2 => 0x80,
                _ => u32::from(channel),
            };
            push(delta, Event::Channel(ChannelMessage { channel, body }));
        }
    }
    // A chord: three Note Ons with the same status byte in a row, so
    // the running-status writer has something to compress.
    for key in [60u8, 64, 67] {
        push(
            0,
            Event::Channel(ChannelMessage {
                channel: 0,
                body: ChannelBody::NoteOn { key, velocity: 90 },
            }),
        );
    }
    push(3, Event::Meta(MetaEvent::EndOfTrack));
    Track { events }
}

fn exhaustive_smf(format: SmfFormat, division: Division, ntrks: u16) -> SmfFile {
    let tracks: Vec<Track> = (0..ntrks).map(|_| exhaustive_track()).collect();
    SmfFile {
        header: SmfHeader {
            format,
            ntrks,
            division,
        },
        tracks,
    }
}

#[test]
fn smf_writers_are_fixed_points_over_every_construct() {
    let files = [
        exhaustive_smf(SmfFormat::SingleTrack, Division::TicksPerQuarter(0x7FFF), 1),
        exhaustive_smf(
            SmfFormat::MultiTrackSimultaneous,
            Division::TicksPerQuarter(1),
            3,
        ),
        exhaustive_smf(
            SmfFormat::MultiTrackIndependent,
            Division::Smpte {
                frames_per_second: 29,
                ticks_per_frame: 40,
            },
            2,
        ),
        exhaustive_smf(
            SmfFormat::SingleTrack,
            Division::Smpte {
                frames_per_second: 24,
                ticks_per_frame: 255,
            },
            1,
        ),
    ];
    for file in &files {
        for (label, bytes) in [
            ("explicit", file.to_bytes().expect("explicit writer")),
            (
                "running-status",
                file.to_bytes_running_status().expect("running writer"),
            ),
        ] {
            let back = smf::parse(&bytes).unwrap_or_else(|e| panic!("{label}: {e:?}"));
            assert_eq!(&back, file, "{label}: parse(write(smf)) == smf");
            let again = match label {
                "explicit" => back.to_bytes().unwrap(),
                _ => back.to_bytes_running_status().unwrap(),
            };
            assert_eq!(again, bytes, "{label}: write is idempotent");
        }
        // The running-status stream is strictly shorter and decodes to
        // the same model.
        let e = file.to_bytes().unwrap();
        let r = file.to_bytes_running_status().unwrap();
        assert!(r.len() < e.len());
        assert_eq!(smf::parse(&r).unwrap(), smf::parse(&e).unwrap());
    }
}

#[test]
fn smf_writer_rejects_what_the_reader_cannot_produce() {
    let mut f = exhaustive_smf(SmfFormat::SingleTrack, Division::TicksPerQuarter(96), 1);
    f.tracks[0].events[0].delta = 0x1000_0000; // beyond the 4-byte VLQ
    assert!(f.to_bytes().is_err());
    let mut f = exhaustive_smf(SmfFormat::SingleTrack, Division::TicksPerQuarter(96), 1);
    f.header.division = Division::TicksPerQuarter(0x8000);
    assert!(f.to_bytes().is_err());
    let mut f = exhaustive_smf(SmfFormat::SingleTrack, Division::TicksPerQuarter(96), 1);
    f.tracks[0].events.pop(); // no End of Track
    assert!(f.to_bytes().is_err());
    let mut f = exhaustive_smf(SmfFormat::SingleTrack, Division::TicksPerQuarter(96), 1);
    f.header.ntrks = 2;
    assert!(f.to_bytes().is_err());
    // The reader's spec-mandated tolerance for unknown chunk types can
    // produce that shape: a file declaring one track whose only chunk
    // is not an `MTrk` parses to zero tracks under `ntrks = 1` — and
    // the writer refuses to re-emit the inconsistent model (fuzz-found).
    let bytes = [
        b"MThd".as_slice(),
        &[0, 0, 0, 6, 0, 0, 0, 1, 0, 0x60],
        b"OTrk",
        &[0, 0, 0, 4, 0, 0xFF, 0x2F, 0],
    ]
    .concat();
    let parsed = smf::parse(&bytes).expect("unknown chunks are skipped");
    assert_eq!(parsed.header.ntrks, 1);
    assert!(parsed.tracks.is_empty());
    assert!(parsed.to_bytes().is_err());
    assert!(parsed.to_bytes_running_status().is_err());
    // Likewise a track chunk that simply ends without `FF 2F 00`: the
    // reader keeps the events it found, the writer wants the End of
    // Track appended first (fuzz-found).
    let bytes = [
        b"MThd".as_slice(),
        &[0, 0, 0, 6, 0, 0, 0, 1, 0, 0x60],
        b"MTrk",
        &[0, 0, 0, 4, 0, 0x90, 0x3C, 0x40],
    ]
    .concat();
    let parsed = smf::parse(&bytes).expect("truncated track tolerated");
    assert_eq!(parsed.tracks[0].events.len(), 1);
    assert!(parsed.to_bytes().is_err());
    // Field *values* the reader keeps verbatim round-trip verbatim — a
    // key-signature mode byte outside {0, 1} included (fuzz-found).
    let bytes = [
        b"MThd".as_slice(),
        &[0, 0, 0, 6, 0, 0, 0, 1, 0, 0x60],
        b"MTrk",
        &[0, 0, 0, 10, 0, 0xFF, 0x59, 2, 0xF9, 47, 0, 0xFF, 0x2F, 0],
    ]
    .concat();
    let parsed = smf::parse(&bytes).unwrap();
    assert_eq!(
        parsed.tracks[0].events[0].kind,
        Event::Meta(MetaEvent::KeySignature {
            sharps_flats: -7,
            mode: 47
        })
    );
    assert_eq!(parsed.to_bytes().unwrap(), bytes);
    // Port / Channel Prefix bytes beyond their spec ranges likewise.
    let bytes = [
        b"MThd".as_slice(),
        &[0, 0, 0, 6, 0, 0, 0, 1, 0, 0x60],
        b"MTrk",
        &[
            0, 0, 0, 14, 0, 0xFF, 0x21, 1, 0x8B, 0, 0xFF, 0x20, 1, 0x90, 0, 0xFF, 0x2F, 0,
        ],
    ]
    .concat();
    let parsed = smf::parse(&bytes).unwrap();
    assert_eq!(
        parsed.tracks[0].events[0].kind,
        Event::Meta(MetaEvent::Port(0x8B))
    );
    assert_eq!(
        parsed.tracks[0].events[1].kind,
        Event::Meta(MetaEvent::ChannelPrefix(0x90))
    );
    assert_eq!(parsed.to_bytes().unwrap(), bytes);
}

#[test]
fn fixture_corpus_round_trips_through_the_writers() {
    for name in ["empty_track.mid", "meta_two_tracks.mid", "note_on_off.mid"] {
        let path = format!("{}/fuzz/corpus/smf/{name}", env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(&path).expect("fixture");
        let parsed = smf::parse(&bytes).expect("fixture parses");
        let written = parsed.to_bytes().expect("writes");
        assert_eq!(smf::parse(&written).unwrap(), parsed, "{name}");
        assert_eq!(parsed.to_bytes().unwrap(), written, "{name}: idempotent");
        let rs = parsed.to_bytes_running_status().expect("writes");
        assert_eq!(smf::parse(&rs).unwrap(), parsed, "{name}: running status");
        // Through the clip concordance (M2-116 Appendix A) and back.
        let clip = ClipFile::from_smf(&parsed).expect("from_smf");
        let clip_bytes = clip::write(&clip).expect("clip writes");
        assert_eq!(clip::parse(&clip_bytes).unwrap(), clip, "{name}: clip");
    }
}

#[test]
fn smf_clip_concordance_is_lossy_once_then_stable() {
    let smf0 = exhaustive_smf(
        SmfFormat::MultiTrackSimultaneous,
        Division::TicksPerQuarter(480),
        2,
    );
    // SMF → clip → SMF loses what M2-116 has no place for (ports,
    // channel prefixes, SMPTE offsets, sequencer-specific, F7
    // continuations, mode of a key signature …) — once.
    let smf1 = ClipFile::from_smf(&smf0).unwrap().to_smf().unwrap();
    let smf2 = ClipFile::from_smf(&smf1).unwrap().to_smf().unwrap();
    assert_eq!(smf1, smf2, "second pass is the identity");
    // And the clip side is stable from the first pass.
    let clip1 = ClipFile::from_smf(&smf1).unwrap();
    let clip2 = ClipFile::from_smf(&clip1.to_smf().unwrap()).unwrap();
    assert_eq!(clip1, clip2);
    // The MIDI 2.0 exhaustive clip translated to SMF and back keeps
    // the MIDI 1.0-expressible content stable too.
    let c = exhaustive_clip();
    let s1 = c.to_smf().unwrap();
    let s2 = ClipFile::from_smf(&s1).unwrap().to_smf().unwrap();
    assert_eq!(s1, s2);
    // Note: a MIDI 1.0 UMP clip's MIDI 1.0 events survive a full
    // round through the SMF model.
    let m1 = Midi1ChannelVoice::NoteOn {
        channel: 2,
        note: 60,
        velocity: 100,
    };
    let clip = ClipFile {
        ticks_per_quarter_note: 96,
        profiles: vec![],
        config: vec![],
        start_tick: 0,
        sequence: vec![ClipEvent {
            tick: 12,
            message: UmpMessage::Midi1 { group: 0, msg: m1 },
        }],
        end_tick: 20,
    };
    let back = ClipFile::from_smf(&clip.to_smf().unwrap()).unwrap();
    assert_eq!(back.sequence, clip.sequence);
    assert_eq!(back.end_tick, clip.end_tick);
    let _ = DataFormat::Complete; // (import used by the sysex helpers' types)
}
