//! MIDI Clip File (SMF2 clip, `.midi2`) — M2-116-U v1.0.
//!
//! The MIDI Clip File is the MIDI 2.0 successor to the SMF Type 0
//! file: a single sequence of Universal MIDI Packets. Its layout
//! (M2-116 §4, Figure 2) is:
//!
//! 1. **File Header** — the 8 ASCII bytes `SMF2CLIP` (§5). Everything
//!    after it is UMP messages, stored big-endian (§3).
//! 2. **Clip Configuration Header** (§6) — optional Set Profile On
//!    SysEx messages (no Delta Clockstamp), then a Delta Clockstamp
//!    Ticks Per Quarter Note (DCTPQ) with a preceding DCS of 0, then
//!    any further receiver-setup messages, each with a preceding DCS.
//! 3. **Clip Sequence Data** (§7) — from the Start of Clip message to
//!    the End of Clip message, every message with an associated DCS.
//!    Nothing may follow End of Clip.
//!
//! Timing uses the two Utility messages of M2-104 §7.2.3: DCTPQ sets
//! the tick unit, and each Delta Clockstamp declares the time of all
//! following messages until the next one; the 20-bit DCS field is
//! restarted with a `DCS + NOOP` pair when a gap exceeds 1 048 575
//! ticks (§3.2.2).
//!
//! [`parse`] and [`write`] round-trip the file form; [`ClipFile::to_smf`]
//! converts a clip into an [`SmfFile`](crate::smf::SmfFile) through the
//! Appendix-D Default Translation (so the existing SMF scheduler +
//! 32-voice mixer render `.midi2` content), and [`ClipFile::from_smf`]
//! builds a clip from an SMF sequence.

use oxideav_core::{Error, Result};

use crate::smf::{
    ChannelBody, ChannelMessage, Division, Event, MetaEvent, SmfBuilder, SmfFile, SmfFormat,
};
use crate::ump::flex::{self, text_status, FlexAddress, FlexDataMessage, FlexTextAssembler};
use crate::ump::message::{Midi1ChannelVoice, UmpMessage, UtilityMessage};
use crate::ump::stream::UmpStreamMessage;
use crate::ump::translator::midi2_to_midi1_messages;
use crate::ump::{sysex7_packets, Sysex7Assembler, UmpStream};

/// The 8-byte File Header (§5, Table 5).
pub const CLIP_MAGIC: &[u8; 8] = b"SMF2CLIP";

/// The file extension mandated by §4.1.
pub const CLIP_EXTENSION: &str = "midi2";

/// Maximum value of the 20-bit Delta Clockstamp field.
const DCS_MAX: u64 = 0x000F_FFFF;

/// One timed message in a MIDI Clip File. `tick` is absolute (the
/// running sum of Delta Clockstamps from the start of the file), in
/// DCTPQ units.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipEvent {
    /// Absolute tick at which the message fires.
    pub tick: u64,
    /// The message itself (Delta Clockstamps are consumed into `tick`
    /// and never appear here).
    pub message: UmpMessage,
}

/// A parsed MIDI Clip File (M2-116).
#[derive(Debug, Clone, PartialEq)]
pub struct ClipFile {
    /// Delta Clockstamp Ticks Per Quarter Note — the tick unit of
    /// every [`ClipEvent::tick`] (§3.2.1).
    pub ticks_per_quarter_note: u16,
    /// Assembled Set Profile On SysEx payloads from the very start of
    /// the Clip Configuration Header (§6.2) — the bytes between 0xF0
    /// and 0xF7, exclusive. Empty for clips that target no Profile.
    pub profiles: Vec<Vec<u8>>,
    /// Receiver-configuration messages between the DCTPQ and the Start
    /// of Clip (§6.3), with absolute ticks.
    pub config: Vec<ClipEvent>,
    /// Absolute tick of the Start of Clip message (§7.1).
    pub start_tick: u64,
    /// The Clip Sequence Data between Start of Clip and End of Clip.
    pub sequence: Vec<ClipEvent>,
    /// Absolute tick of the End of Clip message (§7.3).
    pub end_tick: u64,
}

/// Does this byte buffer begin with the MIDI Clip File header?
#[must_use]
pub fn is_clip_file(data: &[u8]) -> bool {
    data.len() >= CLIP_MAGIC.len() && &data[..CLIP_MAGIC.len()] == CLIP_MAGIC
}

/// Parse a `.midi2` MIDI Clip File.
pub fn parse(data: &[u8]) -> Result<ClipFile> {
    if !is_clip_file(data) {
        return Err(Error::invalid(
            "MIDI Clip File: missing 'SMF2CLIP' File Header",
        ));
    }
    let body = &data[CLIP_MAGIC.len()..];
    if body.len() % 4 != 0 {
        return Err(Error::invalid(
            "MIDI Clip File: UMP data length is not a multiple of 4 bytes",
        ));
    }
    let words: Vec<u32> = body
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut messages = Vec::new();
    for p in UmpStream::new(&words) {
        messages.push(UmpMessage::decode(&p?)?);
    }
    let mut iter = messages.into_iter().peekable();

    // ── Leading Set Profile On messages (no DCS) — §6.2 ──
    let mut profiles = Vec::new();
    let mut profile_asm = Sysex7Assembler::new();
    while let Some(UmpMessage::Sysex7(_)) = iter.peek() {
        let Some(UmpMessage::Sysex7(sx)) = iter.next() else {
            unreachable!()
        };
        if let Some(payload) = profile_asm.push(&sx) {
            profiles.push(payload);
        }
    }
    if profile_asm.in_progress() {
        return Err(Error::invalid(
            "MIDI Clip File: unterminated Set Profile On SysEx before the DCTPQ",
        ));
    }

    // ── DCS(0) + DCTPQ — §6, §3.2.1 ──
    let mut tick: u64 = match iter.next() {
        Some(UmpMessage::Utility(UtilityMessage::DeltaClockstamp {
            ticks_since_last_event,
        })) => u64::from(ticks_since_last_event),
        _ => {
            return Err(Error::invalid(
                "MIDI Clip File: expected the Delta Clockstamp preceding the DCTPQ",
            ));
        }
    };
    let ticks_per_quarter_note = match iter.next() {
        Some(UmpMessage::Utility(UtilityMessage::DeltaClockstampTpq {
            ticks_per_quarter_note,
        })) => ticks_per_quarter_note,
        _ => {
            return Err(Error::invalid(
                "MIDI Clip File: the first message after the File Header (and any \
                 Profile ID) must be Delta Clockstamp Ticks Per Quarter Note",
            ));
        }
    };

    // ── Config events until Start of Clip; then sequence until End ──
    let mut config = Vec::new();
    let mut sequence = Vec::new();
    let mut start_tick: Option<u64> = None;
    let mut end_tick: Option<u64> = None;
    for message in iter.by_ref() {
        if end_tick.is_some() {
            // §7.3: a MIDI Clip File shall not have any data following
            // the End of Clip message.
            return Err(Error::invalid(
                "MIDI Clip File: data present after the End of Clip message",
            ));
        }
        match message {
            UmpMessage::Utility(UtilityMessage::DeltaClockstamp {
                ticks_since_last_event,
            }) => {
                tick += u64::from(ticks_since_last_event);
            }
            // NOOP restarts the delta count (§3.2.2) and carries no
            // playback meaning — consume it.
            UmpMessage::Utility(UtilityMessage::Noop) => {}
            UmpMessage::Stream(UmpStreamMessage::StartOfClip) => {
                if start_tick.is_some() {
                    return Err(Error::invalid(
                        "MIDI Clip File: more than one Start of Clip message",
                    ));
                }
                start_tick = Some(tick);
            }
            UmpMessage::Stream(UmpStreamMessage::EndOfClip) => {
                if start_tick.is_none() {
                    return Err(Error::invalid(
                        "MIDI Clip File: End of Clip before Start of Clip",
                    ));
                }
                end_tick = Some(tick);
            }
            other => {
                let ev = ClipEvent {
                    tick,
                    message: other,
                };
                if start_tick.is_some() {
                    sequence.push(ev);
                } else {
                    config.push(ev);
                }
            }
        }
    }
    let (Some(start_tick), Some(end_tick)) = (start_tick, end_tick) else {
        return Err(Error::invalid(
            "MIDI Clip File: missing Start of Clip / End of Clip messages",
        ));
    };
    Ok(ClipFile {
        ticks_per_quarter_note,
        profiles,
        config,
        start_tick,
        sequence,
        end_tick,
    })
}

/// Append one UMP's words, big-endian.
fn push_ump(out: &mut Vec<u8>, ump: &crate::ump::Ump) {
    for w in ump.words() {
        out.extend_from_slice(&w.to_be_bytes());
    }
}

/// Emit the Delta Clockstamp(s) advancing from `*cursor` to `tick`,
/// restarting the 20-bit count with `DCS + NOOP` pairs when the gap
/// exceeds the field range (§3.2.2).
fn push_delta(out: &mut Vec<u8>, cursor: &mut u64, tick: u64) {
    let mut delta = tick.saturating_sub(*cursor);
    while delta > DCS_MAX {
        push_ump(
            out,
            &UtilityMessage::DeltaClockstamp {
                ticks_since_last_event: DCS_MAX as u32,
            }
            .encode(),
        );
        push_ump(out, &UtilityMessage::Noop.encode());
        delta -= DCS_MAX;
    }
    push_ump(
        out,
        &UtilityMessage::DeltaClockstamp {
            ticks_since_last_event: delta as u32,
        }
        .encode(),
    );
    *cursor = tick;
}

/// Serialise a [`ClipFile`] to `.midi2` bytes.
///
/// Canonical emission: each message gets its own preceding Delta
/// Clockstamp (simultaneous events carry a DCS of 0), Set Profile On
/// messages lead without a DCS (§6.2), and long gaps are chunked with
/// `DCS + NOOP` restarts (§3.2.2). Events must be tick-ordered.
pub fn write(clip: &ClipFile) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(CLIP_MAGIC);

    for payload in &clip.profiles {
        for p in sysex7_packets(0, payload)? {
            push_ump(&mut out, &p);
        }
    }

    let mut cursor: u64 = 0;
    // DCS(0) + DCTPQ (§6).
    push_ump(
        &mut out,
        &UtilityMessage::DeltaClockstamp {
            ticks_since_last_event: 0,
        }
        .encode(),
    );
    push_ump(
        &mut out,
        &UtilityMessage::DeltaClockstampTpq {
            ticks_per_quarter_note: clip.ticks_per_quarter_note,
        }
        .encode(),
    );

    let emit = |out: &mut Vec<u8>, cursor: &mut u64, events: &[ClipEvent]| -> Result<()> {
        for ev in events {
            if ev.tick < *cursor {
                return Err(Error::invalid(
                    "MIDI Clip File: events must be in tick order",
                ));
            }
            push_delta(out, cursor, ev.tick);
            push_ump(out, &ev.message.encode());
        }
        Ok(())
    };

    emit(&mut out, &mut cursor, &clip.config)?;
    if clip.start_tick < cursor {
        return Err(Error::invalid(
            "MIDI Clip File: Start of Clip tick precedes the configuration events",
        ));
    }
    push_delta(&mut out, &mut cursor, clip.start_tick);
    push_ump(&mut out, &UmpStreamMessage::StartOfClip.encode());
    emit(&mut out, &mut cursor, &clip.sequence)?;
    if clip.end_tick < cursor {
        return Err(Error::invalid(
            "MIDI Clip File: End of Clip tick precedes the sequence events",
        ));
    }
    push_delta(&mut out, &mut cursor, clip.end_tick);
    push_ump(&mut out, &UmpStreamMessage::EndOfClip.encode());
    Ok(out)
}

/// Map a UMP MIDI 1.0 Channel Voice message to an SMF channel event.
fn midi1_to_channel(msg: &Midi1ChannelVoice) -> ChannelMessage {
    match *msg {
        Midi1ChannelVoice::NoteOff {
            channel,
            note,
            velocity,
        } => ChannelMessage {
            channel,
            body: ChannelBody::NoteOff {
                key: note,
                velocity,
            },
        },
        Midi1ChannelVoice::NoteOn {
            channel,
            note,
            velocity,
        } => ChannelMessage {
            channel,
            body: ChannelBody::NoteOn {
                key: note,
                velocity,
            },
        },
        Midi1ChannelVoice::PolyPressure {
            channel,
            note,
            data,
        } => ChannelMessage {
            channel,
            body: ChannelBody::PolyAftertouch {
                key: note,
                pressure: data,
            },
        },
        Midi1ChannelVoice::ControlChange {
            channel,
            index,
            data,
        } => ChannelMessage {
            channel,
            body: ChannelBody::ControlChange {
                controller: index,
                value: data,
            },
        },
        Midi1ChannelVoice::ProgramChange { channel, program } => ChannelMessage {
            channel,
            body: ChannelBody::ProgramChange { program },
        },
        Midi1ChannelVoice::ChannelPressure { channel, data } => ChannelMessage {
            channel,
            body: ChannelBody::ChannelAftertouch { pressure: data },
        },
        Midi1ChannelVoice::PitchBend { channel, lsb, msb } => ChannelMessage {
            channel,
            body: ChannelBody::PitchBend {
                value: u16::from(lsb & 0x7F) | (u16::from(msb & 0x7F) << 7),
            },
        },
    }
}

/// Map an SMF channel event to a UMP MIDI 1.0 Channel Voice message.
fn channel_to_midi1(msg: &ChannelMessage) -> Midi1ChannelVoice {
    let channel = msg.channel;
    match msg.body {
        ChannelBody::NoteOff { key, velocity } => Midi1ChannelVoice::NoteOff {
            channel,
            note: key,
            velocity,
        },
        ChannelBody::NoteOn { key, velocity } => Midi1ChannelVoice::NoteOn {
            channel,
            note: key,
            velocity,
        },
        ChannelBody::PolyAftertouch { key, pressure } => Midi1ChannelVoice::PolyPressure {
            channel,
            note: key,
            data: pressure,
        },
        ChannelBody::ControlChange { controller, value } => Midi1ChannelVoice::ControlChange {
            channel,
            index: controller,
            data: value,
        },
        ChannelBody::ProgramChange { program } => {
            Midi1ChannelVoice::ProgramChange { channel, program }
        }
        ChannelBody::ChannelAftertouch { pressure } => Midi1ChannelVoice::ChannelPressure {
            channel,
            data: pressure,
        },
        ChannelBody::PitchBend { value } => Midi1ChannelVoice::PitchBend {
            channel,
            lsb: (value & 0x7F) as u8,
            msb: ((value >> 7) & 0x7F) as u8,
        },
    }
}

/// Map a Flex Data text status to the SMF text-meta type byte, when a
/// counterpart exists (M2-116 Appendix A concordance direction).
fn flex_text_to_meta_kind(bank: u8, status: u8) -> Option<u8> {
    match (bank, status) {
        (text_status::BANK_METADATA, text_status::UNKNOWN_METADATA) => Some(0x01),
        (text_status::BANK_METADATA, text_status::COPYRIGHT_NOTICE) => Some(0x02),
        (text_status::BANK_METADATA, text_status::MIDI_CLIP_NAME) => Some(0x03),
        (text_status::BANK_PERFORMANCE, text_status::LYRICS) => Some(0x05),
        // Other credits have no dedicated SMF meta — carry as generic
        // Text (FF 01).
        (text_status::BANK_METADATA, _) => Some(0x01),
        _ => None,
    }
}

impl ClipFile {
    /// Convert this clip into a format-0 [`SmfFile`] through the
    /// M2-104 Appendix-D Default Translation, so the existing SMF
    /// scheduler / mixer pipeline can render `.midi2` content:
    ///
    /// * MIDI 1.0 Channel Voice UMPs map directly;
    /// * MIDI 2.0 Channel Voice messages translate per §D.2 (including
    ///   the compound RPN/NRPN and Bank/Program expansions);
    /// * Flex Data Set Tempo / Time Signature / Key Signature and the
    ///   text families map to their SMF meta counterparts;
    /// * SysEx7 packet runs reassemble into `F0` SysEx events;
    /// * SysEx8 / Mixed Data Set / Stream messages have no MIDI 1.0
    ///   form (§D.2.9) and are dropped.
    pub fn to_smf(&self) -> Result<SmfFile> {
        let division = self.ticks_per_quarter_note.clamp(1, 0x7FFF);
        let mut builder = SmfBuilder::new();
        builder
            .format(SmfFormat::SingleTrack)
            .division(Division::TicksPerQuarter(division));
        let end_tick = self.end_tick;
        let mut sysex_asm = Sysex7Assembler::new();
        let mut text_asm = FlexTextAssembler::new();
        builder.track(|tb| {
            for ev in self.config.iter().chain(self.sequence.iter()) {
                let tick = ev.tick;
                match &ev.message {
                    UmpMessage::Midi1 { msg, .. } => {
                        tb.push(tick, Event::Channel(midi1_to_channel(msg)));
                    }
                    UmpMessage::Midi2 { msg, .. } => {
                        for m1 in midi2_to_midi1_messages(msg) {
                            tb.push(tick, Event::Channel(midi1_to_channel(&m1)));
                        }
                    }
                    UmpMessage::Sysex7(sx) => {
                        if let Some(mut payload) = sysex_asm.push(sx) {
                            // SMF convention: the F0 event payload keeps
                            // the trailing EOX byte.
                            payload.push(0xF7);
                            tb.push(
                                tick,
                                Event::Sysex {
                                    escape: false,
                                    data: payload,
                                },
                            );
                        }
                    }
                    UmpMessage::Flex(flex_msg) => match flex_msg {
                        FlexDataMessage::SetTempo {
                            ten_ns_per_quarter_note,
                            ..
                        } => {
                            tb.meta(
                                tick,
                                MetaEvent::Tempo(flex::ten_ns_per_quarter_to_usec(
                                    *ten_ns_per_quarter_note,
                                )),
                            );
                        }
                        FlexDataMessage::SetTimeSignature {
                            numerator,
                            denominator,
                            number_of_32nd_notes,
                            ..
                        } => {
                            tb.meta(
                                tick,
                                MetaEvent::TimeSignature {
                                    numerator: *numerator,
                                    denominator_pow2: *denominator,
                                    // 24 MIDI Clocks per metronome click
                                    // is the conventional SMF value; the
                                    // Flex message carries none.
                                    clocks_per_click: 24,
                                    notated_32nd_per_quarter: *number_of_32nd_notes,
                                },
                            );
                        }
                        FlexDataMessage::SetKeySignature { sharps_flats, .. } => {
                            // The SMF meta has no tonic field and the
                            // Flex message has no major/minor mode —
                            // carry the accidental count, mode major.
                            if (-7..=7).contains(sharps_flats) {
                                tb.meta(
                                    tick,
                                    MetaEvent::KeySignature {
                                        sharps_flats: *sharps_flats,
                                        mode: 0,
                                    },
                                );
                            }
                        }
                        FlexDataMessage::Text {
                            format,
                            status_bank,
                            status,
                            bytes,
                            ..
                        } => {
                            if let Some(text) = text_asm.push(*format, bytes) {
                                if let Some(kind) = flex_text_to_meta_kind(*status_bank, *status) {
                                    tb.meta(tick, MetaEvent::Text { kind, text });
                                }
                            }
                        }
                        // Metronome configuration and chord names have
                        // no SMF v1 meta counterpart.
                        _ => {}
                    },
                    // §D.2.9: SysEx8 / Mixed Data Set / Utility have no
                    // non-UMP MIDI 1.0 form; Stream messages address the
                    // endpoint, not the sequence.
                    _ => {}
                }
            }
            tb.meta(end_tick, MetaEvent::EndOfTrack);
        });
        Ok(builder.build())
    }

    /// Build a clip from an SMF sequence (the M2-116 Appendix A
    /// concordance, SMF1 → SMF2 direction). Tracks are merged onto the
    /// shared timebase; channel events ride as MIDI 1.0-in-UMP,
    /// `FF 51` tempo / `FF 58` time signature / `FF 59` key signature
    /// metas become their Flex Data counterparts, text metas with a
    /// Flex mapping become Flex text messages, and `F0` SysEx events
    /// become SysEx7 packet runs.
    pub fn from_smf(smf: &SmfFile) -> Result<Self> {
        let Division::TicksPerQuarter(tpq) = smf.header.division else {
            return Err(Error::invalid(
                "MIDI Clip File: SMPTE-division SMFs have no DCTPQ equivalent",
            ));
        };
        // Merge to absolute ticks, preserving (track, in-track) order
        // for simultaneous events.
        let mut merged: Vec<(u64, usize, usize, &Event)> = Vec::new();
        let mut end_tick: u64 = 0;
        for (ti, track) in smf.tracks.iter().enumerate() {
            let mut tick: u64 = 0;
            for (oi, te) in track.events.iter().enumerate() {
                tick = tick.saturating_add(u64::from(te.delta));
                if matches!(te.kind, Event::Meta(MetaEvent::EndOfTrack)) {
                    end_tick = end_tick.max(tick);
                } else {
                    merged.push((tick, ti, oi, &te.kind));
                }
            }
        }
        merged.sort_by_key(|&(tick, ti, oi, _)| (tick, ti, oi));

        let mut sequence = Vec::new();
        let mut push = |tick: u64, message: UmpMessage| {
            sequence.push(ClipEvent { tick, message });
        };
        for (tick, _, _, event) in &merged {
            let tick = *tick;
            match event {
                Event::Channel(cm) => push(
                    tick,
                    UmpMessage::Midi1 {
                        group: 0,
                        msg: channel_to_midi1(cm),
                    },
                ),
                Event::Meta(MetaEvent::Tempo(us)) => push(
                    tick,
                    UmpMessage::Flex(FlexDataMessage::SetTempo {
                        group: 0,
                        ten_ns_per_quarter_note: flex::usec_per_quarter_to_10ns(*us),
                    }),
                ),
                Event::Meta(MetaEvent::TimeSignature {
                    numerator,
                    denominator_pow2,
                    notated_32nd_per_quarter,
                    ..
                }) => push(
                    tick,
                    UmpMessage::Flex(FlexDataMessage::SetTimeSignature {
                        group: 0,
                        numerator: *numerator,
                        denominator: *denominator_pow2,
                        number_of_32nd_notes: *notated_32nd_per_quarter,
                    }),
                ),
                Event::Meta(MetaEvent::KeySignature { sharps_flats, .. }) => push(
                    tick,
                    UmpMessage::Flex(FlexDataMessage::SetKeySignature {
                        group: 0,
                        address: FlexAddress::Group,
                        sharps_flats: *sharps_flats,
                        tonic: 0,
                    }),
                ),
                Event::Meta(MetaEvent::Text { kind, text }) => {
                    let (bank, status) = match kind {
                        0x02 => (text_status::BANK_METADATA, text_status::COPYRIGHT_NOTICE),
                        0x03 => (text_status::BANK_METADATA, text_status::MIDI_CLIP_NAME),
                        0x05 => (text_status::BANK_PERFORMANCE, text_status::LYRICS),
                        _ => (text_status::BANK_METADATA, text_status::UNKNOWN_METADATA),
                    };
                    let address = if bank == text_status::BANK_PERFORMANCE {
                        FlexAddress::Channel(0)
                    } else {
                        FlexAddress::Group
                    };
                    if let Ok(text) = std::str::from_utf8(text) {
                        if let Ok(packets) = flex::flex_text_packets(0, address, bank, status, text)
                        {
                            for p in packets {
                                push(tick, UmpMessage::decode(&p)?);
                            }
                        }
                    }
                }
                Event::Sysex {
                    escape: false,
                    data,
                } => {
                    let payload = if data.last() == Some(&0xF7) {
                        &data[..data.len() - 1]
                    } else {
                        &data[..]
                    };
                    if let Ok(packets) = sysex7_packets(0, payload) {
                        for p in packets {
                            push(tick, UmpMessage::decode(&p)?);
                        }
                    }
                }
                // F7 continuations, and metas with no Flex counterpart
                // (sequence number, channel prefix, port, SMPTE offset,
                // sequencer-specific), are dropped.
                _ => {}
            }
        }
        let end_tick = end_tick.max(sequence.last().map_or(0, |e| e.tick));
        Ok(ClipFile {
            ticks_per_quarter_note: tpq,
            profiles: Vec::new(),
            config: Vec::new(),
            start_tick: 0,
            sequence,
            end_tick,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ump::flex::FlexDataMessage;
    use crate::ump::message::Midi2ChannelVoice;

    fn note_on(tick: u64, channel: u8, note: u8, velocity: u8) -> ClipEvent {
        ClipEvent {
            tick,
            message: UmpMessage::Midi1 {
                group: 0,
                msg: Midi1ChannelVoice::NoteOn {
                    channel,
                    note,
                    velocity,
                },
            },
        }
    }

    fn note_off(tick: u64, channel: u8, note: u8) -> ClipEvent {
        ClipEvent {
            tick,
            message: UmpMessage::Midi1 {
                group: 0,
                msg: Midi1ChannelVoice::NoteOff {
                    channel,
                    note,
                    velocity: 0x40,
                },
            },
        }
    }

    fn simple_clip() -> ClipFile {
        ClipFile {
            ticks_per_quarter_note: 480,
            profiles: vec![],
            config: vec![ClipEvent {
                tick: 0,
                message: UmpMessage::Flex(FlexDataMessage::SetTempo {
                    group: 0,
                    ten_ns_per_quarter_note: 50_000_000,
                }),
            }],
            start_tick: 0,
            sequence: vec![
                note_on(0, 0, 60, 100),
                note_off(480, 0, 60),
                note_on(480, 0, 64, 100),
                note_off(960, 0, 64),
            ],
            end_tick: 960,
        }
    }

    #[test]
    fn header_bytes_are_smf2clip() {
        let bytes = write(&simple_clip()).unwrap();
        assert_eq!(&bytes[..8], b"SMF2CLIP");
        // Byte 9 onward is the first UMP: DCS(0) then DCTPQ(480) —
        // Utility status nibble at bits 20..24 (Figure 26 / Table 26).
        assert_eq!(&bytes[8..12], &[0x00, 0x40, 0x00, 0x00]);
        assert_eq!(&bytes[12..16], &[0x00, 0x30, 0x01, 0xE0]);
    }

    #[test]
    fn write_parse_round_trips() {
        let clip = simple_clip();
        let bytes = write(&clip).unwrap();
        let back = parse(&bytes).unwrap();
        assert_eq!(back, clip);
    }

    #[test]
    fn profiles_round_trip_without_dcs() {
        let mut clip = simple_clip();
        // A Set Profile On CI payload (opaque bytes for framing test).
        clip.profiles = vec![vec![0x7E, 0x7F, 0x0D, 0x21, 0x01]];
        let bytes = write(&clip).unwrap();
        // First UMP after the header is the SysEx7, not a DCS.
        assert_eq!(bytes[8] >> 4, 0x3);
        let back = parse(&bytes).unwrap();
        assert_eq!(back.profiles, clip.profiles);
        assert_eq!(back.sequence, clip.sequence);
    }

    #[test]
    fn long_gap_restarts_delta_with_noop() {
        let mut clip = simple_clip();
        clip.sequence = vec![note_on(0, 0, 60, 100), note_off(2_000_000, 0, 60)];
        clip.end_tick = 2_000_000;
        let bytes = write(&clip).unwrap();
        let back = parse(&bytes).unwrap();
        assert_eq!(back.sequence[1].tick, 2_000_000);
        assert_eq!(back.end_tick, 2_000_000);
    }

    #[test]
    fn data_after_end_of_clip_is_rejected() {
        let mut bytes = write(&simple_clip()).unwrap();
        // Append one more DCS+NoteOn after End of Clip.
        bytes.extend_from_slice(&[0x00, 0x40, 0x00, 0x00]);
        bytes.extend_from_slice(&[0x20, 0x90, 0x3C, 0x64]);
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn missing_magic_and_missing_dctpq_are_rejected() {
        assert!(parse(b"MThd\x00\x00\x00\x06").is_err());
        // Magic + a DCS followed by a NoteOn instead of the DCTPQ.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CLIP_MAGIC);
        bytes.extend_from_slice(&[0x00, 0x40, 0x00, 0x00]);
        bytes.extend_from_slice(&[0x20, 0x90, 0x3C, 0x64]);
        assert!(parse(&bytes).is_err());
    }

    #[test]
    fn to_smf_translates_midi1_events_and_tempo() {
        let smf = simple_clip().to_smf().unwrap();
        assert_eq!(smf.header.division, Division::TicksPerQuarter(480));
        assert_eq!(smf.tracks.len(), 1);
        let events = &smf.tracks[0].events;
        // Tempo meta first (config), then 4 channel events, then EOT.
        assert!(matches!(
            events[0].kind,
            Event::Meta(MetaEvent::Tempo(500_000))
        ));
        let channel_count = events
            .iter()
            .filter(|e| matches!(e.kind, Event::Channel(_)))
            .count();
        assert_eq!(channel_count, 4);
        assert!(matches!(
            events.last().unwrap().kind,
            Event::Meta(MetaEvent::EndOfTrack)
        ));
    }

    #[test]
    fn to_smf_expands_midi2_rpn_to_cc_sequence() {
        let mut clip = simple_clip();
        clip.sequence.insert(
            0,
            ClipEvent {
                tick: 0,
                message: UmpMessage::Midi2 {
                    group: 0,
                    msg: Midi2ChannelVoice::RegisteredController {
                        channel: 0,
                        bank: 0,
                        index: 0,
                        data: crate::ump::scale_up(2 << 7, 14, 32),
                    },
                },
            },
        );
        let smf = clip.to_smf().unwrap();
        let ccs: Vec<(u8, u8)> = smf.tracks[0]
            .events
            .iter()
            .filter_map(|e| match e.kind {
                Event::Channel(ChannelMessage {
                    body: ChannelBody::ControlChange { controller, value },
                    ..
                }) => Some((controller, value)),
                _ => None,
            })
            .collect();
        assert_eq!(ccs, vec![(101, 0), (100, 0), (6, 2), (38, 0)]);
    }

    #[test]
    fn to_smf_reassembles_sysex7() {
        let mut clip = simple_clip();
        // GM System On as a Complete SysEx7 UMP at tick 0.
        for p in sysex7_packets(0, &[0x7E, 0x7F, 0x09, 0x01]).unwrap() {
            clip.config.insert(
                0,
                ClipEvent {
                    tick: 0,
                    message: UmpMessage::decode(&p).unwrap(),
                },
            );
        }
        let smf = clip.to_smf().unwrap();
        let sysex: Vec<&Vec<u8>> = smf.tracks[0]
            .events
            .iter()
            .filter_map(|e| match &e.kind {
                Event::Sysex {
                    escape: false,
                    data,
                } => Some(data),
                _ => None,
            })
            .collect();
        assert_eq!(sysex, vec![&vec![0x7E, 0x7F, 0x09, 0x01, 0xF7]]);
    }

    #[test]
    fn from_smf_and_back_preserves_notes_and_tempo() {
        let clip = simple_clip();
        let smf = clip.to_smf().unwrap();
        let clip2 = ClipFile::from_smf(&smf).unwrap();
        assert_eq!(clip2.ticks_per_quarter_note, 480);
        // Same channel-voice content (tempo now rides in the sequence).
        let notes: Vec<&ClipEvent> = clip2
            .sequence
            .iter()
            .filter(|e| matches!(e.message, UmpMessage::Midi1 { .. }))
            .collect();
        assert_eq!(notes.len(), 4);
        assert_eq!(notes[0], &note_on(0, 0, 60, 100));
        let tempo = clip2.sequence.iter().find(|e| {
            matches!(
                e.message,
                UmpMessage::Flex(FlexDataMessage::SetTempo { .. })
            )
        });
        assert!(tempo.is_some());
        assert_eq!(clip2.end_tick, 960);
        // And the clip file form still round-trips.
        let bytes = write(&clip2).unwrap();
        assert_eq!(parse(&bytes).unwrap(), clip2);
    }

    #[test]
    fn text_meta_round_trips_through_flex_text() {
        let mut clip = simple_clip();
        clip.config.push(ClipEvent {
            tick: 0,
            message: UmpMessage::decode(
                &flex::flex_text_packets(
                    0,
                    FlexAddress::Group,
                    text_status::BANK_METADATA,
                    text_status::MIDI_CLIP_NAME,
                    "Clip A",
                )
                .unwrap()[0],
            )
            .unwrap(),
        });
        let smf = clip.to_smf().unwrap();
        let has_name = smf.tracks[0].events.iter().any(|e| {
            matches!(&e.kind, Event::Meta(MetaEvent::Text { kind: 3, text }) if text == b"Clip A")
        });
        assert!(has_name, "MIDI Clip Name should become FF 03");
    }
}
