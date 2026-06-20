//! Typed decode and encode of UMP channel-voice and system messages.
//!
//! Covers four Message Types from the spec:
//!
//! * MT 0x0 Utility (§7.2): NOOP, JR Clock, JR Timestamp, Delta
//!   Clockstamp Ticks Per Quarter Note (DCTPQ), Delta Clockstamp (DC).
//! * MT 0x1 System Common / System Real Time (§7.6).
//! * MT 0x2 MIDI 1.0 Channel Voice (§7.3).
//! * MT 0x4 MIDI 2.0 Channel Voice (§7.4).
//!
//! Each typed message exposes a `decode` from an [`Ump`] and an `encode`
//! back to one. The umbrella [`UmpMessage`] dispatches on Message Type
//! so a [`UmpStream`](super::packet::UmpStream) of mixed packets can be
//! decoded uniformly; Message Types this module does not yet model
//! (Data, Flex Data, UMP Stream, Reserved) surface as
//! [`UmpMessage::Unhandled`] carrying the raw packet.

use oxideav_core::{Error, Result};

use super::packet::{MessageType, Ump};

/// A Utility Message (MT 0x0, Groupless, 32-bit) — spec §7.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtilityMessage {
    /// NOOP (status 0x0) — no operation (§7.2.1).
    Noop,
    /// JR Clock (status 0x1) — the Sender's current time, a 16-bit value
    /// in ticks of 1/31250 s (§7.2.2.1).
    JrClock { sender_clock_time: u16 },
    /// JR Timestamp (status 0x2) — the time of the following message(s),
    /// 16-bit ticks of 1/31250 s (§7.2.2.2).
    JrTimestamp { sender_clock_timestamp: u16 },
    /// Delta Clockstamp Ticks Per Quarter Note (status 0x3) — sets the
    /// DC unit of measure, 1..=65535 (0 is Reserved) (§7.2.3.1).
    DeltaClockstampTpq { ticks_per_quarter_note: u16 },
    /// Delta Clockstamp (status 0x4) — ticks since the last event, a
    /// 20-bit field (§7.2.3.2).
    DeltaClockstamp { ticks_since_last_event: u32 },
}

impl UtilityMessage {
    /// Decode a Utility packet (MT 0x0).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::Utility {
            return Err(Error::invalid("UMP: not a Utility message"));
        }
        let w = p.word0();
        // Utility status is the low nibble of the status byte; the high
        // nibble of the status byte is Reserved (§7.2 general format).
        let status = (w >> 16) & 0x0F;
        match status {
            0x0 => Ok(UtilityMessage::Noop),
            0x1 => Ok(UtilityMessage::JrClock {
                sender_clock_time: (w & 0xFFFF) as u16,
            }),
            0x2 => Ok(UtilityMessage::JrTimestamp {
                sender_clock_timestamp: (w & 0xFFFF) as u16,
            }),
            0x3 => Ok(UtilityMessage::DeltaClockstampTpq {
                ticks_per_quarter_note: (w & 0xFFFF) as u16,
            }),
            0x4 => Ok(UtilityMessage::DeltaClockstamp {
                // Figure 34: 20-bit "number of ticks since last event".
                ticks_since_last_event: w & 0x000F_FFFF,
            }),
            other => Err(Error::invalid(format!(
                "UMP Utility: unknown status {other:#x}"
            ))),
        }
    }

    /// Encode this Utility message to its 32-bit packet.
    #[must_use]
    pub fn encode(&self) -> Ump {
        let base = 0x0000_0000u32; // MT=0, group reserved
        let w = match *self {
            UtilityMessage::Noop => base,
            UtilityMessage::JrClock { sender_clock_time } => {
                base | (0x1 << 16) | u32::from(sender_clock_time)
            }
            UtilityMessage::JrTimestamp {
                sender_clock_timestamp,
            } => base | (0x2 << 16) | u32::from(sender_clock_timestamp),
            UtilityMessage::DeltaClockstampTpq {
                ticks_per_quarter_note,
            } => base | (0x3 << 16) | u32::from(ticks_per_quarter_note),
            UtilityMessage::DeltaClockstamp {
                ticks_since_last_event,
            } => base | (0x4 << 16) | (ticks_since_last_event & 0x000F_FFFF),
        };
        Ump::from_parts([w, 0, 0, 0], 1)
    }
}

/// A System Common / System Real Time message (MT 0x1, 32-bit) — §7.6.
///
/// Carries the same data as the MIDI 1.0 system messages, repacked into
/// a single UMP. The `group` field is preserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMessage {
    /// Addressed Group (0..15).
    pub group: u8,
    /// The System status byte, 0xF1..0xFF (0xF0/0xF7 are not used in
    /// UMP — System Exclusive uses MT 0x3 instead, §7.6).
    pub status: u8,
    /// MIDI 1.0 byte 2 (data 1), or 0 if the message has none.
    pub data1: u8,
    /// MIDI 1.0 byte 3 (data 2), or 0 if the message has none.
    pub data2: u8,
}

impl SystemMessage {
    /// Decode a System packet (MT 0x1).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::System {
            return Err(Error::invalid("UMP: not a System message"));
        }
        let status = p.status_byte();
        if status == 0xF0 || status == 0xF7 {
            return Err(Error::invalid(
                "UMP System: status 0xF0/0xF7 not valid in UMP",
            ));
        }
        let w = p.word0();
        Ok(SystemMessage {
            group: p.group().unwrap_or(0),
            status,
            data1: ((w >> 8) & 0x7F) as u8,
            data2: (w & 0x7F) as u8,
        })
    }

    /// The 14-bit Song Position Pointer value (LSB-first per §7.6 note),
    /// valid only when `status == 0xF2`.
    #[must_use]
    pub fn song_position(&self) -> Option<u16> {
        if self.status == 0xF2 {
            Some(u16::from(self.data1) | (u16::from(self.data2) << 7))
        } else {
            None
        }
    }

    /// Encode this System message to its 32-bit packet.
    #[must_use]
    pub fn encode(&self) -> Ump {
        let w = (0x1u32 << 28)
            | (u32::from(self.group & 0x0F) << 24)
            | (u32::from(self.status) << 16)
            | (u32::from(self.data1 & 0x7F) << 8)
            | u32::from(self.data2 & 0x7F);
        Ump::from_parts([w, 0, 0, 0], 1)
    }
}

/// A MIDI 1.0 Channel Voice message in UMP format (MT 0x2, 32-bit) —
/// §7.3. The three MIDI 1.0 status/data bytes are carried verbatim
/// (2-byte messages zero-fill byte 4 per Figure 37).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Midi1ChannelVoice {
    /// Note Off (opcode 0x8).
    NoteOff { channel: u8, note: u8, velocity: u8 },
    /// Note On (opcode 0x9).
    NoteOn { channel: u8, note: u8, velocity: u8 },
    /// Poly Pressure / polyphonic aftertouch (opcode 0xA).
    PolyPressure { channel: u8, note: u8, data: u8 },
    /// Control Change (opcode 0xB).
    ControlChange { channel: u8, index: u8, data: u8 },
    /// Program Change (opcode 0xC).
    ProgramChange { channel: u8, program: u8 },
    /// Channel Pressure / channel aftertouch (opcode 0xD).
    ChannelPressure { channel: u8, data: u8 },
    /// Pitch Bend (opcode 0xE), 14-bit (LSB, MSB).
    PitchBend { channel: u8, lsb: u8, msb: u8 },
}

impl Midi1ChannelVoice {
    /// Decode a MIDI 1.0 Channel Voice packet (MT 0x2).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::Midi1ChannelVoice {
            return Err(Error::invalid("UMP: not a MIDI 1.0 Channel Voice message"));
        }
        let ch = p.channel();
        let w = p.word0();
        let byte3 = ((w >> 8) & 0x7F) as u8;
        let byte4 = (w & 0x7F) as u8;
        match p.opcode() {
            0x8 => Ok(Midi1ChannelVoice::NoteOff {
                channel: ch,
                note: byte3,
                velocity: byte4,
            }),
            0x9 => Ok(Midi1ChannelVoice::NoteOn {
                channel: ch,
                note: byte3,
                velocity: byte4,
            }),
            0xA => Ok(Midi1ChannelVoice::PolyPressure {
                channel: ch,
                note: byte3,
                data: byte4,
            }),
            0xB => Ok(Midi1ChannelVoice::ControlChange {
                channel: ch,
                index: byte3,
                data: byte4,
            }),
            0xC => Ok(Midi1ChannelVoice::ProgramChange {
                channel: ch,
                program: byte3,
            }),
            0xD => Ok(Midi1ChannelVoice::ChannelPressure {
                channel: ch,
                data: byte3,
            }),
            0xE => Ok(Midi1ChannelVoice::PitchBend {
                channel: ch,
                lsb: byte3,
                msb: byte4,
            }),
            other => Err(Error::invalid(format!(
                "UMP MIDI 1.0 CV: unknown opcode {other:#x}"
            ))),
        }
    }

    /// The channel (0..15) this message addresses.
    #[must_use]
    pub fn channel(&self) -> u8 {
        match *self {
            Midi1ChannelVoice::NoteOff { channel, .. }
            | Midi1ChannelVoice::NoteOn { channel, .. }
            | Midi1ChannelVoice::PolyPressure { channel, .. }
            | Midi1ChannelVoice::ControlChange { channel, .. }
            | Midi1ChannelVoice::ProgramChange { channel, .. }
            | Midi1ChannelVoice::ChannelPressure { channel, .. }
            | Midi1ChannelVoice::PitchBend { channel, .. } => channel,
        }
    }

    /// Encode to a MIDI 1.0 Channel Voice packet on the given `group`.
    #[must_use]
    pub fn encode(&self, group: u8) -> Ump {
        let (opcode, ch, byte3, byte4) = match *self {
            Midi1ChannelVoice::NoteOff {
                channel,
                note,
                velocity,
            } => (0x8u8, channel, note, velocity),
            Midi1ChannelVoice::NoteOn {
                channel,
                note,
                velocity,
            } => (0x9, channel, note, velocity),
            Midi1ChannelVoice::PolyPressure {
                channel,
                note,
                data,
            } => (0xA, channel, note, data),
            Midi1ChannelVoice::ControlChange {
                channel,
                index,
                data,
            } => (0xB, channel, index, data),
            Midi1ChannelVoice::ProgramChange { channel, program } => (0xC, channel, program, 0),
            Midi1ChannelVoice::ChannelPressure { channel, data } => (0xD, channel, data, 0),
            Midi1ChannelVoice::PitchBend { channel, lsb, msb } => (0xE, channel, lsb, msb),
        };
        let status = (opcode << 4) | (ch & 0x0F);
        let w = (0x2u32 << 28)
            | (u32::from(group & 0x0F) << 24)
            | (u32::from(status) << 16)
            | (u32::from(byte3 & 0x7F) << 8)
            | u32::from(byte4 & 0x7F);
        Ump::from_parts([w, 0, 0, 0], 1)
    }
}

include!("message_midi2.rs");
include!("translate.rs");

/// A decoded UMP message dispatched on Message Type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UmpMessage {
    /// MT 0x0 Utility.
    Utility(UtilityMessage),
    /// MT 0x1 System Common / System Real Time.
    System(SystemMessage),
    /// MT 0x2 MIDI 1.0 Channel Voice (with its Group).
    Midi1 { group: u8, msg: Midi1ChannelVoice },
    /// MT 0x4 MIDI 2.0 Channel Voice (with its Group).
    Midi2 { group: u8, msg: Midi2ChannelVoice },
    /// Any Message Type not modelled by this layer (Data, Flex Data,
    /// UMP Stream, Reserved) — carries the raw packet for inspection.
    Unhandled(Ump),
}

impl UmpMessage {
    /// Decode any UMP packet, dispatching on its Message Type.
    pub fn decode(p: &Ump) -> Result<Self> {
        match p.message_type() {
            MessageType::Utility => Ok(UmpMessage::Utility(UtilityMessage::decode(p)?)),
            MessageType::System => Ok(UmpMessage::System(SystemMessage::decode(p)?)),
            MessageType::Midi1ChannelVoice => Ok(UmpMessage::Midi1 {
                group: p.group().unwrap_or(0),
                msg: Midi1ChannelVoice::decode(p)?,
            }),
            MessageType::Midi2ChannelVoice => Ok(UmpMessage::Midi2 {
                group: p.group().unwrap_or(0),
                msg: Midi2ChannelVoice::decode(p)?,
            }),
            _ => Ok(UmpMessage::Unhandled(*p)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ump(words: &[u32]) -> Ump {
        Ump::from_words(words).unwrap()
    }

    #[test]
    fn utility_noop_round_trips() {
        let m = UtilityMessage::decode(&ump(&[0x0000_0000])).unwrap();
        assert_eq!(m, UtilityMessage::Noop);
        assert_eq!(m.encode().word0(), 0x0000_0000);
    }

    #[test]
    fn utility_jr_clock() {
        // status 0x1, time 0x1234.
        let m = UtilityMessage::decode(&ump(&[0x0001_1234])).unwrap();
        assert_eq!(
            m,
            UtilityMessage::JrClock {
                sender_clock_time: 0x1234
            }
        );
        assert_eq!(m.encode().word0(), 0x0001_1234);
    }

    #[test]
    fn utility_delta_clockstamp_20bit() {
        // status 0x4, ticks 0xABCDE (20-bit).
        let m = UtilityMessage::decode(&ump(&[0x0004_BCDE])).unwrap();
        assert_eq!(
            m,
            UtilityMessage::DeltaClockstamp {
                ticks_since_last_event: 0x0004_BCDE & 0x000F_FFFF
            }
        );
    }

    #[test]
    fn system_timing_clock() {
        // MT1, group 0, status 0xF8 (timing clock).
        let m = SystemMessage::decode(&ump(&[0x10F8_0000])).unwrap();
        assert_eq!(m.status, 0xF8);
        assert_eq!(m.group, 0);
        assert_eq!(m.encode().word0(), 0x10F8_0000);
    }

    #[test]
    fn system_song_position_lsb_first() {
        // MT1, group 0, status 0xF2, lsb 0x7F, msb 0x01 → 0x7F | (1<<7) = 255.
        let m = SystemMessage::decode(&ump(&[0x10F2_7F01])).unwrap();
        assert_eq!(m.song_position(), Some(0x7F | (1 << 7)));
    }

    #[test]
    fn system_rejects_sysex_status() {
        assert!(SystemMessage::decode(&ump(&[0x10F0_0000])).is_err());
        assert!(SystemMessage::decode(&ump(&[0x10F7_0000])).is_err());
    }

    #[test]
    fn midi1_note_on_round_trips() {
        // MT2, group 3, status 0x91, note 0x40, vel 0x7F.
        let p = ump(&[0x2391_407F]);
        let m = Midi1ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi1ChannelVoice::NoteOn {
                channel: 1,
                note: 0x40,
                velocity: 0x7F
            }
        );
        assert_eq!(m.encode(3), p);
    }

    #[test]
    fn midi1_program_change_zerofills_byte4() {
        let p = ump(&[0x20C5_2000]);
        let m = Midi1ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi1ChannelVoice::ProgramChange {
                channel: 5,
                program: 0x20
            }
        );
        assert_eq!(m.encode(0).word0() & 0xFF, 0);
    }

    #[test]
    fn umpmessage_dispatch_unhandled() {
        // MT5 Data128 — not modelled, surfaces as Unhandled.
        let p = ump(&[0x5000_0000, 0, 0, 0]);
        assert!(matches!(
            UmpMessage::decode(&p).unwrap(),
            UmpMessage::Unhandled(_)
        ));
    }
}
