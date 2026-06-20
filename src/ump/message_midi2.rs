// MIDI 2.0 Channel Voice message decode (MT 0x4, 64-bit) — spec §7.4.
//
// Included into message.rs so it shares the module's imports
// (`Error`, `Result`, `MessageType`, `Ump`). Kept in a separate file
// only to keep each source file at a readable length.

/// A MIDI 2.0 Channel Voice message in UMP format (MT 0x4, 64-bit) —
/// §7.4. Two 32-bit words: word 0 carries MT/group/status/index, word 1
/// carries the 32-bit data payload (or the structured per-field layout
/// for Note On/Off, Program Change, and RPN/NRPN messages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Midi2ChannelVoice {
    /// Registered Per-Note Controller (opcode 0x0) — §7.4.4.
    RegisteredPerNoteController {
        channel: u8,
        note: u8,
        index: u8,
        data: u32,
    },
    /// Assignable Per-Note Controller (opcode 0x1) — §7.4.4.
    AssignablePerNoteController {
        channel: u8,
        note: u8,
        index: u8,
        data: u32,
    },
    /// Registered Controller / RPN (opcode 0x2) — §7.4.7.
    RegisteredController {
        channel: u8,
        bank: u8,
        index: u8,
        data: u32,
    },
    /// Assignable Controller / NRPN (opcode 0x3) — §7.4.7.
    AssignableController {
        channel: u8,
        bank: u8,
        index: u8,
        data: u32,
    },
    /// Relative Registered Controller (opcode 0x4) — §7.4.8. `data` is a
    /// two's-complement relative value.
    RelativeRegisteredController {
        channel: u8,
        bank: u8,
        index: u8,
        data: i32,
    },
    /// Relative Assignable Controller (opcode 0x5) — §7.4.8.
    RelativeAssignableController {
        channel: u8,
        bank: u8,
        index: u8,
        data: i32,
    },
    /// Per-Note Pitch Bend (opcode 0x6) — §7.4.12. Bipolar, centred at
    /// 0x8000_0000.
    PerNotePitchBend { channel: u8, note: u8, data: u32 },
    /// Note Off (opcode 0x8) — §7.4.1. 16-bit velocity + attribute.
    NoteOff {
        channel: u8,
        note: u8,
        attribute_type: u8,
        velocity: u16,
        attribute: u16,
    },
    /// Note On (opcode 0x9) — §7.4.2. Velocity 0 is **not** a Note Off.
    NoteOn {
        channel: u8,
        note: u8,
        attribute_type: u8,
        velocity: u16,
        attribute: u16,
    },
    /// Poly Pressure (opcode 0xA) — §7.4.3. 32-bit data.
    PolyPressure { channel: u8, note: u8, data: u32 },
    /// Per-Note Management (opcode 0xF) — §7.4.5. D = detach, S = reset.
    PerNoteManagement {
        channel: u8,
        note: u8,
        detach: bool,
        reset: bool,
    },
    /// Control Change (opcode 0xB) — §7.4.6. 32-bit data.
    ControlChange { channel: u8, index: u8, data: u32 },
    /// Program Change (opcode 0xC) — §7.4.9. Bank Select folded in via
    /// the Bank Valid (B) option flag.
    ProgramChange {
        channel: u8,
        bank_valid: bool,
        program: u8,
        bank_msb: u8,
        bank_lsb: u8,
    },
    /// Channel Pressure (opcode 0xD) — §7.4.10. 32-bit data.
    ChannelPressure { channel: u8, data: u32 },
    /// Pitch Bend (opcode 0xE) — §7.4.11. Bipolar, centred at
    /// 0x8000_0000.
    PitchBend { channel: u8, data: u32 },
}

impl Midi2ChannelVoice {
    /// Decode a MIDI 2.0 Channel Voice packet (MT 0x4, 2 words).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::Midi2ChannelVoice {
            return Err(Error::invalid("UMP: not a MIDI 2.0 Channel Voice message"));
        }
        let words = p.words();
        if words.len() < 2 {
            return Err(Error::invalid("UMP MIDI 2.0 CV: needs 2 words"));
        }
        let w0 = words[0];
        let w1 = words[1];
        let ch = p.channel();
        // Index field = word-0 bits 8..15; its low byte and high byte.
        let index_hi = ((w0 >> 8) & 0xFF) as u8; // first index byte (note / bank / CC index)
        let index_lo = (w0 & 0xFF) as u8; // second index byte (per-note ctrl idx / RPN-NRPN idx / attr type)
        match p.opcode() {
            0x0 => Ok(Midi2ChannelVoice::RegisteredPerNoteController {
                channel: ch,
                note: index_hi & 0x7F,
                index: index_lo,
                data: w1,
            }),
            0x1 => Ok(Midi2ChannelVoice::AssignablePerNoteController {
                channel: ch,
                note: index_hi & 0x7F,
                index: index_lo,
                data: w1,
            }),
            0x2 => Ok(Midi2ChannelVoice::RegisteredController {
                channel: ch,
                bank: index_hi & 0x7F,
                index: index_lo & 0x7F,
                data: w1,
            }),
            0x3 => Ok(Midi2ChannelVoice::AssignableController {
                channel: ch,
                bank: index_hi & 0x7F,
                index: index_lo & 0x7F,
                data: w1,
            }),
            0x4 => Ok(Midi2ChannelVoice::RelativeRegisteredController {
                channel: ch,
                bank: index_hi & 0x7F,
                index: index_lo & 0x7F,
                data: w1 as i32,
            }),
            0x5 => Ok(Midi2ChannelVoice::RelativeAssignableController {
                channel: ch,
                bank: index_hi & 0x7F,
                index: index_lo & 0x7F,
                data: w1 as i32,
            }),
            0x6 => Ok(Midi2ChannelVoice::PerNotePitchBend {
                channel: ch,
                note: index_hi & 0x7F,
                data: w1,
            }),
            0x8 => Ok(Midi2ChannelVoice::NoteOff {
                channel: ch,
                note: index_hi & 0x7F,
                attribute_type: index_lo,
                velocity: (w1 >> 16) as u16,
                attribute: (w1 & 0xFFFF) as u16,
            }),
            0x9 => Ok(Midi2ChannelVoice::NoteOn {
                channel: ch,
                note: index_hi & 0x7F,
                attribute_type: index_lo,
                velocity: (w1 >> 16) as u16,
                attribute: (w1 & 0xFFFF) as u16,
            }),
            0xA => Ok(Midi2ChannelVoice::PolyPressure {
                channel: ch,
                note: index_hi & 0x7F,
                data: w1,
            }),
            0xB => Ok(Midi2ChannelVoice::ControlChange {
                channel: ch,
                index: index_hi & 0x7F,
                data: w1,
            }),
            0xC => Ok(Midi2ChannelVoice::ProgramChange {
                channel: ch,
                // Option flags occupy word-0 bits 0..7; Bank Valid (B)
                // is bit 0 (Figure 64).
                bank_valid: (index_lo & 0x01) != 0,
                program: ((w1 >> 24) & 0x7F) as u8,
                bank_msb: ((w1 >> 8) & 0x7F) as u8,
                bank_lsb: (w1 & 0x7F) as u8,
            }),
            0xD => Ok(Midi2ChannelVoice::ChannelPressure {
                channel: ch,
                data: w1,
            }),
            0xE => Ok(Midi2ChannelVoice::PitchBend {
                channel: ch,
                data: w1,
            }),
            0xF => Ok(Midi2ChannelVoice::PerNoteManagement {
                channel: ch,
                note: index_hi & 0x7F,
                // Option flags in word-0 bits 0..7: D = bit 1, S = bit 0
                // (Figure 51).
                detach: (index_lo & 0x02) != 0,
                reset: (index_lo & 0x01) != 0,
            }),
            other => Err(Error::invalid(format!(
                "UMP MIDI 2.0 CV: unknown opcode {other:#x}"
            ))),
        }
    }

    /// The channel (0..15) this message addresses.
    #[must_use]
    pub fn channel(&self) -> u8 {
        use Midi2ChannelVoice::*;
        match *self {
            RegisteredPerNoteController { channel, .. }
            | AssignablePerNoteController { channel, .. }
            | RegisteredController { channel, .. }
            | AssignableController { channel, .. }
            | RelativeRegisteredController { channel, .. }
            | RelativeAssignableController { channel, .. }
            | PerNotePitchBend { channel, .. }
            | NoteOff { channel, .. }
            | NoteOn { channel, .. }
            | PolyPressure { channel, .. }
            | PerNoteManagement { channel, .. }
            | ControlChange { channel, .. }
            | ProgramChange { channel, .. }
            | ChannelPressure { channel, .. }
            | PitchBend { channel, .. } => channel,
        }
    }

    /// Encode to a MIDI 2.0 Channel Voice packet on the given `group`.
    #[must_use]
    pub fn encode(&self, group: u8) -> Ump {
        use Midi2ChannelVoice::*;
        let (opcode, ix_hi, ix_lo, w1) = match *self {
            RegisteredPerNoteController {
                note, index, data, ..
            } => (0x0u8, note & 0x7F, index, data),
            AssignablePerNoteController {
                note, index, data, ..
            } => (0x1, note & 0x7F, index, data),
            RegisteredController {
                bank, index, data, ..
            } => (0x2, bank & 0x7F, index & 0x7F, data),
            AssignableController {
                bank, index, data, ..
            } => (0x3, bank & 0x7F, index & 0x7F, data),
            RelativeRegisteredController {
                bank, index, data, ..
            } => (0x4, bank & 0x7F, index & 0x7F, data as u32),
            RelativeAssignableController {
                bank, index, data, ..
            } => (0x5, bank & 0x7F, index & 0x7F, data as u32),
            PerNotePitchBend { note, data, .. } => (0x6, note & 0x7F, 0, data),
            NoteOff {
                note,
                attribute_type,
                velocity,
                attribute,
                ..
            } => (
                0x8,
                note & 0x7F,
                attribute_type,
                (u32::from(velocity) << 16) | u32::from(attribute),
            ),
            NoteOn {
                note,
                attribute_type,
                velocity,
                attribute,
                ..
            } => (
                0x9,
                note & 0x7F,
                attribute_type,
                (u32::from(velocity) << 16) | u32::from(attribute),
            ),
            PolyPressure { note, data, .. } => (0xA, note & 0x7F, 0, data),
            ControlChange { index, data, .. } => (0xB, index & 0x7F, 0, data),
            ProgramChange {
                bank_valid,
                program,
                bank_msb,
                bank_lsb,
                ..
            } => (
                0xC,
                0,
                u8::from(bank_valid),
                (u32::from(program & 0x7F) << 24)
                    | (u32::from(bank_msb & 0x7F) << 8)
                    | u32::from(bank_lsb & 0x7F),
            ),
            ChannelPressure { data, .. } => (0xD, 0, 0, data),
            PitchBend { data, .. } => (0xE, 0, 0, data),
            PerNoteManagement {
                note, detach, reset, ..
            } => (
                0xF,
                note & 0x7F,
                (u8::from(detach) << 1) | u8::from(reset),
                0,
            ),
        };
        let status = (opcode << 4) | (self.channel() & 0x0F);
        let w0 = (0x4u32 << 28)
            | (u32::from(group & 0x0F) << 24)
            | (u32::from(status) << 16)
            | (u32::from(ix_hi) << 8)
            | u32::from(ix_lo);
        Ump::from_parts([w0, w1, 0, 0], 2)
    }
}

#[cfg(test)]
mod midi2_tests {
    use super::*;

    fn ump(words: &[u32]) -> Ump {
        Ump::from_words(words).unwrap()
    }

    #[test]
    fn note_on_decodes_16bit_velocity_and_attribute() {
        // MT4, group 1, status 0x90 (note on ch0), note 0x40, attr type
        // 0x03 (pitch 7.9), velocity 0xFFFF, attribute 0x1234.
        let p = ump(&[0x4190_4003, 0xFFFF_1234]);
        let m = Midi2ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi2ChannelVoice::NoteOn {
                channel: 0,
                note: 0x40,
                attribute_type: 0x03,
                velocity: 0xFFFF,
                attribute: 0x1234,
            }
        );
        assert_eq!(m.encode(1), p);
    }

    #[test]
    fn program_change_bank_valid_flag() {
        // MT4, status 0xC0, option flags = B(bit0)=1, program 0x20,
        // bank msb 0x01, bank lsb 0x02.
        let p = ump(&[0x40C0_0001, 0x2000_0102]);
        let m = Midi2ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi2ChannelVoice::ProgramChange {
                channel: 0,
                bank_valid: true,
                program: 0x20,
                bank_msb: 0x01,
                bank_lsb: 0x02,
            }
        );
        assert_eq!(m.encode(0), p);
    }

    #[test]
    fn registered_controller_rpn_fields() {
        // MT4, status 0x20, bank 0x00, index 0x00 (RPN Pitch Bend Range),
        // data 0x1234_5678.
        let p = ump(&[0x4020_0000, 0x1234_5678]);
        let m = Midi2ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi2ChannelVoice::RegisteredController {
                channel: 0,
                bank: 0,
                index: 0,
                data: 0x1234_5678,
            }
        );
        assert_eq!(m.encode(0), p);
    }

    #[test]
    fn per_note_management_flags() {
        // MT4, status 0xF0, note 0x3C, option flags D=1 S=1.
        let p = ump(&[0x40F0_3C03, 0x0000_0000]);
        let m = Midi2ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi2ChannelVoice::PerNoteManagement {
                channel: 0,
                note: 0x3C,
                detach: true,
                reset: true,
            }
        );
        assert_eq!(m.encode(0), p);
    }

    #[test]
    fn relative_registered_controller_is_signed() {
        // data 0xFFFF_FFFF == -1 two's complement.
        let p = ump(&[0x4040_0001, 0xFFFF_FFFF]);
        let m = Midi2ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi2ChannelVoice::RelativeRegisteredController {
                channel: 0,
                bank: 0,
                index: 1,
                data: -1,
            }
        );
        assert_eq!(m.encode(0), p);
    }

    #[test]
    fn pitch_bend_centre_round_trips() {
        let p = ump(&[0x40E0_0000, 0x8000_0000]);
        let m = Midi2ChannelVoice::decode(&p).unwrap();
        assert_eq!(
            m,
            Midi2ChannelVoice::PitchBend {
                channel: 0,
                data: 0x8000_0000,
            }
        );
        assert_eq!(m.encode(0), p);
    }
}
