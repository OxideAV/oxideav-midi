// Default-mode translation between MIDI 1.0 and MIDI 2.0 Channel Voice
// messages (spec Appendix D.2 / D.3).
//
// Included into message.rs so it shares the module's imports. The
// Default Translation Mode (§D, mandatory for conformance) scales data
// values to the full destination range using the Min-Center-Max
// primitives in `super::scaling`.
//
// Scope: this covers the single-message-in / single-message-out
// translations of D.3 (1.0->2.0) and D.2 (2.0->1.0). The RPN/NRPN
// compound-sequence reassembly (D.3.3, where CC 6/38/98/99/100/101 fold
// into one Registered/Assignable Controller) is inherently stateful
// across multiple input messages and is intentionally left to a higher
// layer; the direct Registered/Assignable Controller messages still
// translate here.

use super::scaling;

impl Midi1ChannelVoice {
    /// Translate this MIDI 1.0 Channel Voice message to its MIDI 2.0
    /// equivalent under the Default Translation Mode (§D.3).
    ///
    /// Returns `None` for messages that the Default Translation Mode
    /// declines to translate as a standalone message:
    ///
    /// * Control Change indices 0 / 32 (Bank Select), 6 / 38 (Data
    ///   Entry), and 98 / 99 / 100 / 101 (NRPN/RPN selectors) — these
    ///   participate in compound sequences that fold into MIDI 2.0
    ///   Program Change or Registered/Assignable Controller messages
    ///   (§D.3.3, §D.3.4) and so are not translated in isolation.
    /// * Increment / Decrement (CC 96 / 97) — translated only as part of
    ///   an RPN/NRPN sequence (§D.3.3).
    #[must_use]
    pub fn to_midi2(&self) -> Option<Midi2ChannelVoice> {
        match *self {
            Midi1ChannelVoice::NoteOff {
                channel,
                note,
                velocity,
            } => Some(Midi2ChannelVoice::NoteOff {
                channel,
                note,
                attribute_type: 0,
                velocity: scaling::scale_7_to_16(velocity),
                attribute: 0,
            }),
            Midi1ChannelVoice::NoteOn {
                channel,
                note,
                velocity,
            } => {
                // §D.3.1: Note On velocity 0x00 is special — it is a Note
                // Off. Translate to a MIDI 2.0 Note Off with velocity
                // 0x8000.
                if velocity == 0 {
                    Some(Midi2ChannelVoice::NoteOff {
                        channel,
                        note,
                        attribute_type: 0,
                        velocity: 0x8000,
                        attribute: 0,
                    })
                } else {
                    Some(Midi2ChannelVoice::NoteOn {
                        channel,
                        note,
                        attribute_type: 0,
                        velocity: scaling::scale_7_to_16(velocity),
                        attribute: 0,
                    })
                }
            }
            Midi1ChannelVoice::PolyPressure {
                channel,
                note,
                data,
            } => Some(Midi2ChannelVoice::PolyPressure {
                channel,
                note,
                data: scaling::scale_7_to_32(data),
            }),
            Midi1ChannelVoice::ControlChange {
                channel,
                index,
                data,
            } => {
                // §D.3.3 / §D.3.4 / "Bank Select Control Change": these
                // controllers do not translate as standalone messages.
                match index {
                    0 | 6 | 32 | 38 | 96 | 97 | 98 | 99 | 100 | 101 => None,
                    _ => Some(Midi2ChannelVoice::ControlChange {
                        channel,
                        index,
                        data: scaling::scale_7_to_32(data),
                    }),
                }
            }
            Midi1ChannelVoice::ProgramChange { channel, program } => {
                // §D.3.4: with no Bank Select information available, set
                // Bank Valid = 0 and zero the bank fields.
                Some(Midi2ChannelVoice::ProgramChange {
                    channel,
                    bank_valid: false,
                    program,
                    bank_msb: 0,
                    bank_lsb: 0,
                })
            }
            Midi1ChannelVoice::ChannelPressure { channel, data } => {
                Some(Midi2ChannelVoice::ChannelPressure {
                    channel,
                    data: scaling::scale_7_to_32(data),
                })
            }
            Midi1ChannelVoice::PitchBend { channel, lsb, msb } => {
                // §D.3.6: MIDI 1.0 pitch bend is LSB-first; combine into a
                // 14-bit value then upscale 14 -> 32.
                let value14 = u16::from(lsb & 0x7F) | (u16::from(msb & 0x7F) << 7);
                Some(Midi2ChannelVoice::PitchBend {
                    channel,
                    data: scaling::scale_14_to_32(value14),
                })
            }
        }
    }
}

impl Midi2ChannelVoice {
    /// Translate this MIDI 2.0 Channel Voice message to its MIDI 1.0
    /// equivalent under the Default Translation Mode (§D.2).
    ///
    /// Returns `None` for MIDI 2.0 messages with no MIDI 1.0
    /// representation (per-note controllers, relative controllers,
    /// per-note pitch bend, per-note management — §D.2.8). A Program
    /// Change with Bank Valid set translates to a single MIDI 1.0
    /// Program Change here; the preceding Bank Select Control Changes a
    /// full translator would also emit are left to a higher layer.
    #[must_use]
    pub fn to_midi1(&self) -> Option<Midi1ChannelVoice> {
        match *self {
            Midi2ChannelVoice::NoteOff {
                channel,
                note,
                velocity,
                ..
            } => Some(Midi1ChannelVoice::NoteOff {
                channel,
                note,
                velocity: scaling::scale_16_to_7(velocity),
            }),
            Midi2ChannelVoice::NoteOn {
                channel,
                note,
                velocity,
                ..
            } => {
                // §D.2.1 / §7.4.2: a translated velocity of 0 would read
                // as a Note Off, so replace it with 1.
                let v = scaling::scale_16_to_7(velocity);
                Some(Midi1ChannelVoice::NoteOn {
                    channel,
                    note,
                    velocity: if v == 0 { 1 } else { v },
                })
            }
            Midi2ChannelVoice::PolyPressure {
                channel,
                note,
                data,
            } => Some(Midi1ChannelVoice::PolyPressure {
                channel,
                note,
                data: scaling::scale_32_to_7(data),
            }),
            Midi2ChannelVoice::ControlChange {
                channel,
                index,
                data,
            } => Some(Midi1ChannelVoice::ControlChange {
                channel,
                index,
                data: scaling::scale_32_to_7(data),
            }),
            Midi2ChannelVoice::ProgramChange {
                channel, program, ..
            } => Some(Midi1ChannelVoice::ProgramChange { channel, program }),
            Midi2ChannelVoice::ChannelPressure { channel, data } => {
                Some(Midi1ChannelVoice::ChannelPressure {
                    channel,
                    data: scaling::scale_32_to_7(data),
                })
            }
            Midi2ChannelVoice::PitchBend { channel, data } => {
                let value14 = scaling::scale_32_to_14(data);
                Some(Midi1ChannelVoice::PitchBend {
                    channel,
                    lsb: (value14 & 0x7F) as u8,
                    msb: ((value14 >> 7) & 0x7F) as u8,
                })
            }
            // §D.2.8 messages that cannot be translated to MIDI 1.0.
            Midi2ChannelVoice::RegisteredPerNoteController { .. }
            | Midi2ChannelVoice::AssignablePerNoteController { .. }
            | Midi2ChannelVoice::RegisteredController { .. }
            | Midi2ChannelVoice::AssignableController { .. }
            | Midi2ChannelVoice::RelativeRegisteredController { .. }
            | Midi2ChannelVoice::RelativeAssignableController { .. }
            | Midi2ChannelVoice::PerNotePitchBend { .. }
            | Midi2ChannelVoice::PerNoteManagement { .. } => None,
        }
    }
}

#[cfg(test)]
mod translate_tests {
    use super::*;

    #[test]
    fn note_on_upscales_velocity_7_to_16() {
        let m1 = Midi1ChannelVoice::NoteOn {
            channel: 2,
            note: 60,
            velocity: 0x7F,
        };
        assert_eq!(
            m1.to_midi2(),
            Some(Midi2ChannelVoice::NoteOn {
                channel: 2,
                note: 60,
                attribute_type: 0,
                velocity: 0xFFFF,
                attribute: 0,
            })
        );
    }

    #[test]
    fn note_on_velocity_zero_becomes_note_off_0x8000() {
        // §D.3.1.
        let m1 = Midi1ChannelVoice::NoteOn {
            channel: 0,
            note: 64,
            velocity: 0,
        };
        assert_eq!(
            m1.to_midi2(),
            Some(Midi2ChannelVoice::NoteOff {
                channel: 0,
                note: 64,
                attribute_type: 0,
                velocity: 0x8000,
                attribute: 0,
            })
        );
    }

    #[test]
    fn midi2_note_on_velocity_floor_is_one() {
        // §7.4.2 / §D.2.1: a downscaled velocity of 0 is bumped to 1.
        let m2 = Midi2ChannelVoice::NoteOn {
            channel: 0,
            note: 64,
            attribute_type: 0,
            velocity: 0x0001, // downscales to 0
            attribute: 0,
        };
        assert_eq!(
            m2.to_midi1(),
            Some(Midi1ChannelVoice::NoteOn {
                channel: 0,
                note: 64,
                velocity: 1,
            })
        );
    }

    #[test]
    fn control_change_round_trips_lossless() {
        // §D.1.1: 1.0 -> 2.0 -> 1.0 is lossless for a non-special CC.
        for data in 0u8..=0x7F {
            let m1 = Midi1ChannelVoice::ControlChange {
                channel: 1,
                index: 74, // not a special controller
                data,
            };
            let back = m1.to_midi2().unwrap().to_midi1().unwrap();
            assert_eq!(back, m1, "data={data}");
        }
    }

    #[test]
    fn pitch_bend_round_trips_lossless() {
        for v in (0u16..=0x3FFF).step_by(37) {
            let m1 = Midi1ChannelVoice::PitchBend {
                channel: 0,
                lsb: (v & 0x7F) as u8,
                msb: ((v >> 7) & 0x7F) as u8,
            };
            let back = m1.to_midi2().unwrap().to_midi1().unwrap();
            assert_eq!(back, m1, "v={v}");
        }
    }

    #[test]
    fn special_control_changes_do_not_translate_standalone() {
        for index in [0u8, 6, 32, 38, 96, 97, 98, 99, 100, 101] {
            let m1 = Midi1ChannelVoice::ControlChange {
                channel: 0,
                index,
                data: 5,
            };
            assert_eq!(m1.to_midi2(), None, "index={index}");
        }
    }

    #[test]
    fn program_change_no_bank_sets_bank_invalid() {
        let m1 = Midi1ChannelVoice::ProgramChange {
            channel: 3,
            program: 42,
        };
        assert_eq!(
            m1.to_midi2(),
            Some(Midi2ChannelVoice::ProgramChange {
                channel: 3,
                bank_valid: false,
                program: 42,
                bank_msb: 0,
                bank_lsb: 0,
            })
        );
    }

    #[test]
    fn midi2_only_messages_have_no_midi1_form() {
        let m2 = Midi2ChannelVoice::PerNotePitchBend {
            channel: 0,
            note: 60,
            data: 0x8000_0000,
        };
        assert_eq!(m2.to_midi1(), None);
    }

    #[test]
    fn channel_pressure_round_trips_lossless() {
        for data in 0u8..=0x7F {
            let m1 = Midi1ChannelVoice::ChannelPressure { channel: 5, data };
            let back = m1.to_midi2().unwrap().to_midi1().unwrap();
            assert_eq!(back, m1, "data={data}");
        }
    }
}
