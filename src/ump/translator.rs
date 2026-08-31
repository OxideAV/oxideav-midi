//! Stateful Default-Translation between the MIDI 1.0 and MIDI 2.0
//! Protocols — the compound sequences of M2-104 Appendix D that the
//! single-message layer (`Midi1ChannelVoice::to_midi2` /
//! `Midi2ChannelVoice::to_midi1`) deliberately leaves to a higher
//! layer:
//!
//! * **§D.3.3** — MIDI 1.0 CC 98/99/100/101 select an (N)RPN and
//!   CC 6/38 carry its data; a well-formed sequence folds into ONE
//!   MIDI 2.0 Registered/Assignable Controller message with the 14-bit
//!   value upscaled to 32 bits.
//! * **§D.3.4** — MIDI 1.0 Bank Select (CC 0/32) folds into the MIDI
//!   2.0 Program Change message's bank fields with Bank Valid set.
//! * **§D.2.3** — one MIDI 2.0 Registered/Assignable Controller
//!   expands to the four MIDI 1.0 CC messages (101/100 or 99/98, then
//!   6 and 38).
//! * **§D.2.4** — one MIDI 2.0 Program Change with Bank Valid expands
//!   to Bank Select MSB, Bank Select LSB, then Program Change.
//!
//! [`Midi1ToMidi2Translator`] holds the per-channel running state
//! (selected parameter, held data bytes, current bank) and turns a
//! MIDI 1.0 message stream into MIDI 2.0 messages;
//! [`midi2_to_midi1_messages`] performs the stateless reverse
//! expansion.

use super::message::{Midi1ChannelVoice, Midi2ChannelVoice};
use super::scaling;

/// Number of MIDI channels tracked per translator.
const CHANNELS: usize = 16;

/// The currently selected parameter on one channel (§D.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParamSelect {
    /// `true` = RPN (CC 101/100), `false` = NRPN (CC 99/98).
    registered: bool,
    /// Parameter MSB → MIDI 2.0 `bank`.
    msb: u8,
    /// Parameter LSB → MIDI 2.0 `index`.
    lsb: u8,
}

impl ParamSelect {
    /// RPN/NRPN Null Function: MSB and LSB both 0x7F — not translated
    /// (§D.3.3).
    fn is_null(self) -> bool {
        self.msb == 0x7F && self.lsb == 0x7F
    }
}

/// Per-channel translation state.
#[derive(Debug, Clone, Copy, Default)]
struct ChannelState {
    /// Selector in progress. `partial_*` track which halves arrived so
    /// a selector is only considered formed once both bytes are known
    /// ("properly formed", §D.3.3).
    sel_registered: bool,
    sel_msb: Option<u8>,
    sel_lsb: Option<u8>,
    /// Latest CC 6 (Data Entry MSB) since the selector was formed.
    data_msb: Option<u8>,
    /// Latest CC 38 (Data Entry LSB).
    data_lsb: u8,
    /// Bank Select CC 0 / CC 32 (§D.3.4). `bank_msb` doubles as the
    /// "bank information available" flag.
    bank_msb: Option<u8>,
    bank_lsb: u8,
}

impl ChannelState {
    fn selector(&self) -> Option<ParamSelect> {
        match (self.sel_msb, self.sel_lsb) {
            (Some(msb), Some(lsb)) => Some(ParamSelect {
                registered: self.sel_registered,
                msb,
                lsb,
            }),
            _ => None,
        }
    }
}

/// Stateful MIDI 1.0 → MIDI 2.0 Default-Translation (Appendix D.3).
///
/// Feed every MIDI 1.0 Channel Voice message through
/// [`push`](Self::push); each call returns zero or more MIDI 2.0
/// messages. The single-message rules of §D.3.1–D.3.6 are inherited
/// from [`Midi1ChannelVoice::to_midi2`]; this layer adds the §D.3.3
/// RPN/NRPN folding and §D.3.4 Bank Select folding.
///
/// Per §D.3.3, an (N)RPN message is emitted when:
/// * a CC 38 is received (Data Entry MSB + LSB known),
/// * a subsequent CC 6 is received (the standard MIDI 1.0 controller
///   convention applies: a new MSB resets the held LSB to zero),
/// * a CC 98/99/100/101 arrives while data for the previous parameter
///   is still held un-emitted (the previous parameter is flushed).
///
/// The Null Function selector (MSB = LSB = 0x7F) is never translated.
#[derive(Debug, Default)]
pub struct Midi1ToMidi2Translator {
    channels: [ChannelState; CHANNELS],
}

impl Midi1ToMidi2Translator {
    /// A fresh translator with no held state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all held selectors, data bytes, and banks.
    pub fn reset(&mut self) {
        self.channels = [ChannelState::default(); CHANNELS];
    }

    /// Emit the (N)RPN for the current selector + held data, if any.
    fn emit_param(st: &ChannelState, channel: u8, out: &mut Vec<Midi2ChannelVoice>) {
        let Some(sel) = st.selector() else { return };
        if sel.is_null() {
            return;
        }
        let Some(msb) = st.data_msb else { return };
        let value14 = (u16::from(msb & 0x7F) << 7) | u16::from(st.data_lsb & 0x7F);
        let data = scaling::scale_up(u32::from(value14), 14, 32);
        out.push(if sel.registered {
            Midi2ChannelVoice::RegisteredController {
                channel,
                bank: sel.msb,
                index: sel.lsb,
                data,
            }
        } else {
            Midi2ChannelVoice::AssignableController {
                channel,
                bank: sel.msb,
                index: sel.lsb,
                data,
            }
        });
    }

    /// Feed one MIDI 1.0 Channel Voice message; returns the MIDI 2.0
    /// message(s) the Default Translation produces for it (possibly
    /// none while a compound sequence is still forming).
    pub fn push(&mut self, msg: &Midi1ChannelVoice) -> Vec<Midi2ChannelVoice> {
        let mut out = Vec::new();
        match *msg {
            Midi1ChannelVoice::ControlChange {
                channel,
                index,
                data,
            } => {
                let st = &mut self.channels[usize::from(channel & 0x0F)];
                match index {
                    // Bank Select (§D.3.4): held for the next Program
                    // Change; never translated standalone. A new MSB
                    // resets the LSB (MIDI 1.0 controller-pair rule).
                    0 => {
                        st.bank_msb = Some(data & 0x7F);
                        st.bank_lsb = 0;
                    }
                    32 => st.bank_lsb = data & 0x7F,
                    // (N)RPN selectors (§D.3.3). A selector byte both
                    // ends the previous parameter (flushing held data)
                    // and starts forming the next one.
                    98..=101 => {
                        let registered = index >= 100;
                        let is_msb = index == 99 || index == 101;
                        if st.data_msb.is_some() {
                            let flushed = *st;
                            Self::emit_param(&flushed, channel, &mut out);
                            st.data_msb = None;
                            st.data_lsb = 0;
                        }
                        if st.sel_registered != registered {
                            st.sel_msb = None;
                            st.sel_lsb = None;
                        }
                        st.sel_registered = registered;
                        if is_msb {
                            st.sel_msb = Some(data & 0x7F);
                        } else {
                            st.sel_lsb = Some(data & 0x7F);
                        }
                        // A new selector abandons old data bytes.
                        st.data_msb = None;
                        st.data_lsb = 0;
                    }
                    // Data Entry MSB (§D.3.3): a subsequent CC 6 emits
                    // immediately; the first waits for CC 38 or a
                    // flush trigger.
                    6 => {
                        if st.selector().is_some() {
                            let subsequent = st.data_msb.is_some();
                            st.data_msb = Some(data & 0x7F);
                            st.data_lsb = 0;
                            if subsequent {
                                let now = *st;
                                Self::emit_param(&now, channel, &mut out);
                            }
                        }
                    }
                    // Data Entry LSB (§D.3.3): emit with the held MSB.
                    38 => {
                        if st.selector().is_some() && st.data_msb.is_some() {
                            st.data_lsb = data & 0x7F;
                            let now = *st;
                            Self::emit_param(&now, channel, &mut out);
                        }
                    }
                    // Everything else (including Increment/Decrement
                    // CC 96/97, §D.3.3) is a single-message translation.
                    _ => out.extend(msg.to_midi2()),
                }
            }
            Midi1ChannelVoice::ProgramChange { channel, program } => {
                let st = &self.channels[usize::from(channel & 0x0F)];
                out.push(match st.bank_msb {
                    // §D.3.4: fold the current valid Bank Select in.
                    Some(msb) => Midi2ChannelVoice::ProgramChange {
                        channel,
                        bank_valid: true,
                        program,
                        bank_msb: msb,
                        bank_lsb: st.bank_lsb,
                    },
                    None => Midi2ChannelVoice::ProgramChange {
                        channel,
                        bank_valid: false,
                        program,
                        bank_msb: 0,
                        bank_lsb: 0,
                    },
                });
            }
            _ => out.extend(msg.to_midi2()),
        }
        out
    }
}

/// Expand one MIDI 2.0 Channel Voice message into its MIDI 1.0
/// Default-Translation sequence (Appendix D.2).
///
/// * Registered / Assignable Controller → four CC messages (§D.2.3,
///   Figure 97): CC 101+100 (RPN) or CC 99+98 (NRPN) selecting the
///   parameter, then CC 6 / CC 38 carrying the 32→14-bit downscaled
///   value.
/// * Program Change with Bank Valid → Bank Select MSB, Bank Select
///   LSB, Program Change, in that order (§D.2.4).
/// * §D.2.8 messages with no MIDI 1.0 equivalent (per-note
///   controllers/pitch bend/management, relative controllers) return
///   an empty vector.
/// * Everything else is the single-message translation of
///   [`Midi2ChannelVoice::to_midi1`].
#[must_use]
pub fn midi2_to_midi1_messages(msg: &Midi2ChannelVoice) -> Vec<Midi1ChannelVoice> {
    match *msg {
        Midi2ChannelVoice::RegisteredController {
            channel,
            bank,
            index,
            data,
        }
        | Midi2ChannelVoice::AssignableController {
            channel,
            bank,
            index,
            data,
        } => {
            let registered = matches!(msg, Midi2ChannelVoice::RegisteredController { .. });
            let value14 = scaling::scale_down(data, 32, 14) as u16;
            let (cc_msb, cc_lsb) = if registered { (101, 100) } else { (99, 98) };
            vec![
                Midi1ChannelVoice::ControlChange {
                    channel,
                    index: cc_msb,
                    data: bank & 0x7F,
                },
                Midi1ChannelVoice::ControlChange {
                    channel,
                    index: cc_lsb,
                    data: index & 0x7F,
                },
                Midi1ChannelVoice::ControlChange {
                    channel,
                    index: 6,
                    data: ((value14 >> 7) & 0x7F) as u8,
                },
                Midi1ChannelVoice::ControlChange {
                    channel,
                    index: 38,
                    data: (value14 & 0x7F) as u8,
                },
            ]
        }
        Midi2ChannelVoice::ProgramChange {
            channel,
            bank_valid: true,
            program,
            bank_msb,
            bank_lsb,
        } => vec![
            Midi1ChannelVoice::ControlChange {
                channel,
                index: 0,
                data: bank_msb & 0x7F,
            },
            Midi1ChannelVoice::ControlChange {
                channel,
                index: 32,
                data: bank_lsb & 0x7F,
            },
            Midi1ChannelVoice::ProgramChange { channel, program },
        ],
        _ => msg.to_midi1().into_iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(channel: u8, index: u8, data: u8) -> Midi1ChannelVoice {
        Midi1ChannelVoice::ControlChange {
            channel,
            index,
            data,
        }
    }

    #[test]
    fn rpn_sequence_folds_into_one_registered_controller() {
        // Classic Pitch Bend Range = 2 semitones: CC101=0 CC100=0
        // CC6=2 CC38=0.
        let mut t = Midi1ToMidi2Translator::new();
        assert!(t.push(&cc(0, 101, 0)).is_empty());
        assert!(t.push(&cc(0, 100, 0)).is_empty());
        assert!(t.push(&cc(0, 6, 2)).is_empty()); // waits for CC38/trigger
        let out = t.push(&cc(0, 38, 0));
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::RegisteredController {
                channel: 0,
                bank: 0,
                index: 0,
                data: scaling::scale_up(2 << 7, 14, 32),
            }]
        );
    }

    #[test]
    fn nrpn_sequence_folds_into_assignable_controller() {
        let mut t = Midi1ToMidi2Translator::new();
        t.push(&cc(3, 99, 0x10));
        t.push(&cc(3, 98, 0x20));
        t.push(&cc(3, 6, 0x40));
        let out = t.push(&cc(3, 38, 0x01));
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::AssignableController {
                channel: 3,
                bank: 0x10,
                index: 0x20,
                data: scaling::scale_up((0x40 << 7) | 0x01, 14, 32),
            }]
        );
    }

    #[test]
    fn subsequent_cc6_emits_immediately() {
        // §D.3.3 second trigger: after a first CC 6, another CC 6 for
        // the same parameter emits at once (LSB reset to 0).
        let mut t = Midi1ToMidi2Translator::new();
        t.push(&cc(0, 101, 0));
        t.push(&cc(0, 100, 0));
        t.push(&cc(0, 6, 2));
        let out = t.push(&cc(0, 6, 3));
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::RegisteredController {
                channel: 0,
                bank: 0,
                index: 0,
                data: scaling::scale_up(3 << 7, 14, 32),
            }]
        );
    }

    #[test]
    fn new_selector_flushes_pending_data() {
        // §D.3.3 third trigger: CC 100 for a new RPN flushes the held
        // CC 6 of the previous one.
        let mut t = Midi1ToMidi2Translator::new();
        t.push(&cc(0, 101, 0));
        t.push(&cc(0, 100, 0));
        t.push(&cc(0, 6, 12));
        let out = t.push(&cc(0, 100, 1)); // start RPN 0x0001
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::RegisteredController {
                channel: 0,
                bank: 0,
                index: 0,
                data: scaling::scale_up(12 << 7, 14, 32),
            }]
        );
        // And the new parameter still works.
        let out = t.push(&cc(0, 38, 5));
        assert!(out.is_empty(), "no CC6 yet for the new parameter");
        t.push(&cc(0, 6, 1));
        let out = t.push(&cc(0, 38, 5));
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::RegisteredController {
                channel: 0,
                bank: 0,
                index: 1,
                data: scaling::scale_up((1 << 7) | 5, 14, 32),
            }]
        );
    }

    #[test]
    fn null_function_is_not_translated() {
        let mut t = Midi1ToMidi2Translator::new();
        t.push(&cc(0, 101, 0x7F));
        t.push(&cc(0, 100, 0x7F));
        assert!(t.push(&cc(0, 6, 10)).is_empty());
        assert!(t.push(&cc(0, 38, 10)).is_empty());
    }

    #[test]
    fn lone_data_entry_does_not_translate() {
        // §D.3.3: individual CC 6/38 without a properly formed
        // selector do not translate.
        let mut t = Midi1ToMidi2Translator::new();
        assert!(t.push(&cc(0, 6, 10)).is_empty());
        assert!(t.push(&cc(0, 38, 10)).is_empty());
        // Half-formed selector (MSB only) is not "properly formed".
        t.push(&cc(0, 101, 0));
        assert!(t.push(&cc(0, 6, 10)).is_empty());
        assert!(t.push(&cc(0, 38, 10)).is_empty());
    }

    #[test]
    fn increment_decrement_translate_as_plain_cc() {
        // §D.3.3: CC 96/97 are translated to MIDI 2.0 Control Change,
        // never to Relative Registered/Assignable Controllers.
        let mut t = Midi1ToMidi2Translator::new();
        let out = t.push(&cc(0, 96, 1));
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::ControlChange {
                channel: 0,
                index: 96,
                data: scaling::scale_7_to_32(1),
            }]
        );
    }

    #[test]
    fn bank_select_folds_into_program_change() {
        // §D.3.4 with bank: CC0, CC32, PC → one MIDI 2.0 PC, B=1.
        let mut t = Midi1ToMidi2Translator::new();
        assert!(t.push(&cc(0, 0, 0x78)).is_empty());
        assert!(t.push(&cc(0, 32, 0x01)).is_empty());
        let out = t.push(&Midi1ChannelVoice::ProgramChange {
            channel: 0,
            program: 40,
        });
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::ProgramChange {
                channel: 0,
                bank_valid: true,
                program: 40,
                bank_msb: 0x78,
                bank_lsb: 0x01,
            }]
        );
    }

    #[test]
    fn program_change_without_bank_sets_bank_invalid() {
        let mut t = Midi1ToMidi2Translator::new();
        let out = t.push(&Midi1ChannelVoice::ProgramChange {
            channel: 5,
            program: 7,
        });
        assert_eq!(
            out,
            vec![Midi2ChannelVoice::ProgramChange {
                channel: 5,
                bank_valid: false,
                program: 7,
                bank_msb: 0,
                bank_lsb: 0,
            }]
        );
    }

    #[test]
    fn registered_controller_expands_to_four_ccs() {
        // §D.2.3 Figure 97.
        let m = Midi2ChannelVoice::RegisteredController {
            channel: 2,
            bank: 0,
            index: 0,
            data: scaling::scale_up(2 << 7, 14, 32),
        };
        assert_eq!(
            midi2_to_midi1_messages(&m),
            vec![cc(2, 101, 0), cc(2, 100, 0), cc(2, 6, 2), cc(2, 38, 0)]
        );
    }

    #[test]
    fn assignable_controller_expands_to_nrpn_ccs() {
        let m = Midi2ChannelVoice::AssignableController {
            channel: 0,
            bank: 0x10,
            index: 0x20,
            data: scaling::scale_up((0x40 << 7) | 0x01, 14, 32),
        };
        assert_eq!(
            midi2_to_midi1_messages(&m),
            vec![
                cc(0, 99, 0x10),
                cc(0, 98, 0x20),
                cc(0, 6, 0x40),
                cc(0, 38, 0x01)
            ]
        );
    }

    #[test]
    fn program_change_with_bank_expands_to_three_messages() {
        // §D.2.4: Bank MSB, Bank LSB, Program Change — in that order.
        let m = Midi2ChannelVoice::ProgramChange {
            channel: 1,
            bank_valid: true,
            program: 5,
            bank_msb: 0x78,
            bank_lsb: 0x00,
        };
        assert_eq!(
            midi2_to_midi1_messages(&m),
            vec![
                cc(1, 0, 0x78),
                cc(1, 32, 0x00),
                Midi1ChannelVoice::ProgramChange {
                    channel: 1,
                    program: 5
                },
            ]
        );
    }

    #[test]
    fn per_note_messages_expand_to_nothing() {
        // §D.2.8.
        let m = Midi2ChannelVoice::PerNotePitchBend {
            channel: 0,
            note: 60,
            data: 0x8000_0000,
        };
        assert!(midi2_to_midi1_messages(&m).is_empty());
    }

    #[test]
    fn rpn_round_trip_1_to_2_to_1() {
        // MIDI1 RPN sequence → MIDI2 → back to the same four CCs.
        let mut t = Midi1ToMidi2Translator::new();
        t.push(&cc(0, 101, 0));
        t.push(&cc(0, 100, 2));
        t.push(&cc(0, 6, 0x33));
        let m2 = t.push(&cc(0, 38, 0x11));
        assert_eq!(m2.len(), 1);
        assert_eq!(
            midi2_to_midi1_messages(&m2[0]),
            vec![
                cc(0, 101, 0),
                cc(0, 100, 2),
                cc(0, 6, 0x33),
                cc(0, 38, 0x11)
            ]
        );
    }
}
