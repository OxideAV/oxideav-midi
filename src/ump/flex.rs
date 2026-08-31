//! Flex Data messages (MT 0xD, 128-bit) — M2-104 §7.5.
//!
//! Flex Data carries the "sequence metadata" vocabulary that replaced
//! SMF v1 Meta Events in the MIDI Clip File world: Set Tempo, Set Time
//! Signature, Set Metronome, Set Key Signature, Set Chord Name
//! (Status Bank 0x00), plus the UTF-8 text families — Metadata Text
//! (Status Bank 0x01: project / song / clip names, copyright, credits,
//! recording date & location) and Performance Text (Status Bank 0x02:
//! lyrics, ruby, and their BCP 47 language tags).
//!
//! General format (§7.5.1, Figure 69):
//!
//! ```text
//!   word 0: mt=0xD(4) | group(4) | form(2) | addrs(2) | channel(4)
//!           | status bank(8) | status(8)
//!   words 1..3: 12 payload bytes
//! ```
//!
//! `form` plays the Complete / Start / Continue / End role (Table 9,
//! same semantics as SysEx — reused here as
//! [`DataFormat`](super::data::DataFormat)); a message spans at most 32
//! UMPs (§7.5.1). `addrs` selects Channel or Group addressing
//! ([`FlexAddress`], Table 10).

use oxideav_core::{Error, Result};

use super::data::DataFormat;
use super::packet::{MessageType, Ump};

/// Flex Data Address field (§7.5.1, Table 10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlexAddress {
    /// 0x0 — the message is sent to one MIDI Channel (0..=15).
    Channel(u8),
    /// 0x1 — the message is sent to the whole Group.
    Group,
}

impl FlexAddress {
    fn addrs_bits(self) -> u8 {
        match self {
            FlexAddress::Channel(_) => 0,
            FlexAddress::Group => 1,
        }
    }

    fn channel_bits(self) -> u8 {
        match self {
            FlexAddress::Channel(c) => c & 0x0F,
            FlexAddress::Group => 0,
        }
    }
}

/// Status values for the Status Bank 0x01 (Metadata Text) and 0x02
/// (Performance Text) families — §7.5.9.1, Table 16.
pub mod text_status {
    /// Bank 0x01 — Metadata Text.
    pub const BANK_METADATA: u8 = 0x01;
    /// Bank 0x02 — Performance Text Events.
    pub const BANK_PERFORMANCE: u8 = 0x02;

    /// 0x01/0x00 Unknown Metadata Text Event.
    pub const UNKNOWN_METADATA: u8 = 0x00;
    /// 0x01/0x01 Project Name (Group-addressed).
    pub const PROJECT_NAME: u8 = 0x01;
    /// 0x01/0x02 Composition (Song) Name.
    pub const COMPOSITION_NAME: u8 = 0x02;
    /// 0x01/0x03 MIDI Clip Name (Group-addressed).
    pub const MIDI_CLIP_NAME: u8 = 0x03;
    /// 0x01/0x04 Copyright Notice.
    pub const COPYRIGHT_NOTICE: u8 = 0x04;
    /// 0x01/0x05 Composer Name.
    pub const COMPOSER_NAME: u8 = 0x05;
    /// 0x01/0x06 Lyricist Name.
    pub const LYRICIST_NAME: u8 = 0x06;
    /// 0x01/0x07 Arranger Name.
    pub const ARRANGER_NAME: u8 = 0x07;
    /// 0x01/0x08 Publisher Name.
    pub const PUBLISHER_NAME: u8 = 0x08;
    /// 0x01/0x09 Primary Performer Name.
    pub const PRIMARY_PERFORMER_NAME: u8 = 0x09;
    /// 0x01/0x0A Accompanying Performer Name.
    pub const ACCOMPANYING_PERFORMER_NAME: u8 = 0x0A;
    /// 0x01/0x0B Recording/Concert Date (ISO 8601, §7.5.9.2).
    pub const RECORDING_DATE: u8 = 0x0B;
    /// 0x01/0x0C Recording/Concert Location.
    pub const RECORDING_LOCATION: u8 = 0x0C;

    /// 0x02/0x00 Unknown Performance Text Event.
    pub const UNKNOWN_PERFORMANCE: u8 = 0x00;
    /// 0x02/0x01 Lyric Data (§7.5.10).
    pub const LYRICS: u8 = 0x01;
    /// 0x02/0x02 Lyric Language — BCP 47 tag (§7.5.11).
    pub const LYRICS_LANGUAGE: u8 = 0x02;
    /// 0x02/0x03 Ruby Data (§7.5.12).
    pub const RUBY: u8 = 0x03;
    /// 0x02/0x04 Ruby Language — BCP 47 tag (§7.5.13).
    pub const RUBY_LANGUAGE: u8 = 0x04;
}

/// Payload bytes carried per Flex Data UMP (words 1..3).
pub const FLEX_PAYLOAD_BYTES: usize = 12;
/// A Flex Data Message spans at most 32 UMPs (§7.5.1), so a text
/// payload caps at 32 × 12 bytes.
pub const MAX_FLEX_TEXT_BYTES: usize = 32 * FLEX_PAYLOAD_BYTES;

/// One chord alteration: an Alteration Type nibble (0 none, 1 add,
/// 2 subtract, 3 raise, 4 lower) paired with the chord degree it
/// applies to (§7.5.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChordAlteration {
    /// Alteration Type (4-bit): 0 = no alteration.
    pub alteration_type: u8,
    /// Degree of the chord being altered (1 = root, 3 = third, …).
    pub degree: u8,
}

impl ChordAlteration {
    fn byte(self) -> u8 {
        ((self.alteration_type & 0x0F) << 4) | (self.degree & 0x0F)
    }

    fn from_byte(b: u8) -> Self {
        ChordAlteration {
            alteration_type: b >> 4,
            degree: b & 0x0F,
        }
    }
}

/// The payload of a Set Chord Name message (§7.5.8, Figure 75).
///
/// Note fields use the §7.5.7 encoding: 0x0 unknown, 0x1..=0x7 = A..G
/// (for `bass_note`, 0x0 = same as the chord tonic). Sharps/flats
/// fields are 4-bit two's complement counts (+ = sharps, − = flats;
/// −8 = unknown/non-standard, and for the bass = "same as tonic").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChordName {
    /// Sharps (+) / flats (−) applied to the chord tonic.
    pub tonic_sharps_flats: i8,
    /// Chord tonic note (0x0 unknown, 0x1..=0x7 = A..G).
    pub tonic: u8,
    /// Chord Type (Table 14: 0x00 no chord, 0x01 Major, … 0x1B 7sus4).
    pub chord_type: u8,
    /// Up to 4 alterations to the main chord.
    pub alterations: [ChordAlteration; 4],
    /// Sharps/flats applied to the bass note (−8 = same as tonic).
    pub bass_sharps_flats: i8,
    /// Bass note (0x0 = same as the chord tonic note).
    pub bass_note: u8,
    /// Bass Chord Type (0x00 = no bass chord; else as Table 14).
    pub bass_chord_type: u8,
    /// Up to 2 alterations to the bass chord.
    pub bass_alterations: [ChordAlteration; 2],
}

/// Decode a 4-bit two's-complement field to a signed value (−8..=7).
fn nib_i8(v: u8) -> i8 {
    let v = (v & 0x0F) as i8;
    if v >= 8 {
        v - 16
    } else {
        v
    }
}

/// Encode a signed value (−8..=7) as a 4-bit two's-complement nibble.
fn i8_nib(v: i8) -> u8 {
    (v as u8) & 0x0F
}

/// A decoded Flex Data message (MT 0xD) — §7.5.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlexDataMessage {
    /// Set Tempo (bank 0x00, status 0x00) — §7.5.3. Group-addressed.
    SetTempo {
        /// The Group this message is addressed to.
        group: u8,
        /// Time per quarter note in 10-nanosecond units.
        ten_ns_per_quarter_note: u32,
    },
    /// Set Time Signature (bank 0x00, status 0x01) — §7.5.4.
    /// Group-addressed.
    SetTimeSignature {
        /// The Group this message is addressed to.
        group: u8,
        /// Beats per bar. The wire byte supports 1..=256 beats; the
        /// raw byte is kept (0 encodes 256 in the 8-bit field).
        numerator: u8,
        /// Denominator as a negative power of two (2 = quarter note,
        /// 3 = eighth note, …; 0 = non-standard).
        denominator: u8,
        /// Number of 1/32 notes in 24 MIDI Clocks (SMF v1 carry-over).
        number_of_32nd_notes: u8,
    },
    /// Set Metronome (bank 0x00, status 0x02) — §7.5.5. Group-addressed.
    SetMetronome {
        /// The Group this message is addressed to.
        group: u8,
        /// MIDI Clocks per primary click.
        clocks_per_primary_click: u8,
        /// Bar Accent Parts 1..3 (sum equals beats per bar; parts 2/3
        /// zero when the bar is undivided).
        bar_accents: [u8; 3],
        /// The two overlapping Number-of-Subdivision-Clicks fields
        /// (0 = no subdivision clicks).
        subdivision_clicks: [u8; 2],
    },
    /// Set Key Signature (bank 0x00, status 0x05) — §7.5.7.
    SetKeySignature {
        /// The Group this message is addressed to.
        group: u8,
        /// Channel or Group destination.
        address: FlexAddress,
        /// Sharps (+1..=+7) / flats (−1..=−7); 0 = no accidentals,
        /// −8 = unknown or non-standard.
        sharps_flats: i8,
        /// Tonic note (0x0 unknown, 0x1..=0x7 = A..G).
        tonic: u8,
    },
    /// Set Chord Name (bank 0x00, status 0x06) — §7.5.8.
    SetChordName {
        /// The Group this message is addressed to.
        group: u8,
        /// Channel or Group destination.
        address: FlexAddress,
        /// The chord description payload.
        chord: ChordName,
    },
    /// One packet of a text-family message (Status Banks 0x01 / 0x02)
    /// — §7.5.9. Text is UTF-8 without a BOM; 0x00 bytes pad the tail
    /// of a Complete/End UMP. A Lyric/Ruby Complete UMP that is all
    /// zeros is a melisma event (§7.5.10.1) — `bytes` is empty then.
    Text {
        /// The Group this message is addressed to.
        group: u8,
        /// Channel or Group destination.
        address: FlexAddress,
        /// Complete / Start / Continue / End role of this UMP.
        format: DataFormat,
        /// Status Bank (0x01 metadata, 0x02 performance text).
        status_bank: u8,
        /// Status within the bank ([`text_status`]).
        status: u8,
        /// Up to 12 text bytes in this UMP (trailing 0x00 stripped).
        bytes: Vec<u8>,
    },
    /// Any other (bank, status) — reserved by MMA/AMEI; kept raw.
    Unknown {
        /// The full raw packet.
        raw: Ump,
    },
}

/// Build word 0 of a Flex Data UMP.
fn word0(group: u8, form: DataFormat, address: FlexAddress, bank: u8, status: u8) -> u32 {
    0xD000_0000
        | (u32::from(group & 0x0F) << 24)
        | (u32::from(form.bits()) << 22)
        | (u32::from(address.addrs_bits()) << 20)
        | (u32::from(address.channel_bits()) << 16)
        | (u32::from(bank) << 8)
        | u32::from(status)
}

impl FlexDataMessage {
    /// Decode a Flex Data packet (MT 0xD, 4 words).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::FlexData {
            return Err(Error::invalid("UMP: not a Flex Data message"));
        }
        let w = p.words();
        if w.len() < 4 {
            return Err(Error::invalid("UMP Flex Data: needs 4 words"));
        }
        let group = p.group().unwrap_or(0);
        let form = DataFormat::from_bits(((w[0] >> 22) & 0x3) as u8);
        let addrs = ((w[0] >> 20) & 0x3) as u8;
        let channel = ((w[0] >> 16) & 0x0F) as u8;
        let bank = ((w[0] >> 8) & 0xFF) as u8;
        let status = (w[0] & 0xFF) as u8;
        let address = match addrs {
            0 => FlexAddress::Channel(channel),
            1 => FlexAddress::Group,
            other => {
                return Err(Error::invalid(format!(
                    "UMP Flex Data: reserved Address field value {other}"
                )));
            }
        };
        Ok(match (bank, status) {
            (0x00, 0x00) => FlexDataMessage::SetTempo {
                group,
                ten_ns_per_quarter_note: w[1],
            },
            (0x00, 0x01) => FlexDataMessage::SetTimeSignature {
                group,
                numerator: (w[1] >> 24) as u8,
                denominator: ((w[1] >> 16) & 0xFF) as u8,
                number_of_32nd_notes: ((w[1] >> 8) & 0xFF) as u8,
            },
            (0x00, 0x02) => FlexDataMessage::SetMetronome {
                group,
                clocks_per_primary_click: (w[1] >> 24) as u8,
                bar_accents: [
                    ((w[1] >> 16) & 0xFF) as u8,
                    ((w[1] >> 8) & 0xFF) as u8,
                    (w[1] & 0xFF) as u8,
                ],
                subdivision_clicks: [(w[2] >> 24) as u8, ((w[2] >> 16) & 0xFF) as u8],
            },
            (0x00, 0x05) => FlexDataMessage::SetKeySignature {
                group,
                address,
                sharps_flats: nib_i8((w[1] >> 28) as u8),
                tonic: ((w[1] >> 24) & 0x0F) as u8,
            },
            (0x00, 0x06) => FlexDataMessage::SetChordName {
                group,
                address,
                chord: ChordName {
                    tonic_sharps_flats: nib_i8((w[1] >> 28) as u8),
                    tonic: ((w[1] >> 24) & 0x0F) as u8,
                    chord_type: ((w[1] >> 16) & 0xFF) as u8,
                    alterations: [
                        ChordAlteration::from_byte(((w[1] >> 8) & 0xFF) as u8),
                        ChordAlteration::from_byte((w[1] & 0xFF) as u8),
                        ChordAlteration::from_byte((w[2] >> 24) as u8),
                        ChordAlteration::from_byte(((w[2] >> 16) & 0xFF) as u8),
                    ],
                    bass_sharps_flats: nib_i8((w[3] >> 28) as u8),
                    bass_note: ((w[3] >> 24) & 0x0F) as u8,
                    bass_chord_type: ((w[3] >> 16) & 0xFF) as u8,
                    bass_alterations: [
                        ChordAlteration::from_byte(((w[3] >> 8) & 0xFF) as u8),
                        ChordAlteration::from_byte((w[3] & 0xFF) as u8),
                    ],
                },
            },
            (0x01 | 0x02, _) => {
                let mut bytes: Vec<u8> = (0..FLEX_PAYLOAD_BYTES)
                    .map(|i| {
                        let off = 4 + i;
                        ((w[off / 4] >> (24 - 8 * (off % 4))) & 0xFF) as u8
                    })
                    .collect();
                while bytes.last() == Some(&0) {
                    bytes.pop();
                }
                FlexDataMessage::Text {
                    group,
                    address,
                    format: form,
                    status_bank: bank,
                    status,
                    bytes,
                }
            }
            _ => FlexDataMessage::Unknown { raw: *p },
        })
    }

    /// Encode back into a 4-word Flex Data packet. The "shall" fields
    /// of §7.5.3–7.5.5 (Format = Complete, Address = Group) are
    /// enforced on encode.
    #[must_use]
    pub fn encode(&self) -> Ump {
        let mut words = [0u32; 4];
        match self {
            FlexDataMessage::SetTempo {
                group,
                ten_ns_per_quarter_note,
            } => {
                words[0] = word0(*group, DataFormat::Complete, FlexAddress::Group, 0x00, 0x00);
                words[1] = *ten_ns_per_quarter_note;
            }
            FlexDataMessage::SetTimeSignature {
                group,
                numerator,
                denominator,
                number_of_32nd_notes,
            } => {
                words[0] = word0(*group, DataFormat::Complete, FlexAddress::Group, 0x00, 0x01);
                words[1] = (u32::from(*numerator) << 24)
                    | (u32::from(*denominator) << 16)
                    | (u32::from(*number_of_32nd_notes) << 8);
            }
            FlexDataMessage::SetMetronome {
                group,
                clocks_per_primary_click,
                bar_accents,
                subdivision_clicks,
            } => {
                words[0] = word0(*group, DataFormat::Complete, FlexAddress::Group, 0x00, 0x02);
                words[1] = (u32::from(*clocks_per_primary_click) << 24)
                    | (u32::from(bar_accents[0]) << 16)
                    | (u32::from(bar_accents[1]) << 8)
                    | u32::from(bar_accents[2]);
                words[2] = (u32::from(subdivision_clicks[0]) << 24)
                    | (u32::from(subdivision_clicks[1]) << 16);
            }
            FlexDataMessage::SetKeySignature {
                group,
                address,
                sharps_flats,
                tonic,
            } => {
                words[0] = word0(*group, DataFormat::Complete, *address, 0x00, 0x05);
                words[1] =
                    (u32::from(i8_nib(*sharps_flats)) << 28) | (u32::from(*tonic & 0x0F) << 24);
            }
            FlexDataMessage::SetChordName {
                group,
                address,
                chord,
            } => {
                words[0] = word0(*group, DataFormat::Complete, *address, 0x00, 0x06);
                words[1] = (u32::from(i8_nib(chord.tonic_sharps_flats)) << 28)
                    | (u32::from(chord.tonic & 0x0F) << 24)
                    | (u32::from(chord.chord_type) << 16)
                    | (u32::from(chord.alterations[0].byte()) << 8)
                    | u32::from(chord.alterations[1].byte());
                words[2] = (u32::from(chord.alterations[2].byte()) << 24)
                    | (u32::from(chord.alterations[3].byte()) << 16);
                words[3] = (u32::from(i8_nib(chord.bass_sharps_flats)) << 28)
                    | (u32::from(chord.bass_note & 0x0F) << 24)
                    | (u32::from(chord.bass_chord_type) << 16)
                    | (u32::from(chord.bass_alterations[0].byte()) << 8)
                    | u32::from(chord.bass_alterations[1].byte());
            }
            FlexDataMessage::Text {
                group,
                address,
                format,
                status_bank,
                status,
                bytes,
            } => {
                words[0] = word0(*group, *format, *address, *status_bank, *status);
                let n = bytes.len().min(FLEX_PAYLOAD_BYTES);
                for (i, &b) in bytes[..n].iter().enumerate() {
                    let off = 4 + i;
                    words[off / 4] |= u32::from(b) << (24 - 8 * (off % 4));
                }
            }
            FlexDataMessage::Unknown { raw } => return *raw,
        }
        Ump::from_parts(words, 4)
    }
}

/// Split a UTF-8 text into a §7.5.9 Flex Data text packet run
/// (Complete, or Start / Continue* / End; 12 bytes per UMP; at most
/// 32 UMPs).
pub fn flex_text_packets(
    group: u8,
    address: FlexAddress,
    status_bank: u8,
    status: u8,
    text: &str,
) -> Result<Vec<Ump>> {
    let bytes = text.as_bytes();
    if bytes.len() > MAX_FLEX_TEXT_BYTES {
        return Err(Error::invalid(format!(
            "UMP Flex Data: text longer than {MAX_FLEX_TEXT_BYTES} bytes ({} given)",
            bytes.len()
        )));
    }
    let chunks: Vec<&[u8]> = if bytes.is_empty() {
        vec![&[][..]]
    } else {
        bytes.chunks(FLEX_PAYLOAD_BYTES).collect()
    };
    let last = chunks.len() - 1;
    Ok(chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let format = if last == 0 {
                DataFormat::Complete
            } else if i == 0 {
                DataFormat::Start
            } else if i == last {
                DataFormat::End
            } else {
                DataFormat::Continue
            };
            FlexDataMessage::Text {
                group,
                address,
                format,
                status_bank,
                status,
                bytes: c.to_vec(),
            }
            .encode()
        })
        .collect())
}

/// Reassembles a multi-UMP Flex Data text payload (§7.5.9). Returns
/// the raw bytes so lyric semantics (trailing space = word end, CR =
/// line end, CR LF = paragraph end, empty Complete = melisma) stay
/// observable.
#[derive(Debug, Default)]
pub struct FlexTextAssembler {
    buf: Vec<u8>,
    in_progress: bool,
}

impl FlexTextAssembler {
    /// Fresh assembler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one packet's `(format, bytes)`. Returns the completed
    /// payload on Complete / End, `None` while in progress.
    pub fn push(&mut self, format: DataFormat, bytes: &[u8]) -> Option<Vec<u8>> {
        match format {
            DataFormat::Complete => {
                self.buf.clear();
                self.in_progress = false;
                Some(bytes.to_vec())
            }
            DataFormat::Start => {
                self.buf.clear();
                self.buf.extend_from_slice(bytes);
                self.in_progress = true;
                None
            }
            DataFormat::Continue => {
                if self.in_progress {
                    self.buf.extend_from_slice(bytes);
                }
                None
            }
            DataFormat::End => {
                if !self.in_progress {
                    return None;
                }
                self.buf.extend_from_slice(bytes);
                self.in_progress = false;
                Some(std::mem::take(&mut self.buf))
            }
        }
    }
}

/// Convert an SMF `FF 51` tempo (microseconds per quarter note) to the
/// Flex Data Set Tempo unit (10 ns per quarter note).
#[must_use]
pub fn usec_per_quarter_to_10ns(usec: u32) -> u32 {
    usec.saturating_mul(100)
}

/// Convert a Flex Data Set Tempo value (10 ns per quarter note) to the
/// SMF `FF 51` unit (microseconds per quarter note), rounding to the
/// nearest microsecond.
#[must_use]
pub fn ten_ns_per_quarter_to_usec(ten_ns: u32) -> u32 {
    (ten_ns + 50) / 100
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(m: &FlexDataMessage) {
        let p = m.encode();
        assert_eq!(p.len(), 4);
        assert_eq!(&FlexDataMessage::decode(&p).unwrap(), m);
    }

    #[test]
    fn set_tempo_pins_words() {
        // 120 BPM = 500_000 µs/qn = 50_000_000 × 10 ns.
        let m = FlexDataMessage::SetTempo {
            group: 0,
            ten_ns_per_quarter_note: 50_000_000,
        };
        let p = m.encode();
        // form=0, addrs=1 (Group), channel=0, bank 0, status 0.
        assert_eq!(p.words()[0], 0xD010_0000);
        assert_eq!(p.words()[1], 50_000_000);
        round_trip(&m);
    }

    #[test]
    fn set_time_signature_fields() {
        let m = FlexDataMessage::SetTimeSignature {
            group: 3,
            numerator: 6,
            denominator: 3, // eighth note
            number_of_32nd_notes: 8,
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0xD310_0001);
        assert_eq!(p.words()[1], 0x0603_0800);
        round_trip(&m);
    }

    #[test]
    fn set_metronome_matches_spec_example() {
        // Figure 73 first example: 4/4 with bar accent, no subdivisions
        // — 24 clocks per primary click, accent every 4th beat.
        let m = FlexDataMessage::SetMetronome {
            group: 0,
            clocks_per_primary_click: 24,
            bar_accents: [4, 0, 0],
            subdivision_clicks: [0, 0],
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0xD010_0002);
        assert_eq!(p.words()[1], 0x1804_0000);
        assert_eq!(p.words()[2], 0);
        round_trip(&m);
    }

    #[test]
    fn key_signature_two_complement_sharps_flats() {
        // Four flats, tonic D → D Flat (Table 12).
        let m = FlexDataMessage::SetKeySignature {
            group: 0,
            address: FlexAddress::Group,
            sharps_flats: -4,
            tonic: 0x4,
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0xD010_0005);
        assert_eq!(p.words()[1], 0xC400_0000);
        round_trip(&m);

        let unknown = FlexDataMessage::SetKeySignature {
            group: 0,
            address: FlexAddress::Channel(9),
            sharps_flats: -8,
            tonic: 0,
        };
        assert_eq!(unknown.encode().words()[0], 0xD009_0005);
        round_trip(&unknown);
    }

    #[test]
    fn chord_name_matches_spec_bb_min_example() {
        // Figure 76 first example: Set Chord Name B♭ Min —
        // t.sf=0xF (one flat), tonic 0x2 (B), chord type 0x07 (Minor),
        // no alterations, bass same as tonic (b.sf=0x8, bass 0x0).
        let m = FlexDataMessage::SetChordName {
            group: 0,
            address: FlexAddress::Channel(0),
            chord: ChordName {
                tonic_sharps_flats: -1,
                tonic: 0x2,
                chord_type: 0x07,
                alterations: [ChordAlteration::default(); 4],
                bass_sharps_flats: -8,
                bass_note: 0x0,
                bass_chord_type: 0x00,
                bass_alterations: [ChordAlteration::default(); 2],
            },
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0xD000_0006);
        assert_eq!(p.words()[1], 0xF207_0000);
        assert_eq!(p.words()[2], 0x0000_0000);
        assert_eq!(p.words()[3], 0x8000_0000);
        round_trip(&m);
    }

    #[test]
    fn chord_name_cmaj7_sharp11_example() {
        // Figure 76 third example: CMaj7 #11 — tonic C natural, Major
        // 7th, alteration 1 = raise (3) degree 11 (0xB).
        let m = FlexDataMessage::SetChordName {
            group: 0,
            address: FlexAddress::Channel(0),
            chord: ChordName {
                tonic_sharps_flats: 0,
                tonic: 0x3,
                chord_type: 0x03,
                alterations: [
                    ChordAlteration {
                        alteration_type: 3,
                        degree: 0xB,
                    },
                    ChordAlteration::default(),
                    ChordAlteration::default(),
                    ChordAlteration::default(),
                ],
                bass_sharps_flats: -8,
                bass_note: 0x0,
                bass_chord_type: 0x00,
                bass_alterations: [ChordAlteration::default(); 2],
            },
        };
        let p = m.encode();
        assert_eq!(p.words()[1], 0x0303_3B00);
        round_trip(&m);
    }

    #[test]
    fn short_text_is_one_complete_ump() {
        let packets = flex_text_packets(
            0,
            FlexAddress::Group,
            text_status::BANK_METADATA,
            text_status::MIDI_CLIP_NAME,
            "Intro",
        )
        .unwrap();
        assert_eq!(packets.len(), 1);
        match FlexDataMessage::decode(&packets[0]).unwrap() {
            FlexDataMessage::Text {
                format,
                status_bank,
                status,
                bytes,
                ..
            } => {
                assert_eq!(format, DataFormat::Complete);
                assert_eq!(status_bank, 0x01);
                assert_eq!(status, 0x03);
                assert_eq!(bytes, b"Intro");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn long_text_reassembles_across_umps() {
        let text = "A Copyright Notice long enough to span several Flex Data UMPs © 2026";
        let packets = flex_text_packets(
            1,
            FlexAddress::Group,
            text_status::BANK_METADATA,
            text_status::COPYRIGHT_NOTICE,
            text,
        )
        .unwrap();
        assert!(packets.len() > 2);
        let mut asm = FlexTextAssembler::new();
        let mut out = None;
        for p in &packets {
            match FlexDataMessage::decode(p).unwrap() {
                FlexDataMessage::Text { format, bytes, .. } => {
                    out = asm.push(format, &bytes);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(out.as_deref(), Some(text.as_bytes()));
    }

    #[test]
    fn lyric_melisma_is_empty_complete() {
        // §7.5.10.1: an all-zero Complete Lyric Data UMP is a melisma.
        let m = FlexDataMessage::Text {
            group: 0,
            address: FlexAddress::Channel(1),
            format: DataFormat::Complete,
            status_bank: text_status::BANK_PERFORMANCE,
            status: text_status::LYRICS,
            bytes: vec![],
        };
        let p = m.encode();
        assert_eq!(p.words(), &[0xD001_0201, 0, 0, 0]);
        round_trip(&m);
    }

    #[test]
    fn reserved_bank_survives_round_trip() {
        let raw = Ump::from_words(&[0xD010_7F42, 1, 2, 3]).unwrap();
        let m = FlexDataMessage::decode(&raw).unwrap();
        assert!(matches!(m, FlexDataMessage::Unknown { .. }));
        assert_eq!(m.encode(), raw);
    }

    #[test]
    fn tempo_unit_conversions() {
        assert_eq!(usec_per_quarter_to_10ns(500_000), 50_000_000);
        assert_eq!(ten_ns_per_quarter_to_usec(50_000_000), 500_000);
        // Round-to-nearest on non-multiples.
        assert_eq!(ten_ns_per_quarter_to_usec(50_000_049), 500_000);
        assert_eq!(ten_ns_per_quarter_to_usec(50_000_050), 500_001);
    }
}
