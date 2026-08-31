//! UMP Data messages — System Exclusive (7-bit, MT 0x3, §7.7),
//! System Exclusive 8 (8-bit, MT 0x5, §7.8), the Mixed Data Set
//! (MT 0x5 statuses 0x8/0x9, §7.9), and the 16-bit Manufacturer ID
//! encoding shared by SysEx8 / MDS (§7.10).
//!
//! The MIDI 1.0 bracketing bytes 0xF0 / 0xF7 are **not** carried in the
//! UMP Format (§7.7): a SysEx payload is the bytes *between* them,
//! split across one or more packets whose 4-bit Status field plays the
//! Complete / Start / Continue / End role (Tables 18 / 19).
//!
//! * [`Sysex7`] — one 64-bit packet carrying up to 6 seven-bit bytes.
//! * [`Data128Message`] — one 128-bit packet: SysEx8 (up to 13 data
//!   bytes plus the mandatory Stream ID), the §7.8.1 abort marker, or
//!   a Mixed Data Set Header / Payload chunk.
//! * [`sysex7_packets`] / [`sysex8_packets`] — split a full payload
//!   into a spec-conformant packet run; [`Sysex7Assembler`] /
//!   [`Sysex8Assembler`] reassemble one.

use oxideav_core::{Error, Result};

use super::packet::{MessageType, Ump};

/// The 4-bit Status role shared by SysEx7 and SysEx8 packets
/// (Table 18 / Table 19).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataFormat {
    /// 0x0 — complete message in one UMP.
    Complete,
    /// 0x1 — start of a message spanning two or more UMPs.
    Start,
    /// 0x2 — continuation (there may be many in one message).
    Continue,
    /// 0x3 — end of a multi-UMP message.
    End,
}

impl DataFormat {
    /// Decode the low two bits of the status nibble.
    #[must_use]
    pub fn from_bits(v: u8) -> Self {
        match v & 0x3 {
            0 => DataFormat::Complete,
            1 => DataFormat::Start,
            2 => DataFormat::Continue,
            _ => DataFormat::End,
        }
    }

    /// The raw status-nibble value.
    #[must_use]
    pub fn bits(self) -> u8 {
        match self {
            DataFormat::Complete => 0,
            DataFormat::Start => 1,
            DataFormat::Continue => 2,
            DataFormat::End => 3,
        }
    }
}

/// Maximum payload bytes in one SysEx7 packet (§7.7 "# of bytes").
pub const SYSEX7_MAX_BYTES: usize = 6;
/// Maximum data bytes (after the Stream ID) in one SysEx8 packet
/// (§7.8: # of bytes counts the Stream ID plus 0..=13 data bytes).
pub const SYSEX8_MAX_BYTES: usize = 13;
/// Payload bytes carried by one Mixed Data Set Payload UMP (§7.9).
pub const MDS_PAYLOAD_BYTES: usize = 14;

/// One System Exclusive (7-bit) packet — MT 0x3, 64-bit (§7.7,
/// Figure 84). Carries up to [`SYSEX7_MAX_BYTES`] payload bytes; each
/// byte is 7-bit (high bit zero). The 0xF0/0xF7 brackets are omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sysex7 {
    /// The Group this packet is addressed to.
    pub group: u8,
    /// Complete / Start / Continue / End role (Table 18).
    pub format: DataFormat,
    /// 0..=6 payload bytes (7-bit each). Unused wire bytes are zero.
    pub data: Vec<u8>,
}

impl Sysex7 {
    /// Decode a Data-64 packet (MT 0x3, 2 words).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::Data64 {
            return Err(Error::invalid("UMP: not a Data 64 (SysEx7) message"));
        }
        let w = p.words();
        if w.len() < 2 {
            return Err(Error::invalid("UMP SysEx7: needs 2 words"));
        }
        let status = ((w[0] >> 20) & 0x0F) as u8;
        if status > 0x3 {
            return Err(Error::invalid(format!(
                "UMP SysEx7: reserved status {status:#x}"
            )));
        }
        let n = ((w[0] >> 16) & 0x0F) as usize;
        if n > SYSEX7_MAX_BYTES {
            return Err(Error::invalid(format!(
                "UMP SysEx7: # of bytes {n} exceeds 6"
            )));
        }
        let all = [
            ((w[0] >> 8) & 0x7F) as u8,
            (w[0] & 0x7F) as u8,
            ((w[1] >> 24) & 0x7F) as u8,
            ((w[1] >> 16) & 0x7F) as u8,
            ((w[1] >> 8) & 0x7F) as u8,
            (w[1] & 0x7F) as u8,
        ];
        Ok(Sysex7 {
            group: p.group().unwrap_or(0),
            format: DataFormat::from_bits(status),
            data: all[..n].to_vec(),
        })
    }

    /// Encode back into a 2-word Data-64 packet. At most
    /// [`SYSEX7_MAX_BYTES`] bytes are carried; data bytes are masked
    /// to 7 bits per §7.7.
    #[must_use]
    pub fn encode(&self) -> Ump {
        let n = self.data.len().min(SYSEX7_MAX_BYTES);
        let mut b = [0u8; 6];
        for (i, &x) in self.data[..n].iter().enumerate() {
            b[i] = x & 0x7F;
        }
        let w0 = 0x3000_0000
            | (u32::from(self.group & 0x0F) << 24)
            | (u32::from(self.format.bits()) << 20)
            | ((n as u32) << 16)
            | (u32::from(b[0]) << 8)
            | u32::from(b[1]);
        let w1 = (u32::from(b[2]) << 24)
            | (u32::from(b[3]) << 16)
            | (u32::from(b[4]) << 8)
            | u32::from(b[5]);
        Ump::from_parts([w0, w1, 0, 0], 2)
    }
}

/// Split a full SysEx payload (the bytes between 0xF0 and 0xF7,
/// exclusive) into a §7.7 packet run: one Complete packet, or
/// Start / Continue* / End. Every byte must be 7-bit.
pub fn sysex7_packets(group: u8, payload: &[u8]) -> Result<Vec<Ump>> {
    if let Some(bad) = payload.iter().find(|&&b| b > 0x7F) {
        return Err(Error::invalid(format!(
            "UMP SysEx7: payload byte {bad:#04x} has the high bit set"
        )));
    }
    let chunks: Vec<&[u8]> = if payload.is_empty() {
        vec![&[][..]]
    } else {
        payload.chunks(SYSEX7_MAX_BYTES).collect()
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
            Sysex7 {
                group,
                format,
                data: c.to_vec(),
            }
            .encode()
        })
        .collect())
}

/// Reassembles a SysEx7 payload from its packet run (§7.7).
#[derive(Debug, Default)]
pub struct Sysex7Assembler {
    buf: Vec<u8>,
    in_progress: bool,
}

impl Sysex7Assembler {
    /// Fresh assembler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a Start has been seen without its End yet.
    #[must_use]
    pub fn in_progress(&self) -> bool {
        self.in_progress
    }

    /// Discard any partial message (§7.7.1: any interspersed
    /// non-continuation UMP on the Group terminates the message).
    pub fn abort(&mut self) {
        self.buf.clear();
        self.in_progress = false;
    }

    /// Feed one packet. Returns the completed payload on a Complete or
    /// End packet, `None` while the message is still in progress.
    pub fn push(&mut self, p: &Sysex7) -> Option<Vec<u8>> {
        match p.format {
            DataFormat::Complete => {
                self.abort();
                Some(p.data.clone())
            }
            DataFormat::Start => {
                self.buf.clear();
                self.buf.extend_from_slice(&p.data);
                self.in_progress = true;
                None
            }
            DataFormat::Continue => {
                if self.in_progress {
                    self.buf.extend_from_slice(&p.data);
                }
                None
            }
            DataFormat::End => {
                if !self.in_progress {
                    return None;
                }
                self.buf.extend_from_slice(&p.data);
                self.in_progress = false;
                Some(std::mem::take(&mut self.buf))
            }
        }
    }
}

/// A decoded Data-128 packet (MT 0x5) — SysEx8 (§7.8) or Mixed Data
/// Set (§7.9).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Data128Message {
    /// System Exclusive 8 (statuses 0x0..=0x3) — all 8 bits of every
    /// data byte are significant. `data` excludes the Stream ID.
    Sysex8 {
        /// The Group this packet is addressed to.
        group: u8,
        /// Complete / Start / Continue / End role (Table 19).
        format: DataFormat,
        /// Stream ID interleaving simultaneous SysEx8 messages (0 for
        /// devices supporting a single stream).
        stream_id: u8,
        /// 0..=13 data bytes.
        data: Vec<u8>,
    },
    /// §7.8.1 abort marker: an End UMP whose `# of bytes` field is 0xF
    /// — previous data is incomplete or of unknown quality.
    Sysex8Abort {
        /// The Group this packet is addressed to.
        group: u8,
        /// Stream ID of the aborted message.
        stream_id: u8,
    },
    /// Mixed Data Set Header (status 0x8) — §7.9, Figure 86.
    MixedDataSetHeader {
        /// The Group this chunk is addressed to.
        group: u8,
        /// MDS ID tying the chunks of one message together (0..=15).
        mds_id: u8,
        /// Size of this chunk in bytes, including the 16 header bytes.
        valid_bytes_in_chunk: u16,
        /// Number of chunks in the data set (0 = unknown / streaming).
        num_chunks: u16,
        /// This chunk's 1-based number.
        chunk_num: u16,
        /// 16-bit Manufacturer ID (§7.10) — see
        /// [`manufacturer_id_to_16bit`].
        manufacturer_id: u16,
        /// Device ID (0xFFFF = "all call").
        device_id: u16,
        /// Sub ID #1 (defined by the Universal SysEx ID in use).
        sub_id_1: u16,
        /// Sub ID #2.
        sub_id_2: u16,
    },
    /// Mixed Data Set Payload (status 0x9) — 14 payload bytes,
    /// zero-padded in the final UMP of a chunk.
    MixedDataSetPayload {
        /// The Group this chunk is addressed to.
        group: u8,
        /// MDS ID of the parent Mixed Data Set message.
        mds_id: u8,
        /// The 14 payload bytes (§7.9 pads the tail with zeros).
        data: [u8; MDS_PAYLOAD_BYTES],
    },
    /// A reserved MT 0x5 status — kept raw so readers can skip it.
    Unknown {
        /// The full raw packet.
        raw: Ump,
    },
}

impl Data128Message {
    /// Decode a Data-128 packet (MT 0x5, 4 words).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::Data128 {
            return Err(Error::invalid("UMP: not a Data 128 message"));
        }
        let w = p.words();
        if w.len() < 4 {
            return Err(Error::invalid("UMP Data 128: needs 4 words"));
        }
        let group = p.group().unwrap_or(0);
        let status = ((w[0] >> 20) & 0x0F) as u8;
        let low = ((w[0] >> 16) & 0x0F) as u8;
        Ok(match status {
            0x0..=0x3 => {
                let stream_id = ((w[0] >> 8) & 0xFF) as u8;
                if low == 0xF {
                    if status != 0x3 {
                        return Err(Error::invalid(
                            "UMP SysEx8: # of bytes 0xF is only valid in an End UMP",
                        ));
                    }
                    return Ok(Data128Message::Sysex8Abort { group, stream_id });
                }
                let n = low as usize;
                if n == 0 {
                    return Err(Error::invalid(
                        "UMP SysEx8: # of bytes 0x0 is invalid (Stream ID is mandatory)",
                    ));
                }
                // Bytes count the Stream ID; data bytes follow it.
                let mut data = Vec::with_capacity(n - 1);
                for i in 0..(n - 1) {
                    data.push(byte_at(w, 3 + i));
                }
                Data128Message::Sysex8 {
                    group,
                    format: DataFormat::from_bits(status),
                    stream_id,
                    data,
                }
            }
            0x8 => Data128Message::MixedDataSetHeader {
                group,
                mds_id: low,
                valid_bytes_in_chunk: (w[0] & 0xFFFF) as u16,
                num_chunks: (w[1] >> 16) as u16,
                chunk_num: (w[1] & 0xFFFF) as u16,
                manufacturer_id: (w[2] >> 16) as u16,
                device_id: (w[2] & 0xFFFF) as u16,
                sub_id_1: (w[3] >> 16) as u16,
                sub_id_2: (w[3] & 0xFFFF) as u16,
            },
            0x9 => {
                let mut data = [0u8; MDS_PAYLOAD_BYTES];
                for (i, b) in data.iter_mut().enumerate() {
                    *b = byte_at(w, 2 + i);
                }
                Data128Message::MixedDataSetPayload {
                    group,
                    mds_id: low,
                    data,
                }
            }
            _ => Data128Message::Unknown { raw: *p },
        })
    }

    /// Encode back into a 4-word Data-128 packet.
    #[must_use]
    pub fn encode(&self) -> Ump {
        let mut words = [0u32; 4];
        match self {
            Data128Message::Sysex8 {
                group,
                format,
                stream_id,
                data,
            } => {
                let n = data.len().min(SYSEX8_MAX_BYTES);
                words[0] = 0x5000_0000
                    | (u32::from(group & 0x0F) << 24)
                    | (u32::from(format.bits()) << 20)
                    | (((n as u32) + 1) << 16)
                    | (u32::from(*stream_id) << 8);
                pack_from(&mut words, 3, &data[..n]);
            }
            Data128Message::Sysex8Abort { group, stream_id } => {
                words[0] = 0x5000_0000
                    | (u32::from(group & 0x0F) << 24)
                    | (u32::from(DataFormat::End.bits()) << 20)
                    | (0xF << 16)
                    | (u32::from(*stream_id) << 8);
            }
            Data128Message::MixedDataSetHeader {
                group,
                mds_id,
                valid_bytes_in_chunk,
                num_chunks,
                chunk_num,
                manufacturer_id,
                device_id,
                sub_id_1,
                sub_id_2,
            } => {
                words[0] = 0x5080_0000
                    | (u32::from(group & 0x0F) << 24)
                    | (u32::from(mds_id & 0x0F) << 16)
                    | u32::from(*valid_bytes_in_chunk);
                words[1] = (u32::from(*num_chunks) << 16) | u32::from(*chunk_num);
                words[2] = (u32::from(*manufacturer_id) << 16) | u32::from(*device_id);
                words[3] = (u32::from(*sub_id_1) << 16) | u32::from(*sub_id_2);
            }
            Data128Message::MixedDataSetPayload {
                group,
                mds_id,
                data,
            } => {
                words[0] = 0x5090_0000
                    | (u32::from(group & 0x0F) << 24)
                    | (u32::from(mds_id & 0x0F) << 16);
                pack_from(&mut words, 2, data);
            }
            Data128Message::Unknown { raw } => return *raw,
        }
        Ump::from_parts(words, 4)
    }
}

/// Extract the byte at big-endian byte offset `at` (0 = MSB of word 0).
fn byte_at(words: &[u32], at: usize) -> u8 {
    ((words[at / 4] >> (24 - 8 * (at % 4))) & 0xFF) as u8
}

/// OR `src` into `words` starting at big-endian byte offset `at`.
fn pack_from(words: &mut [u32; 4], at: usize, src: &[u8]) {
    for (i, &b) in src.iter().enumerate() {
        let off = at + i;
        words[off / 4] |= u32::from(b) << (24 - 8 * (off % 4));
    }
}

/// Split a full SysEx8 payload into a §7.8 packet run on one Stream ID.
pub fn sysex8_packets(group: u8, stream_id: u8, payload: &[u8]) -> Vec<Ump> {
    let chunks: Vec<&[u8]> = if payload.is_empty() {
        vec![&[][..]]
    } else {
        payload.chunks(SYSEX8_MAX_BYTES).collect()
    };
    let last = chunks.len() - 1;
    chunks
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
            Data128Message::Sysex8 {
                group,
                format,
                stream_id,
                data: c.to_vec(),
            }
            .encode()
        })
        .collect()
}

/// Reassembles one SysEx8 stream (a single Stream ID) — §7.8.
#[derive(Debug, Default)]
pub struct Sysex8Assembler {
    buf: Vec<u8>,
    in_progress: bool,
}

impl Sysex8Assembler {
    /// Fresh assembler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a Start has been seen without its End yet.
    #[must_use]
    pub fn in_progress(&self) -> bool {
        self.in_progress
    }

    /// Feed one packet's `(format, data)`. Returns the completed
    /// payload on Complete / End; `None` in progress. A
    /// [`Data128Message::Sysex8Abort`] should be routed to
    /// [`abort`](Self::abort) instead.
    pub fn push(&mut self, format: DataFormat, data: &[u8]) -> Option<Vec<u8>> {
        match format {
            DataFormat::Complete => {
                self.abort();
                Some(data.to_vec())
            }
            DataFormat::Start => {
                self.buf.clear();
                self.buf.extend_from_slice(data);
                self.in_progress = true;
                None
            }
            DataFormat::Continue => {
                if self.in_progress {
                    self.buf.extend_from_slice(data);
                }
                None
            }
            DataFormat::End => {
                if !self.in_progress {
                    return None;
                }
                self.buf.extend_from_slice(data);
                self.in_progress = false;
                Some(std::mem::take(&mut self.buf))
            }
        }
    }

    /// Discard any partial message (§7.8.1 unexpected end of data).
    pub fn abort(&mut self) {
        self.buf.clear();
        self.in_progress = false;
    }
}

/// Encode a MIDI 1.0 Manufacturer / Special / Universal System
/// Exclusive ID as the 16-bit MfrID used by SysEx8 and Mixed Data Set
/// messages (§7.10, Figure 87).
///
/// `id` is the MIDI 1.0 form: `[x, 0, 0]` for a 1-byte ID (including
/// the Special IDs 0x7D / 0x7E / 0x7F) or `[0, b2, b3]` for a 3-byte
/// ID.
#[must_use]
pub fn manufacturer_id_to_16bit(id: [u8; 3]) -> u16 {
    if id[0] != 0 {
        // 1-byte ID: high byte 0x00, low byte the 7-bit value.
        u16::from(id[0] & 0x7F)
    } else {
        // 3-byte ID (leading 0x00): byte 2 → high byte, byte 3 → low
        // byte with its most significant bit set high.
        (u16::from(id[1] & 0x7F) << 8) | u16::from(id[2] & 0x7F) | 0x0080
    }
}

/// Decode a §7.10 16-bit MfrID back to the MIDI 1.0 3-byte form
/// (`[x, 0, 0]` for 1-byte IDs, `[0, b2, b3]` for 3-byte IDs).
#[must_use]
pub fn manufacturer_id_from_16bit(mfr: u16) -> [u8; 3] {
    if mfr & 0x0080 != 0 {
        [0, ((mfr >> 8) & 0x7F) as u8, (mfr & 0x7F) as u8]
    } else {
        [(mfr & 0x7F) as u8, 0, 0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gm_system_on_is_one_complete_sysex7_packet() {
        // GM System On: F0 7E 7F 09 01 F7 → UMP payload 7E 7F 09 01.
        let packets = sysex7_packets(0, &[0x7E, 0x7F, 0x09, 0x01]).unwrap();
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].words(), &[0x3004_7E7F, 0x0901_0000]);
        let m = Sysex7::decode(&packets[0]).unwrap();
        assert_eq!(m.format, DataFormat::Complete);
        assert_eq!(m.data, vec![0x7E, 0x7F, 0x09, 0x01]);
    }

    #[test]
    fn sysex7_13_bytes_becomes_start_continue_end() {
        let payload: Vec<u8> = (0..13u8).collect();
        let packets = sysex7_packets(2, &payload).unwrap();
        assert_eq!(packets.len(), 3);
        let mut asm = Sysex7Assembler::new();
        let mut out = None;
        for p in &packets {
            out = asm.push(&Sysex7::decode(p).unwrap());
        }
        assert_eq!(out.as_deref(), Some(&payload[..]));
        // Start carries 6 bytes, End carries the single remainder.
        assert_eq!((packets[0].words()[0] >> 16) & 0xFF, 0x16);
        assert_eq!((packets[2].words()[0] >> 16) & 0xFF, 0x31);
    }

    #[test]
    fn sysex7_rejects_8bit_payload_bytes() {
        assert!(sysex7_packets(0, &[0x80]).is_err());
    }

    #[test]
    fn sysex7_interrupted_message_terminates() {
        let payload: Vec<u8> = (0..13u8).collect();
        let packets = sysex7_packets(0, &payload).unwrap();
        let mut asm = Sysex7Assembler::new();
        asm.push(&Sysex7::decode(&packets[0]).unwrap());
        assert!(asm.in_progress());
        // §7.7.1: a non-SysEx UMP on the Group terminates the message.
        asm.abort();
        assert!(!asm.in_progress());
        assert_eq!(asm.push(&Sysex7::decode(&packets[2]).unwrap()), None);
    }

    #[test]
    fn sysex8_round_trips_with_stream_id() {
        let payload: Vec<u8> = (0..30u8).map(|x| x.wrapping_mul(9)).collect();
        let packets = sysex8_packets(1, 7, &payload);
        assert_eq!(packets.len(), 3); // 13 + 13 + 4 bytes
        let mut asm = Sysex8Assembler::new();
        let mut out = None;
        for p in &packets {
            match Data128Message::decode(p).unwrap() {
                Data128Message::Sysex8 {
                    group,
                    format,
                    stream_id,
                    data,
                } => {
                    assert_eq!(group, 1);
                    assert_eq!(stream_id, 7);
                    out = asm.push(format, &data);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(out.as_deref(), Some(&payload[..]));
        // First word pins: MT5, group 1, Start, #bytes=14 (stream id + 13).
        assert_eq!(packets[0].words()[0] >> 16, 0x511E);
    }

    #[test]
    fn sysex8_abort_is_end_with_0xf_bytes() {
        let m = Data128Message::Sysex8Abort {
            group: 0,
            stream_id: 3,
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0x503F_0300);
        assert_eq!(Data128Message::decode(&p).unwrap(), m);
    }

    #[test]
    fn sysex8_zero_byte_count_is_invalid() {
        let p = Ump::from_words(&[0x5000_0000, 0, 0, 0]).unwrap();
        assert!(Data128Message::decode(&p).is_err());
    }

    #[test]
    fn mds_header_round_trips() {
        let m = Data128Message::MixedDataSetHeader {
            group: 4,
            mds_id: 0xA,
            valid_bytes_in_chunk: 0x0120,
            num_chunks: 3,
            chunk_num: 1,
            manufacturer_id: 0x007E,
            device_id: 0xFFFF,
            sub_id_1: 0x0001,
            sub_id_2: 0x0002,
        };
        let p = m.encode();
        assert_eq!(
            p.words(),
            &[0x548A_0120, 0x0003_0001, 0x007E_FFFF, 0x0001_0002]
        );
        assert_eq!(Data128Message::decode(&p).unwrap(), m);
    }

    #[test]
    fn mds_payload_round_trips() {
        let mut data = [0u8; MDS_PAYLOAD_BYTES];
        for (i, b) in data.iter_mut().enumerate() {
            *b = 0xF0 | i as u8;
        }
        let m = Data128Message::MixedDataSetPayload {
            group: 0,
            mds_id: 0xA,
            data,
        };
        let p = m.encode();
        assert_eq!(p.words()[0] >> 16, 0x509A);
        assert_eq!(Data128Message::decode(&p).unwrap(), m);
    }

    #[test]
    fn manufacturer_id_16bit_translations_pin_table20() {
        // Table 20 Special IDs.
        assert_eq!(manufacturer_id_to_16bit([0x7D, 0, 0]), 0x007D);
        assert_eq!(manufacturer_id_to_16bit([0x7E, 0, 0]), 0x007E);
        assert_eq!(manufacturer_id_to_16bit([0x7F, 0, 0]), 0x007F);
        // 3-byte ID: high bit of the LOW byte marks the long form.
        let long = manufacturer_id_to_16bit([0x00, 0x21, 0x09]);
        assert_eq!(long, 0x2189);
        assert_eq!(manufacturer_id_from_16bit(long), [0x00, 0x21, 0x09]);
        assert_eq!(manufacturer_id_from_16bit(0x007D), [0x7D, 0, 0]);
    }
}
