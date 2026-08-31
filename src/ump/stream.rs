//! UMP Stream Messages (MT 0xF) — M2-104 §7.1.
//!
//! UMP Stream Messages are addressed to the UMP Endpoint as a whole,
//! without a Group or Channel assignment (they are Groupless, §1.4).
//! All UMP Stream Messages are 128-bit (4-word) packets whose first
//! word carries a 2-bit Format field and a 10-bit Status field
//! (§7.1, Figure 10):
//!
//! ```text
//!   word 0:  mt=0xF(4) | format(2) | status(10) | data(16)
//!   words 1..3: per-status payload
//! ```
//!
//! This module models every Stream message defined by M2-104 v1.1.2
//! (Table 34): Endpoint Discovery / Info / Device Identity / Name /
//! Product Instance Id, Stream Configuration Request + Notification,
//! Function Block Discovery / Info / Name, and the Start of Clip /
//! End of Clip markers used by the MIDI Clip File (M2-116).
//!
//! Multi-packet text payloads (Endpoint Name, Product Instance Id,
//! Function Block Name) are carried as per-packet byte chunks plus the
//! §7.1 Format field; [`StreamTextAssembler`] reassembles a chunk
//! sequence into the full string, and the `*_packets` helpers split a
//! string into a spec-conformant packet run.

use oxideav_core::{Error, Result};

use super::packet::{MessageType, Ump};

/// The 2-bit Format field common to all UMP Stream Messages (§7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamFormat {
    /// 0x0 — complete message in one UMP.
    Complete,
    /// 0x1 — start of a message which spans two or more UMPs.
    Start,
    /// 0x2 — continuation of a message spanning three or more UMPs.
    Continue,
    /// 0x3 — end of a message which spans two or more UMPs.
    End,
}

impl StreamFormat {
    /// Decode the 2-bit field value.
    #[must_use]
    pub fn from_bits(v: u8) -> Self {
        match v & 0x3 {
            0 => StreamFormat::Complete,
            1 => StreamFormat::Start,
            2 => StreamFormat::Continue,
            _ => StreamFormat::End,
        }
    }

    /// The raw 2-bit field value.
    #[must_use]
    pub fn bits(self) -> u8 {
        match self {
            StreamFormat::Complete => 0,
            StreamFormat::Start => 1,
            StreamFormat::Continue => 2,
            StreamFormat::End => 3,
        }
    }
}

/// Endpoint Discovery Filter bitmap bits (§7.1.1, Figure 12). Each set
/// bit requests one individual notification in reply.
pub mod discovery_filter {
    /// e — request an Endpoint Info Notification.
    pub const ENDPOINT_INFO: u8 = 0x01;
    /// d — request a Device Identity Notification.
    pub const DEVICE_IDENTITY: u8 = 0x02;
    /// n — request an Endpoint Name Notification.
    pub const ENDPOINT_NAME: u8 = 0x04;
    /// i — request a Product Instance Id Notification.
    pub const PRODUCT_INSTANCE_ID: u8 = 0x08;
    /// s — request a Stream Configuration Notification.
    pub const STREAM_CONFIGURATION: u8 = 0x10;
    /// All five request bits.
    pub const ALL: u8 = 0x1F;
}

/// Function Block Discovery Filter bitmap bits (§7.1.7, Figure 21).
pub mod function_block_filter {
    /// i — request a Function Block Info Notification.
    pub const INFO: u8 = 0x01;
    /// n — request a Function Block Name Notification.
    pub const NAME: u8 = 0x02;
}

/// Stream Configuration Protocol field values (§7.1.6.2 / §7.1.6.3).
pub mod protocol {
    /// 0x01 — MIDI 1.0 Protocol.
    pub const MIDI1: u8 = 0x01;
    /// 0x02 — MIDI 2.0 Protocol.
    pub const MIDI2: u8 = 0x02;
}

/// Maximum UMP Endpoint Name length in bytes (§7.1.4).
pub const MAX_ENDPOINT_NAME_BYTES: usize = 98;
/// Maximum Product Instance Id length in bytes (§7.1.5).
pub const MAX_PRODUCT_INSTANCE_ID_BYTES: usize = 42;
/// Maximum Function Block Name length in bytes (§7.1.9).
pub const MAX_FUNCTION_BLOCK_NAME_BYTES: usize = 91;

/// A decoded UMP Stream Message (MT 0xF) — §7.1, Table 34.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UmpStreamMessage {
    /// Endpoint Discovery (status 0x00) — §7.1.1. Requests basic
    /// information about the UMP Endpoint; `filter` is the
    /// [`discovery_filter`] bitmap.
    EndpointDiscovery {
        /// UMP specification major version (0x01 for v1.1).
        ump_version_major: u8,
        /// UMP specification minor version (0x01 for v1.1).
        ump_version_minor: u8,
        /// [`discovery_filter`] bitmap of requested notifications.
        filter: u8,
    },
    /// Endpoint Info Notification (status 0x01) — §7.1.2.
    EndpointInfoNotification {
        /// UMP specification major version.
        ump_version_major: u8,
        /// UMP specification minor version.
        ump_version_minor: u8,
        /// S — the Endpoint's Function Blocks are static.
        static_function_blocks: bool,
        /// Number of Function Blocks (0..=32).
        num_function_blocks: u8,
        /// M2 — supports the MIDI 2.0 Protocol.
        midi2_capable: bool,
        /// M1 — supports the MIDI 1.0 Protocol.
        midi1_capable: bool,
        /// RXJR — supports receiving JR Timestamps.
        rx_jr: bool,
        /// TXJR — supports sending JR Timestamps.
        tx_jr: bool,
    },
    /// Device Identity Notification (status 0x02) — §7.1.3. Same data
    /// as the MIDI 1.0 Device Inquiry Universal SysEx reply.
    DeviceIdentityNotification {
        /// 3-byte System Exclusive ID of the manufacturer (1-byte IDs
        /// are `[id, 0, 0]`). Each byte is 7-bit.
        device_manufacturer: [u8; 3],
        /// 14-bit Device Family (LSB-first pair on the wire).
        device_family: u16,
        /// 14-bit Device Family Model Number (LSB-first on the wire).
        device_family_model: u16,
        /// 4 × 7-bit Software Revision Level bytes.
        software_revision: [u8; 4],
    },
    /// Endpoint Name Notification (status 0x03) — §7.1.4. One packet's
    /// worth (up to 14 bytes) of the UTF-8 endpoint name.
    EndpointNameNotification {
        /// Multi-packet Format role of this packet.
        format: StreamFormat,
        /// Up to 14 name bytes carried by this packet (trailing zero
        /// padding stripped on decode).
        bytes: Vec<u8>,
    },
    /// Product Instance Id Notification (status 0x04) — §7.1.5. One
    /// packet's worth (up to 14 bytes) of the ASCII instance id.
    ProductInstanceIdNotification {
        /// Multi-packet Format role of this packet.
        format: StreamFormat,
        /// Up to 14 id bytes carried by this packet.
        bytes: Vec<u8>,
    },
    /// Stream Configuration Request (status 0x05) — §7.1.6.2.
    StreamConfigurationRequest {
        /// Requested [`protocol`] (0x01 MIDI 1.0, 0x02 MIDI 2.0).
        protocol: u8,
        /// RXJR — the receiving Endpoint can expect JR Timestamps.
        rx_jr: bool,
        /// TXJR — the receiving Endpoint shall send JR Timestamps.
        tx_jr: bool,
    },
    /// Stream Configuration Notification (status 0x06) — §7.1.6.3.
    StreamConfigurationNotification {
        /// Declared [`protocol`] (0x01 MIDI 1.0, 0x02 MIDI 2.0).
        protocol: u8,
        /// RXJR — the sender expects to receive JR Timestamps.
        rx_jr: bool,
        /// TXJR — the sender will send JR Timestamps.
        tx_jr: bool,
    },
    /// Function Block Discovery (status 0x10) — §7.1.7. `block` 0xFF
    /// requests all Function Blocks; `filter` is the
    /// [`function_block_filter`] bitmap.
    FunctionBlockDiscovery {
        /// Function Block number to query (0x00..=0x1F, or 0xFF = all).
        block: u8,
        /// [`function_block_filter`] bitmap of requested notifications.
        filter: u8,
    },
    /// Function Block Info Notification (status 0x11) — §7.1.8.
    FunctionBlockInfoNotification {
        /// a — this Function Block is currently active.
        active: bool,
        /// Function Block number (0..=31).
        block: u8,
        /// 2-bit User Interface Hint (0b00 unknown, 0b01 receiver,
        /// 0b10 sender, 0b11 both).
        ui_hint: u8,
        /// 2-bit MIDI 1.0 field (0x00 not MIDI 1.0, 0x01 unrestricted,
        /// 0x02 restricted to 31.25 kbps).
        midi1_port: u8,
        /// 2-bit Direction (0b01 input, 0b10 output, 0b11 bidirectional).
        direction: u8,
        /// Lowest Group number that is a member of this Function Block.
        first_group: u8,
        /// Count of Groups spanned (1..=16).
        groups_spanned: u8,
        /// MIDI-CI Message Version/Format (0x00 = none or unknown).
        ci_version: u8,
        /// Max number of simultaneous SysEx8 streams supported.
        max_sysex8_streams: u8,
    },
    /// Function Block Name Notification (status 0x12) — §7.1.9. One
    /// packet's worth (up to 13 bytes) of the UTF-8 block name.
    FunctionBlockNameNotification {
        /// Multi-packet Format role of this packet.
        format: StreamFormat,
        /// Function Block number this name belongs to.
        block: u8,
        /// Up to 13 name bytes carried by this packet.
        bytes: Vec<u8>,
    },
    /// Start of Clip (status 0x20) — §7.1.10. First event in a MIDI
    /// Clip File's Clip Sequence Data.
    StartOfClip,
    /// End of Clip (status 0x21) — §7.1.11. Last event in a MIDI Clip
    /// File's Clip Sequence Data.
    EndOfClip,
    /// A Status value not defined by M2-104 v1.1.2 — kept raw so a
    /// stream reader can skip it (receivers ignore unknown statuses).
    Unknown {
        /// The 2-bit Format field.
        format: StreamFormat,
        /// The 10-bit Status field.
        status: u16,
        /// The full raw packet.
        raw: Ump,
    },
}

/// Build word 0 for a Stream message: `0xF` | format | status | data16.
fn word0(format: StreamFormat, status: u16, data16: u16) -> u32 {
    0xF000_0000
        | (u32::from(format.bits()) << 26)
        | (u32::from(status & 0x03FF) << 16)
        | u32::from(data16)
}

/// Pack big-endian bytes from `src` into `words[..]` starting at byte
/// offset `at` (offset 0 = the most significant byte of `words[0]`).
fn pack_bytes(words: &mut [u32; 4], at: usize, src: &[u8]) {
    for (i, &b) in src.iter().enumerate() {
        let off = at + i;
        let w = off / 4;
        let sh = 24 - 8 * (off % 4);
        words[w] |= u32::from(b) << sh;
    }
}

/// Extract the byte at big-endian byte offset `at` from the packet
/// words (offset 0 = most significant byte of word 0).
fn byte_at(words: &[u32], at: usize) -> u8 {
    let w = at / 4;
    let sh = 24 - 8 * (at % 4);
    ((words[w] >> sh) & 0xFF) as u8
}

/// Collect `n` payload bytes starting at byte offset `at`, stripping
/// trailing 0x00 padding (§7.1.4: zero bytes mark the end of the text).
fn text_bytes(words: &[u32], at: usize, n: usize) -> Vec<u8> {
    let mut v: Vec<u8> = (0..n).map(|i| byte_at(words, at + i)).collect();
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

impl UmpStreamMessage {
    /// Decode a UMP Stream packet (MT 0xF, 4 words).
    pub fn decode(p: &Ump) -> Result<Self> {
        if p.message_type() != MessageType::UmpStream {
            return Err(Error::invalid("UMP: not a UMP Stream message"));
        }
        let w = p.words();
        if w.len() < 4 {
            return Err(Error::invalid("UMP Stream: needs 4 words"));
        }
        let format = StreamFormat::from_bits(((w[0] >> 26) & 0x3) as u8);
        let status = ((w[0] >> 16) & 0x03FF) as u16;
        let data16 = (w[0] & 0xFFFF) as u16;
        Ok(match status {
            0x00 => UmpStreamMessage::EndpointDiscovery {
                ump_version_major: (data16 >> 8) as u8,
                ump_version_minor: (data16 & 0xFF) as u8,
                filter: (w[1] & 0x1F) as u8,
            },
            0x01 => UmpStreamMessage::EndpointInfoNotification {
                ump_version_major: (data16 >> 8) as u8,
                ump_version_minor: (data16 & 0xFF) as u8,
                static_function_blocks: (w[1] >> 31) != 0,
                num_function_blocks: ((w[1] >> 24) & 0x7F) as u8,
                midi2_capable: (w[1] >> 9) & 1 != 0,
                midi1_capable: (w[1] >> 8) & 1 != 0,
                rx_jr: (w[1] >> 1) & 1 != 0,
                tx_jr: w[1] & 1 != 0,
            },
            0x02 => UmpStreamMessage::DeviceIdentityNotification {
                device_manufacturer: [
                    ((w[1] >> 16) & 0x7F) as u8,
                    ((w[1] >> 8) & 0x7F) as u8,
                    (w[1] & 0x7F) as u8,
                ],
                device_family: u16::from(((w[2] >> 24) & 0x7F) as u8)
                    | (u16::from(((w[2] >> 16) & 0x7F) as u8) << 7),
                device_family_model: u16::from(((w[2] >> 8) & 0x7F) as u8)
                    | (u16::from((w[2] & 0x7F) as u8) << 7),
                software_revision: [
                    ((w[3] >> 24) & 0x7F) as u8,
                    ((w[3] >> 16) & 0x7F) as u8,
                    ((w[3] >> 8) & 0x7F) as u8,
                    (w[3] & 0x7F) as u8,
                ],
            },
            0x03 => UmpStreamMessage::EndpointNameNotification {
                format,
                bytes: text_bytes(w, 2, 14),
            },
            0x04 => UmpStreamMessage::ProductInstanceIdNotification {
                format,
                bytes: text_bytes(w, 2, 14),
            },
            0x05 | 0x06 => {
                let protocol = (data16 >> 8) as u8;
                let rx_jr = (data16 >> 1) & 1 != 0;
                let tx_jr = data16 & 1 != 0;
                if status == 0x05 {
                    UmpStreamMessage::StreamConfigurationRequest {
                        protocol,
                        rx_jr,
                        tx_jr,
                    }
                } else {
                    UmpStreamMessage::StreamConfigurationNotification {
                        protocol,
                        rx_jr,
                        tx_jr,
                    }
                }
            }
            0x10 => UmpStreamMessage::FunctionBlockDiscovery {
                block: (data16 >> 8) as u8,
                filter: (data16 & 0x03) as u8,
            },
            0x11 => UmpStreamMessage::FunctionBlockInfoNotification {
                active: (data16 >> 15) & 1 != 0,
                block: ((data16 >> 8) & 0x7F) as u8,
                ui_hint: ((data16 >> 4) & 0x3) as u8,
                midi1_port: ((data16 >> 2) & 0x3) as u8,
                direction: (data16 & 0x3) as u8,
                first_group: ((w[1] >> 24) & 0xFF) as u8,
                groups_spanned: ((w[1] >> 16) & 0xFF) as u8,
                ci_version: ((w[1] >> 8) & 0xFF) as u8,
                max_sysex8_streams: (w[1] & 0xFF) as u8,
            },
            0x12 => UmpStreamMessage::FunctionBlockNameNotification {
                format,
                block: (data16 >> 8) as u8,
                bytes: text_bytes(w, 3, 13),
            },
            0x20 => UmpStreamMessage::StartOfClip,
            0x21 => UmpStreamMessage::EndOfClip,
            other => UmpStreamMessage::Unknown {
                format,
                status: other,
                raw: *p,
            },
        })
    }

    /// Encode back into a 4-word UMP Stream packet.
    #[must_use]
    pub fn encode(&self) -> Ump {
        let mut words = [0u32; 4];
        match self {
            UmpStreamMessage::EndpointDiscovery {
                ump_version_major,
                ump_version_minor,
                filter,
            } => {
                words[0] = word0(
                    StreamFormat::Complete,
                    0x00,
                    (u16::from(*ump_version_major) << 8) | u16::from(*ump_version_minor),
                );
                words[1] = u32::from(*filter & 0x1F);
            }
            UmpStreamMessage::EndpointInfoNotification {
                ump_version_major,
                ump_version_minor,
                static_function_blocks,
                num_function_blocks,
                midi2_capable,
                midi1_capable,
                rx_jr,
                tx_jr,
            } => {
                words[0] = word0(
                    StreamFormat::Complete,
                    0x01,
                    (u16::from(*ump_version_major) << 8) | u16::from(*ump_version_minor),
                );
                words[1] = (u32::from(*static_function_blocks) << 31)
                    | (u32::from(*num_function_blocks & 0x7F) << 24)
                    | (u32::from(*midi2_capable) << 9)
                    | (u32::from(*midi1_capable) << 8)
                    | (u32::from(*rx_jr) << 1)
                    | u32::from(*tx_jr);
            }
            UmpStreamMessage::DeviceIdentityNotification {
                device_manufacturer,
                device_family,
                device_family_model,
                software_revision,
            } => {
                words[0] = word0(StreamFormat::Complete, 0x02, 0);
                words[1] = (u32::from(device_manufacturer[0] & 0x7F) << 16)
                    | (u32::from(device_manufacturer[1] & 0x7F) << 8)
                    | u32::from(device_manufacturer[2] & 0x7F);
                words[2] = (u32::from((device_family & 0x7F) as u8) << 24)
                    | (u32::from(((device_family >> 7) & 0x7F) as u8) << 16)
                    | (u32::from((device_family_model & 0x7F) as u8) << 8)
                    | u32::from(((device_family_model >> 7) & 0x7F) as u8);
                words[3] = (u32::from(software_revision[0] & 0x7F) << 24)
                    | (u32::from(software_revision[1] & 0x7F) << 16)
                    | (u32::from(software_revision[2] & 0x7F) << 8)
                    | u32::from(software_revision[3] & 0x7F);
            }
            UmpStreamMessage::EndpointNameNotification { format, bytes } => {
                let n = bytes.len().min(14);
                words[0] = word0(*format, 0x03, 0);
                pack_bytes(&mut words, 2, &bytes[..n]);
            }
            UmpStreamMessage::ProductInstanceIdNotification { format, bytes } => {
                let n = bytes.len().min(14);
                words[0] = word0(*format, 0x04, 0);
                pack_bytes(&mut words, 2, &bytes[..n]);
            }
            UmpStreamMessage::StreamConfigurationRequest {
                protocol,
                rx_jr,
                tx_jr,
            } => {
                words[0] = word0(
                    StreamFormat::Complete,
                    0x05,
                    (u16::from(*protocol) << 8) | (u16::from(*rx_jr) << 1) | u16::from(*tx_jr),
                );
            }
            UmpStreamMessage::StreamConfigurationNotification {
                protocol,
                rx_jr,
                tx_jr,
            } => {
                words[0] = word0(
                    StreamFormat::Complete,
                    0x06,
                    (u16::from(*protocol) << 8) | (u16::from(*rx_jr) << 1) | u16::from(*tx_jr),
                );
            }
            UmpStreamMessage::FunctionBlockDiscovery { block, filter } => {
                words[0] = word0(
                    StreamFormat::Complete,
                    0x10,
                    (u16::from(*block) << 8) | u16::from(*filter & 0x03),
                );
            }
            UmpStreamMessage::FunctionBlockInfoNotification {
                active,
                block,
                ui_hint,
                midi1_port,
                direction,
                first_group,
                groups_spanned,
                ci_version,
                max_sysex8_streams,
            } => {
                words[0] = word0(
                    StreamFormat::Complete,
                    0x11,
                    (u16::from(*active) << 15)
                        | (u16::from(*block & 0x7F) << 8)
                        | (u16::from(*ui_hint & 0x3) << 4)
                        | (u16::from(*midi1_port & 0x3) << 2)
                        | u16::from(*direction & 0x3),
                );
                words[1] = (u32::from(*first_group) << 24)
                    | (u32::from(*groups_spanned) << 16)
                    | (u32::from(*ci_version) << 8)
                    | u32::from(*max_sysex8_streams);
            }
            UmpStreamMessage::FunctionBlockNameNotification {
                format,
                block,
                bytes,
            } => {
                let n = bytes.len().min(13);
                words[0] = word0(*format, 0x12, u16::from(*block) << 8);
                pack_bytes(&mut words, 3, &bytes[..n]);
            }
            UmpStreamMessage::StartOfClip => {
                words[0] = word0(StreamFormat::Complete, 0x20, 0);
            }
            UmpStreamMessage::EndOfClip => {
                words[0] = word0(StreamFormat::Complete, 0x21, 0);
            }
            UmpStreamMessage::Unknown { raw, .. } => return *raw,
        }
        Ump::from_parts(words, 4)
    }
}

/// Split `text` into per-packet chunks of `per_packet` bytes and build
/// one packet per chunk with the correct §7.1 Format run
/// (Complete, or Start / Continue* / End).
fn text_packets(
    text: &[u8],
    per_packet: usize,
    max_bytes: usize,
    what: &str,
    mut make: impl FnMut(StreamFormat, &[u8]) -> UmpStreamMessage,
) -> Result<Vec<Ump>> {
    if text.len() > max_bytes {
        return Err(Error::invalid(format!(
            "UMP Stream: {what} longer than {max_bytes} bytes ({} given)",
            text.len()
        )));
    }
    let chunks: Vec<&[u8]> = if text.is_empty() {
        vec![&[][..]]
    } else {
        text.chunks(per_packet).collect()
    };
    let last = chunks.len() - 1;
    Ok(chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let fmt = if last == 0 {
                StreamFormat::Complete
            } else if i == 0 {
                StreamFormat::Start
            } else if i == last {
                StreamFormat::End
            } else {
                StreamFormat::Continue
            };
            make(fmt, c).encode()
        })
        .collect())
}

/// Build the packet run declaring a UMP Endpoint Name (§7.1.4).
/// UTF-8, at most [`MAX_ENDPOINT_NAME_BYTES`] bytes.
pub fn endpoint_name_packets(name: &str) -> Result<Vec<Ump>> {
    text_packets(
        name.as_bytes(),
        14,
        MAX_ENDPOINT_NAME_BYTES,
        "Endpoint Name",
        |format, c| UmpStreamMessage::EndpointNameNotification {
            format,
            bytes: c.to_vec(),
        },
    )
}

/// Build the packet run declaring a Product Instance Id (§7.1.5).
/// ASCII in the ordinal range 32..=126, at most
/// [`MAX_PRODUCT_INSTANCE_ID_BYTES`] bytes.
pub fn product_instance_id_packets(id: &str) -> Result<Vec<Ump>> {
    if !id.bytes().all(|b| (32..=126).contains(&b)) {
        return Err(Error::invalid(
            "UMP Stream: Product Instance Id must be ASCII 32..=126",
        ));
    }
    text_packets(
        id.as_bytes(),
        14,
        MAX_PRODUCT_INSTANCE_ID_BYTES,
        "Product Instance Id",
        |format, c| UmpStreamMessage::ProductInstanceIdNotification {
            format,
            bytes: c.to_vec(),
        },
    )
}

/// Build the packet run declaring a Function Block Name (§7.1.9).
/// UTF-8, at most [`MAX_FUNCTION_BLOCK_NAME_BYTES`] bytes; 13 bytes
/// per packet.
pub fn function_block_name_packets(block: u8, name: &str) -> Result<Vec<Ump>> {
    text_packets(
        name.as_bytes(),
        13,
        MAX_FUNCTION_BLOCK_NAME_BYTES,
        "Function Block Name",
        |format, c| UmpStreamMessage::FunctionBlockNameNotification {
            format,
            block,
            bytes: c.to_vec(),
        },
    )
}

/// Reassembles a multi-packet UMP Stream text payload (Endpoint Name,
/// Product Instance Id, or Function Block Name) from its per-packet
/// chunks (§7.1.4 / §7.1.5 / §7.1.9).
///
/// Feed each packet's `(format, bytes)` in arrival order; a
/// `Complete` chunk or an `End` chunk yields the finished string.
#[derive(Debug, Default)]
pub struct StreamTextAssembler {
    buf: Vec<u8>,
}

impl StreamTextAssembler {
    /// Fresh assembler with an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one packet's chunk. Returns the completed text (lossily
    /// decoded as UTF-8) when `format` is `Complete` or `End`, `None`
    /// while the message is still in progress.
    pub fn push(&mut self, format: StreamFormat, bytes: &[u8]) -> Option<String> {
        match format {
            StreamFormat::Complete => {
                let s = String::from_utf8_lossy(bytes).into_owned();
                self.buf.clear();
                Some(s)
            }
            StreamFormat::Start => {
                self.buf.clear();
                self.buf.extend_from_slice(bytes);
                None
            }
            StreamFormat::Continue => {
                self.buf.extend_from_slice(bytes);
                None
            }
            StreamFormat::End => {
                self.buf.extend_from_slice(bytes);
                let s = String::from_utf8_lossy(&self.buf).into_owned();
                self.buf.clear();
                Some(s)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(m: &UmpStreamMessage) {
        let p = m.encode();
        assert_eq!(p.len(), 4, "stream packets are 128-bit");
        let back = UmpStreamMessage::decode(&p).unwrap();
        assert_eq!(&back, m);
    }

    #[test]
    fn endpoint_discovery_round_trips_and_pins_word0() {
        let m = UmpStreamMessage::EndpointDiscovery {
            ump_version_major: 1,
            ump_version_minor: 1,
            filter: discovery_filter::ALL,
        };
        let p = m.encode();
        // MT F, form 0, status 0x00, ver 0x0101; filter in word 1.
        assert_eq!(p.words()[0], 0xF000_0101);
        assert_eq!(p.words()[1], 0x0000_001F);
        round_trip(&m);
    }

    #[test]
    fn endpoint_info_notification_field_packing() {
        let m = UmpStreamMessage::EndpointInfoNotification {
            ump_version_major: 1,
            ump_version_minor: 1,
            static_function_blocks: true,
            num_function_blocks: 3,
            midi2_capable: true,
            midi1_capable: true,
            rx_jr: false,
            tx_jr: true,
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0xF001_0101);
        // s=1, nfb=3 in the top byte; m2|m1 bits 9..8; txjr bit 0.
        assert_eq!(p.words()[1], 0x8300_0301);
        round_trip(&m);
    }

    #[test]
    fn device_identity_14bit_fields_are_lsb_first() {
        let m = UmpStreamMessage::DeviceIdentityNotification {
            device_manufacturer: [0x41, 0x00, 0x00],
            device_family: 0x1234,
            device_family_model: 0x0577,
            software_revision: [1, 2, 3, 4],
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0xF002_0000);
        assert_eq!(p.words()[1], 0x0041_0000);
        // family 0x1234 = lsb 0x34, msb 0x24; model 0x0577 = lsb 0x77 msb 0x0A.
        assert_eq!(p.words()[2], 0x3424_770A);
        assert_eq!(p.words()[3], 0x0102_0304);
        round_trip(&m);
    }

    #[test]
    fn short_endpoint_name_is_one_complete_packet() {
        let packets = endpoint_name_packets("OxideAV").unwrap();
        assert_eq!(packets.len(), 1);
        let m = UmpStreamMessage::decode(&packets[0]).unwrap();
        match m {
            UmpStreamMessage::EndpointNameNotification { format, bytes } => {
                assert_eq!(format, StreamFormat::Complete);
                assert_eq!(bytes, b"OxideAV");
            }
            other => panic!("unexpected {other:?}"),
        }
        // "O" lands in word-0's second-lowest byte per Figure 15.
        assert_eq!(
            packets[0].words()[0] & 0xFFFF,
            u32::from(b'O') << 8 | u32::from(b'x')
        );
    }

    #[test]
    fn long_endpoint_name_reassembles() {
        let name = "A UMP Endpoint With A Rather Long Name Indeed!";
        let packets = endpoint_name_packets(name).unwrap();
        assert!(packets.len() > 1);
        let mut asm = StreamTextAssembler::new();
        let mut out = None;
        for p in &packets {
            match UmpStreamMessage::decode(p).unwrap() {
                UmpStreamMessage::EndpointNameNotification { format, bytes } => {
                    out = asm.push(format, &bytes);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(out.as_deref(), Some(name));
    }

    #[test]
    fn endpoint_name_over_98_bytes_is_rejected() {
        let name = "x".repeat(99);
        assert!(endpoint_name_packets(&name).is_err());
    }

    #[test]
    fn product_instance_id_rejects_non_ascii() {
        assert!(product_instance_id_packets("héllo").is_err());
        assert!(product_instance_id_packets("SN-12345").is_ok());
    }

    #[test]
    fn stream_configuration_round_trips() {
        round_trip(&UmpStreamMessage::StreamConfigurationRequest {
            protocol: protocol::MIDI2,
            rx_jr: true,
            tx_jr: false,
        });
        let m = UmpStreamMessage::StreamConfigurationNotification {
            protocol: protocol::MIDI1,
            rx_jr: false,
            tx_jr: true,
        };
        assert_eq!(m.encode().words()[0], 0xF006_0101);
        round_trip(&m);
    }

    #[test]
    fn function_block_info_round_trips() {
        let m = UmpStreamMessage::FunctionBlockInfoNotification {
            active: true,
            block: 2,
            ui_hint: 0b11,
            midi1_port: 0x00,
            direction: 0b11,
            first_group: 0,
            groups_spanned: 16,
            ci_version: 0x01,
            max_sysex8_streams: 2,
        };
        let p = m.encode();
        assert_eq!(p.words()[0], 0xF011_8233);
        assert_eq!(p.words()[1], 0x0010_0102);
        round_trip(&m);
    }

    #[test]
    fn function_block_name_carries_block_and_13_bytes() {
        let packets = function_block_name_packets(5, "Synth").unwrap();
        assert_eq!(packets.len(), 1);
        match UmpStreamMessage::decode(&packets[0]).unwrap() {
            UmpStreamMessage::FunctionBlockNameNotification {
                format,
                block,
                bytes,
            } => {
                assert_eq!(format, StreamFormat::Complete);
                assert_eq!(block, 5);
                assert_eq!(bytes, b"Synth");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn clip_markers_pin_status_words() {
        assert_eq!(
            UmpStreamMessage::StartOfClip.encode().words(),
            &[0xF020_0000, 0, 0, 0]
        );
        assert_eq!(
            UmpStreamMessage::EndOfClip.encode().words(),
            &[0xF021_0000, 0, 0, 0]
        );
        round_trip(&UmpStreamMessage::StartOfClip);
        round_trip(&UmpStreamMessage::EndOfClip);
    }

    #[test]
    fn unknown_status_survives_round_trip() {
        let raw = Ump::from_words(&[0xF3FF_0000, 1, 2, 3]).unwrap();
        let m = UmpStreamMessage::decode(&raw).unwrap();
        match &m {
            UmpStreamMessage::Unknown { status, .. } => assert_eq!(*status, 0x3FF),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(m.encode(), raw);
    }
}
