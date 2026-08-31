//! MIDI Capability Inquiry (MIDI-CI) — typed SysEx surface per
//! M2-101-UM v1.2.1, with the Profile ID rules of M2-102-U and the
//! Property Exchange base messages whose header/property payloads are
//! defined by M2-103-UM.
//!
//! MIDI-CI messages are Universal System Exclusive messages with
//! Sub-ID#1 `0x0D` (§5.2.1, Table 5). This module parses and emits the
//! **payload** form — the bytes between the 0xF0/0xF7 brackets,
//! exclusive — which is exactly what rides in a UMP SysEx7 packet run
//! (§5.2: "When MIDI-CI messages are carried inside a Universal MIDI
//! Packet, the F0 and F7 are omitted") and what
//! [`Event::Sysex`](crate::smf::Event::Sysex) carries minus its
//! trailing EOX.
//!
//! Scope: the common envelope (Device ID, Sub-ID#2, version, MUIDs),
//! the Management category (Discovery, Endpoint, Invalidate MUID,
//! ACK/NAK), Profile Configuration (inquiry/reply, add/remove,
//! details, enable/disable + reports, profile-specific data), the
//! Property Exchange base messages (capabilities + chunked
//! Get/Set/Subscribe carrying their M2-103 header/property blobs as
//! opaque data), and Process Inquiry. No session state machine is
//! provided — this is a wire surface.

use oxideav_core::{Error, Result};

/// Universal System Exclusive (Non-Real Time) leading byte.
pub const UNIVERSAL_SYSEX: u8 = 0x7E;
/// Universal SysEx Sub-ID#1 assigned to MIDI-CI (§5.2.1).
pub const SUB_ID_1_MIDI_CI: u8 = 0x0D;
/// The Broadcast MUID `7F 7F 7F 7F` (§3.3.3) as a 28-bit value.
pub const BROADCAST_MUID: u32 = 0x0FFF_FFFF;
/// MIDI-CI Message Version/Format for M2-101 v1.1.
pub const CI_VERSION_1: u8 = 0x01;
/// MIDI-CI Message Version/Format for M2-101 v1.2 / v1.2.1.
pub const CI_VERSION_2: u8 = 0x02;
/// Device ID value addressing the whole Group (§5.2.1).
pub const DEVICE_ID_GROUP: u8 = 0x7E;
/// Device ID value addressing the whole Function Block (§5.2.1).
pub const DEVICE_ID_FUNCTION_BLOCK: u8 = 0x7F;

/// Universal SysEx Sub-ID#2 values for every MIDI-CI message
/// (M2-101 Tables 6..46). The high nibble is the Category (Table 4).
pub mod sub_id2 {
    /// 0x70 Discovery (Management).
    pub const DISCOVERY: u8 = 0x70;
    /// 0x71 Reply to Discovery.
    pub const REPLY_TO_DISCOVERY: u8 = 0x71;
    /// 0x72 Inquiry: Endpoint Information.
    pub const INQUIRY_ENDPOINT: u8 = 0x72;
    /// 0x73 Reply to Endpoint Information.
    pub const REPLY_TO_ENDPOINT: u8 = 0x73;
    /// 0x7D MIDI-CI ACK.
    pub const ACK: u8 = 0x7D;
    /// 0x7E Invalidate MUID.
    pub const INVALIDATE_MUID: u8 = 0x7E;
    /// 0x7F MIDI-CI NAK.
    pub const NAK: u8 = 0x7F;

    /// 0x20 Profile Inquiry.
    pub const PROFILE_INQUIRY: u8 = 0x20;
    /// 0x21 Reply to Profile Inquiry.
    pub const REPLY_TO_PROFILE_INQUIRY: u8 = 0x21;
    /// 0x22 Set Profile On.
    pub const SET_PROFILE_ON: u8 = 0x22;
    /// 0x23 Set Profile Off.
    pub const SET_PROFILE_OFF: u8 = 0x23;
    /// 0x24 Profile Enabled Report.
    pub const PROFILE_ENABLED_REPORT: u8 = 0x24;
    /// 0x25 Profile Disabled Report.
    pub const PROFILE_DISABLED_REPORT: u8 = 0x25;
    /// 0x26 Profile Added Report.
    pub const PROFILE_ADDED_REPORT: u8 = 0x26;
    /// 0x27 Profile Removed Report ("Profile List Remove").
    pub const PROFILE_REMOVED_REPORT: u8 = 0x27;
    /// 0x28 Profile Details Inquiry.
    pub const PROFILE_DETAILS_INQUIRY: u8 = 0x28;
    /// 0x29 Reply to Profile Details Inquiry.
    pub const REPLY_TO_PROFILE_DETAILS: u8 = 0x29;
    /// 0x2F Profile Specific Data.
    pub const PROFILE_SPECIFIC_DATA: u8 = 0x2F;

    /// 0x30 Inquiry: Property Exchange Capabilities.
    pub const PE_CAPABILITIES: u8 = 0x30;
    /// 0x31 Reply to Property Exchange Capabilities.
    pub const REPLY_TO_PE_CAPABILITIES: u8 = 0x31;
    /// 0x34 Inquiry: Get Property Data.
    pub const GET_PROPERTY_DATA: u8 = 0x34;
    /// 0x35 Reply to Get Property Data.
    pub const REPLY_TO_GET_PROPERTY_DATA: u8 = 0x35;
    /// 0x36 Inquiry: Set Property Data.
    pub const SET_PROPERTY_DATA: u8 = 0x36;
    /// 0x37 Reply to Set Property Data.
    pub const REPLY_TO_SET_PROPERTY_DATA: u8 = 0x37;
    /// 0x38 Subscription.
    pub const SUBSCRIPTION: u8 = 0x38;
    /// 0x39 Reply to Subscription.
    pub const REPLY_TO_SUBSCRIPTION: u8 = 0x39;
    /// 0x3F Notify (replaced by ACK/NAK in v1.2.1; still honoured).
    pub const NOTIFY: u8 = 0x3F;

    /// 0x40 Inquiry: Process Inquiry Capabilities.
    pub const PROCESS_INQUIRY_CAPABILITIES: u8 = 0x40;
    /// 0x41 Reply to Process Inquiry Capabilities.
    pub const REPLY_TO_PROCESS_INQUIRY_CAPABILITIES: u8 = 0x41;
    /// 0x42 Inquiry: MIDI Message Report.
    pub const MIDI_MESSAGE_REPORT: u8 = 0x42;
    /// 0x43 Reply to MIDI Message Report.
    pub const REPLY_TO_MIDI_MESSAGE_REPORT: u8 = 0x43;
    /// 0x44 End of MIDI Message Report.
    pub const END_OF_MIDI_MESSAGE_REPORT: u8 = 0x44;
}

/// Capability Inquiry Category Supported bitmap bits (§5.5.2,
/// Table 7).
pub mod category {
    /// D2 — Profile Configuration supported.
    pub const PROFILE_CONFIGURATION: u8 = 0x04;
    /// D3 — Property Exchange supported.
    pub const PROPERTY_EXCHANGE: u8 = 0x08;
    /// D4 — Process Inquiry supported.
    pub const PROCESS_INQUIRY: u8 = 0x10;
}

/// A 5-byte MIDI-CI Profile ID (M2-101 §7.3.1, M2-102 Table).
///
/// Standard Defined Profiles carry `0x7E` in byte 1, then Profile
/// Bank, Profile Number, Profile Version, and Profile Level;
/// Manufacturer Specific Profiles carry the 3-byte Manufacturer SysEx
/// ID then two bytes of manufacturer-specific info.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProfileId(pub [u8; 5]);

impl ProfileId {
    /// A Standard Defined Profile (byte 1 = 0x7E).
    #[must_use]
    pub fn standard(bank: u8, number: u8, version: u8, level: u8) -> Self {
        ProfileId([
            0x7E,
            bank & 0x7F,
            number & 0x7F,
            version & 0x7F,
            level & 0x7F,
        ])
    }

    /// Whether this is an MMA/AMEI Standard Defined Profile.
    #[must_use]
    pub fn is_standard_defined(&self) -> bool {
        self.0[0] == 0x7E
    }
}

/// The four Device Identification fields shared by Discovery and its
/// reply (§5.5.1) — the same data as the MIDI 1.0 Device Inquiry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceIdentity {
    /// 3-byte Manufacturer SysEx ID (1-byte IDs are `[id, 0, 0]`).
    pub manufacturer: [u8; 3],
    /// 14-bit Device Family (LSB-first pair on the wire).
    pub family: u16,
    /// 14-bit Device Family Model Number.
    pub family_model: u16,
    /// 4-byte Software Revision Level (device-specific format).
    pub software_revision: [u8; 4],
}

/// The chunked payload shared by the Property Exchange data messages
/// (Get/Set/Subscribe and their replies, and Notify) — §8.7..§8.13.
/// Header Data and Property Data are carried opaquely; their JSON
/// semantics live in M2-103 (Common Rules for Property Exchange).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PePayload {
    /// Request ID tying the chunks of one PE Transaction together.
    pub request_id: u8,
    /// Header Data (first chunk only; M2-103 defines the contents).
    pub header: Vec<u8>,
    /// Number of Chunks in Message (0 = unknown/streaming).
    pub num_chunks: u16,
    /// Number of This Chunk (counting starts at 1).
    pub this_chunk: u16,
    /// Property Data carried by this chunk.
    pub data: Vec<u8>,
}

/// The body of a MIDI-CI message, discriminated by Sub-ID#2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CiBody {
    /// 0x70 Discovery (§5.5) — broadcast by an Initiator.
    Discovery {
        /// Device identification fields.
        identity: DeviceIdentity,
        /// Capability Inquiry Category Supported bitmap ([`category`]).
        categories: u8,
        /// Receivable Maximum SysEx Message Size (28-bit, ≥128).
        max_sysex_size: u32,
        /// Initiator's Output Path ID (added in CI version 2).
        output_path_id: Option<u8>,
    },
    /// 0x71 Reply to Discovery (§5.6).
    ReplyToDiscovery {
        /// Device identification fields.
        identity: DeviceIdentity,
        /// Capability Inquiry Category Supported bitmap.
        categories: u8,
        /// Receivable Maximum SysEx Message Size.
        max_sysex_size: u32,
        /// Echoed Output Path ID (version 2).
        output_path_id: Option<u8>,
        /// Function Block number, 0x7F = none (version 2).
        function_block: Option<u8>,
    },
    /// 0x72 Inquiry: Endpoint Information (§5.7).
    InquiryEndpoint {
        /// Requested information (0x00 = Product Instance ID).
        status: u8,
    },
    /// 0x73 Reply to Endpoint Information (§5.8).
    ReplyToEndpoint {
        /// Echoed Status value.
        status: u8,
        /// Information Data (for Status 0x00: the ASCII Product
        /// Instance ID, ≤42 bytes).
        data: Vec<u8>,
    },
    /// 0x7E Invalidate MUID (§5.9) — broadcast.
    InvalidateMuid {
        /// The MUID being invalidated.
        target_muid: u32,
    },
    /// 0x7D MIDI-CI ACK (§5.10).
    Ack {
        /// Original Transaction Sub-ID#2 Classification.
        original_sub_id2: u8,
        /// ACK Status Code (Table 14).
        status_code: u8,
        /// ACK Status Data.
        status_data: u8,
        /// 5 details bytes (Profile ID / PE stream+chunk / reserved).
        details: [u8; 5],
        /// Human-readable message text (§5.10.4 encoding).
        message: Vec<u8>,
    },
    /// 0x7F MIDI-CI NAK (§5.11). The classification fields were added
    /// in CI version 2; a version-1 NAK is just the envelope.
    Nak {
        /// Original Transaction Sub-ID#2 Classification (version 2).
        original_sub_id2: Option<u8>,
        /// NAK Status Code (Table 16) (version 2).
        status_code: Option<u8>,
        /// NAK Status Data (version 2).
        status_data: Option<u8>,
        /// 5 details bytes (version 2).
        details: Option<[u8; 5]>,
        /// Human-readable message text (version 2).
        message: Vec<u8>,
    },
    /// 0x20 Profile Inquiry (§7.2) — envelope only.
    ProfileInquiry,
    /// 0x21 Reply to Profile Inquiry (§7.3).
    ReplyToProfileInquiry {
        /// Profiles supported and currently enabled.
        enabled: Vec<ProfileId>,
        /// Profiles supported but currently disabled.
        disabled: Vec<ProfileId>,
    },
    /// 0x22 Set Profile On (§7.8).
    SetProfileOn {
        /// The Profile to enable.
        profile: ProfileId,
        /// Number of Channels requested (version 2; 0 when addressed
        /// to a Group / Function Block).
        num_channels: Option<u16>,
    },
    /// 0x23 Set Profile Off (§7.9).
    SetProfileOff {
        /// The Profile to disable.
        profile: ProfileId,
    },
    /// 0x24 Profile Enabled Report (§7.10) — broadcast.
    ProfileEnabledReport {
        /// The Profile now enabled.
        profile: ProfileId,
        /// Channels enabled (version 2).
        num_channels: Option<u16>,
    },
    /// 0x25 Profile Disabled Report (§7.11) — broadcast.
    ProfileDisabledReport {
        /// The Profile now disabled.
        profile: ProfileId,
        /// Channels disabled (version 2).
        num_channels: Option<u16>,
    },
    /// 0x26 Profile Added Report (§7.4) — broadcast.
    ProfileAddedReport {
        /// The Profile newly supported.
        profile: ProfileId,
    },
    /// 0x27 Profile Removed Report (§7.5) — broadcast.
    ProfileRemovedReport {
        /// The Profile no longer supported.
        profile: ProfileId,
    },
    /// 0x28 Profile Details Inquiry (§7.6).
    ProfileDetailsInquiry {
        /// The Profile being queried.
        profile: ProfileId,
        /// Inquiry Target (0x00..=0x3F registered, 0x40..=0x7F
        /// profile-specific).
        target: u8,
    },
    /// 0x29 Reply to Profile Details Inquiry (§7.7).
    ReplyToProfileDetails {
        /// The Profile being described.
        profile: ProfileId,
        /// Echoed Inquiry Target.
        target: u8,
        /// Inquiry Target Data.
        data: Vec<u8>,
    },
    /// 0x2F Profile Specific Data (§7.12).
    ProfileSpecificData {
        /// The Profile the data belongs to.
        profile: ProfileId,
        /// Profile-defined payload.
        data: Vec<u8>,
    },
    /// 0x30 / 0x31 Property Exchange Capabilities + reply (§8.5/§8.6).
    PeCapabilities {
        /// `true` for the 0x31 reply, `false` for the 0x30 inquiry.
        reply: bool,
        /// Number of Simultaneous PE Requests Supported.
        num_requests: u8,
        /// PE Major/Minor version (version 2).
        pe_version: Option<(u8, u8)>,
    },
    /// The chunked PE data messages: 0x34..=0x39 and 0x3F (Notify).
    /// The concrete message is the envelope's `sub_id2`.
    PropertyData(PePayload),
    /// 0x40 Inquiry: Process Inquiry Capabilities (§9.2) — envelope
    /// only.
    ProcessInquiryCapabilities,
    /// 0x41 Reply to Process Inquiry Capabilities (§9.3).
    ReplyToProcessInquiryCapabilities {
        /// Supported-features bitmap (D0 = MIDI Message Report).
        features: u8,
    },
    /// 0x42 Inquiry: MIDI Message Report (§9.5).
    MidiMessageReport {
        /// Message Data Control (0x00 none / 0x01 non-default /
        /// 0x7F full — Table 44).
        data_control: u8,
        /// Requested System Messages bitmap.
        system: u8,
        /// Reserved second System bitmap byte.
        other_system: u8,
        /// Requested Channel Controller Messages bitmap.
        channel_controller: u8,
        /// Requested Note Data Messages bitmap.
        note_data: u8,
    },
    /// 0x43 Reply to MIDI Message Report (§9.6) — same bitmaps, no
    /// Message Data Control.
    ReplyToMidiMessageReport {
        /// Granted System Messages bitmap.
        system: u8,
        /// Reserved second System bitmap byte.
        other_system: u8,
        /// Granted Channel Controller Messages bitmap.
        channel_controller: u8,
        /// Granted Note Data Messages bitmap.
        note_data: u8,
    },
    /// 0x44 End of MIDI Message Report (§9.8) — envelope only.
    EndOfMidiMessageReport,
    /// Any Sub-ID#2 not modelled — body bytes kept raw.
    Unknown {
        /// The message body after the Destination MUID.
        data: Vec<u8>,
    },
}

/// A complete MIDI-CI message: the §5.2.1 envelope plus a typed body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CiMessage {
    /// Device ID: 0x00..=0x0F channels, 0x7E Group, 0x7F Function
    /// Block.
    pub device_id: u8,
    /// Universal SysEx Sub-ID#2 ([`sub_id2`]).
    pub sub_id2: u8,
    /// MIDI-CI Message Version/Format (0x01 / 0x02).
    pub version: u8,
    /// Source MUID (28-bit).
    pub source_muid: u32,
    /// Destination MUID (28-bit; [`BROADCAST_MUID`] for broadcast).
    pub destination_muid: u32,
    /// The typed body.
    pub body: CiBody,
}

/// Cursor-style reader over 7-bit SysEx payload bytes.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u7(&mut self) -> Result<u8> {
        let v = *self
            .b
            .get(self.pos)
            .ok_or_else(|| Error::invalid("MIDI-CI: truncated message"))?;
        self.pos += 1;
        Ok(v & 0x7F)
    }

    fn u14(&mut self) -> Result<u16> {
        let lo = self.u7()?;
        let hi = self.u7()?;
        Ok(u16::from(lo) | (u16::from(hi) << 7))
    }

    fn u28(&mut self) -> Result<u32> {
        let mut v = 0u32;
        for i in 0..4 {
            v |= u32::from(self.u7()?) << (7 * i);
        }
        Ok(v)
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.b.len() {
            return Err(Error::invalid("MIDI-CI: truncated message"));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn remaining(&self) -> usize {
        self.b.len() - self.pos
    }
}

fn push_u14(out: &mut Vec<u8>, v: u16) {
    out.push((v & 0x7F) as u8);
    out.push(((v >> 7) & 0x7F) as u8);
}

fn push_u28(out: &mut Vec<u8>, v: u32) {
    for i in 0..4 {
        out.push(((v >> (7 * i)) & 0x7F) as u8);
    }
}

fn read_identity(r: &mut Reader) -> Result<DeviceIdentity> {
    let m = r.bytes(3)?;
    Ok(DeviceIdentity {
        manufacturer: [m[0] & 0x7F, m[1] & 0x7F, m[2] & 0x7F],
        family: r.u14()?,
        family_model: r.u14()?,
        software_revision: {
            let s = r.bytes(4)?;
            [s[0], s[1], s[2], s[3]]
        },
    })
}

fn push_identity(out: &mut Vec<u8>, id: &DeviceIdentity) {
    out.extend_from_slice(&[
        id.manufacturer[0] & 0x7F,
        id.manufacturer[1] & 0x7F,
        id.manufacturer[2] & 0x7F,
    ]);
    push_u14(out, id.family);
    push_u14(out, id.family_model);
    out.extend_from_slice(&id.software_revision);
}

fn read_profile(r: &mut Reader) -> Result<ProfileId> {
    let p = r.bytes(5)?;
    Ok(ProfileId([p[0], p[1], p[2], p[3], p[4]]))
}

fn read_pe_payload(r: &mut Reader) -> Result<PePayload> {
    let request_id = r.u7()?;
    let hl = r.u14()? as usize;
    let header = r.bytes(hl)?.to_vec();
    let num_chunks = r.u14()?;
    let this_chunk = r.u14()?;
    let dl = r.u14()? as usize;
    let data = r.bytes(dl)?.to_vec();
    Ok(PePayload {
        request_id,
        header,
        num_chunks,
        this_chunk,
        data,
    })
}

fn push_pe_payload(out: &mut Vec<u8>, p: &PePayload) {
    out.push(p.request_id & 0x7F);
    push_u14(out, p.header.len() as u16);
    out.extend_from_slice(&p.header);
    push_u14(out, p.num_chunks);
    push_u14(out, p.this_chunk);
    push_u14(out, p.data.len() as u16);
    out.extend_from_slice(&p.data);
}

impl CiMessage {
    /// Parse a MIDI-CI message from its SysEx **payload** bytes (the
    /// bytes between 0xF0 and 0xF7, exclusive — a trailing 0xF7 is
    /// tolerated and stripped).
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let payload = if payload.last() == Some(&0xF7) {
            &payload[..payload.len() - 1]
        } else {
            payload
        };
        let mut r = Reader { b: payload, pos: 0 };
        if r.u7()? != UNIVERSAL_SYSEX {
            return Err(Error::invalid(
                "MIDI-CI: not a Universal System Exclusive (0x7E) message",
            ));
        }
        let device_id = r.u7()?;
        if r.u7()? != SUB_ID_1_MIDI_CI {
            return Err(Error::invalid("MIDI-CI: Sub-ID#1 is not 0x0D"));
        }
        let sub = r.u7()?;
        let version = r.u7()?;
        let source_muid = r.u28()?;
        let destination_muid = r.u28()?;

        use sub_id2::*;
        let body = match sub {
            DISCOVERY | REPLY_TO_DISCOVERY => {
                let identity = read_identity(&mut r)?;
                let categories = r.u7()?;
                let max_sysex_size = r.u28()?;
                let output_path_id = if r.remaining() >= 1 {
                    Some(r.u7()?)
                } else {
                    None
                };
                if sub == DISCOVERY {
                    CiBody::Discovery {
                        identity,
                        categories,
                        max_sysex_size,
                        output_path_id,
                    }
                } else {
                    let function_block = if r.remaining() >= 1 {
                        Some(r.u7()?)
                    } else {
                        None
                    };
                    CiBody::ReplyToDiscovery {
                        identity,
                        categories,
                        max_sysex_size,
                        output_path_id,
                        function_block,
                    }
                }
            }
            INQUIRY_ENDPOINT => CiBody::InquiryEndpoint { status: r.u7()? },
            REPLY_TO_ENDPOINT => {
                let status = r.u7()?;
                let lid = r.u14()? as usize;
                CiBody::ReplyToEndpoint {
                    status,
                    data: r.bytes(lid)?.to_vec(),
                }
            }
            INVALIDATE_MUID => CiBody::InvalidateMuid {
                target_muid: r.u28()?,
            },
            ACK => {
                let original_sub_id2 = r.u7()?;
                let status_code = r.u7()?;
                let status_data = r.u7()?;
                let d = r.bytes(5)?;
                let details = [d[0], d[1], d[2], d[3], d[4]];
                let ml = r.u14()? as usize;
                CiBody::Ack {
                    original_sub_id2,
                    status_code,
                    status_data,
                    details,
                    message: r.bytes(ml)?.to_vec(),
                }
            }
            NAK => {
                if r.remaining() == 0 {
                    // Version-1 NAK: envelope only (§5.11).
                    CiBody::Nak {
                        original_sub_id2: None,
                        status_code: None,
                        status_data: None,
                        details: None,
                        message: Vec::new(),
                    }
                } else {
                    let original_sub_id2 = Some(r.u7()?);
                    let status_code = Some(r.u7()?);
                    let status_data = Some(r.u7()?);
                    let d = r.bytes(5)?;
                    let details = Some([d[0], d[1], d[2], d[3], d[4]]);
                    let ml = r.u14()? as usize;
                    CiBody::Nak {
                        original_sub_id2,
                        status_code,
                        status_data,
                        details,
                        message: r.bytes(ml)?.to_vec(),
                    }
                }
            }
            PROFILE_INQUIRY => CiBody::ProfileInquiry,
            REPLY_TO_PROFILE_INQUIRY => {
                let cep = r.u14()? as usize;
                let mut enabled = Vec::with_capacity(cep.min(64));
                for _ in 0..cep {
                    enabled.push(read_profile(&mut r)?);
                }
                let cdp = r.u14()? as usize;
                let mut disabled = Vec::with_capacity(cdp.min(64));
                for _ in 0..cdp {
                    disabled.push(read_profile(&mut r)?);
                }
                CiBody::ReplyToProfileInquiry { enabled, disabled }
            }
            SET_PROFILE_ON => {
                let profile = read_profile(&mut r)?;
                let num_channels = if r.remaining() >= 2 {
                    Some(r.u14()?)
                } else {
                    None
                };
                CiBody::SetProfileOn {
                    profile,
                    num_channels,
                }
            }
            SET_PROFILE_OFF => CiBody::SetProfileOff {
                profile: read_profile(&mut r)?,
            },
            PROFILE_ENABLED_REPORT | PROFILE_DISABLED_REPORT => {
                let profile = read_profile(&mut r)?;
                let num_channels = if r.remaining() >= 2 {
                    Some(r.u14()?)
                } else {
                    None
                };
                if sub == PROFILE_ENABLED_REPORT {
                    CiBody::ProfileEnabledReport {
                        profile,
                        num_channels,
                    }
                } else {
                    CiBody::ProfileDisabledReport {
                        profile,
                        num_channels,
                    }
                }
            }
            PROFILE_ADDED_REPORT => CiBody::ProfileAddedReport {
                profile: read_profile(&mut r)?,
            },
            PROFILE_REMOVED_REPORT => CiBody::ProfileRemovedReport {
                profile: read_profile(&mut r)?,
            },
            PROFILE_DETAILS_INQUIRY => CiBody::ProfileDetailsInquiry {
                profile: read_profile(&mut r)?,
                target: r.u7()?,
            },
            REPLY_TO_PROFILE_DETAILS => {
                let profile = read_profile(&mut r)?;
                let target = r.u7()?;
                let dl = r.u14()? as usize;
                CiBody::ReplyToProfileDetails {
                    profile,
                    target,
                    data: r.bytes(dl)?.to_vec(),
                }
            }
            PROFILE_SPECIFIC_DATA => {
                let profile = read_profile(&mut r)?;
                let dl = r.u28()? as usize;
                CiBody::ProfileSpecificData {
                    profile,
                    data: r.bytes(dl)?.to_vec(),
                }
            }
            PE_CAPABILITIES | REPLY_TO_PE_CAPABILITIES => {
                let num_requests = r.u7()?;
                let pe_version = if r.remaining() >= 2 {
                    Some((r.u7()?, r.u7()?))
                } else {
                    None
                };
                CiBody::PeCapabilities {
                    reply: sub == REPLY_TO_PE_CAPABILITIES,
                    num_requests,
                    pe_version,
                }
            }
            GET_PROPERTY_DATA
            | REPLY_TO_GET_PROPERTY_DATA
            | SET_PROPERTY_DATA
            | REPLY_TO_SET_PROPERTY_DATA
            | SUBSCRIPTION
            | REPLY_TO_SUBSCRIPTION
            | NOTIFY => CiBody::PropertyData(read_pe_payload(&mut r)?),
            PROCESS_INQUIRY_CAPABILITIES => CiBody::ProcessInquiryCapabilities,
            REPLY_TO_PROCESS_INQUIRY_CAPABILITIES => {
                CiBody::ReplyToProcessInquiryCapabilities { features: r.u7()? }
            }
            MIDI_MESSAGE_REPORT => CiBody::MidiMessageReport {
                data_control: r.u7()?,
                system: r.u7()?,
                other_system: r.u7()?,
                channel_controller: r.u7()?,
                note_data: r.u7()?,
            },
            REPLY_TO_MIDI_MESSAGE_REPORT => CiBody::ReplyToMidiMessageReport {
                system: r.u7()?,
                other_system: r.u7()?,
                channel_controller: r.u7()?,
                note_data: r.u7()?,
            },
            END_OF_MIDI_MESSAGE_REPORT => CiBody::EndOfMidiMessageReport,
            _ => CiBody::Unknown {
                data: r.b[r.pos..].to_vec(),
            },
        };
        Ok(CiMessage {
            device_id,
            sub_id2: sub,
            version,
            source_muid,
            destination_muid,
            body,
        })
    }

    /// Emit the SysEx payload bytes (no 0xF0/0xF7 brackets). Feed
    /// through [`sysex7_packets`](crate::ump::sysex7_packets) for UMP
    /// transport or append 0xF7 for an SMF `F0` event.
    #[must_use]
    pub fn emit(&self) -> Vec<u8> {
        let mut out = vec![
            UNIVERSAL_SYSEX,
            self.device_id & 0x7F,
            SUB_ID_1_MIDI_CI,
            self.sub_id2 & 0x7F,
            self.version & 0x7F,
        ];
        push_u28(&mut out, self.source_muid);
        push_u28(&mut out, self.destination_muid);
        match &self.body {
            CiBody::Discovery {
                identity,
                categories,
                max_sysex_size,
                output_path_id,
            } => {
                push_identity(&mut out, identity);
                out.push(*categories & 0x7F);
                push_u28(&mut out, *max_sysex_size);
                if let Some(p) = output_path_id {
                    out.push(*p & 0x7F);
                }
            }
            CiBody::ReplyToDiscovery {
                identity,
                categories,
                max_sysex_size,
                output_path_id,
                function_block,
            } => {
                push_identity(&mut out, identity);
                out.push(*categories & 0x7F);
                push_u28(&mut out, *max_sysex_size);
                if let Some(p) = output_path_id {
                    out.push(*p & 0x7F);
                    out.push(function_block.unwrap_or(0x7F) & 0x7F);
                }
            }
            CiBody::InquiryEndpoint { status } => out.push(*status & 0x7F),
            CiBody::ReplyToEndpoint { status, data } => {
                out.push(*status & 0x7F);
                push_u14(&mut out, data.len() as u16);
                out.extend_from_slice(data);
            }
            CiBody::InvalidateMuid { target_muid } => push_u28(&mut out, *target_muid),
            CiBody::Ack {
                original_sub_id2,
                status_code,
                status_data,
                details,
                message,
            } => {
                out.push(*original_sub_id2 & 0x7F);
                out.push(*status_code & 0x7F);
                out.push(*status_data & 0x7F);
                out.extend_from_slice(details);
                push_u14(&mut out, message.len() as u16);
                out.extend_from_slice(message);
            }
            CiBody::Nak {
                original_sub_id2,
                status_code,
                status_data,
                details,
                message,
            } => {
                if let Some(o) = original_sub_id2 {
                    out.push(*o & 0x7F);
                    out.push(status_code.unwrap_or(0) & 0x7F);
                    out.push(status_data.unwrap_or(0) & 0x7F);
                    out.extend_from_slice(&details.unwrap_or([0; 5]));
                    push_u14(&mut out, message.len() as u16);
                    out.extend_from_slice(message);
                }
            }
            CiBody::ProfileInquiry
            | CiBody::ProcessInquiryCapabilities
            | CiBody::EndOfMidiMessageReport => {}
            CiBody::ReplyToProfileInquiry { enabled, disabled } => {
                push_u14(&mut out, enabled.len() as u16);
                for p in enabled {
                    out.extend_from_slice(&p.0);
                }
                push_u14(&mut out, disabled.len() as u16);
                for p in disabled {
                    out.extend_from_slice(&p.0);
                }
            }
            CiBody::SetProfileOn {
                profile,
                num_channels,
            } => {
                out.extend_from_slice(&profile.0);
                if let Some(n) = num_channels {
                    push_u14(&mut out, *n);
                }
            }
            CiBody::SetProfileOff { profile } => {
                out.extend_from_slice(&profile.0);
                if self.version >= CI_VERSION_2 {
                    push_u14(&mut out, 0); // reserved pair
                }
            }
            CiBody::ProfileEnabledReport {
                profile,
                num_channels,
            }
            | CiBody::ProfileDisabledReport {
                profile,
                num_channels,
            } => {
                out.extend_from_slice(&profile.0);
                if let Some(n) = num_channels {
                    push_u14(&mut out, *n);
                }
            }
            CiBody::ProfileAddedReport { profile } | CiBody::ProfileRemovedReport { profile } => {
                out.extend_from_slice(&profile.0);
            }
            CiBody::ProfileDetailsInquiry { profile, target } => {
                out.extend_from_slice(&profile.0);
                out.push(*target & 0x7F);
            }
            CiBody::ReplyToProfileDetails {
                profile,
                target,
                data,
            } => {
                out.extend_from_slice(&profile.0);
                out.push(*target & 0x7F);
                push_u14(&mut out, data.len() as u16);
                out.extend_from_slice(data);
            }
            CiBody::ProfileSpecificData { profile, data } => {
                out.extend_from_slice(&profile.0);
                push_u28(&mut out, data.len() as u32);
                out.extend_from_slice(data);
            }
            CiBody::PeCapabilities {
                num_requests,
                pe_version,
                ..
            } => {
                out.push(*num_requests & 0x7F);
                if let Some((maj, min)) = pe_version {
                    out.push(*maj & 0x7F);
                    out.push(*min & 0x7F);
                }
            }
            CiBody::PropertyData(p) => push_pe_payload(&mut out, p),
            CiBody::ReplyToProcessInquiryCapabilities { features } => {
                out.push(*features & 0x7F);
            }
            CiBody::MidiMessageReport {
                data_control,
                system,
                other_system,
                channel_controller,
                note_data,
            } => {
                out.push(*data_control & 0x7F);
                out.push(*system & 0x7F);
                out.push(*other_system & 0x7F);
                out.push(*channel_controller & 0x7F);
                out.push(*note_data & 0x7F);
            }
            CiBody::ReplyToMidiMessageReport {
                system,
                other_system,
                channel_controller,
                note_data,
            } => {
                out.push(*system & 0x7F);
                out.push(*other_system & 0x7F);
                out.push(*channel_controller & 0x7F);
                out.push(*note_data & 0x7F);
            }
            CiBody::Unknown { data } => out.extend_from_slice(data),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(m: &CiMessage) {
        let bytes = m.emit();
        let back = CiMessage::parse(&bytes).unwrap();
        assert_eq!(&back, m);
    }

    fn envelope(sub: u8, body: CiBody) -> CiMessage {
        CiMessage {
            device_id: DEVICE_ID_FUNCTION_BLOCK,
            sub_id2: sub,
            version: CI_VERSION_2,
            source_muid: 0x0123_4567 & 0x0FFF_FFFF,
            destination_muid: BROADCAST_MUID,
            body,
        }
    }

    #[test]
    fn discovery_round_trips_and_pins_envelope() {
        let m = envelope(
            sub_id2::DISCOVERY,
            CiBody::Discovery {
                identity: DeviceIdentity {
                    manufacturer: [0x41, 0, 0],
                    family: 0x0102,
                    family_model: 0x0304,
                    software_revision: [1, 2, 3, 4],
                },
                categories: category::PROFILE_CONFIGURATION | category::PROPERTY_EXCHANGE,
                max_sysex_size: 512,
                output_path_id: Some(0),
            },
        );
        let bytes = m.emit();
        // §5.2.1 envelope: 7E <dev> 0D <sub> <ver> <src×4> <dst×4>.
        assert_eq!(&bytes[..5], &[0x7E, 0x7F, 0x0D, 0x70, 0x02]);
        // Broadcast destination MUID = 7F 7F 7F 7F (LSB first).
        assert_eq!(&bytes[9..13], &[0x7F, 0x7F, 0x7F, 0x7F]);
        // Every byte is 7-bit clean.
        assert!(bytes.iter().all(|&b| b < 0x80));
        round_trip(&m);
    }

    #[test]
    fn muid_is_lsb_first_7bit() {
        let m = envelope(
            sub_id2::INVALIDATE_MUID,
            CiBody::InvalidateMuid {
                target_muid: 0x0ABC_DEF0 & 0x0FFF_FFFF,
            },
        );
        let bytes = m.emit();
        let target = &bytes[13..17];
        let v = u32::from(target[0])
            | (u32::from(target[1]) << 7)
            | (u32::from(target[2]) << 14)
            | (u32::from(target[3]) << 21);
        assert_eq!(v, 0x0ABC_DEF0 & 0x0FFF_FFFF);
        round_trip(&m);
    }

    #[test]
    fn profile_inquiry_reply_lists_round_trip() {
        let m = envelope(
            sub_id2::REPLY_TO_PROFILE_INQUIRY,
            CiBody::ReplyToProfileInquiry {
                enabled: vec![ProfileId::standard(0x00, 0x21, 0x01, 0x01)],
                disabled: vec![
                    ProfileId::standard(0x00, 0x22, 0x01, 0x01),
                    ProfileId([0x41, 0x00, 0x00, 0x01, 0x02]),
                ],
            },
        );
        round_trip(&m);
        let bytes = m.emit();
        // cep = 1 (LSB first) right after the envelope.
        assert_eq!(&bytes[13..15], &[0x01, 0x00]);
    }

    #[test]
    fn set_profile_on_v2_carries_channel_count() {
        let m = envelope(
            sub_id2::SET_PROFILE_ON,
            CiBody::SetProfileOn {
                profile: ProfileId::standard(0x00, 0x21, 0x01, 0x01),
                num_channels: Some(0),
            },
        );
        round_trip(&m);
        // A v1-style message without the field also parses.
        let m1 = envelope(
            sub_id2::SET_PROFILE_ON,
            CiBody::SetProfileOn {
                profile: ProfileId::standard(0x00, 0x21, 0x01, 0x01),
                num_channels: None,
            },
        );
        round_trip(&m1);
    }

    #[test]
    fn standard_profile_id_flag() {
        assert!(ProfileId::standard(0, 0x21, 1, 1).is_standard_defined());
        assert!(!ProfileId([0x41, 0, 0, 0, 0]).is_standard_defined());
    }

    #[test]
    fn nak_v1_is_envelope_only() {
        let m = CiMessage {
            device_id: DEVICE_ID_FUNCTION_BLOCK,
            sub_id2: sub_id2::NAK,
            version: CI_VERSION_1,
            source_muid: 1,
            destination_muid: 2,
            body: CiBody::Nak {
                original_sub_id2: None,
                status_code: None,
                status_data: None,
                details: None,
                message: Vec::new(),
            },
        };
        let bytes = m.emit();
        assert_eq!(bytes.len(), 13, "v1 NAK carries no body");
        round_trip(&m);
    }

    #[test]
    fn nak_v2_classification_round_trips() {
        let m = envelope(
            sub_id2::NAK,
            CiBody::Nak {
                original_sub_id2: Some(sub_id2::GET_PROPERTY_DATA),
                status_code: Some(0x41),
                status_data: Some(0),
                details: Some([1, 2, 3, 0, 0]),
                message: b"Message was malformed".to_vec(),
            },
        );
        round_trip(&m);
    }

    #[test]
    fn pe_get_property_data_chunk_round_trips() {
        let m = envelope(
            sub_id2::GET_PROPERTY_DATA,
            CiBody::PropertyData(PePayload {
                request_id: 1,
                header: br#"{"resource":"DeviceInfo"}"#.to_vec(),
                num_chunks: 1,
                this_chunk: 1,
                data: Vec::new(),
            }),
        );
        round_trip(&m);
        // Reply with chunked property data.
        let m = envelope(
            sub_id2::REPLY_TO_GET_PROPERTY_DATA,
            CiBody::PropertyData(PePayload {
                request_id: 1,
                header: br#"{"status":200}"#.to_vec(),
                num_chunks: 2,
                this_chunk: 1,
                data: vec![0x7B, 0x22],
            }),
        );
        round_trip(&m);
    }

    #[test]
    fn midi_message_report_round_trips() {
        let m = envelope(
            sub_id2::MIDI_MESSAGE_REPORT,
            CiBody::MidiMessageReport {
                data_control: 0x7F,
                system: 0b0000_0111,
                other_system: 0,
                channel_controller: 0b0011_1111,
                note_data: 0b0001_1111,
            },
        );
        round_trip(&m);
        let r = envelope(
            sub_id2::REPLY_TO_MIDI_MESSAGE_REPORT,
            CiBody::ReplyToMidiMessageReport {
                system: 0b0000_0011,
                other_system: 0,
                channel_controller: 0b0000_0011,
                note_data: 0b0000_0001,
            },
        );
        round_trip(&r);
        round_trip(&envelope(
            sub_id2::END_OF_MIDI_MESSAGE_REPORT,
            CiBody::EndOfMidiMessageReport,
        ));
    }

    #[test]
    fn unknown_sub_id_survives() {
        let m = envelope(
            0x50,
            CiBody::Unknown {
                data: vec![1, 2, 3],
            },
        );
        round_trip(&m);
    }

    #[test]
    fn trailing_eox_is_tolerated() {
        let mut bytes = envelope(sub_id2::PROFILE_INQUIRY, CiBody::ProfileInquiry).emit();
        bytes.push(0xF7);
        let m = CiMessage::parse(&bytes).unwrap();
        assert_eq!(m.body, CiBody::ProfileInquiry);
    }

    #[test]
    fn truncated_message_is_rejected() {
        let bytes = envelope(sub_id2::DISCOVERY, CiBody::InquiryEndpoint { status: 0 }).emit();
        // Sub-ID says Discovery but the body is one byte — truncated.
        assert!(CiMessage::parse(&bytes).is_err());
        assert!(CiMessage::parse(&[0x7E, 0x7F]).is_err());
        assert!(CiMessage::parse(&[0x7E, 0x7F, 0x0C, 0x70, 0x02]).is_err());
    }

    #[test]
    fn ci_rides_in_sysex7_packets() {
        // The emitted payload is what a UMP SysEx7 run carries.
        let m = envelope(
            sub_id2::SET_PROFILE_ON,
            CiBody::SetProfileOn {
                profile: ProfileId::standard(0x00, 0x21, 0x01, 0x01),
                num_channels: Some(0),
            },
        );
        let payload = m.emit();
        let packets = crate::ump::sysex7_packets(0, &payload).unwrap();
        let mut asm = crate::ump::Sysex7Assembler::new();
        let mut out = None;
        for p in &packets {
            out = asm.push(&crate::ump::Sysex7::decode(p).unwrap());
        }
        assert_eq!(out.as_deref(), Some(&payload[..]));
        assert_eq!(CiMessage::parse(&out.unwrap()).unwrap(), m);
    }
}
