//! Universal MIDI Packet (UMP) Format and MIDI 2.0 Protocol.
//!
//! Implements the container and message vocabulary of the MIDI
//! Association *Universal MIDI Packet (UMP) Format and MIDI 2.0
//! Protocol* specification (M2-104-UM v1.1.2). The UMP is a
//! transport-independent 32-bit-word container that carries **all**
//! MIDI 1.0 Protocol and MIDI 2.0 Protocol messages in a single,
//! uniform packet format.
//!
//! This module is organised in layers:
//!
//! * [`packet`] — the raw [`Ump`] word container: Message Type (MT)
//!   allocation, group/status field extraction, per-MT size derivation,
//!   and a [`UmpStream`] reader that walks a flat `&[u32]` word buffer
//!   into individual packets (handling the variable 1/2/3/4-word sizes).
//! * [`message`] — typed decode of the channel-voice and system
//!   vocabularies: MIDI 1.0 Channel Voice (MT 0x2), System Common /
//!   System Real Time (MT 0x1), Utility (MT 0x0), and the full MIDI 2.0
//!   Channel Voice set (MT 0x4).
//! * [`scaling`] — the spec Appendix D bit-scaling primitives
//!   (Min-Center-Max upscaling, truncating downscaling) plus
//!   default-mode translation between MIDI 1.0 and MIDI 2.0 channel
//!   voice messages.
//!
//! The word order follows the spec §2.1.1 convention: each diagram
//! line is one 32-bit word, most-significant bit leftmost, and the
//! first word carries the Message Type in its top nibble. Byte order
//! on any concrete transport is out of scope (§2.1.1 "Scope of Bit,
//! Byte, and Word Order Guidance"); this module operates purely on
//! `u32` words.

pub mod message;
pub mod packet;
pub mod scaling;

pub use message::{
    Midi1ChannelVoice, Midi2ChannelVoice, SystemMessage, UmpMessage, UtilityMessage,
};
pub use packet::{MessageType, Ump, UmpStream};
