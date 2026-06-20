//! The raw Universal MIDI Packet word container.
//!
//! A UMP is one, two, three, or four 32-bit words (spec §2.1: "Each
//! UMP shall be one, two, three, or four 32-bit words long"). The
//! most-significant 4 bits of the first word are the Message Type (MT),
//! which determines the packet's total size per Table 4
//! ("Message Type (MT) Allocation").
//!
//! This layer is deliberately untyped above the word level: it exposes
//! the common fields (MT, group, status) and the raw words, and offers
//! [`UmpStream`] to split a flat `&[u32]` word buffer into packets. The
//! [`super::message`] layer decodes the per-MT payload.

use oxideav_core::{Error, Result};

/// Universal MIDI Packet Message Type (MT), the top nibble of word 0.
///
/// Table 4 ("Message Type (MT) Allocation") assigns each of the 16 MT
/// values a fixed UMP size, so a reader can determine packet length
/// from the first nibble alone — even for message types it does not yet
/// decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    /// MT 0x0 — Utility Messages (32-bit). Groupless (§1.4).
    Utility,
    /// MT 0x1 — System Real Time and System Common Messages (32-bit).
    System,
    /// MT 0x2 — MIDI 1.0 Channel Voice Messages (32-bit).
    Midi1ChannelVoice,
    /// MT 0x3 — Data Messages including System Exclusive (7-bit) (64-bit).
    Data64,
    /// MT 0x4 — MIDI 2.0 Channel Voice Messages (64-bit).
    Midi2ChannelVoice,
    /// MT 0x5 — Data Messages (System Exclusive 8 / Mixed Data Set) (128-bit).
    Data128,
    /// MT 0xD — Flex Data Messages (128-bit).
    FlexData,
    /// MT 0xF — UMP Stream Messages (128-bit). Groupless (§1.4).
    UmpStream,
    /// A Message Type marked Reserved in Table 4 (0x6..0xC, 0xE).
    /// The associated value is the raw MT nibble; the size is still
    /// well-defined per Table 4 so the packet can be skipped.
    Reserved(u8),
}

impl MessageType {
    /// Decode the 4-bit Message Type nibble (0x0..=0xF).
    ///
    /// Only the low 4 bits of `mt` are considered.
    #[must_use]
    pub fn from_nibble(mt: u8) -> Self {
        match mt & 0x0F {
            0x0 => MessageType::Utility,
            0x1 => MessageType::System,
            0x2 => MessageType::Midi1ChannelVoice,
            0x3 => MessageType::Data64,
            0x4 => MessageType::Midi2ChannelVoice,
            0x5 => MessageType::Data128,
            0xD => MessageType::FlexData,
            0xF => MessageType::UmpStream,
            other => MessageType::Reserved(other),
        }
    }

    /// The raw 4-bit MT nibble for this Message Type.
    #[must_use]
    pub fn nibble(self) -> u8 {
        match self {
            MessageType::Utility => 0x0,
            MessageType::System => 0x1,
            MessageType::Midi1ChannelVoice => 0x2,
            MessageType::Data64 => 0x3,
            MessageType::Midi2ChannelVoice => 0x4,
            MessageType::Data128 => 0x5,
            MessageType::FlexData => 0xD,
            MessageType::UmpStream => 0xF,
            MessageType::Reserved(n) => n & 0x0F,
        }
    }

    /// Number of 32-bit words this packet occupies, per Table 4.
    ///
    /// | MT          | words | MT          | words |
    /// |-------------|-------|-------------|-------|
    /// | 0x0..0x2    | 1     | 0x8..0xA    | 2     |
    /// | 0x3..0x4    | 2     | 0xB..0xC    | 3     |
    /// | 0x5         | 4     | 0xD..0xF    | 4     |
    /// | 0x6..0x7    | 1     |             |       |
    #[must_use]
    pub fn word_count(self) -> usize {
        match self.nibble() {
            0x0 | 0x1 | 0x2 | 0x6 | 0x7 => 1,
            0x3 | 0x4 | 0x8 | 0x9 | 0xA => 2,
            0xB | 0xC => 3,
            // 0x5, 0xD, 0xE, 0xF
            _ => 4,
        }
    }

    /// Whether this Message Type carries a Group field (§1.4 / §2.1.2).
    ///
    /// Utility (MT 0x0) and UMP Stream (MT 0xF) are Groupless; all other
    /// defined Message Types address one of 16 Groups via word-0 bits
    /// 24..27.
    #[must_use]
    pub fn has_group(self) -> bool {
        !matches!(self, MessageType::Utility | MessageType::UmpStream)
    }
}

/// A single Universal MIDI Packet: 1 to 4 32-bit words.
///
/// Stored as a fixed `[u32; 4]` plus a `len` so the type is `Copy` and
/// allocation-free. Only the first `len` words are meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ump {
    words: [u32; 4],
    len: u8,
}

impl Ump {
    /// Build a UMP from a word slice, validating the length against the
    /// Message Type's required size (Table 4).
    ///
    /// Returns [`Error::InvalidData`] if `words` is empty, longer than 4
    /// words, or shorter than the size the leading Message Type
    /// mandates.
    pub fn from_words(words: &[u32]) -> Result<Self> {
        if words.is_empty() {
            return Err(Error::invalid("UMP: empty word slice"));
        }
        if words.len() > 4 {
            return Err(Error::invalid("UMP: more than 4 words"));
        }
        let mt = MessageType::from_nibble((words[0] >> 28) as u8);
        let need = mt.word_count();
        if words.len() < need {
            return Err(Error::invalid(format!(
                "UMP: MT {:#x} needs {} words, got {}",
                mt.nibble(),
                need,
                words.len()
            )));
        }
        let mut buf = [0u32; 4];
        buf[..need].copy_from_slice(&words[..need]);
        Ok(Ump {
            words: buf,
            len: need as u8,
        })
    }

    /// Construct directly from already-validated parts (internal use by
    /// the encoders in [`super::message`]).
    #[must_use]
    pub(crate) fn from_parts(words: [u32; 4], len: usize) -> Self {
        Ump {
            words,
            len: len as u8,
        }
    }

    /// The meaningful words of this packet (`1..=4` long).
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words[..self.len as usize]
    }

    /// Word 0, always present.
    #[must_use]
    pub fn word0(&self) -> u32 {
        self.words[0]
    }

    /// The decoded Message Type.
    #[must_use]
    pub fn message_type(&self) -> MessageType {
        MessageType::from_nibble((self.words[0] >> 28) as u8)
    }

    /// The 4-bit Group field (word-0 bits 24..27), or `None` for
    /// Groupless Message Types (Utility, UMP Stream).
    #[must_use]
    pub fn group(&self) -> Option<u8> {
        if self.message_type().has_group() {
            Some(((self.words[0] >> 24) & 0x0F) as u8)
        } else {
            None
        }
    }

    /// The 8-bit Status byte (word-0 bits 16..23).
    ///
    /// For channel-voice Message Types this byte packs a 4-bit opcode in
    /// the high nibble and a 4-bit Channel in the low nibble. For System
    /// (MT 0x1) it is the full System status (0xF0..0xFF). Utility
    /// (MT 0x0) uses only the low nibble as a status code; the high
    /// nibble is Reserved. UMP Stream / Flex Data lay out their status
    /// differently — callers should decode those per their own format.
    #[must_use]
    pub fn status_byte(&self) -> u8 {
        ((self.words[0] >> 16) & 0xFF) as u8
    }

    /// The high nibble of the status byte — the channel-voice opcode.
    #[must_use]
    pub fn opcode(&self) -> u8 {
        ((self.words[0] >> 20) & 0x0F) as u8
    }

    /// The low nibble of the status byte — the MIDI Channel (0..15) for
    /// channel-voice Message Types.
    #[must_use]
    pub fn channel(&self) -> u8 {
        ((self.words[0] >> 16) & 0x0F) as u8
    }

    /// The number of 32-bit words in this packet.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Always false — a UMP is at minimum one word. Present for the
    /// clippy `len_without_is_empty` lint.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// A forward reader that splits a flat `&[u32]` word buffer into
/// individual [`Ump`] packets.
///
/// Each `next()` reads the leading word's Message Type, determines the
/// packet size from Table 4, and yields a packet of exactly that many
/// words — so a heterogeneous stream of mixed-size packets is walked
/// correctly without a separate length prefix (the UMP format is
/// self-delimiting through its MT field).
///
/// A trailing run of words too short for the Message Type they begin
/// surfaces as an `Err` on the final `next()`; well-formed prefixes are
/// still yielded.
#[derive(Debug, Clone)]
pub struct UmpStream<'a> {
    words: &'a [u32],
    pos: usize,
}

impl<'a> UmpStream<'a> {
    /// Wrap a word buffer for packet-by-packet reading.
    #[must_use]
    pub fn new(words: &'a [u32]) -> Self {
        UmpStream { words, pos: 0 }
    }

    /// Words not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> &'a [u32] {
        &self.words[self.pos..]
    }
}

impl Iterator for UmpStream<'_> {
    type Item = Result<Ump>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.words.len() {
            return None;
        }
        let mt = MessageType::from_nibble((self.words[self.pos] >> 28) as u8);
        let need = mt.word_count();
        let avail = self.words.len() - self.pos;
        if avail < need {
            // Consume the remainder so the iterator terminates after
            // surfacing the truncation error exactly once.
            let slice = &self.words[self.pos..];
            self.pos = self.words.len();
            return Some(Ump::from_words(slice));
        }
        let slice = &self.words[self.pos..self.pos + need];
        self.pos += need;
        Some(Ump::from_words(slice))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_size_table_matches_spec_table_4() {
        // Spec Table 4 sizes, in words.
        assert_eq!(MessageType::Utility.word_count(), 1);
        assert_eq!(MessageType::System.word_count(), 1);
        assert_eq!(MessageType::Midi1ChannelVoice.word_count(), 1);
        assert_eq!(MessageType::Data64.word_count(), 2);
        assert_eq!(MessageType::Midi2ChannelVoice.word_count(), 2);
        assert_eq!(MessageType::Data128.word_count(), 4);
        assert_eq!(MessageType::FlexData.word_count(), 4);
        assert_eq!(MessageType::UmpStream.word_count(), 4);
        // Reserved 0x6/0x7 = 1 word; 0x8..0xA = 2; 0xB/0xC = 3; 0xE = 4.
        assert_eq!(MessageType::Reserved(0x6).word_count(), 1);
        assert_eq!(MessageType::Reserved(0x7).word_count(), 1);
        assert_eq!(MessageType::Reserved(0x8).word_count(), 2);
        assert_eq!(MessageType::Reserved(0xA).word_count(), 2);
        assert_eq!(MessageType::Reserved(0xB).word_count(), 3);
        assert_eq!(MessageType::Reserved(0xC).word_count(), 3);
        assert_eq!(MessageType::Reserved(0xE).word_count(), 4);
    }

    #[test]
    fn nibble_round_trips() {
        for n in 0u8..=0x0F {
            assert_eq!(MessageType::from_nibble(n).nibble(), n);
        }
    }

    #[test]
    fn groupless_message_types() {
        assert!(!MessageType::Utility.has_group());
        assert!(!MessageType::UmpStream.has_group());
        assert!(MessageType::System.has_group());
        assert!(MessageType::Midi1ChannelVoice.has_group());
        assert!(MessageType::Midi2ChannelVoice.has_group());
    }

    #[test]
    fn parse_midi1_note_on_word() {
        // MT=2, group=3, status=0x91 (Note On ch 1), note 0x40, vel 0x7F.
        let w = 0x2391_407F;
        let p = Ump::from_words(&[w]).unwrap();
        assert_eq!(p.message_type(), MessageType::Midi1ChannelVoice);
        assert_eq!(p.group(), Some(3));
        assert_eq!(p.status_byte(), 0x91);
        assert_eq!(p.opcode(), 0x9);
        assert_eq!(p.channel(), 1);
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn utility_is_groupless() {
        // MT=0, NOOP.
        let p = Ump::from_words(&[0x0000_0000]).unwrap();
        assert_eq!(p.message_type(), MessageType::Utility);
        assert_eq!(p.group(), None);
    }

    #[test]
    fn from_words_rejects_truncated_64bit() {
        // MT=4 (MIDI 2.0 CV) needs 2 words, give 1.
        let err = Ump::from_words(&[0x4090_0000]).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn from_words_rejects_empty_and_overlong() {
        assert!(matches!(
            Ump::from_words(&[]).unwrap_err(),
            Error::InvalidData(_)
        ));
        assert!(matches!(
            Ump::from_words(&[0, 0, 0, 0, 0]).unwrap_err(),
            Error::InvalidData(_)
        ));
    }

    #[test]
    fn stream_walks_mixed_sizes() {
        // A 1-word MIDI 1.0 CV, then a 2-word MIDI 2.0 CV, then a
        // 1-word System message.
        let words = [
            0x2090_407F, // MT2 note on
            0x4090_4000, // MT4 note on word0
            0x7FFF_0000, // MT4 note on word1
            0x10F8_0000, // MT1 system real time (timing clock)
        ];
        let mut s = UmpStream::new(&words);
        let a = s.next().unwrap().unwrap();
        assert_eq!(a.message_type(), MessageType::Midi1ChannelVoice);
        assert_eq!(a.len(), 1);
        let b = s.next().unwrap().unwrap();
        assert_eq!(b.message_type(), MessageType::Midi2ChannelVoice);
        assert_eq!(b.len(), 2);
        assert_eq!(b.words(), &[0x4090_4000, 0x7FFF_0000]);
        let c = s.next().unwrap().unwrap();
        assert_eq!(c.message_type(), MessageType::System);
        assert!(s.next().is_none());
    }

    #[test]
    fn stream_surfaces_trailing_truncation_once() {
        // Valid 1-word packet, then a 2-word MT begun with only 1 word.
        let words = [0x2090_407F, 0x4090_4000];
        let mut s = UmpStream::new(&words);
        assert!(s.next().unwrap().is_ok());
        assert!(s.next().unwrap().is_err());
        assert!(s.next().is_none());
    }
}
