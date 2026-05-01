//! Standard MIDI File (SMF) parser.
//!
//! SMF is the on-disk container for MIDI sequences (`.mid` / `.midi`).
//! A file is a sequence of RIFF-style chunks, each `<4-char tag><u32 BE
//! length><payload>`:
//!
//! - **`MThd`** — the file header. Always 6 bytes of payload:
//!   * `format` (u16 BE): 0 = single track; 1 = multi-track, all tracks
//!     played simultaneously; 2 = multi-track, tracks are independent.
//!   * `ntrks` (u16 BE): number of `MTrk` chunks that follow.
//!   * `division` (u16 BE): time division. If the high bit is clear,
//!     the value is "ticks per quarter note" (1..=32767). If the high
//!     bit is set, the upper byte is a negative SMPTE frame rate
//!     (-24, -25, -29, -30) and the lower byte is "ticks per frame".
//! - **`MTrk`** — a track. Payload is a sequence of `<delta-time><event>`
//!   pairs, terminated by an `End-of-Track` meta event (`FF 2F 00`).
//!   Multiple tracks can appear; format-1 tracks share one timebase, so
//!   playback is the merged time-ordered union of all tracks.
//!
//! A delta time is a **variable-length quantity** (VLQ): 1-4 bytes,
//! big-endian, with bit 7 set on every byte except the last. The MIDI
//! file spec restricts VLQs to a maximum of 4 bytes (28-bit value), so
//! `0x0FFFFFFF` is the largest legal value. We enforce that bound to
//! prevent malicious files from forcing unbounded reads.
//!
//! Three event flavours appear inside a track:
//!
//! 1. **MIDI channel events** — the same status/data bytes used on the
//!    wire (`8n` Note Off, `9n` Note On, `An` Poly Aftertouch, `Bn`
//!    Control Change, `Cn` Program Change, `Dn` Channel Aftertouch,
//!    `En` Pitch Bend). Running status applies inside a track: if a
//!    byte where a status would appear has its high bit clear, the
//!    previous channel-event status is reused. Running status is
//!    invalidated by every meta event, sysex event, and the start of a
//!    new track. (Real-Time messages do not appear in SMF, so we do not
//!    need the wire-format "real-time messages preserve running status"
//!    rule.)
//! 2. **Sysex events** — `F0 <varlen length> <data...>` (with the
//!    optional trailing `F7`) or `F7 <varlen length> <data...>` for
//!    "escape" / continuation sysex. The varlen-declared length must
//!    not exceed the bytes remaining in the chunk.
//! 3. **Meta events** — `FF <type> <varlen length> <data...>`. Common
//!    types we decode: `00` sequence number, `01..=0F` text events
//!    (text, copyright, track name, instrument name, lyric, marker, cue
//!    point, …), `20` channel prefix, `21` MIDI port, `2F` end of
//!    track, `51` set tempo (3-byte microseconds-per-quarter), `54`
//!    SMPTE offset, `58` time signature, `59` key signature, `7F`
//!    sequencer-specific. Unknown meta types are preserved verbatim
//!    (`MetaEvent::Unknown { type_byte, data }`) — the spec mandates
//!    that decoders ignore unknown meta types rather than fail.
//!
//! # Bounds and limits
//!
//! Every parser entry point operates on a borrowed slice and never
//! reads past its end. Three caps are enforced before allocating:
//!
//! - VLQs accept at most 4 bytes (spec).
//! - A varlen-declared length (sysex / meta payload) must fit in the
//!   bytes remaining in the enclosing chunk.
//! - Total decoded events per file are capped at [`MAX_EVENTS_PER_FILE`]
//!   (one million) to avoid runaway allocations on a malicious file.
//!
//! These three caps together mean a SMF parse is bounded in both time
//! and memory by the input length.

use oxideav_core::{Error, Result};

/// Maximum legal VLQ length per the SMF spec (yields a 28-bit value).
pub const MAX_VLQ_BYTES: usize = 4;

/// Maximum total events the parser will emit for a single file.
/// Beyond this, we abort with `Error::InvalidData` rather than risk
/// unbounded allocation.
pub const MAX_EVENTS_PER_FILE: usize = 1_000_000;

// ───────────────────────── public types ─────────────────────────

/// SMF file format value from the `MThd` chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmfFormat {
    /// 0 — a single multi-channel track.
    SingleTrack,
    /// 1 — multiple tracks played simultaneously (one shared timebase).
    MultiTrackSimultaneous,
    /// 2 — multiple independent single-track sequences.
    MultiTrackIndependent,
}

impl SmfFormat {
    fn from_u16(v: u16) -> Result<Self> {
        match v {
            0 => Ok(Self::SingleTrack),
            1 => Ok(Self::MultiTrackSimultaneous),
            2 => Ok(Self::MultiTrackIndependent),
            other => Err(Error::invalid(format!(
                "SMF: unknown format value {other} (expected 0, 1, or 2)",
            ))),
        }
    }
}

/// SMF time division — either musical (ticks per quarter note) or
/// real-time (SMPTE frames + subdivisions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Division {
    /// Musical time. Value is "ticks per quarter note" — combined with
    /// the most recent tempo meta event to yield real time. Range is
    /// `1..=0x7FFF` per the spec.
    TicksPerQuarter(u16),
    /// SMPTE time. `frames_per_second` is the (positive) frame rate
    /// (24, 25, 29, or 30 — note that 29 stands for 29.97 drop-frame in
    /// the spec) and `ticks_per_frame` is the per-frame subdivision.
    Smpte {
        frames_per_second: u8,
        ticks_per_frame: u8,
    },
}

/// Header chunk parsed from the leading `MThd`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmfHeader {
    pub format: SmfFormat,
    pub ntrks: u16,
    pub division: Division,
}

/// One decoded event with its absolute delta-time (relative to the
/// previous event in the same track).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackEvent {
    /// Delta in division units from the previous event.
    pub delta: u32,
    pub kind: Event,
}

/// The decoded body of a track event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// MIDI channel-voice message.
    Channel(ChannelMessage),
    /// `F0 ... F7` sysex (or `F7` continuation/escape).
    Sysex {
        /// `true` for `F7` (continuation/escape), `false` for `F0` (start).
        escape: bool,
        data: Vec<u8>,
    },
    /// `FF nn <varlen> ...` meta event.
    Meta(MetaEvent),
}

/// Decoded channel-voice message. `channel` is the low nibble of the
/// status byte, in `0..=15` (channel "1" in human-facing tools).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelMessage {
    pub channel: u8,
    pub body: ChannelBody,
}

/// The shape of a channel-voice message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelBody {
    /// `8n key vel` — release the key.
    NoteOff { key: u8, velocity: u8 },
    /// `9n key vel` — strike the key. A velocity of 0 is conventionally
    /// treated as Note-Off (the spec's "running status optimisation");
    /// we surface it as-is and let the consumer decide.
    NoteOn { key: u8, velocity: u8 },
    /// `An key pressure` — polyphonic key pressure (per-key aftertouch).
    PolyAftertouch { key: u8, pressure: u8 },
    /// `Bn cc value` — control change.
    ControlChange { controller: u8, value: u8 },
    /// `Cn program` — program change. Single data byte.
    ProgramChange { program: u8 },
    /// `Dn pressure` — channel pressure (mono aftertouch). Single data byte.
    ChannelAftertouch { pressure: u8 },
    /// `En lsb msb` — pitch bend. Combined as a 14-bit value, centre
    /// `0x2000`.
    PitchBend { value: u16 },
}

/// Decoded meta event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetaEvent {
    /// `FF 00 02 ssss` — explicit sequence number.
    SequenceNumber(u16),
    /// `FF 01..=0F len text` — text-flavour meta. `kind` carries the
    /// raw type byte (`0x01..=0x0F`) and `text` holds the bytes
    /// untouched (callers decode as Latin-1 or UTF-8 per their needs).
    Text { kind: u8, text: Vec<u8> },
    /// `FF 20 01 cc` — channel prefix (deprecated but seen in the wild).
    ChannelPrefix(u8),
    /// `FF 21 01 pp` — MIDI port.
    Port(u8),
    /// `FF 2F 00` — end of track. The parser stops the track here.
    EndOfTrack,
    /// `FF 51 03 tt tt tt` — set tempo, in microseconds per quarter note.
    Tempo(u32),
    /// `FF 54 05 hr mn se fr ff` — SMPTE offset of the track's start.
    SmpteOffset {
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        subframes: u8,
    },
    /// `FF 58 04 nn dd cc bb` — time signature. `numerator` is `nn`,
    /// `denominator_pow2` is `dd` (so the denominator is `1 << dd`),
    /// `clocks_per_click` is `cc`, `notated_32nd_per_quarter` is `bb`.
    TimeSignature {
        numerator: u8,
        denominator_pow2: u8,
        clocks_per_click: u8,
        notated_32nd_per_quarter: u8,
    },
    /// `FF 59 02 sf mi` — key signature. `sharps_flats` is signed
    /// (`-7..=+7`, negative = flats); `mode` is 0 (major) or 1 (minor).
    KeySignature { sharps_flats: i8, mode: u8 },
    /// `FF 7F len data` — sequencer-specific.
    SequencerSpecific(Vec<u8>),
    /// Anything else. Per spec, unknown meta types are ignored — we
    /// preserve them so callers with deeper knowledge can recover them.
    Unknown { type_byte: u8, data: Vec<u8> },
}

/// One track parsed out of an `MTrk` chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Track {
    pub events: Vec<TrackEvent>,
}

/// A fully parsed SMF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmfFile {
    pub header: SmfHeader,
    pub tracks: Vec<Track>,
}

// ───────────────────────── public entry point ─────────────────────────

/// Parse a complete SMF file from a byte slice.
///
/// Returns `Error::InvalidData` for any structural problem (bad chunk
/// tag, truncated payload, illegal VLQ, illegal status byte, varlen
/// declaring more bytes than remain in the chunk, total event count
/// over [`MAX_EVENTS_PER_FILE`], …).
///
/// Memory cost is bounded by the input length plus per-event overhead;
/// the parser never trusts a length field over the bytes it can
/// actually read.
pub fn parse(bytes: &[u8]) -> Result<SmfFile> {
    let mut cursor = Cursor::new(bytes);
    let header = parse_header(&mut cursor)?;

    let mut tracks: Vec<Track> = Vec::new();
    let mut total_events: usize = 0;

    // The spec says to stop reading when `ntrks` chunks have been
    // collected, but it also says to skip unknown chunks — so we walk
    // every chunk and only recognise `MTrk` for tracks.
    while !cursor.is_empty() {
        let tag = cursor.take(4)?;
        let chunk_len = read_u32_be(&mut cursor)? as usize;
        if chunk_len > cursor.remaining() {
            return Err(Error::invalid(format!(
                "SMF: chunk '{}' declares {chunk_len} bytes but only {} remain",
                fmt_tag(tag),
                cursor.remaining(),
            )));
        }
        let payload = cursor.take(chunk_len)?;
        if tag == b"MTrk" {
            let track = parse_track(payload, total_events)?;
            total_events += track.events.len();
            if total_events > MAX_EVENTS_PER_FILE {
                return Err(Error::invalid(format!(
                    "SMF: cumulative event count {total_events} exceeds cap of \
                     {MAX_EVENTS_PER_FILE}",
                )));
            }
            tracks.push(track);
        }
        // unknown chunk — silently skip per spec
    }

    if (tracks.len() as u16) != header.ntrks {
        // Some files in the wild get this wrong; report rather than
        // bail. Caller can decide whether to trust `header.ntrks`.
        // We DO trust the actual track count we walked.
    }

    Ok(SmfFile { header, tracks })
}

// ───────────────────────── header ─────────────────────────

fn parse_header(cursor: &mut Cursor<'_>) -> Result<SmfHeader> {
    let tag = cursor.take(4)?;
    if tag != b"MThd" {
        return Err(Error::invalid(format!(
            "SMF: expected 'MThd' header chunk, got '{}'",
            fmt_tag(tag),
        )));
    }
    let chunk_len = read_u32_be(cursor)? as usize;
    if chunk_len < 6 {
        return Err(Error::invalid(format!(
            "SMF: MThd chunk length is {chunk_len}, expected at least 6",
        )));
    }
    if chunk_len > cursor.remaining() {
        return Err(Error::invalid(format!(
            "SMF: MThd declares {chunk_len} bytes but only {} remain",
            cursor.remaining(),
        )));
    }
    let body = cursor.take(chunk_len)?;
    let format = SmfFormat::from_u16(u16::from_be_bytes([body[0], body[1]]))?;
    let ntrks = u16::from_be_bytes([body[2], body[3]]);
    let div_raw = u16::from_be_bytes([body[4], body[5]]);
    let division = if div_raw & 0x8000 == 0 {
        if div_raw == 0 {
            return Err(Error::invalid(
                "SMF: division of 0 ticks-per-quarter is not legal",
            ));
        }
        Division::TicksPerQuarter(div_raw)
    } else {
        // Upper byte is the negative SMPTE frame rate (two's complement).
        let upper = (div_raw >> 8) as i8;
        let frames_per_second = (-(upper as i16)) as u8;
        let ticks_per_frame = (div_raw & 0xFF) as u8;
        if !matches!(frames_per_second, 24 | 25 | 29 | 30) {
            return Err(Error::invalid(format!(
                "SMF: SMPTE frame rate {frames_per_second} not in {{24, 25, 29, 30}}",
            )));
        }
        Division::Smpte {
            frames_per_second,
            ticks_per_frame,
        }
    };

    Ok(SmfHeader {
        format,
        ntrks,
        division,
    })
}

// ───────────────────────── track ─────────────────────────

fn parse_track(payload: &[u8], events_so_far: usize) -> Result<Track> {
    let mut cursor = Cursor::new(payload);
    let mut events: Vec<TrackEvent> = Vec::new();
    let mut running_status: Option<u8> = None;
    let mut local_total = events_so_far;

    while !cursor.is_empty() {
        let delta = read_vlq(&mut cursor)?;
        let evt = read_event(&mut cursor, &mut running_status)?;
        let is_eot = matches!(&evt, Event::Meta(MetaEvent::EndOfTrack));
        events.push(TrackEvent { delta, kind: evt });
        local_total = local_total.saturating_add(1);
        if local_total > MAX_EVENTS_PER_FILE {
            return Err(Error::invalid(format!(
                "SMF: cumulative event count {local_total} exceeds cap of \
                 {MAX_EVENTS_PER_FILE}",
            )));
        }
        if is_eot {
            // Spec says End-of-Track must be the last event; trailing
            // bytes after it are ignored.
            break;
        }
    }

    Ok(Track { events })
}

fn read_event(cursor: &mut Cursor<'_>, running: &mut Option<u8>) -> Result<Event> {
    let first = cursor.peek_u8()?;
    if first == 0xFF {
        // Meta event. Invalidates running status.
        cursor.advance(1)?;
        let type_byte = cursor.read_u8()?;
        let len = read_vlq(cursor)? as usize;
        if len > cursor.remaining() {
            return Err(Error::invalid(format!(
                "SMF: meta event 0x{type_byte:02X} declares {len} bytes but only {} remain",
                cursor.remaining(),
            )));
        }
        let data = cursor.take(len)?;
        *running = None;
        Ok(Event::Meta(parse_meta(type_byte, data)?))
    } else if first == 0xF0 || first == 0xF7 {
        cursor.advance(1)?;
        let len = read_vlq(cursor)? as usize;
        if len > cursor.remaining() {
            return Err(Error::invalid(format!(
                "SMF: sysex 0x{first:02X} declares {len} bytes but only {} remain",
                cursor.remaining(),
            )));
        }
        let data = cursor.take(len)?.to_vec();
        *running = None;
        Ok(Event::Sysex {
            escape: first == 0xF7,
            data,
        })
    } else if first & 0x80 != 0 {
        // New status byte for a channel event.
        cursor.advance(1)?;
        // Channel-voice status bytes only — `F1..=F6` are System Common
        // and have no place in SMF tracks.
        if first >= 0xF1 {
            return Err(Error::invalid(format!(
                "SMF: status byte 0x{first:02X} is System Common/Real-Time, \
                 not legal inside an MTrk chunk",
            )));
        }
        *running = Some(first);
        read_channel_message(cursor, first)
    } else {
        // Running status: high bit clear, reuse previous status.
        let status = running.ok_or_else(|| {
            Error::invalid(format!(
                "SMF: data byte 0x{first:02X} appeared without a prior status byte \
                 (no running status to inherit)",
            ))
        })?;
        read_channel_message(cursor, status)
    }
}

fn read_channel_message(cursor: &mut Cursor<'_>, status: u8) -> Result<Event> {
    let channel = status & 0x0F;
    let kind = status & 0xF0;
    let body = match kind {
        0x80 => {
            let key = cursor.read_data_byte()?;
            let velocity = cursor.read_data_byte()?;
            ChannelBody::NoteOff { key, velocity }
        }
        0x90 => {
            let key = cursor.read_data_byte()?;
            let velocity = cursor.read_data_byte()?;
            ChannelBody::NoteOn { key, velocity }
        }
        0xA0 => {
            let key = cursor.read_data_byte()?;
            let pressure = cursor.read_data_byte()?;
            ChannelBody::PolyAftertouch { key, pressure }
        }
        0xB0 => {
            let controller = cursor.read_data_byte()?;
            let value = cursor.read_data_byte()?;
            ChannelBody::ControlChange { controller, value }
        }
        0xC0 => {
            let program = cursor.read_data_byte()?;
            ChannelBody::ProgramChange { program }
        }
        0xD0 => {
            let pressure = cursor.read_data_byte()?;
            ChannelBody::ChannelAftertouch { pressure }
        }
        0xE0 => {
            let lsb = cursor.read_data_byte()? as u16;
            let msb = cursor.read_data_byte()? as u16;
            ChannelBody::PitchBend {
                value: (msb << 7) | lsb,
            }
        }
        _ => unreachable!("status nibble {kind:02X} is not a channel-voice message"),
    };
    Ok(Event::Channel(ChannelMessage { channel, body }))
}

fn parse_meta(type_byte: u8, data: &[u8]) -> Result<MetaEvent> {
    Ok(match type_byte {
        0x00 if data.len() == 2 => {
            MetaEvent::SequenceNumber(u16::from_be_bytes([data[0], data[1]]))
        }
        0x01..=0x0F => MetaEvent::Text {
            kind: type_byte,
            text: data.to_vec(),
        },
        0x20 if data.len() == 1 => MetaEvent::ChannelPrefix(data[0]),
        0x21 if data.len() == 1 => MetaEvent::Port(data[0]),
        0x2F if data.is_empty() => MetaEvent::EndOfTrack,
        0x51 if data.len() == 3 => {
            MetaEvent::Tempo(((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32))
        }
        0x54 if data.len() == 5 => MetaEvent::SmpteOffset {
            hours: data[0],
            minutes: data[1],
            seconds: data[2],
            frames: data[3],
            subframes: data[4],
        },
        0x58 if data.len() == 4 => MetaEvent::TimeSignature {
            numerator: data[0],
            denominator_pow2: data[1],
            clocks_per_click: data[2],
            notated_32nd_per_quarter: data[3],
        },
        0x59 if data.len() == 2 => MetaEvent::KeySignature {
            sharps_flats: data[0] as i8,
            mode: data[1],
        },
        0x7F => MetaEvent::SequencerSpecific(data.to_vec()),
        _ => MetaEvent::Unknown {
            type_byte,
            data: data.to_vec(),
        },
    })
}

// ───────────────────────── helpers ─────────────────────────

/// Cursor over a borrowed byte slice. The methods all check bounds and
/// surface `Error::InvalidData` rather than panicking — every length /
/// offset that can come from input must be validated against this.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::invalid(format!(
                "SMF: short read — wanted {n} bytes, {} remain",
                self.remaining()
            )));
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Like `read_u8` but enforces the MIDI rule that a data byte's
    /// high bit must be clear.
    fn read_data_byte(&mut self) -> Result<u8> {
        let b = self.read_u8()?;
        if b & 0x80 != 0 {
            return Err(Error::invalid(format!(
                "SMF: expected data byte (high bit clear), got 0x{b:02X}",
            )));
        }
        Ok(b)
    }

    fn peek_u8(&self) -> Result<u8> {
        if self.is_empty() {
            return Err(Error::invalid("SMF: short read — wanted 1 byte, 0 remain"));
        }
        Ok(self.bytes[self.pos])
    }

    fn advance(&mut self, n: usize) -> Result<()> {
        if self.remaining() < n {
            return Err(Error::invalid(format!(
                "SMF: short advance — wanted {n} bytes, {} remain",
                self.remaining()
            )));
        }
        self.pos += n;
        Ok(())
    }
}

fn read_u32_be(cursor: &mut Cursor<'_>) -> Result<u32> {
    let s = cursor.take(4)?;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// Read a SMF variable-length quantity. Bounded to [`MAX_VLQ_BYTES`]
/// per the spec — every byte but the last has bit 7 set; the final
/// byte clears bit 7.
fn read_vlq(cursor: &mut Cursor<'_>) -> Result<u32> {
    let mut value: u32 = 0;
    for i in 0..MAX_VLQ_BYTES {
        let b = cursor.read_u8()?;
        value = (value << 7) | ((b & 0x7F) as u32);
        if b & 0x80 == 0 {
            return Ok(value);
        }
        if i == MAX_VLQ_BYTES - 1 {
            // Continuation bit was still set on the 4th byte — over the cap.
            return Err(Error::invalid(format!(
                "SMF: VLQ exceeded {MAX_VLQ_BYTES}-byte cap (continuation bit set on final byte)",
            )));
        }
    }
    unreachable!("loop returns or errors before this point");
}

fn fmt_tag(tag: &[u8]) -> String {
    String::from_utf8_lossy(tag).into_owned()
}

// ───────────────────────── tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a value as a SMF VLQ. Caller's responsibility to keep
    /// the value in `0..=0x0FFFFFFF`.
    fn encode_vlq(mut v: u32) -> Vec<u8> {
        let mut buf = vec![v & 0x7F];
        v >>= 7;
        while v != 0 {
            buf.push((v & 0x7F) | 0x80);
            v >>= 7;
        }
        // we built it LSB-first; reverse to get MSB-first wire order.
        buf.into_iter().rev().map(|b| b as u8).collect()
    }

    fn header_chunk(format: u16, ntrks: u16, division: u16) -> Vec<u8> {
        let mut b = vec![];
        b.extend_from_slice(b"MThd");
        b.extend_from_slice(&6u32.to_be_bytes());
        b.extend_from_slice(&format.to_be_bytes());
        b.extend_from_slice(&ntrks.to_be_bytes());
        b.extend_from_slice(&division.to_be_bytes());
        b
    }

    fn track_chunk(events: &[u8]) -> Vec<u8> {
        let mut b = vec![];
        b.extend_from_slice(b"MTrk");
        b.extend_from_slice(&(events.len() as u32).to_be_bytes());
        b.extend_from_slice(events);
        b
    }

    #[test]
    fn vlq_one_byte() {
        let mut c = Cursor::new(&[0x00]);
        assert_eq!(read_vlq(&mut c).unwrap(), 0);
        let mut c = Cursor::new(&[0x40]);
        assert_eq!(read_vlq(&mut c).unwrap(), 0x40);
        let mut c = Cursor::new(&[0x7F]);
        assert_eq!(read_vlq(&mut c).unwrap(), 0x7F);
    }

    #[test]
    fn vlq_multi_byte() {
        // From the SMF spec's worked examples:
        //   0x00000080 → 81 00
        //   0x00002000 → C0 00
        //   0x00003FFF → FF 7F
        //   0x00100000 → C0 80 00
        //   0x001FFFFF → FF FF 7F
        //   0x00200000 → C0 80 80 00
        //   0x0FFFFFFF → FF FF FF 7F
        let cases: &[(u32, &[u8])] = &[
            (0x80, &[0x81, 0x00]),
            (0x2000, &[0xC0, 0x00]),
            (0x3FFF, &[0xFF, 0x7F]),
            (0x10_0000, &[0xC0, 0x80, 0x00]),
            (0x1F_FFFF, &[0xFF, 0xFF, 0x7F]),
            (0x20_0000, &[0x81, 0x80, 0x80, 0x00]),
            (0x0FFF_FFFF, &[0xFF, 0xFF, 0xFF, 0x7F]),
        ];
        for (v, bytes) in cases {
            let mut c = Cursor::new(bytes);
            assert_eq!(
                read_vlq(&mut c).unwrap(),
                *v,
                "decode VLQ {v:#x} from {bytes:?}",
            );
            assert_eq!(encode_vlq(*v), bytes.to_vec(), "round-trip VLQ {v:#x}");
        }
    }

    #[test]
    fn vlq_rejects_5_byte() {
        // Continuation bit on every byte: should be rejected at the 4th byte.
        let mut c = Cursor::new(&[0xFF, 0xFF, 0xFF, 0xFF, 0x7F]);
        let err = read_vlq(&mut c).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn header_format_0_ticks_per_quarter() {
        let mut blob = header_chunk(0, 1, 480);
        // empty track
        blob.extend(track_chunk(&[0x00, 0xFF, 0x2F, 0x00]));
        let smf = parse(&blob).unwrap();
        assert_eq!(smf.header.format, SmfFormat::SingleTrack);
        assert_eq!(smf.header.ntrks, 1);
        assert_eq!(smf.header.division, Division::TicksPerQuarter(480));
        assert_eq!(smf.tracks.len(), 1);
        // The lone event is end-of-track.
        assert_eq!(smf.tracks[0].events.len(), 1);
        assert!(matches!(
            smf.tracks[0].events[0].kind,
            Event::Meta(MetaEvent::EndOfTrack)
        ));
    }

    #[test]
    fn header_smpte_division() {
        // -25 fps with 40 ticks per frame.
        // Upper byte: -25 as i8 → 0xE7. Lower byte: 40 → 0x28.
        let div = u16::from_be_bytes([0xE7, 0x28]);
        let mut blob = header_chunk(0, 1, div);
        blob.extend(track_chunk(&[0x00, 0xFF, 0x2F, 0x00]));
        let smf = parse(&blob).unwrap();
        assert_eq!(
            smf.header.division,
            Division::Smpte {
                frames_per_second: 25,
                ticks_per_frame: 40,
            },
        );
    }

    #[test]
    fn type_0_single_track_with_note_pair_and_tempo() {
        // Hand-built track:
        //   delta=0  FF 51 03 07 A1 20    set tempo 500000us/qn (120 BPM)
        //   delta=0  FF 58 04 04 02 18 08 time signature 4/4
        //   delta=0  90 3C 64              note on chan 0, key 60, vel 100
        //   delta=480 80 3C 40             note off (status byte explicit)
        //   delta=0  FF 2F 00              end of track
        let mut events = vec![];
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        events.extend_from_slice(&[0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));

        let smf = parse(&blob).unwrap();
        assert_eq!(smf.header.format, SmfFormat::SingleTrack);
        let evs = &smf.tracks[0].events;
        assert_eq!(evs.len(), 5);

        assert!(matches!(
            evs[0].kind,
            Event::Meta(MetaEvent::Tempo(500_000))
        ));
        assert!(matches!(
            evs[1].kind,
            Event::Meta(MetaEvent::TimeSignature {
                numerator: 4,
                denominator_pow2: 2,
                clocks_per_click: 24,
                notated_32nd_per_quarter: 8,
            })
        ));
        match &evs[2].kind {
            Event::Channel(ChannelMessage {
                channel: 0,
                body:
                    ChannelBody::NoteOn {
                        key: 60,
                        velocity: 100,
                    },
            }) => {}
            other => panic!("unexpected event #2: {other:?}"),
        }
        assert_eq!(evs[3].delta, 480);
        match &evs[3].kind {
            Event::Channel(ChannelMessage {
                channel: 0,
                body:
                    ChannelBody::NoteOff {
                        key: 60,
                        velocity: 0x40,
                    },
            }) => {}
            other => panic!("unexpected event #3: {other:?}"),
        }
        assert!(matches!(evs[4].kind, Event::Meta(MetaEvent::EndOfTrack)));
    }

    #[test]
    fn running_status_is_honoured() {
        // delta=0  90 3C 64        note on (status set)
        // delta=10 3D 64           running status: another note on (key 61)
        // delta=10 3E 64           running status: another note on (key 62)
        // delta=0  FF 2F 00        end of track (clears running status)
        let events: &[u8] = &[
            0x00, 0x90, 0x3C, 0x64, 0x0A, 0x3D, 0x64, 0x0A, 0x3E, 0x64, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(events));

        let smf = parse(&blob).unwrap();
        let evs = &smf.tracks[0].events;
        assert_eq!(evs.len(), 4);
        for (i, &expected_key) in [60u8, 61, 62].iter().enumerate() {
            match &evs[i].kind {
                Event::Channel(ChannelMessage {
                    channel: 0,
                    body: ChannelBody::NoteOn { key, velocity: 100 },
                }) if *key == expected_key => {}
                other => panic!("event #{i}: expected NoteOn key={expected_key}, got {other:?}"),
            }
        }
    }

    #[test]
    fn type_1_multi_track() {
        // Track 1: tempo + time sig + EOT
        let track1: &[u8] = &[
            0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20, 0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18,
            0x08, 0x00, 0xFF, 0x2F, 0x00,
        ];
        // Track 2: note on chan 1, key 64, vel 90; long-delta note off; EOT.
        let mut track2 = vec![0x00, 0x91, 0x40, 0x5A];
        // delta=0x2000 (two-byte VLQ: 0xC0 0x00)
        track2.extend_from_slice(&encode_vlq(0x2000));
        track2.extend_from_slice(&[0x81, 0x40, 0x40, 0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(track1));
        blob.extend(track_chunk(&track2));

        let smf = parse(&blob).unwrap();
        assert_eq!(smf.header.format, SmfFormat::MultiTrackSimultaneous);
        assert_eq!(smf.tracks.len(), 2);
        // Track 1 has 3 events.
        assert_eq!(smf.tracks[0].events.len(), 3);
        // Track 2: note on, note off, EOT.
        assert_eq!(smf.tracks[1].events.len(), 3);
        match &smf.tracks[1].events[0].kind {
            Event::Channel(ChannelMessage {
                channel: 1,
                body:
                    ChannelBody::NoteOn {
                        key: 64,
                        velocity: 90,
                    },
            }) => {}
            other => panic!("track 2 event 0 unexpected: {other:?}"),
        }
        assert_eq!(smf.tracks[1].events[1].delta, 0x2000);
    }

    #[test]
    fn unknown_chunk_is_skipped() {
        let mut blob = header_chunk(0, 1, 96);
        // Inject an unknown chunk between header and track.
        blob.extend_from_slice(b"XYZW");
        blob.extend_from_slice(&3u32.to_be_bytes());
        blob.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        blob.extend(track_chunk(&[0x00, 0xFF, 0x2F, 0x00]));
        let smf = parse(&blob).unwrap();
        assert_eq!(smf.tracks.len(), 1);
    }

    #[test]
    fn meta_text_events() {
        // delta=0 FF 03 06 "Track1"
        // delta=0 FF 2F 00
        let mut events = vec![0x00, 0xFF, 0x03, 0x06];
        events.extend_from_slice(b"Track1");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));

        let smf = parse(&blob).unwrap();
        match &smf.tracks[0].events[0].kind {
            Event::Meta(MetaEvent::Text { kind: 0x03, text }) => {
                assert_eq!(text, b"Track1");
            }
            other => panic!("expected text event, got {other:?}"),
        }
    }

    #[test]
    fn pitch_bend_combines_lsb_msb() {
        // delta=0 E0 00 40   pitch bend chan 0, value=0x2000 (centre)
        // delta=0 FF 2F 00
        let events = [0x00, 0xE0, 0x00, 0x40, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert_eq!(smf.tracks[0].events[0].delta, 0);
        match &smf.tracks[0].events[0].kind {
            Event::Channel(ChannelMessage {
                channel: 0,
                body: ChannelBody::PitchBend { value: 0x2000 },
            }) => {}
            other => panic!("expected pitch bend 0x2000, got {other:?}"),
        }
    }

    #[test]
    fn sysex_event() {
        // delta=0 F0 04 7E 7F 09 01    GM-on universal sysex (no closing F7)
        // delta=0 FF 2F 00
        let events = [
            0x00, 0xF0, 0x04, 0x7E, 0x7F, 0x09, 0x01, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        match &smf.tracks[0].events[0].kind {
            Event::Sysex {
                escape: false,
                data,
            } => assert_eq!(data, &[0x7E, 0x7F, 0x09, 0x01]),
            other => panic!("expected sysex, got {other:?}"),
        }
    }

    #[test]
    fn rejects_chunk_length_overrun() {
        // MThd lies about its length (claims 60 bytes when 6 are present).
        let mut blob = vec![];
        blob.extend_from_slice(b"MThd");
        blob.extend_from_slice(&60u32.to_be_bytes());
        blob.extend_from_slice(&[0; 6]);
        let err = parse(&blob).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_meta_length_overrun() {
        // FF 03 FF 7F  ← claims 16383 bytes of text in a 4-byte chunk
        let events: &[u8] = &[0x00, 0xFF, 0x03, 0xFF, 0x7F];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(events));
        let err = parse(&blob).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_data_byte_without_status() {
        // Track starts with a delta + data byte (no prior status).
        let events: &[u8] = &[0x00, 0x40, 0x40];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(events));
        let err = parse(&blob).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rejects_system_common_in_track() {
        // F1 (MIDI Time Code Quarter Frame) is illegal in SMF tracks.
        let events: &[u8] = &[0x00, 0xF1, 0x40];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(events));
        let err = parse(&blob).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }
}
