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

/// One time-signature change pinned to the absolute tick (relative to
/// the start of its parent track) at which the
/// [`FF 58 04 nn dd cc bb`](MetaEvent::TimeSignature) meta event
/// fires.
///
/// Returned by [`SmfFile::time_signatures`] — see that method for the
/// merge semantics across multiple tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeSignatureChange {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files conventionally place all
    /// time signatures on track 0; format-2 files keep them
    /// per-track.
    pub track: usize,
    /// `nn` — beats per measure (numerator).
    pub numerator: u8,
    /// `dd` — denominator as a negative power of two
    /// (`denominator = 1 << denominator_pow2`).
    pub denominator_pow2: u8,
    /// `cc` — MIDI clocks between metronome clicks (24 per quarter
    /// note in the default mapping).
    pub clocks_per_click: u8,
    /// `bb` — number of notated 32nd notes per MIDI quarter note
    /// (8 in the conventional mapping).
    pub notated_32nd_per_quarter: u8,
}

impl TimeSignatureChange {
    /// Decoded denominator value (`1 << denominator_pow2`). Clamped to
    /// fit a `u32` so a pathological `dd >= 32` stays in-range without
    /// overflowing — the spec doesn't bound `dd`, but every
    /// real-world file uses small values (`0..=6`, i.e. whole-note
    /// through 64th-note).
    pub fn denominator(&self) -> u32 {
        if self.denominator_pow2 >= 32 {
            // Saturate rather than overflow on absurd input.
            u32::MAX
        } else {
            1u32 << self.denominator_pow2
        }
    }
}

/// One tempo change pinned to the absolute tick (relative to the
/// start of its parent track) at which the
/// [`FF 51 03 tt tt tt`](MetaEvent::Tempo) Set Tempo meta event fires.
///
/// Returned by [`SmfFile::tempo_map`] — see that method for the merge
/// semantics across multiple tracks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoChange {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files conventionally place every
    /// tempo change on track 0; format-2 files keep them per-track.
    pub track: usize,
    /// `tt tt tt` decoded as a 24-bit big-endian unsigned integer —
    /// microseconds per quarter note. The default (assumed before the
    /// first explicit Set Tempo) is 500 000 µs/qn = 120 BPM.
    pub microseconds_per_quarter_note: u32,
    /// Cached `60_000_000 / microseconds_per_quarter_note` evaluated in
    /// `f64`. Zero is mapped to `f64::INFINITY` so a malformed
    /// `tt tt tt == 0` payload doesn't divide-by-zero — the spec
    /// doesn't forbid it but no real file uses it.
    pub bpm: f64,
}

impl TempoChange {
    /// Build a `TempoChange` from its three fields. Pre-computes
    /// [`bpm`](Self::bpm) so callers can read it cheaply.
    ///
    /// `microseconds_per_quarter_note == 0` maps to `bpm = f64::INFINITY`
    /// rather than panicking on the division — see the type-level note.
    pub fn new(tick: u64, track: usize, microseconds_per_quarter_note: u32) -> Self {
        let bpm = if microseconds_per_quarter_note == 0 {
            f64::INFINITY
        } else {
            60_000_000.0 / microseconds_per_quarter_note as f64
        };
        Self {
            tick,
            track,
            microseconds_per_quarter_note,
            bpm,
        }
    }
}

/// One key-signature change pinned to the absolute tick (relative to
/// the start of its parent track) at which the
/// [`FF 59 02 sf mi`](MetaEvent::KeySignature) meta event fires.
///
/// Returned by [`SmfFile::key_signatures`] — see that method for the
/// merge semantics across multiple tracks.
///
/// Per the SMF spec, `sharps_flats` is a signed count in `-7..=+7`:
/// negative means flats, positive means sharps, `0` is the natural
/// scale (C major / A minor). `mode` is `0` for major and `1` for
/// minor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeySignatureChange {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files conventionally place key
    /// signatures on track 0; format-2 files keep them per-track.
    pub track: usize,
    /// `sf` — accidental count. Negative = flats, positive = sharps;
    /// legal range per spec is `-7..=+7`.
    pub sharps_flats: i8,
    /// `mi` — `0` major, `1` minor. Anything else is preserved
    /// verbatim (the spec doesn't reserve other values, but real
    /// files have been seen with junk here).
    pub mode: u8,
}

impl KeySignatureChange {
    /// `true` when `mode == 1` (minor). Major / any-non-one is `false`.
    pub fn is_minor(&self) -> bool {
        self.mode == 1
    }

    /// `true` when `mode == 0` (major).
    pub fn is_major(&self) -> bool {
        self.mode == 0
    }

    /// Human-readable tonic name (e.g. `"C"`, `"F#"`, `"Bb"`).
    ///
    /// Returns `None` when `sharps_flats` is outside the spec range
    /// `-7..=+7` or when `mode` is neither `0` (major) nor `1` (minor).
    ///
    /// Derived from the circle-of-fifths positions documented in the
    /// SMF spec § "FF 59 02 sf mi Key Signature": each `+1` step adds
    /// a sharp following `F# C# G# D# A# E# B#`; each `-1` step adds
    /// a flat following `Bb Eb Ab Db Gb Cb Fb`. The minor-key column
    /// is the relative minor (a sixth below the major tonic).
    pub fn tonic_name(&self) -> Option<&'static str> {
        // Index `sf + 7` maps `-7..=+7` to `0..=14`.
        if !(-7..=7).contains(&self.sharps_flats) {
            return None;
        }
        let idx = (self.sharps_flats + 7) as usize;
        let names = match self.mode {
            0 => MAJOR_TONICS,
            1 => MINOR_TONICS,
            _ => return None,
        };
        Some(names[idx])
    }

    /// Full key name (e.g. `"C major"`, `"A minor"`, `"F# minor"`,
    /// `"Bb major"`). `None` under the same conditions as
    /// [`tonic_name`](Self::tonic_name).
    pub fn name(&self) -> Option<&'static str> {
        if !(-7..=7).contains(&self.sharps_flats) {
            return None;
        }
        let idx = (self.sharps_flats + 7) as usize;
        let table = match self.mode {
            0 => MAJOR_KEY_NAMES,
            1 => MINOR_KEY_NAMES,
            _ => return None,
        };
        Some(table[idx])
    }
}

// Circle-of-fifths labels for the SMF FF 59 02 sf mi meta event.
//
// Index = `sf + 7`, so the entries run `-7 -6 -5 -4 -3 -2 -1  0  +1
// +2 +3 +4 +5 +6 +7`. The major and minor rows are the standard
// musical mapping (see e.g. SMF 1.0 spec page 11): minor is the
// relative minor of the major three semitones below, so a 4-sharp
// signature is E major / C# minor, etc. Spelling follows the
// conventional accidental for each direction (sharp-side keys use
// sharps, flat-side keys use flats).
const MAJOR_TONICS: [&str; 15] = [
    "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", // sf = -7..-1
    "C", // sf = 0
    "G", "D", "A", "E", "B", "F#", "C#", // sf = +1..+7
];

const MINOR_TONICS: [&str; 15] = [
    "Ab", "Eb", "Bb", "F", "C", "G", "D", // sf = -7..-1
    "A", // sf = 0
    "E", "B", "F#", "C#", "G#", "D#", "A#", // sf = +1..+7
];

const MAJOR_KEY_NAMES: [&str; 15] = [
    "Cb major", "Gb major", "Db major", "Ab major", "Eb major", "Bb major", "F major", "C major",
    "G major", "D major", "A major", "E major", "B major", "F# major", "C# major",
];

const MINOR_KEY_NAMES: [&str; 15] = [
    "Ab minor", "Eb minor", "Bb minor", "F minor", "C minor", "G minor", "D minor", "A minor",
    "E minor", "B minor", "F# minor", "C# minor", "G# minor", "D# minor", "A# minor",
];

/// One marker meta event pinned to the absolute tick (relative to the
/// start of its parent track) at which the
/// [`FF 06 len text`](MetaEvent::Text) meta event fires.
///
/// Returned by [`SmfFile::markers`] — see that method for the merge
/// semantics across multiple tracks.
///
/// The marker text is preserved byte-for-byte from the SMF stream. The
/// spec does not pin a character set, so callers that need a Rust
/// string should call [`MarkerEvent::text_lossy`] (UTF-8 with U+FFFD
/// substitutes) or pick their own decoding strategy from the raw
/// bytes returned by [`MarkerEvent::text_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files conventionally place all
    /// markers on track 0; format-2 files keep them per-track.
    pub track: usize,
    /// Raw marker text bytes (the `text` payload of `FF 06 len text`).
    /// The SMF spec leaves the encoding unspecified — historically
    /// Latin-1 was conventional, modern DAWs emit UTF-8. Stored as
    /// `Vec<u8>` so we don't fabricate a decoding.
    pub text: Vec<u8>,
}

impl MarkerEvent {
    /// Borrow the raw marker bytes.
    pub fn text_bytes(&self) -> &[u8] {
        &self.text
    }

    /// Lossy UTF-8 decode of the marker text. Invalid sequences are
    /// replaced with `U+FFFD` (REPLACEMENT CHARACTER), so this never
    /// fails. Callers that need a strict decoding should call
    /// [`std::str::from_utf8`] on [`text_bytes`](Self::text_bytes)
    /// themselves.
    pub fn text_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.text)
    }
}

/// One lyric meta event pinned to the absolute tick (relative to the
/// start of its parent track) at which the
/// [`FF 05 len text`](MetaEvent::Text) meta event fires.
///
/// Returned by [`SmfFile::lyrics`] — see that method for the merge
/// semantics across multiple tracks.
///
/// Lyric meta events are the karaoke staple: each event carries one
/// syllable (or fragment) of the song text, pinned to the tick at which
/// the player should display it. The `.kar` convention layers a syllable
/// stream onto an otherwise ordinary SMF; players render lyrics by
/// walking the time-ordered list in step with playback.
///
/// The lyric text is preserved byte-for-byte from the SMF stream. The
/// spec does not pin a character set, so callers that need a Rust
/// string should call [`LyricEvent::text_lossy`] (UTF-8 with `U+FFFD`
/// substitutes) or pick their own decoding strategy from the raw bytes
/// returned by [`LyricEvent::text_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LyricEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). The `.kar` convention places every lyric
    /// on a single track (often track 1, named `"Words"` / `"Lyrics"`
    /// via a preceding `FF 03` track-name meta event); format-2 files
    /// keep them per-track.
    pub track: usize,
    /// Raw lyric text bytes (the `text` payload of `FF 05 len text`).
    /// The SMF spec leaves the encoding unspecified — historically
    /// Latin-1 was conventional, modern DAWs emit UTF-8. Stored as
    /// `Vec<u8>` so we don't fabricate a decoding.
    pub text: Vec<u8>,
}

impl LyricEvent {
    /// Borrow the raw lyric bytes.
    pub fn text_bytes(&self) -> &[u8] {
        &self.text
    }

    /// Lossy UTF-8 decode of the lyric text. Invalid sequences are
    /// replaced with `U+FFFD` (REPLACEMENT CHARACTER), so this never
    /// fails. Callers that need a strict decoding should call
    /// [`std::str::from_utf8`] on [`text_bytes`](Self::text_bytes)
    /// themselves.
    pub fn text_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.text)
    }
}

/// One cue-point meta event pinned to the absolute tick (relative to
/// the start of its parent track) at which the
/// [`FF 07 len text`](MetaEvent::Text) meta event fires.
///
/// Returned by [`SmfFile::cue_points`] — see that method for the merge
/// semantics across multiple tracks.
///
/// Cue points are the film-score / theatrical sync convention from the
/// original Standard MIDI File Specification 1.0: each event labels a
/// point in the sequence where some external action should occur
/// (a scene change, an SFX trigger, a video cue, …). They share the
/// same byte shape as markers (`FF 06`) and lyrics (`FF 05`) but
/// indicate "sync with an external event" rather than "label a song
/// section" or "display a karaoke syllable". This helper isolates the
/// `FF 07` stream so callers driving external synchronisation don't
/// have to discriminate against neighbouring text-meta kinds.
///
/// The cue text is preserved byte-for-byte from the SMF stream. The
/// spec does not pin a character set, so callers that need a Rust
/// string should call [`CueEvent::text_lossy`] (UTF-8 with `U+FFFD`
/// substitutes) or pick their own decoding strategy from the raw bytes
/// returned by [`CueEvent::text_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CueEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files conventionally place sync
    /// cues on a dedicated track (often the conductor / tempo track);
    /// format-2 files keep them per-track.
    pub track: usize,
    /// Raw cue-point text bytes (the `text` payload of `FF 07 len
    /// text`). The SMF spec leaves the encoding unspecified —
    /// historically Latin-1 was conventional, modern editors emit
    /// UTF-8. Stored as `Vec<u8>` so we don't fabricate a decoding.
    pub text: Vec<u8>,
}

impl CueEvent {
    /// Borrow the raw cue-point bytes.
    pub fn text_bytes(&self) -> &[u8] {
        &self.text
    }

    /// Lossy UTF-8 decode of the cue-point text. Invalid sequences are
    /// replaced with `U+FFFD` (REPLACEMENT CHARACTER), so this never
    /// fails. Callers that need a strict decoding should call
    /// [`std::str::from_utf8`] on [`text_bytes`](Self::text_bytes)
    /// themselves.
    pub fn text_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.text)
    }
}

/// One track-name meta event pinned to the absolute tick (relative to
/// the start of its parent track) at which the
/// [`FF 03 len text`](MetaEvent::Text) meta event fires.
///
/// Returned by [`SmfFile::track_names`] — see that method for the merge
/// semantics across multiple tracks.
///
/// `FF 03` declares the name of a track (or, on a format-0 file, the
/// name of the sequence as a whole — the original SMF specification
/// allows either reading). DAWs surface it as the visible label in the
/// track list; this helper isolates the `FF 03` stream so callers
/// populating that label don't have to discriminate against the
/// neighbouring text-meta kinds (free-form text, copyright, instrument
/// name [see [`SmfFile::instrument_names`]], lyric, marker, cue
/// point).
///
/// SMF authoring tools conventionally place at most one `FF 03` per
/// track at tick 0, but the spec doesn't constrain count or placement;
/// this helper surfaces every occurrence so callers that want
/// "first only" can take `.next()` on the iterator while callers that
/// want the full history can read the whole `Vec`.
///
/// The name text is preserved byte-for-byte from the SMF stream. The
/// spec does not pin a character set, so callers that need a Rust
/// string should call [`TrackNameEvent::text_lossy`] (UTF-8 with
/// `U+FFFD` substitutes) or pick their own decoding strategy from the
/// raw bytes returned by [`TrackNameEvent::text_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackNameEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase. Typically `0`
    /// since track names conventionally land at the head of the track,
    /// but the spec permits later placement.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). On a format-1 file each music track
    /// usually carries its own `FF 03`; on a format-0 file the single
    /// track's `FF 03` is conventionally read as the sequence title.
    pub track: usize,
    /// Raw track-name text bytes (the `text` payload of `FF 03 len
    /// text`). The SMF spec leaves the encoding unspecified —
    /// historically Latin-1 was conventional, modern DAWs emit UTF-8.
    /// Stored as `Vec<u8>` so we don't fabricate a decoding.
    pub text: Vec<u8>,
}

impl TrackNameEvent {
    /// Borrow the raw track-name bytes.
    pub fn text_bytes(&self) -> &[u8] {
        &self.text
    }

    /// Lossy UTF-8 decode of the track-name text. Invalid sequences are
    /// replaced with `U+FFFD` (REPLACEMENT CHARACTER), so this never
    /// fails. Callers that need a strict decoding should call
    /// [`std::str::from_utf8`] on [`text_bytes`](Self::text_bytes)
    /// themselves.
    pub fn text_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.text)
    }
}

/// One instrument-name meta event pinned to the absolute tick (relative
/// to the start of its parent track) at which the
/// [`FF 04 len text`](MetaEvent::Text) meta event fires.
///
/// Returned by [`SmfFile::instrument_names`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `FF 04` declares the name of the *instrument* a track is targeting
/// (e.g. `"Grand Piano"`, `"Drum Kit"`, `"Trumpet"`), as distinct from
/// the *track* name (`FF 03`) which labels the track itself in the DAW
/// track list. A single track may legally carry both — the track-list
/// label and the instrument it is voicing. Authoring tools
/// conventionally place the instrument name near the head of the track
/// (often at tick 0, before the first note), but the spec doesn't
/// constrain count or placement; this helper surfaces every occurrence
/// so callers that only want the first per track can take `.next()` on
/// the iterator while callers that want the full history can read the
/// whole `Vec`.
///
/// The name text is preserved byte-for-byte from the SMF stream. The
/// spec does not pin a character set, so callers that need a Rust
/// string should call [`InstrumentNameEvent::text_lossy`] (UTF-8 with
/// `U+FFFD` substitutes) or pick their own decoding strategy from the
/// raw bytes returned by [`InstrumentNameEvent::text_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstrumentNameEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase. Typically `0`
    /// since instrument names conventionally land at the head of the
    /// track, but the spec permits later placement (e.g. to label a
    /// patch change mid-track).
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). On a format-1 file each music track that
    /// targets a specific instrument usually carries its own `FF 04`;
    /// the conductor / tempo track typically does not.
    pub track: usize,
    /// Raw instrument-name text bytes (the `text` payload of `FF 04
    /// len text`). The SMF spec leaves the encoding unspecified —
    /// historically Latin-1 was conventional, modern DAWs emit UTF-8.
    /// Stored as `Vec<u8>` so we don't fabricate a decoding.
    pub text: Vec<u8>,
}

impl InstrumentNameEvent {
    /// Borrow the raw instrument-name bytes.
    pub fn text_bytes(&self) -> &[u8] {
        &self.text
    }

    /// Lossy UTF-8 decode of the instrument-name text. Invalid
    /// sequences are replaced with `U+FFFD` (REPLACEMENT CHARACTER),
    /// so this never fails. Callers that need a strict decoding should
    /// call [`std::str::from_utf8`] on
    /// [`text_bytes`](Self::text_bytes) themselves.
    pub fn text_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.text)
    }
}

/// One free-form text meta event pinned to the absolute tick (relative
/// to the start of its parent track) at which the
/// [`FF 01 len text`](MetaEvent::Text) meta event fires.
///
/// Returned by [`SmfFile::texts`] — see that method for the merge
/// semantics across multiple tracks.
///
/// `FF 01` is the generic / free-form text meta event: a catch-all for
/// annotations that don't fit one of the more specific text-meta kinds
/// (`FF 02` copyright, `FF 03` track name, `FF 04` instrument name,
/// `FF 05` lyric, `FF 06` marker, `FF 07` cue point). Authoring tools
/// emit it for production notes, mix-engineer comments, "do not edit",
/// version stamps, and anything else the editor wants to keep next to
/// the music without it being recognised as one of the structured kinds.
/// This helper isolates the `FF 01` stream so callers reading annotation
/// text don't have to discriminate against the neighbouring text-meta
/// kinds.
///
/// The text is preserved byte-for-byte from the SMF stream. The spec
/// does not pin a character set, so callers that need a Rust string
/// should call [`TextEvent::text_lossy`] (UTF-8 with `U+FFFD`
/// substitutes) or pick their own decoding strategy from the raw bytes
/// returned by [`TextEvent::text_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Authoring tools may place free-form text
    /// on any track; format-1 files frequently keep production
    /// annotations on the conductor track.
    pub track: usize,
    /// Raw text bytes (the `text` payload of `FF 01 len text`). The
    /// SMF spec leaves the encoding unspecified — historically Latin-1
    /// was conventional, modern DAWs emit UTF-8. Stored as `Vec<u8>`
    /// so we don't fabricate a decoding.
    pub text: Vec<u8>,
}

impl TextEvent {
    /// Borrow the raw text bytes.
    pub fn text_bytes(&self) -> &[u8] {
        &self.text
    }

    /// Lossy UTF-8 decode of the free-form text. Invalid sequences are
    /// replaced with `U+FFFD` (REPLACEMENT CHARACTER), so this never
    /// fails. Callers that need a strict decoding should call
    /// [`std::str::from_utf8`] on [`text_bytes`](Self::text_bytes)
    /// themselves.
    pub fn text_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.text)
    }
}

/// One copyright-notice meta event pinned to the absolute tick (relative
/// to the start of its parent track) at which the
/// [`FF 02 len text`](MetaEvent::Text) meta event fires.
///
/// Returned by [`SmfFile::copyrights`] — see that method for the merge
/// semantics across multiple tracks.
///
/// `FF 02` declares a copyright notice for the sequence. The original
/// Standard MIDI File Specification 1.0 recommends placing it at the
/// head of the first track (tick 0) so a player can surface authorship
/// without scanning the whole file. The spec does not forbid multiple
/// occurrences or later placement, so this helper surfaces every
/// occurrence in time order; callers that only want the first notice
/// can take `.next()` on the iterator.
///
/// The notice text is preserved byte-for-byte from the SMF stream. The
/// spec does not pin a character set, so callers that need a Rust
/// string should call [`CopyrightEvent::text_lossy`] (UTF-8 with
/// `U+FFFD` substitutes) or pick their own decoding strategy from the
/// raw bytes returned by [`CopyrightEvent::text_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyrightEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase. Typically `0`
    /// since copyright notices conventionally land at the head of the
    /// first track, but the spec permits later placement.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). The SMF specification recommends placing
    /// the copyright on the first track; this helper surfaces the
    /// notice wherever it actually appears.
    pub track: usize,
    /// Raw copyright text bytes (the `text` payload of `FF 02 len
    /// text`). The SMF spec leaves the encoding unspecified —
    /// historically Latin-1 was conventional, modern DAWs emit UTF-8.
    /// Stored as `Vec<u8>` so we don't fabricate a decoding.
    pub text: Vec<u8>,
}

impl CopyrightEvent {
    /// Borrow the raw copyright bytes.
    pub fn text_bytes(&self) -> &[u8] {
        &self.text
    }

    /// Lossy UTF-8 decode of the copyright text. Invalid sequences are
    /// replaced with `U+FFFD` (REPLACEMENT CHARACTER), so this never
    /// fails. Callers that need a strict decoding should call
    /// [`std::str::from_utf8`] on [`text_bytes`](Self::text_bytes)
    /// themselves.
    pub fn text_lossy(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.text)
    }
}

impl SmfFile {
    /// Collect every [`MetaEvent::Tempo`] from every track, pinned to
    /// the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two changes at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::time_signatures`] and the
    /// scheduler use (`scheduler.rs` §"merged event list, sorted by
    /// absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a Set Tempo meta
    /// event. Per the SMF convention, a player that needs an initial
    /// tempo should assume **500 000 µs/qn = 120 BPM** until the first
    /// change fires.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn tempo_map(&self) -> Vec<TempoChange> {
        let mut out: Vec<TempoChange> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Tempo(us_per_qn)) = &ev.kind {
                    out.push(TempoChange::new(abs, track_idx, *us_per_qn));
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every [`MetaEvent::TimeSignature`] from every track,
    /// pinned to the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two changes at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule the scheduler uses (`scheduler.rs` §"merged
    /// event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a time-signature
    /// meta event. Per the SMF convention, a player that needs an
    /// initial time signature should assume **4/4** until the first
    /// change fires.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn time_signatures(&self) -> Vec<TimeSignatureChange> {
        let mut out: Vec<TimeSignatureChange> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::TimeSignature {
                    numerator,
                    denominator_pow2,
                    clocks_per_click,
                    notated_32nd_per_quarter,
                }) = &ev.kind
                {
                    out.push(TimeSignatureChange {
                        tick: abs,
                        track: track_idx,
                        numerator: *numerator,
                        denominator_pow2: *denominator_pow2,
                        clocks_per_click: *clocks_per_click,
                        notated_32nd_per_quarter: *notated_32nd_per_quarter,
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the
        // per-track insertion order survives the sort (so track 0
        // wins over track 1 at the same tick — matches the
        // scheduler's merge convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every [`MetaEvent::KeySignature`] from every track,
    /// pinned to the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two changes at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::time_signatures`] /
    /// [`SmfFile::tempo_map`] and the scheduler use (`scheduler.rs`
    /// §"merged event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a Key Signature
    /// meta event. Per the SMF convention, a player that needs an
    /// initial key signature should assume **C major** (`sf = 0,
    /// mi = 0`) until the first change fires.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn key_signatures(&self) -> Vec<KeySignatureChange> {
        let mut out: Vec<KeySignatureChange> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::KeySignature { sharps_flats, mode }) = &ev.kind {
                    out.push(KeySignatureChange {
                        tick: abs,
                        track: track_idx,
                        sharps_flats: *sharps_flats,
                        mode: *mode,
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every marker meta event (`FF 06 len text`, surfaced as
    /// [`MetaEvent::Text`] with `kind == 0x06`) from every track,
    /// pinned to the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two markers at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] and
    /// the scheduler use (`scheduler.rs` §"merged event list, sorted
    /// by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a marker meta
    /// event. The spec does not bound how many markers a file may
    /// declare; the overall cap remains [`MAX_EVENTS_PER_FILE`] (the
    /// same cap the parser enforces, applied across every event kind).
    ///
    /// Markers are conventionally used by DAWs to label song sections
    /// (e.g. `"Verse"`, `"Chorus"`) and are distinct from the more
    /// general text events (`FF 01..=05, 07..=0F`) which carry
    /// copyright, track name, instrument name, lyric, cue point, and
    /// program name payloads — only `FF 06` is selected here. For
    /// `FF 03` track names see [`SmfFile::track_names`]; for `FF 04`
    /// instrument names see [`SmfFile::instrument_names`]; for `FF 05`
    /// karaoke syllables see [`SmfFile::lyrics`]; for `FF 07`
    /// film-score sync points see [`SmfFile::cue_points`].
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn markers(&self) -> Vec<MarkerEvent> {
        let mut out: Vec<MarkerEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Text { kind: 0x06, text }) = &ev.kind {
                    out.push(MarkerEvent {
                        tick: abs,
                        track: track_idx,
                        text: text.clone(),
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every lyric meta event (`FF 05 len text`, surfaced as
    /// [`MetaEvent::Text`] with `kind == 0x05`) from every track,
    /// pinned to the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two lyrics at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::markers`] / [`SmfFile::tempo_map`]
    /// / [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`]
    /// and the scheduler use (`scheduler.rs` §"merged event list,
    /// sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a lyric meta
    /// event. The spec does not bound how many lyrics a file may
    /// declare; the overall cap remains [`MAX_EVENTS_PER_FILE`] (the
    /// same cap the parser enforces, applied across every event kind).
    ///
    /// Lyrics are the karaoke (`.kar`) convention — one syllable
    /// fragment per event, pinned to its display tick. They are
    /// distinct from the neighbouring text events:
    ///
    /// - `FF 01` general text — free-form annotation
    /// - `FF 02` copyright notice
    /// - `FF 03` track name (see [`SmfFile::track_names`])
    /// - `FF 04` instrument name (see [`SmfFile::instrument_names`])
    /// - `FF 06` marker — song-section labels (see
    ///   [`SmfFile::markers`])
    /// - `FF 07` cue point — film-score sync markers (see
    ///   [`SmfFile::cue_points`])
    ///
    /// Only `FF 05` is selected here so callers iterating karaoke
    /// syllables don't have to discriminate themselves.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn lyrics(&self) -> Vec<LyricEvent> {
        let mut out: Vec<LyricEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Text { kind: 0x05, text }) = &ev.kind {
                    out.push(LyricEvent {
                        tick: abs,
                        track: track_idx,
                        text: text.clone(),
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every cue-point meta event (`FF 07 len text`, surfaced
    /// as [`MetaEvent::Text`] with `kind == 0x07`) from every track,
    /// pinned to the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two cues at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::tempo_map`] / [`SmfFile::time_signatures`] /
    /// [`SmfFile::key_signatures`] and the scheduler use
    /// (`scheduler.rs` §"merged event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a cue-point meta
    /// event. The spec does not bound how many cue points a file may
    /// declare; the overall cap remains [`MAX_EVENTS_PER_FILE`] (the
    /// same cap the parser enforces, applied across every event kind).
    ///
    /// Cue points are the film-score / theatrical sync convention
    /// from the Standard MIDI File Specification 1.0 — each event
    /// names an external action point (scene change, SFX trigger,
    /// video cue, …). They are distinct from the neighbouring
    /// text-meta kinds:
    ///
    /// - `FF 01` general text — free-form annotation
    /// - `FF 02` copyright notice
    /// - `FF 03` track name (see [`SmfFile::track_names`])
    /// - `FF 04` instrument name (see [`SmfFile::instrument_names`])
    /// - `FF 05` lyric — karaoke syllables (see [`SmfFile::lyrics`])
    /// - `FF 06` marker — song-section labels (see
    ///   [`SmfFile::markers`])
    ///
    /// Only `FF 07` is selected here so callers driving external
    /// synchronisation don't have to discriminate themselves.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn cue_points(&self) -> Vec<CueEvent> {
        let mut out: Vec<CueEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Text { kind: 0x07, text }) = &ev.kind {
                    out.push(CueEvent {
                        tick: abs,
                        track: track_idx,
                        text: text.clone(),
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every track-name meta event (`FF 03 len text`, surfaced
    /// as [`MetaEvent::Text`] with `kind == 0x03`) from every track,
    /// pinned to the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two names at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::cue_points`] / [`SmfFile::markers`]
    /// / [`SmfFile::lyrics`] / [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] and
    /// the scheduler use (`scheduler.rs` §"merged event list, sorted
    /// by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a track-name meta
    /// event. The spec does not bound how many names a file may
    /// declare; the overall cap remains [`MAX_EVENTS_PER_FILE`] (the
    /// same cap the parser enforces, applied across every event kind).
    ///
    /// Track names populate the DAW's track-list label. On format-0
    /// files the single track's `FF 03` is conventionally read as the
    /// sequence title; on format-1 files each music track usually
    /// carries its own `FF 03` at tick 0. They are distinct from the
    /// neighbouring text-meta kinds:
    ///
    /// - `FF 01` general text — free-form annotation
    /// - `FF 02` copyright notice
    /// - `FF 04` instrument name (see [`SmfFile::instrument_names`])
    /// - `FF 05` lyric — karaoke syllables (see [`SmfFile::lyrics`])
    /// - `FF 06` marker — song-section labels (see
    ///   [`SmfFile::markers`])
    /// - `FF 07` cue point — film-score sync markers (see
    ///   [`SmfFile::cue_points`])
    ///
    /// Only `FF 03` is selected here so callers populating a track
    /// list don't have to discriminate themselves. Authoring tools
    /// conventionally emit at most one `FF 03` per track at tick 0;
    /// callers that only want the first name per track can collect
    /// into a `HashMap<usize, TrackNameEvent>` keyed on
    /// [`TrackNameEvent::track`].
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn track_names(&self) -> Vec<TrackNameEvent> {
        let mut out: Vec<TrackNameEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Text { kind: 0x03, text }) = &ev.kind {
                    out.push(TrackNameEvent {
                        tick: abs,
                        track: track_idx,
                        text: text.clone(),
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every instrument-name meta event (`FF 04 len text`,
    /// surfaced as [`MetaEvent::Text`] with `kind == 0x04`) from every
    /// track, pinned to the absolute tick at which it fires, in time
    /// order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two names at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::track_names`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::markers`] /
    /// [`SmfFile::lyrics`] / [`SmfFile::texts`] /
    /// [`SmfFile::copyrights`] / [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] and
    /// the scheduler use (`scheduler.rs` §"merged event list, sorted
    /// by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries an instrument-name
    /// meta event. The spec does not bound how many names a file may
    /// declare; the overall cap remains [`MAX_EVENTS_PER_FILE`] (the
    /// same cap the parser enforces, applied across every event kind).
    ///
    /// Instrument names label the *voice* a track is targeting (e.g.
    /// `"Grand Piano"`, `"Drum Kit"`, `"Trumpet"`) and are distinct
    /// from the *track* names (`FF 03`) that label the track itself in
    /// the DAW track list. A single track may legally carry both; this
    /// helper isolates the `FF 04` stream so callers populating
    /// patch / preset metadata don't have to discriminate against the
    /// neighbouring text-meta kinds:
    ///
    /// - `FF 01` general text (see [`SmfFile::texts`])
    /// - `FF 02` copyright notice (see [`SmfFile::copyrights`])
    /// - `FF 03` track name (see [`SmfFile::track_names`])
    /// - `FF 05` lyric — karaoke syllables (see [`SmfFile::lyrics`])
    /// - `FF 06` marker — song-section labels (see
    ///   [`SmfFile::markers`])
    /// - `FF 07` cue point — film-score sync markers (see
    ///   [`SmfFile::cue_points`])
    ///
    /// Only `FF 04` is selected here so callers populating per-track
    /// instrument metadata get a clean per-track stream. Authoring
    /// tools conventionally emit at most one `FF 04` per track at
    /// tick 0 (often paired with an opening Program Change); callers
    /// that only want the first name per track can collect into a
    /// `HashMap<usize, InstrumentNameEvent>` keyed on
    /// [`InstrumentNameEvent::track`].
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn instrument_names(&self) -> Vec<InstrumentNameEvent> {
        let mut out: Vec<InstrumentNameEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Text { kind: 0x04, text }) = &ev.kind {
                    out.push(InstrumentNameEvent {
                        tick: abs,
                        track: track_idx,
                        text: text.clone(),
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every free-form text meta event (`FF 01 len text`,
    /// surfaced as [`MetaEvent::Text`] with `kind == 0x01`) from every
    /// track, pinned to the absolute tick at which it fires, in time
    /// order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two annotations at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::copyrights`] /
    /// [`SmfFile::track_names`] / [`SmfFile::instrument_names`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::markers`] /
    /// [`SmfFile::lyrics`] / [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] and
    /// the scheduler use (`scheduler.rs` §"merged event list, sorted
    /// by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a free-form text
    /// meta event. The spec does not bound how many annotations a file
    /// may declare; the overall cap remains [`MAX_EVENTS_PER_FILE`]
    /// (the same cap the parser enforces, applied across every event
    /// kind).
    ///
    /// `FF 01` is the generic / free-form text meta event — production
    /// notes, mix-engineer comments, "do not edit", version stamps,
    /// anything that doesn't fit one of the more specific text-meta
    /// kinds. It is distinct from the neighbouring text-meta kinds:
    ///
    /// - `FF 02` copyright notice (see [`SmfFile::copyrights`])
    /// - `FF 03` track name (see [`SmfFile::track_names`])
    /// - `FF 04` instrument name (see [`SmfFile::instrument_names`])
    /// - `FF 05` lyric — karaoke syllables (see [`SmfFile::lyrics`])
    /// - `FF 06` marker — song-section labels (see
    ///   [`SmfFile::markers`])
    /// - `FF 07` cue point — film-score sync markers (see
    ///   [`SmfFile::cue_points`])
    ///
    /// Only `FF 01` is selected here so callers reading annotations
    /// don't have to discriminate themselves.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn texts(&self) -> Vec<TextEvent> {
        let mut out: Vec<TextEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Text { kind: 0x01, text }) = &ev.kind {
                    out.push(TextEvent {
                        tick: abs,
                        track: track_idx,
                        text: text.clone(),
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every copyright-notice meta event (`FF 02 len text`,
    /// surfaced as [`MetaEvent::Text`] with `kind == 0x02`) from every
    /// track, pinned to the absolute tick at which it fires, in time
    /// order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two notices at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::texts`] / [`SmfFile::track_names`]
    /// / [`SmfFile::instrument_names`] / [`SmfFile::cue_points`] /
    /// [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::tempo_map`] / [`SmfFile::time_signatures`] /
    /// [`SmfFile::key_signatures`] and the scheduler use
    /// (`scheduler.rs` §"merged event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a copyright-notice
    /// meta event. The spec does not bound how many notices a file may
    /// declare; the overall cap remains [`MAX_EVENTS_PER_FILE`] (the
    /// same cap the parser enforces, applied across every event kind).
    ///
    /// `FF 02` declares a copyright notice for the sequence. The SMF
    /// specification recommends placing it on the first track at
    /// tick 0 so players can surface authorship without scanning the
    /// full file; this helper surfaces every occurrence in time order
    /// regardless. It is distinct from the neighbouring text-meta
    /// kinds:
    ///
    /// - `FF 01` general text (see [`SmfFile::texts`])
    /// - `FF 03` track name (see [`SmfFile::track_names`])
    /// - `FF 04` instrument name (see [`SmfFile::instrument_names`])
    /// - `FF 05` lyric — karaoke syllables (see [`SmfFile::lyrics`])
    /// - `FF 06` marker — song-section labels (see
    ///   [`SmfFile::markers`])
    /// - `FF 07` cue point — film-score sync markers (see
    ///   [`SmfFile::cue_points`])
    ///
    /// Only `FF 02` is selected here so callers populating
    /// authorship metadata don't have to discriminate themselves.
    /// Callers that only want the first notice can take `.next()` on
    /// the iterator; callers that want the full history can read the
    /// whole `Vec`.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn copyrights(&self) -> Vec<CopyrightEvent> {
        let mut out: Vec<CopyrightEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Text { kind: 0x02, text }) = &ev.kind {
                    out.push(CopyrightEvent {
                        tick: abs,
                        track: track_idx,
                        text: text.clone(),
                    });
                }
            }
        }
        // Stable sort by absolute tick. Within a tick, the per-track
        // insertion order survives the sort (so track 0 wins over
        // track 1 at the same tick — matches the scheduler's merge
        // convention).
        out.sort_by_key(|c| c.tick);
        out
    }
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

    // ───────── TimeSignatureChange / SmfFile::time_signatures ─────────

    #[test]
    fn time_signatures_empty_when_no_meta_event_present() {
        // Track has just a note-on + note-off + EOT — no FF 58.
        let events: &[u8] = &[
            0x00, 0x90, 0x3C, 0x64, 0x40, 0x80, 0x3C, 0x40, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        assert!(smf.time_signatures().is_empty());
    }

    #[test]
    fn time_signatures_single_change_at_tick_zero() {
        // delta=0 FF 58 04 04 02 18 08   time signature 4/4
        // delta=0 FF 2F 00
        let events: &[u8] = &[
            0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        let sigs = smf.time_signatures();
        assert_eq!(sigs.len(), 1);
        let ts = sigs[0];
        assert_eq!(ts.tick, 0);
        assert_eq!(ts.track, 0);
        assert_eq!(ts.numerator, 4);
        assert_eq!(ts.denominator_pow2, 2);
        assert_eq!(ts.denominator(), 4);
        assert_eq!(ts.clocks_per_click, 24);
        assert_eq!(ts.notated_32nd_per_quarter, 8);
    }

    #[test]
    fn time_signatures_multiple_changes_within_one_track_are_in_order() {
        // delta=0   FF 58 04 04 02 18 08   4/4
        // delta=480 FF 58 04 03 02 18 08   3/4
        // delta=480 FF 58 04 06 03 18 08   6/8
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x58, 0x04, 0x03, 0x02, 0x18, 0x08]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x58, 0x04, 0x06, 0x03, 0x18, 0x08]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sigs = smf.time_signatures();
        assert_eq!(sigs.len(), 3);
        assert_eq!(sigs[0].tick, 0);
        assert_eq!((sigs[0].numerator, sigs[0].denominator()), (4, 4));
        assert_eq!(sigs[1].tick, 480);
        assert_eq!((sigs[1].numerator, sigs[1].denominator()), (3, 4));
        assert_eq!(sigs[2].tick, 960);
        assert_eq!((sigs[2].numerator, sigs[2].denominator()), (6, 8));
    }

    #[test]
    fn time_signatures_merge_across_tracks_sorted_by_tick() {
        // Track 0: tick 0 = 4/4, tick 1920 = 7/8, EOT at 1920.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        t0.extend_from_slice(&encode_vlq(1920));
        t0.extend_from_slice(&[0xFF, 0x58, 0x04, 0x07, 0x03, 0x18, 0x08]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        // Track 1: tick 960 = 3/4, EOT at 1920.
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(960));
        t1.extend_from_slice(&[0xFF, 0x58, 0x04, 0x03, 0x02, 0x18, 0x08]);
        t1.extend_from_slice(&encode_vlq(960));
        t1.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sigs = smf.time_signatures();
        assert_eq!(sigs.len(), 3);
        assert_eq!((sigs[0].tick, sigs[0].track, sigs[0].numerator), (0, 0, 4));
        assert_eq!(
            (sigs[1].tick, sigs[1].track, sigs[1].numerator),
            (960, 1, 3),
        );
        assert_eq!(
            (sigs[2].tick, sigs[2].track, sigs[2].numerator),
            (1920, 0, 7),
        );
    }

    #[test]
    fn time_signatures_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two tracks both place a time signature at tick 240. Track 0
        // must appear first in the merged result (stable sort by tick,
        // insertion order otherwise — track 0 was walked first).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x58, 0x04, 0x02, 0x02, 0x18, 0x08]); // 2/4
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x58, 0x04, 0x05, 0x02, 0x18, 0x08]); // 5/4
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sigs = smf.time_signatures();
        assert_eq!(sigs.len(), 2);
        // Both fire at tick 240; track 0 (2/4) precedes track 1 (5/4).
        assert_eq!(sigs[0].tick, 240);
        assert_eq!(sigs[0].track, 0);
        assert_eq!(sigs[0].numerator, 2);
        assert_eq!(sigs[1].tick, 240);
        assert_eq!(sigs[1].track, 1);
        assert_eq!(sigs[1].numerator, 5);
    }

    #[test]
    fn time_signature_after_channel_events_tracks_absolute_tick() {
        // A channel event uses running status mid-track; the time
        // signature appears later. Make sure absolute-tick accounting
        // doesn't lose the pre-event delta.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 note-off (explicit status to clear running)
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]);
        // delta=240 time signature 12/8 (numerator=12, dd=3)
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x58, 0x04, 0x0C, 0x03, 0x18, 0x08]);
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sigs = smf.time_signatures();
        assert_eq!(sigs.len(), 1);
        // 0 + 120 + 120 + 240 = 480 ticks.
        assert_eq!(sigs[0].tick, 480);
        assert_eq!((sigs[0].numerator, sigs[0].denominator()), (12, 8));
    }

    #[test]
    fn time_signature_denominator_saturates_on_huge_pow2() {
        // Construct a `TimeSignatureChange` directly with the spec-
        // illegal `dd = 250`; the helper must return u32::MAX without
        // overflowing the `1 << dd` shift.
        let ts = TimeSignatureChange {
            tick: 0,
            track: 0,
            numerator: 4,
            denominator_pow2: 250,
            clocks_per_click: 24,
            notated_32nd_per_quarter: 8,
        };
        assert_eq!(ts.denominator(), u32::MAX);
    }

    // ───────── TempoChange / SmfFile::tempo_map ─────────

    #[test]
    fn tempo_map_empty_when_no_meta_event_present() {
        // Track has just a note-on + note-off + EOT — no FF 51.
        let events: &[u8] = &[
            0x00, 0x90, 0x3C, 0x64, 0x40, 0x80, 0x3C, 0x40, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        assert!(smf.tempo_map().is_empty());
    }

    #[test]
    fn tempo_map_single_change_at_tick_zero() {
        // delta=0 FF 51 03 07 A1 20   set tempo 500000us/qn (120 BPM)
        // delta=0 FF 2F 00
        let events: &[u8] = &[
            0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        let map = smf.tempo_map();
        assert_eq!(map.len(), 1);
        let tc = map[0];
        assert_eq!(tc.tick, 0);
        assert_eq!(tc.track, 0);
        assert_eq!(tc.microseconds_per_quarter_note, 500_000);
        // 60_000_000 / 500_000 = 120.0 exactly.
        assert!((tc.bpm - 120.0).abs() < 1e-9);
    }

    #[test]
    fn tempo_map_multiple_changes_within_one_track_are_in_order() {
        // delta=0   FF 51 03 07 A1 20   500000 µs/qn   120 BPM
        // delta=480 FF 51 03 03 D0 90   250000 µs/qn   240 BPM
        // delta=480 FF 51 03 0F 42 40  1000000 µs/qn    60 BPM
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x51, 0x03, 0x0F, 0x42, 0x40]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let map = smf.tempo_map();
        assert_eq!(map.len(), 3);
        assert_eq!(map[0].tick, 0);
        assert_eq!(map[0].microseconds_per_quarter_note, 500_000);
        assert!((map[0].bpm - 120.0).abs() < 1e-9);
        assert_eq!(map[1].tick, 480);
        assert_eq!(map[1].microseconds_per_quarter_note, 250_000);
        assert!((map[1].bpm - 240.0).abs() < 1e-9);
        assert_eq!(map[2].tick, 960);
        assert_eq!(map[2].microseconds_per_quarter_note, 1_000_000);
        assert!((map[2].bpm - 60.0).abs() < 1e-9);
    }

    #[test]
    fn tempo_map_merge_across_tracks_sorted_by_tick() {
        // Track 0: tick 0 = 120 BPM, tick 1920 = 90 BPM, EOT at 1920.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 500_000
        t0.extend_from_slice(&encode_vlq(1920));
        // 60_000_000 / 90 = 666_666.66.. → use 666_667 (0x0A 0x2C 0x2B)
        t0.extend_from_slice(&[0xFF, 0x51, 0x03, 0x0A, 0x2C, 0x2B]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        // Track 1: tick 960 = 240 BPM, EOT at 1920.
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(960));
        t1.extend_from_slice(&[0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]); // 250_000
        t1.extend_from_slice(&encode_vlq(960));
        t1.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let map = smf.tempo_map();
        assert_eq!(map.len(), 3);
        assert_eq!(
            (
                map[0].tick,
                map[0].track,
                map[0].microseconds_per_quarter_note
            ),
            (0, 0, 500_000)
        );
        assert_eq!(
            (
                map[1].tick,
                map[1].track,
                map[1].microseconds_per_quarter_note
            ),
            (960, 1, 250_000)
        );
        assert_eq!(
            (
                map[2].tick,
                map[2].track,
                map[2].microseconds_per_quarter_note
            ),
            (1920, 0, 666_667)
        );
    }

    #[test]
    fn tempo_map_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two tracks both place a tempo change at tick 240. Track 0
        // must appear first in the merged result (stable sort by tick,
        // insertion order otherwise — track 0 was walked first).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 500_000 = 120 BPM
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]); // 250_000 = 240 BPM
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let map = smf.tempo_map();
        assert_eq!(map.len(), 2);
        // Both fire at tick 240; track 0 (500_000) precedes track 1 (250_000).
        assert_eq!(map[0].tick, 240);
        assert_eq!(map[0].track, 0);
        assert_eq!(map[0].microseconds_per_quarter_note, 500_000);
        assert_eq!(map[1].tick, 240);
        assert_eq!(map[1].track, 1);
        assert_eq!(map[1].microseconds_per_quarter_note, 250_000);
    }

    #[test]
    fn tempo_after_channel_events_tracks_absolute_tick() {
        // A channel event uses running status mid-track; the tempo
        // change appears later. Make sure absolute-tick accounting
        // doesn't lose the pre-event delta.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 note-off (explicit status to clear running)
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]);
        // delta=240 set tempo 400_000 µs/qn (= 150 BPM): 06 1A 80
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x51, 0x03, 0x06, 0x1A, 0x80]);
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let map = smf.tempo_map();
        assert_eq!(map.len(), 1);
        // 0 + 120 + 120 + 240 = 480 ticks.
        assert_eq!(map[0].tick, 480);
        assert_eq!(map[0].microseconds_per_quarter_note, 400_000);
        assert!((map[0].bpm - 150.0).abs() < 1e-9);
    }

    #[test]
    fn tempo_change_zero_us_maps_to_infinite_bpm_without_panic() {
        // Construct a `TempoChange` directly with a degenerate
        // microseconds-per-quarter of 0 — the helper must return
        // `f64::INFINITY` for BPM rather than dividing by zero.
        let tc = TempoChange::new(0, 0, 0);
        assert_eq!(tc.microseconds_per_quarter_note, 0);
        assert!(tc.bpm.is_infinite());
        assert!(tc.bpm.is_sign_positive());
    }

    // ───────── KeySignatureChange / SmfFile::key_signatures ─────────

    #[test]
    fn key_signatures_empty_when_no_meta_event_present() {
        // Track has just a note-on + note-off + EOT — no FF 59.
        let events: &[u8] = &[
            0x00, 0x90, 0x3C, 0x64, 0x40, 0x80, 0x3C, 0x40, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        assert!(smf.key_signatures().is_empty());
    }

    #[test]
    fn key_signatures_single_change_at_tick_zero_c_major() {
        // delta=0 FF 59 02 00 00   C major (no accidentals)
        // delta=0 FF 2F 00
        let events: &[u8] = &[0x00, 0xFF, 0x59, 0x02, 0x00, 0x00, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        let keys = smf.key_signatures();
        assert_eq!(keys.len(), 1);
        let ks = keys[0];
        assert_eq!(ks.tick, 0);
        assert_eq!(ks.track, 0);
        assert_eq!(ks.sharps_flats, 0);
        assert_eq!(ks.mode, 0);
        assert!(ks.is_major());
        assert!(!ks.is_minor());
        assert_eq!(ks.tonic_name(), Some("C"));
        assert_eq!(ks.name(), Some("C major"));
    }

    #[test]
    fn key_signatures_multiple_changes_within_one_track_are_in_order() {
        // delta=0   FF 59 02 00 00  C major
        // delta=480 FF 59 02 03 00  A major  (3 sharps, major)
        // delta=480 FF 59 02 FD 01  C minor  (sf=-3, minor)
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x59, 0x02, 0x00, 0x00]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x59, 0x02, 0x03, 0x00]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x59, 0x02, 0xFD, 0x01]); // sf = -3
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let keys = smf.key_signatures();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0].tick, 0);
        assert_eq!((keys[0].sharps_flats, keys[0].mode), (0, 0));
        assert_eq!(keys[0].name(), Some("C major"));
        assert_eq!(keys[1].tick, 480);
        assert_eq!((keys[1].sharps_flats, keys[1].mode), (3, 0));
        assert_eq!(keys[1].name(), Some("A major"));
        assert_eq!(keys[2].tick, 960);
        assert_eq!((keys[2].sharps_flats, keys[2].mode), (-3, 1));
        assert_eq!(keys[2].name(), Some("C minor"));
        assert!(keys[2].is_minor());
    }

    #[test]
    fn key_signatures_merge_across_tracks_sorted_by_tick() {
        // Track 0: tick 0 = C major, tick 1920 = E major (sf=4 mi=0), EOT at 1920.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x59, 0x02, 0x00, 0x00]);
        t0.extend_from_slice(&encode_vlq(1920));
        t0.extend_from_slice(&[0xFF, 0x59, 0x02, 0x04, 0x00]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        // Track 1: tick 960 = D minor (sf=-1 mi=1), EOT at 1920.
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(960));
        t1.extend_from_slice(&[0xFF, 0x59, 0x02, 0xFF, 0x01]); // sf = -1
        t1.extend_from_slice(&encode_vlq(960));
        t1.extend_from_slice(&[0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let keys = smf.key_signatures();
        assert_eq!(keys.len(), 3);
        assert_eq!(
            (
                keys[0].tick,
                keys[0].track,
                keys[0].sharps_flats,
                keys[0].mode
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(
            (
                keys[1].tick,
                keys[1].track,
                keys[1].sharps_flats,
                keys[1].mode
            ),
            (960, 1, -1, 1)
        );
        assert_eq!(keys[1].name(), Some("D minor"));
        assert_eq!(
            (
                keys[2].tick,
                keys[2].track,
                keys[2].sharps_flats,
                keys[2].mode
            ),
            (1920, 0, 4, 0)
        );
        assert_eq!(keys[2].name(), Some("E major"));
    }

    #[test]
    fn key_signatures_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two tracks both place a key signature at tick 240. Track 0
        // must appear first in the merged result.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x59, 0x02, 0x02, 0x00]); // 2 sharps major = D major
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x59, 0x02, 0xFE, 0x01]); // sf=-2 minor = G minor
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let keys = smf.key_signatures();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].tick, 240);
        assert_eq!(keys[0].track, 0);
        assert_eq!(keys[0].name(), Some("D major"));
        assert_eq!(keys[1].tick, 240);
        assert_eq!(keys[1].track, 1);
        assert_eq!(keys[1].name(), Some("G minor"));
    }

    #[test]
    fn key_signature_after_channel_events_tracks_absolute_tick() {
        // A channel event uses running status mid-track; the key
        // signature appears later. Make sure absolute-tick accounting
        // doesn't lose the pre-event delta.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 note-off (explicit status to clear running)
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]);
        // delta=240 key signature F# major (sf=6 mi=0)
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x59, 0x02, 0x06, 0x00]);
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let keys = smf.key_signatures();
        assert_eq!(keys.len(), 1);
        // 0 + 120 + 120 + 240 = 480 ticks.
        assert_eq!(keys[0].tick, 480);
        assert_eq!(keys[0].name(), Some("F# major"));
    }

    #[test]
    fn key_signature_tonic_table_covers_full_circle_of_fifths() {
        // Walk every sf in -7..=+7 for both modes and confirm the
        // names match the textbook circle of fifths. The major / minor
        // rows are the spec's standard mapping (e.g. SMF 1.0 spec
        // page 11): minor is the relative minor of the major a sixth
        // up from its tonic.
        let expected_major = [
            (-7i8, "Cb major"),
            (-6, "Gb major"),
            (-5, "Db major"),
            (-4, "Ab major"),
            (-3, "Eb major"),
            (-2, "Bb major"),
            (-1, "F major"),
            (0, "C major"),
            (1, "G major"),
            (2, "D major"),
            (3, "A major"),
            (4, "E major"),
            (5, "B major"),
            (6, "F# major"),
            (7, "C# major"),
        ];
        for (sf, name) in expected_major {
            let ks = KeySignatureChange {
                tick: 0,
                track: 0,
                sharps_flats: sf,
                mode: 0,
            };
            assert_eq!(ks.name(), Some(name), "major sf={sf}");
        }
        let expected_minor = [
            (-7i8, "Ab minor"),
            (-6, "Eb minor"),
            (-5, "Bb minor"),
            (-4, "F minor"),
            (-3, "C minor"),
            (-2, "G minor"),
            (-1, "D minor"),
            (0, "A minor"),
            (1, "E minor"),
            (2, "B minor"),
            (3, "F# minor"),
            (4, "C# minor"),
            (5, "G# minor"),
            (6, "D# minor"),
            (7, "A# minor"),
        ];
        for (sf, name) in expected_minor {
            let ks = KeySignatureChange {
                tick: 0,
                track: 0,
                sharps_flats: sf,
                mode: 1,
            };
            assert_eq!(ks.name(), Some(name), "minor sf={sf}");
        }
    }

    #[test]
    fn key_signature_out_of_range_or_unknown_mode_yields_none() {
        // sf outside -7..=+7 — name() returns None.
        let ks = KeySignatureChange {
            tick: 0,
            track: 0,
            sharps_flats: 8,
            mode: 0,
        };
        assert_eq!(ks.tonic_name(), None);
        assert_eq!(ks.name(), None);

        let ks = KeySignatureChange {
            tick: 0,
            track: 0,
            sharps_flats: -8,
            mode: 1,
        };
        assert_eq!(ks.tonic_name(), None);
        assert_eq!(ks.name(), None);

        // mode neither 0 nor 1 — name() returns None even with a
        // valid sf. is_major() / is_minor() both report false.
        let ks = KeySignatureChange {
            tick: 0,
            track: 0,
            sharps_flats: 0,
            mode: 2,
        };
        assert_eq!(ks.tonic_name(), None);
        assert_eq!(ks.name(), None);
        assert!(!ks.is_major());
        assert!(!ks.is_minor());
    }

    // ───────── MarkerEvent / SmfFile::markers ─────────

    #[test]
    fn markers_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.markers().is_empty());
    }

    #[test]
    fn markers_single_event_at_tick_zero() {
        // delta=0 FF 06 05 "Verse"
        // delta=0 FF 2F 00
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x06, 0x05];
        events.extend_from_slice(b"Verse");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let mk = smf.markers();
        assert_eq!(mk.len(), 1);
        assert_eq!(mk[0].tick, 0);
        assert_eq!(mk[0].track, 0);
        assert_eq!(mk[0].text_bytes(), b"Verse");
        assert_eq!(mk[0].text_lossy(), "Verse");
    }

    #[test]
    fn markers_multiple_events_within_one_track_are_in_order() {
        // delta=0 FF 06 5 "Intro"
        // delta=240 FF 06 5 "Verse"
        // delta=240 FF 06 6 "Chorus"
        // delta=0 FF 2F 00
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x06, 0x05];
        events.extend_from_slice(b"Intro");
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x06, 0x05]);
        events.extend_from_slice(b"Verse");
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x06, 0x06]);
        events.extend_from_slice(b"Chorus");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let mk = smf.markers();
        assert_eq!(mk.len(), 3);
        assert_eq!(mk[0].tick, 0);
        assert_eq!(mk[0].text_bytes(), b"Intro");
        assert_eq!(mk[1].tick, 240);
        assert_eq!(mk[1].text_bytes(), b"Verse");
        assert_eq!(mk[2].tick, 480);
        assert_eq!(mk[2].text_bytes(), b"Chorus");
    }

    #[test]
    fn markers_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 marker "A"
        // track 1: delta=120 marker "B"; delta=240 marker "C"
        // → A is at tick 240 on track 0; B at 120, C at 360 on track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x06, 0x01]);
        t0.extend_from_slice(b"A");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x06, 0x01]);
        t1.extend_from_slice(b"B");
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x06, 0x01]);
        t1.extend_from_slice(b"C");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let mk = smf.markers();
        assert_eq!(mk.len(), 3);
        assert_eq!(mk[0].tick, 120);
        assert_eq!(mk[0].track, 1);
        assert_eq!(mk[0].text_bytes(), b"B");
        assert_eq!(mk[1].tick, 240);
        assert_eq!(mk[1].track, 0);
        assert_eq!(mk[1].text_bytes(), b"A");
        assert_eq!(mk[2].tick, 360);
        assert_eq!(mk[2].track, 1);
        assert_eq!(mk[2].text_bytes(), b"C");
    }

    #[test]
    fn markers_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Both tracks land a marker at tick 240. Stable sort keeps
        // track 0 first (matches the scheduler's merge convention).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x06, 0x04]);
        t0.extend_from_slice(b"trk0");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x06, 0x04]);
        t1.extend_from_slice(b"trk1");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let mk = smf.markers();
        assert_eq!(mk.len(), 2);
        assert_eq!(mk[0].tick, 240);
        assert_eq!(mk[0].track, 0);
        assert_eq!(mk[0].text_bytes(), b"trk0");
        assert_eq!(mk[1].tick, 240);
        assert_eq!(mk[1].track, 1);
        assert_eq!(mk[1].text_bytes(), b"trk1");
    }

    #[test]
    fn markers_filter_excludes_other_text_kinds() {
        // FF 03 "Track1" (track name) + FF 06 "Mark" (marker) + FF 05
        // "lyric" (lyric) — only the marker should land in
        // SmfFile::markers().
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x03, 0x06];
        events.extend_from_slice(b"Track1");
        events.extend_from_slice(&[0x00, 0xFF, 0x06, 0x04]);
        events.extend_from_slice(b"Mark");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x05]);
        events.extend_from_slice(b"lyric");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let mk = smf.markers();
        assert_eq!(mk.len(), 1);
        assert_eq!(mk[0].text_bytes(), b"Mark");
    }

    #[test]
    fn marker_after_channel_events_tracks_absolute_tick() {
        // Running-status note-ons interleaved with delta-positioned
        // markers — verifies absolute tick accounting matches the
        // tempo/time/key helpers.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 marker "X"
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x06, 0x01]);
        events.extend_from_slice(b"X");
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let mk = smf.markers();
        assert_eq!(mk.len(), 1);
        // 0 + 120 + 120 = 240
        assert_eq!(mk[0].tick, 240);
        assert_eq!(mk[0].text_bytes(), b"X");
    }

    #[test]
    fn marker_text_lossy_replaces_invalid_utf8() {
        // 0xFF 0xFE is not a valid UTF-8 sequence — text_lossy() must
        // not panic and must surface U+FFFD substitutes.
        let mk = MarkerEvent {
            tick: 0,
            track: 0,
            text: vec![0xFF, 0xFE],
        };
        // U+FFFD is 3 bytes in UTF-8, two replacement chars = 6 bytes.
        let lossy = mk.text_lossy();
        assert!(lossy.contains('\u{FFFD}'));
        assert_eq!(mk.text_bytes(), &[0xFF, 0xFE]);
    }

    // ───────── LyricEvent / SmfFile::lyrics ─────────

    #[test]
    fn lyrics_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.lyrics().is_empty());
    }

    #[test]
    fn lyrics_single_event_at_tick_zero() {
        // delta=0 FF 05 04 "love"
        // delta=0 FF 2F 00
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x05, 0x04];
        events.extend_from_slice(b"love");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ly = smf.lyrics();
        assert_eq!(ly.len(), 1);
        assert_eq!(ly[0].tick, 0);
        assert_eq!(ly[0].track, 0);
        assert_eq!(ly[0].text_bytes(), b"love");
        assert_eq!(ly[0].text_lossy(), "love");
    }

    #[test]
    fn lyrics_multiple_syllables_within_one_track_are_in_order() {
        // .kar convention: one syllable per event, pinned per tick.
        // delta=0   FF 05 4 "Twin"
        // delta=120 FF 05 4 "kle "
        // delta=120 FF 05 4 "twin"
        // delta=120 FF 05 4 "kle "
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x04]);
        events.extend_from_slice(b"Twin");
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x05, 0x04]);
        events.extend_from_slice(b"kle ");
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x05, 0x04]);
        events.extend_from_slice(b"twin");
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x05, 0x04]);
        events.extend_from_slice(b"kle ");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ly = smf.lyrics();
        assert_eq!(ly.len(), 4);
        assert_eq!(ly[0].tick, 0);
        assert_eq!(ly[0].text_bytes(), b"Twin");
        assert_eq!(ly[1].tick, 120);
        assert_eq!(ly[1].text_bytes(), b"kle ");
        assert_eq!(ly[2].tick, 240);
        assert_eq!(ly[2].text_bytes(), b"twin");
        assert_eq!(ly[3].tick, 360);
        assert_eq!(ly[3].text_bytes(), b"kle ");
    }

    #[test]
    fn lyrics_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 lyric "A"
        // track 1: delta=120 lyric "B"; delta=240 lyric "C"
        // → A is at tick 240 on track 0; B at 120, C at 360 on track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x05, 0x01]);
        t0.extend_from_slice(b"A");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x05, 0x01]);
        t1.extend_from_slice(b"B");
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x05, 0x01]);
        t1.extend_from_slice(b"C");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let ly = smf.lyrics();
        assert_eq!(ly.len(), 3);
        assert_eq!(ly[0].tick, 120);
        assert_eq!(ly[0].track, 1);
        assert_eq!(ly[0].text_bytes(), b"B");
        assert_eq!(ly[1].tick, 240);
        assert_eq!(ly[1].track, 0);
        assert_eq!(ly[1].text_bytes(), b"A");
        assert_eq!(ly[2].tick, 360);
        assert_eq!(ly[2].track, 1);
        assert_eq!(ly[2].text_bytes(), b"C");
    }

    #[test]
    fn lyrics_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Both tracks land a lyric at tick 240. Stable sort keeps
        // track 0 first (matches the scheduler's merge convention).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x05, 0x04]);
        t0.extend_from_slice(b"trk0");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x05, 0x04]);
        t1.extend_from_slice(b"trk1");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let ly = smf.lyrics();
        assert_eq!(ly.len(), 2);
        assert_eq!(ly[0].tick, 240);
        assert_eq!(ly[0].track, 0);
        assert_eq!(ly[0].text_bytes(), b"trk0");
        assert_eq!(ly[1].tick, 240);
        assert_eq!(ly[1].track, 1);
        assert_eq!(ly[1].text_bytes(), b"trk1");
    }

    #[test]
    fn lyrics_filter_excludes_other_text_kinds() {
        // FF 03 "Track1" (track name) + FF 06 "Mark" (marker) + FF 05
        // "syll" (lyric) — only the lyric should land in
        // SmfFile::lyrics().
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x03, 0x06];
        events.extend_from_slice(b"Track1");
        events.extend_from_slice(&[0x00, 0xFF, 0x06, 0x04]);
        events.extend_from_slice(b"Mark");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x04]);
        events.extend_from_slice(b"syll");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ly = smf.lyrics();
        assert_eq!(ly.len(), 1);
        assert_eq!(ly[0].text_bytes(), b"syll");
    }

    #[test]
    fn lyric_after_channel_events_tracks_absolute_tick() {
        // Running-status note-ons interleaved with delta-positioned
        // lyrics — verifies absolute tick accounting matches the
        // tempo/time/key/markers helpers.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 lyric "la"
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ly = smf.lyrics();
        assert_eq!(ly.len(), 1);
        // 0 + 120 + 120 = 240
        assert_eq!(ly[0].tick, 240);
        assert_eq!(ly[0].text_bytes(), b"la");
    }

    #[test]
    fn lyric_text_lossy_replaces_invalid_utf8() {
        // 0xFF 0xFE is not a valid UTF-8 sequence — text_lossy() must
        // not panic and must surface U+FFFD substitutes.
        let ly = LyricEvent {
            tick: 0,
            track: 0,
            text: vec![0xFF, 0xFE],
        };
        let lossy = ly.text_lossy();
        assert!(lossy.contains('\u{FFFD}'));
        assert_eq!(ly.text_bytes(), &[0xFF, 0xFE]);
    }

    // ───────── CueEvent / SmfFile::cue_points ─────────

    #[test]
    fn cue_points_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.cue_points().is_empty());
    }

    #[test]
    fn cue_points_single_event_at_tick_zero() {
        // delta=0 FF 07 05 "Scene"
        // delta=0 FF 2F 00
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x07, 0x05];
        events.extend_from_slice(b"Scene");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.cue_points();
        assert_eq!(cp.len(), 1);
        assert_eq!(cp[0].tick, 0);
        assert_eq!(cp[0].track, 0);
        assert_eq!(cp[0].text_bytes(), b"Scene");
        assert_eq!(cp[0].text_lossy(), "Scene");
    }

    #[test]
    fn cue_points_multiple_within_one_track_are_in_order() {
        // Film-score convention: a stream of named sync points.
        // delta=0   FF 07 5 "Intro"
        // delta=240 FF 07 5 "SceneA"  (relative-encoded len of 6)
        // delta=240 FF 07 5 "SceneB"
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x07, 0x05]);
        events.extend_from_slice(b"Intro");
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x07, 0x06]);
        events.extend_from_slice(b"SceneA");
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x07, 0x06]);
        events.extend_from_slice(b"SceneB");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.cue_points();
        assert_eq!(cp.len(), 3);
        assert_eq!(cp[0].tick, 0);
        assert_eq!(cp[0].text_bytes(), b"Intro");
        assert_eq!(cp[1].tick, 240);
        assert_eq!(cp[1].text_bytes(), b"SceneA");
        assert_eq!(cp[2].tick, 480);
        assert_eq!(cp[2].text_bytes(), b"SceneB");
    }

    #[test]
    fn cue_points_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 cue "A"
        // track 1: delta=120 cue "B"; delta=240 cue "C"
        // → A is at tick 240 on track 0; B at 120, C at 360 on track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x07, 0x01]);
        t0.extend_from_slice(b"A");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x07, 0x01]);
        t1.extend_from_slice(b"B");
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x07, 0x01]);
        t1.extend_from_slice(b"C");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cp = smf.cue_points();
        assert_eq!(cp.len(), 3);
        assert_eq!(cp[0].tick, 120);
        assert_eq!(cp[0].track, 1);
        assert_eq!(cp[0].text_bytes(), b"B");
        assert_eq!(cp[1].tick, 240);
        assert_eq!(cp[1].track, 0);
        assert_eq!(cp[1].text_bytes(), b"A");
        assert_eq!(cp[2].tick, 360);
        assert_eq!(cp[2].track, 1);
        assert_eq!(cp[2].text_bytes(), b"C");
    }

    #[test]
    fn cue_points_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Both tracks land a cue at tick 240. Stable sort keeps
        // track 0 first (matches the scheduler's merge convention).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x07, 0x04]);
        t0.extend_from_slice(b"trk0");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x07, 0x04]);
        t1.extend_from_slice(b"trk1");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cp = smf.cue_points();
        assert_eq!(cp.len(), 2);
        assert_eq!(cp[0].tick, 240);
        assert_eq!(cp[0].track, 0);
        assert_eq!(cp[0].text_bytes(), b"trk0");
        assert_eq!(cp[1].tick, 240);
        assert_eq!(cp[1].track, 1);
        assert_eq!(cp[1].text_bytes(), b"trk1");
    }

    #[test]
    fn cue_points_filter_excludes_other_text_kinds() {
        // FF 03 "Track1" (track name) + FF 06 "Mark" (marker) +
        // FF 05 "syll" (lyric) + FF 07 "Cue!" (cue point). Only the
        // cue point should land in SmfFile::cue_points().
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x03, 0x06];
        events.extend_from_slice(b"Track1");
        events.extend_from_slice(&[0x00, 0xFF, 0x06, 0x04]);
        events.extend_from_slice(b"Mark");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x04]);
        events.extend_from_slice(b"syll");
        events.extend_from_slice(&[0x00, 0xFF, 0x07, 0x04]);
        events.extend_from_slice(b"Cue!");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.cue_points();
        assert_eq!(cp.len(), 1);
        assert_eq!(cp[0].text_bytes(), b"Cue!");
        // And neither the marker nor the lyric should be polluted.
        assert_eq!(smf.markers().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
    }

    #[test]
    fn cue_point_after_channel_events_tracks_absolute_tick() {
        // Running-status note-ons interleaved with a delta-positioned
        // cue — verifies absolute tick accounting matches the
        // tempo/time/key/markers/lyrics helpers.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 cue "go"
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x07, 0x02]);
        events.extend_from_slice(b"go");
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.cue_points();
        assert_eq!(cp.len(), 1);
        // 0 + 120 + 120 = 240
        assert_eq!(cp[0].tick, 240);
        assert_eq!(cp[0].text_bytes(), b"go");
    }

    #[test]
    fn cue_point_text_lossy_replaces_invalid_utf8() {
        // 0xFF 0xFE is not a valid UTF-8 sequence — text_lossy() must
        // not panic and must surface U+FFFD substitutes.
        let cp = CueEvent {
            tick: 0,
            track: 0,
            text: vec![0xFF, 0xFE],
        };
        let lossy = cp.text_lossy();
        assert!(lossy.contains('\u{FFFD}'));
        assert_eq!(cp.text_bytes(), &[0xFF, 0xFE]);
    }

    // ───────── TrackNameEvent / SmfFile::track_names ─────────

    #[test]
    fn track_names_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.track_names().is_empty());
    }

    #[test]
    fn track_names_single_event_at_tick_zero() {
        // delta=0 FF 03 06 "Melody"
        // delta=0 FF 2F 00
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x03, 0x06];
        events.extend_from_slice(b"Melody");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tn = smf.track_names();
        assert_eq!(tn.len(), 1);
        assert_eq!(tn[0].tick, 0);
        assert_eq!(tn[0].track, 0);
        assert_eq!(tn[0].text_bytes(), b"Melody");
        assert_eq!(tn[0].text_lossy(), "Melody");
    }

    #[test]
    fn track_names_per_track_in_format_1() {
        // Format-1 convention: each track carries its own FF 03 at
        // tick 0. The merge keeps track 0's name before track 1's.
        let mut t0: Vec<u8> = vec![0x00, 0xFF, 0x03, 0x05];
        t0.extend_from_slice(b"Drums");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = vec![0x00, 0xFF, 0x03, 0x04];
        t1.extend_from_slice(b"Bass");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let tn = smf.track_names();
        assert_eq!(tn.len(), 2);
        assert_eq!(tn[0].tick, 0);
        assert_eq!(tn[0].track, 0);
        assert_eq!(tn[0].text_bytes(), b"Drums");
        assert_eq!(tn[1].tick, 0);
        assert_eq!(tn[1].track, 1);
        assert_eq!(tn[1].text_bytes(), b"Bass");
    }

    #[test]
    fn track_names_multiple_within_one_track_are_in_order() {
        // The spec doesn't forbid multiple FF 03 on one track. We
        // surface every occurrence in time order so callers that
        // only want the first can take .next() on the iterator.
        // delta=0   FF 03 5 "Intro"
        // delta=480 FF 03 4 "Main"
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x05]);
        events.extend_from_slice(b"Intro");
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Main");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tn = smf.track_names();
        assert_eq!(tn.len(), 2);
        assert_eq!(tn[0].tick, 0);
        assert_eq!(tn[0].text_bytes(), b"Intro");
        assert_eq!(tn[1].tick, 480);
        assert_eq!(tn[1].text_bytes(), b"Main");
    }

    #[test]
    fn track_names_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Both tracks land an FF 03 at tick 240. Stable sort keeps
        // track 0 first (matches the scheduler's merge convention).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x03, 0x04]);
        t0.extend_from_slice(b"trk0");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x03, 0x04]);
        t1.extend_from_slice(b"trk1");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let tn = smf.track_names();
        assert_eq!(tn.len(), 2);
        assert_eq!(tn[0].tick, 240);
        assert_eq!(tn[0].track, 0);
        assert_eq!(tn[0].text_bytes(), b"trk0");
        assert_eq!(tn[1].tick, 240);
        assert_eq!(tn[1].track, 1);
        assert_eq!(tn[1].text_bytes(), b"trk1");
    }

    #[test]
    fn track_names_filter_excludes_other_text_kinds() {
        // FF 01 "Note" (general text) + FF 02 "(c)26" (copyright) +
        // FF 03 "Lead" (track name) + FF 04 "Piano" (instrument) +
        // FF 05 "la" (lyric) + FF 06 "Verse" (marker) +
        // FF 07 "Sync" (cue). Only the track name lands here.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x01, 0x04];
        events.extend_from_slice(b"Note");
        events.extend_from_slice(&[0x00, 0xFF, 0x02, 0x05]);
        events.extend_from_slice(b"(c)26");
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x04, 0x05]);
        events.extend_from_slice(b"Piano");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        events.extend_from_slice(&[0x00, 0xFF, 0x06, 0x05]);
        events.extend_from_slice(b"Verse");
        events.extend_from_slice(&[0x00, 0xFF, 0x07, 0x04]);
        events.extend_from_slice(b"Sync");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tn = smf.track_names();
        assert_eq!(tn.len(), 1);
        assert_eq!(tn[0].text_bytes(), b"Lead");
        // And neither marker / lyric / cue helper should be polluted.
        assert_eq!(smf.markers().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
        assert_eq!(smf.cue_points().len(), 1);
    }

    #[test]
    fn track_name_after_channel_events_tracks_absolute_tick() {
        // Running-status note-ons followed by a late-positioned
        // FF 03 — verifies absolute tick accounting matches the
        // tempo/time/key/markers/lyrics/cue_points helpers. The
        // spec permits FF 03 anywhere in the track.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 FF 03 04 "name"
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"name");
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tn = smf.track_names();
        assert_eq!(tn.len(), 1);
        // 0 + 120 + 120 = 240
        assert_eq!(tn[0].tick, 240);
        assert_eq!(tn[0].text_bytes(), b"name");
    }

    #[test]
    fn track_name_text_lossy_replaces_invalid_utf8() {
        // 0xFF 0xFE is not a valid UTF-8 sequence — text_lossy() must
        // not panic and must surface U+FFFD substitutes.
        let tn = TrackNameEvent {
            tick: 0,
            track: 0,
            text: vec![0xFF, 0xFE],
        };
        let lossy = tn.text_lossy();
        assert!(lossy.contains('\u{FFFD}'));
        assert_eq!(tn.text_bytes(), &[0xFF, 0xFE]);
    }

    // ───────── InstrumentNameEvent / SmfFile::instrument_names ─────────

    #[test]
    fn instrument_names_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.instrument_names().is_empty());
    }

    #[test]
    fn instrument_names_single_event_at_tick_zero() {
        // delta=0 FF 04 0B "Grand Piano"
        // delta=0 FF 2F 00
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x04, 0x0B];
        events.extend_from_slice(b"Grand Piano");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let inst = smf.instrument_names();
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].tick, 0);
        assert_eq!(inst[0].track, 0);
        assert_eq!(inst[0].text_bytes(), b"Grand Piano");
        assert_eq!(inst[0].text_lossy(), "Grand Piano");
    }

    #[test]
    fn instrument_names_per_track_in_format_1() {
        // Format-1 convention: each music track that targets a
        // specific instrument carries its own FF 04 at tick 0. The
        // merge keeps track 0's name before track 1's.
        let mut t0: Vec<u8> = vec![0x00, 0xFF, 0x04, 0x08];
        t0.extend_from_slice(b"Drum Kit");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = vec![0x00, 0xFF, 0x04, 0x07];
        t1.extend_from_slice(b"Trumpet");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let inst = smf.instrument_names();
        assert_eq!(inst.len(), 2);
        assert_eq!(inst[0].tick, 0);
        assert_eq!(inst[0].track, 0);
        assert_eq!(inst[0].text_bytes(), b"Drum Kit");
        assert_eq!(inst[1].tick, 0);
        assert_eq!(inst[1].track, 1);
        assert_eq!(inst[1].text_bytes(), b"Trumpet");
    }

    #[test]
    fn instrument_names_multiple_within_one_track_are_in_order() {
        // The spec doesn't forbid multiple FF 04 on one track. We
        // surface every occurrence in time order so callers that
        // only want the first can take .next() on the iterator.
        // delta=0   FF 04 5 "Piano"
        // delta=480 FF 04 6 "Organ"
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x04, 0x05]);
        events.extend_from_slice(b"Piano");
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x04, 0x05]);
        events.extend_from_slice(b"Organ");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let inst = smf.instrument_names();
        assert_eq!(inst.len(), 2);
        assert_eq!(inst[0].tick, 0);
        assert_eq!(inst[0].text_bytes(), b"Piano");
        assert_eq!(inst[1].tick, 480);
        assert_eq!(inst[1].text_bytes(), b"Organ");
    }

    #[test]
    fn instrument_names_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 FF 04 "A"
        // track 1: delta=120 FF 04 "B"; delta=240 FF 04 "C"
        // → A is at tick 240 on track 0; B at 120, C at 360 on track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x04, 0x01]);
        t0.extend_from_slice(b"A");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x04, 0x01]);
        t1.extend_from_slice(b"B");
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x04, 0x01]);
        t1.extend_from_slice(b"C");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let inst = smf.instrument_names();
        assert_eq!(inst.len(), 3);
        assert_eq!(inst[0].tick, 120);
        assert_eq!(inst[0].track, 1);
        assert_eq!(inst[0].text_bytes(), b"B");
        assert_eq!(inst[1].tick, 240);
        assert_eq!(inst[1].track, 0);
        assert_eq!(inst[1].text_bytes(), b"A");
        assert_eq!(inst[2].tick, 360);
        assert_eq!(inst[2].track, 1);
        assert_eq!(inst[2].text_bytes(), b"C");
    }

    #[test]
    fn instrument_names_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Both tracks land an FF 04 at tick 240. Stable sort keeps
        // track 0 first (matches the scheduler's merge convention).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x04, 0x04]);
        t0.extend_from_slice(b"trk0");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x04, 0x04]);
        t1.extend_from_slice(b"trk1");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let inst = smf.instrument_names();
        assert_eq!(inst.len(), 2);
        assert_eq!(inst[0].tick, 240);
        assert_eq!(inst[0].track, 0);
        assert_eq!(inst[0].text_bytes(), b"trk0");
        assert_eq!(inst[1].tick, 240);
        assert_eq!(inst[1].track, 1);
        assert_eq!(inst[1].text_bytes(), b"trk1");
    }

    #[test]
    fn instrument_names_filter_excludes_other_text_kinds() {
        // FF 01 "Note" (general text) + FF 02 "(c)26" (copyright) +
        // FF 03 "Lead" (track name) + FF 04 "Piano" (instrument) +
        // FF 05 "la" (lyric) + FF 06 "Verse" (marker) +
        // FF 07 "Sync" (cue). Only the instrument name lands here.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x01, 0x04];
        events.extend_from_slice(b"Note");
        events.extend_from_slice(&[0x00, 0xFF, 0x02, 0x05]);
        events.extend_from_slice(b"(c)26");
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x04, 0x05]);
        events.extend_from_slice(b"Piano");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        events.extend_from_slice(&[0x00, 0xFF, 0x06, 0x05]);
        events.extend_from_slice(b"Verse");
        events.extend_from_slice(&[0x00, 0xFF, 0x07, 0x04]);
        events.extend_from_slice(b"Sync");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let inst = smf.instrument_names();
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].text_bytes(), b"Piano");
        // The sibling helpers must not pick up the instrument-name
        // event either.
        assert_eq!(smf.track_names().len(), 1);
        assert_eq!(smf.markers().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
        assert_eq!(smf.cue_points().len(), 1);
    }

    #[test]
    fn instrument_name_after_channel_events_tracks_absolute_tick() {
        // Running-status note-ons followed by a late-positioned
        // FF 04 — verifies absolute tick accounting matches the
        // tempo/time/key/markers/lyrics/cue_points/track_names
        // helpers. The spec permits FF 04 anywhere in the track.
        let mut events: Vec<u8> = Vec::new();
        // tick 0 note-on key 60 vel 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        // delta=120 running-status note-on key 64 vel 80
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        // delta=120 FF 04 04 "harp"
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x04, 0x04]);
        events.extend_from_slice(b"harp");
        // delta=0 EOT
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let inst = smf.instrument_names();
        assert_eq!(inst.len(), 1);
        // 0 + 120 + 120 = 240
        assert_eq!(inst[0].tick, 240);
        assert_eq!(inst[0].text_bytes(), b"harp");
    }

    #[test]
    fn instrument_name_coexists_with_track_name_on_same_track() {
        // A track may legally carry both FF 03 (track-list label) and
        // FF 04 (voice/instrument name). The two helpers must surface
        // them independently — neither shall be polluted by the other.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x03, 0x04];
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x04, 0x05]);
        events.extend_from_slice(b"Piano");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tn = smf.track_names();
        let inst = smf.instrument_names();
        assert_eq!(tn.len(), 1);
        assert_eq!(tn[0].text_bytes(), b"Lead");
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].text_bytes(), b"Piano");
    }

    #[test]
    fn instrument_name_text_lossy_replaces_invalid_utf8() {
        // 0xFF 0xFE is not a valid UTF-8 sequence — text_lossy() must
        // not panic and must surface U+FFFD substitutes.
        let inst = InstrumentNameEvent {
            tick: 0,
            track: 0,
            text: vec![0xFF, 0xFE],
        };
        let lossy = inst.text_lossy();
        assert!(lossy.contains('\u{FFFD}'));
        assert_eq!(inst.text_bytes(), &[0xFF, 0xFE]);
    }

    // ───────── TextEvent / SmfFile::texts (FF 01) ─────────

    #[test]
    fn texts_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.texts().is_empty());
    }

    #[test]
    fn texts_single_event_at_tick_zero() {
        // delta=0 FF 01 0B "do not edit"
        // delta=0 FF 2F 00
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x01, 0x0B];
        events.extend_from_slice(b"do not edit");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tx = smf.texts();
        assert_eq!(tx.len(), 1);
        assert_eq!(tx[0].tick, 0);
        assert_eq!(tx[0].track, 0);
        assert_eq!(tx[0].text_bytes(), b"do not edit");
        assert_eq!(tx[0].text_lossy(), "do not edit");
    }

    #[test]
    fn texts_multiple_within_one_track_are_in_order() {
        // delta=0   FF 01 4 "head"
        // delta=480 FF 01 4 "mid1"
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x01, 0x04]);
        events.extend_from_slice(b"head");
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x01, 0x04]);
        events.extend_from_slice(b"mid1");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tx = smf.texts();
        assert_eq!(tx.len(), 2);
        assert_eq!(tx[0].tick, 0);
        assert_eq!(tx[0].text_bytes(), b"head");
        assert_eq!(tx[1].tick, 480);
        assert_eq!(tx[1].text_bytes(), b"mid1");
    }

    #[test]
    fn texts_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 FF 01 "A"
        // track 1: delta=120 FF 01 "B"; delta=240 FF 01 "C"
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x01, 0x01]);
        t0.extend_from_slice(b"A");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x01, 0x01]);
        t1.extend_from_slice(b"B");
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x01, 0x01]);
        t1.extend_from_slice(b"C");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let tx = smf.texts();
        assert_eq!(tx.len(), 3);
        assert_eq!(tx[0].tick, 120);
        assert_eq!(tx[0].track, 1);
        assert_eq!(tx[0].text_bytes(), b"B");
        assert_eq!(tx[1].tick, 240);
        assert_eq!(tx[1].track, 0);
        assert_eq!(tx[1].text_bytes(), b"A");
        assert_eq!(tx[2].tick, 360);
        assert_eq!(tx[2].track, 1);
        assert_eq!(tx[2].text_bytes(), b"C");
    }

    #[test]
    fn texts_stable_sort_keeps_track0_before_track1_at_same_tick() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x01, 0x04]);
        t0.extend_from_slice(b"trk0");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x01, 0x04]);
        t1.extend_from_slice(b"trk1");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let tx = smf.texts();
        assert_eq!(tx.len(), 2);
        assert_eq!(tx[0].track, 0);
        assert_eq!(tx[0].text_bytes(), b"trk0");
        assert_eq!(tx[1].track, 1);
        assert_eq!(tx[1].text_bytes(), b"trk1");
    }

    #[test]
    fn texts_filter_excludes_other_text_kinds() {
        // FF 01 "Note" picked up; FF 02..=FF 07 all filtered out.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x01, 0x04];
        events.extend_from_slice(b"Note");
        events.extend_from_slice(&[0x00, 0xFF, 0x02, 0x05]);
        events.extend_from_slice(b"(c)26");
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x04, 0x05]);
        events.extend_from_slice(b"Piano");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        events.extend_from_slice(&[0x00, 0xFF, 0x06, 0x05]);
        events.extend_from_slice(b"Verse");
        events.extend_from_slice(&[0x00, 0xFF, 0x07, 0x04]);
        events.extend_from_slice(b"Sync");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tx = smf.texts();
        assert_eq!(tx.len(), 1);
        assert_eq!(tx[0].text_bytes(), b"Note");
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.copyrights().len(), 1);
        assert_eq!(smf.track_names().len(), 1);
        assert_eq!(smf.instrument_names().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
        assert_eq!(smf.markers().len(), 1);
        assert_eq!(smf.cue_points().len(), 1);
    }

    #[test]
    fn text_after_channel_events_tracks_absolute_tick() {
        // Running-status note-ons followed by a late-positioned FF 01.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x01, 0x04]);
        events.extend_from_slice(b"note");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tx = smf.texts();
        assert_eq!(tx.len(), 1);
        assert_eq!(tx[0].tick, 240);
        assert_eq!(tx[0].text_bytes(), b"note");
    }

    #[test]
    fn text_text_lossy_replaces_invalid_utf8() {
        let ev = TextEvent {
            tick: 0,
            track: 0,
            text: vec![0xFF, 0xFE],
        };
        let lossy = ev.text_lossy();
        assert!(lossy.contains('\u{FFFD}'));
        assert_eq!(ev.text_bytes(), &[0xFF, 0xFE]);
    }

    // ───────── CopyrightEvent / SmfFile::copyrights (FF 02) ─────────

    #[test]
    fn copyrights_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.copyrights().is_empty());
    }

    #[test]
    fn copyrights_single_event_at_tick_zero() {
        // delta=0 FF 02 0E "(c) 2026 KLB"
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x02, 0x0C];
        events.extend_from_slice(b"(c) 2026 KLB");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.copyrights();
        assert_eq!(cp.len(), 1);
        assert_eq!(cp[0].tick, 0);
        assert_eq!(cp[0].track, 0);
        assert_eq!(cp[0].text_bytes(), b"(c) 2026 KLB");
        assert_eq!(cp[0].text_lossy(), "(c) 2026 KLB");
    }

    #[test]
    fn copyrights_multiple_within_one_track_are_in_order() {
        // delta=0   FF 02 5 "(c)A"
        // delta=480 FF 02 4 "(c)B"
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x02, 0x04]);
        events.extend_from_slice(b"(c)A");
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x02, 0x04]);
        events.extend_from_slice(b"(c)B");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.copyrights();
        assert_eq!(cp.len(), 2);
        assert_eq!(cp[0].tick, 0);
        assert_eq!(cp[0].text_bytes(), b"(c)A");
        assert_eq!(cp[1].tick, 480);
        assert_eq!(cp[1].text_bytes(), b"(c)B");
    }

    #[test]
    fn copyrights_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 FF 02 "A"
        // track 1: delta=120 FF 02 "B"; delta=240 FF 02 "C"
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x02, 0x01]);
        t0.extend_from_slice(b"A");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x02, 0x01]);
        t1.extend_from_slice(b"B");
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x02, 0x01]);
        t1.extend_from_slice(b"C");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cp = smf.copyrights();
        assert_eq!(cp.len(), 3);
        assert_eq!(cp[0].tick, 120);
        assert_eq!(cp[0].track, 1);
        assert_eq!(cp[0].text_bytes(), b"B");
        assert_eq!(cp[1].tick, 240);
        assert_eq!(cp[1].track, 0);
        assert_eq!(cp[1].text_bytes(), b"A");
        assert_eq!(cp[2].tick, 360);
        assert_eq!(cp[2].track, 1);
        assert_eq!(cp[2].text_bytes(), b"C");
    }

    #[test]
    fn copyrights_stable_sort_keeps_track0_before_track1_at_same_tick() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x02, 0x04]);
        t0.extend_from_slice(b"trk0");
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x02, 0x04]);
        t1.extend_from_slice(b"trk1");
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cp = smf.copyrights();
        assert_eq!(cp.len(), 2);
        assert_eq!(cp[0].track, 0);
        assert_eq!(cp[0].text_bytes(), b"trk0");
        assert_eq!(cp[1].track, 1);
        assert_eq!(cp[1].text_bytes(), b"trk1");
    }

    #[test]
    fn copyrights_filter_excludes_other_text_kinds() {
        // FF 01..07 — only FF 02 picked up.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x01, 0x04];
        events.extend_from_slice(b"Note");
        events.extend_from_slice(&[0x00, 0xFF, 0x02, 0x05]);
        events.extend_from_slice(b"(c)26");
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x04, 0x05]);
        events.extend_from_slice(b"Piano");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        events.extend_from_slice(&[0x00, 0xFF, 0x06, 0x05]);
        events.extend_from_slice(b"Verse");
        events.extend_from_slice(&[0x00, 0xFF, 0x07, 0x04]);
        events.extend_from_slice(b"Sync");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.copyrights();
        assert_eq!(cp.len(), 1);
        assert_eq!(cp[0].text_bytes(), b"(c)26");
        // The seven sibling helpers must each see exactly their one
        // event — none of them contaminate each other.
        assert_eq!(smf.texts().len(), 1);
        assert_eq!(smf.track_names().len(), 1);
        assert_eq!(smf.instrument_names().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
        assert_eq!(smf.markers().len(), 1);
        assert_eq!(smf.cue_points().len(), 1);
    }

    #[test]
    fn copyright_after_channel_events_tracks_absolute_tick() {
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x02, 0x04]);
        events.extend_from_slice(b"(c)2");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cp = smf.copyrights();
        assert_eq!(cp.len(), 1);
        assert_eq!(cp[0].tick, 240);
        assert_eq!(cp[0].text_bytes(), b"(c)2");
    }

    #[test]
    fn copyright_text_lossy_replaces_invalid_utf8() {
        let ev = CopyrightEvent {
            tick: 0,
            track: 0,
            text: vec![0xFF, 0xFE],
        };
        let lossy = ev.text_lossy();
        assert!(lossy.contains('\u{FFFD}'));
        assert_eq!(ev.text_bytes(), &[0xFF, 0xFE]);
    }

    #[test]
    fn texts_and_copyrights_independent_on_same_track() {
        // A track may legally carry both FF 01 (free text) and FF 02
        // (copyright). The two helpers must surface them independently.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x01, 0x04];
        events.extend_from_slice(b"note");
        events.extend_from_slice(&[0x00, 0xFF, 0x02, 0x05]);
        events.extend_from_slice(b"(c)KL");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tx = smf.texts();
        let cp = smf.copyrights();
        assert_eq!(tx.len(), 1);
        assert_eq!(tx[0].text_bytes(), b"note");
        assert_eq!(cp.len(), 1);
        assert_eq!(cp[0].text_bytes(), b"(c)KL");
    }
}
