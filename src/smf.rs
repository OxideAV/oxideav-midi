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

/// Decoded SMPTE frame-rate enumeration. The MIDI Time Code spec
/// (RP-004/008 §"HOURS COUNT") packs the rate into bits 5-6 of the
/// `hr` byte used by both the `FF 54` SMPTE Offset meta event and the
/// MTC Full / Quarter-Frame messages:
///
/// | `yy` | rate                                |
/// |------|-------------------------------------|
/// | `0`  | [`FrameRate::Fps24`]                |
/// | `1`  | [`FrameRate::Fps25`]                |
/// | `2`  | [`FrameRate::Fps30DropFrame`]       |
/// | `3`  | [`FrameRate::Fps30NonDrop`]         |
///
/// Returned by [`SmpteOffsetEvent::frame_rate`] / decoded directly via
/// [`FrameRate::from_hours_byte`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameRate {
    /// 24 frames per second — classic film rate.
    Fps24,
    /// 25 frames per second — PAL/SECAM video rate.
    Fps25,
    /// 30 frames per second, drop-frame — NTSC compensation that drops
    /// two frame numbers each minute (except every tenth minute) to
    /// keep the time-of-day clock aligned with 29.97-Hz playback.
    Fps30DropFrame,
    /// 30 frames per second, non-drop — strict 30-Hz numbering.
    Fps30NonDrop,
}

impl FrameRate {
    /// Decode the rate from a raw `hr` byte (bits 5-6 hold the type
    /// per RP-004/008). The hours value itself lives in bits 0-4 and
    /// is exposed via [`SmpteOffsetEvent::hours_count`].
    pub fn from_hours_byte(hr: u8) -> Self {
        match (hr >> 5) & 0b11 {
            0 => FrameRate::Fps24,
            1 => FrameRate::Fps25,
            2 => FrameRate::Fps30DropFrame,
            _ => FrameRate::Fps30NonDrop,
        }
    }

    /// Nominal frame count per second. For drop-frame this is the
    /// 30-Hz frame-numbering rate (the underlying playback runs at
    /// 29.97 Hz, but the SMPTE counter still labels 30 frames per
    /// second of wall time, skipping two numbers per minute to stay
    /// aligned).
    pub fn frames_per_second(&self) -> u32 {
        match self {
            FrameRate::Fps24 => 24,
            FrameRate::Fps25 => 25,
            FrameRate::Fps30DropFrame | FrameRate::Fps30NonDrop => 30,
        }
    }

    /// Whether this rate uses drop-frame numbering.
    pub fn is_drop_frame(&self) -> bool {
        matches!(self, FrameRate::Fps30DropFrame)
    }
}

/// One SMPTE Offset meta event pinned to the absolute tick (relative
/// to the start of its parent track) at which the
/// [`FF 54 05 hr mn se fr ff`](MetaEvent::SmpteOffset) meta event
/// fires.
///
/// Returned by [`SmfFile::smpte_offsets`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `FF 54` declares the SMPTE wall-clock time at which the *parent
/// track's first event* is meant to fire. The Standard MIDI File
/// specification places the event at the start of the track (tick 0)
/// so a sequencer cueing to SMPTE can autolocate without reading the
/// rest of the track; later placement is uncommon but not forbidden,
/// so this helper surfaces every occurrence in time order.
///
/// The `hr` byte packs the SMPTE frame rate in bits 5-6 (per the
/// MIDI Time Code spec, RP-004/008 §"HOURS COUNT"). The raw byte is
/// preserved in [`hours_raw`](Self::hours_raw); the decoded rate is
/// available via [`frame_rate`](Self::frame_rate) and the hours
/// value via [`hours_count`](Self::hours_count). `minutes`,
/// `seconds`, `frames` are the plain counter values; `subframes`
/// counts 1/100 of a frame (the conventional MTC encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmpteOffsetEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase. Typically `0`
    /// since the SMPTE offset conventionally lands at the head of the
    /// track, but the spec permits later placement.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). The SMF spec applies the offset to the
    /// parent track's first event; this helper preserves which track
    /// each offset belongs to so format-2 multi-pattern files can be
    /// scheduled independently.
    pub track: usize,
    /// Raw `hr` byte from the meta event payload. Bits 5-6 hold the
    /// frame-rate type (`yy`); bits 0-4 hold the hours count
    /// (`0..=23`). Bit 7 is reserved per RP-004/008 and ignored by
    /// receivers.
    pub hours_raw: u8,
    /// Minutes count (`0..=59` per spec; preserved as-is from the
    /// stream — pathological out-of-range values stay visible to the
    /// caller).
    pub minutes: u8,
    /// Seconds count (`0..=59` per spec).
    pub seconds: u8,
    /// Frames count (`0..=29` for 30 Hz; `0..=24` for 25 Hz;
    /// `0..=23` for 24 Hz — preserved as-is).
    pub frames: u8,
    /// Sub-frames — fractional frames in 1/100 units (`0..=99` per
    /// the SMF spec, matching the MTC cueing "ff" field).
    pub subframes: u8,
}

impl SmpteOffsetEvent {
    /// Decoded frame rate from the packed `hr` byte (bits 5-6).
    pub fn frame_rate(&self) -> FrameRate {
        FrameRate::from_hours_byte(self.hours_raw)
    }

    /// Decoded hours count from the `hr` byte (bits 0-4). The SMPTE
    /// spec bounds this to `0..=23`; values above 23 indicate a
    /// malformed file and are returned as-is so the caller can
    /// inspect them (no clamp, no panic).
    pub fn hours_count(&self) -> u8 {
        self.hours_raw & 0b0001_1111
    }

    /// Total wall-clock offset in seconds, computed from the hours,
    /// minutes, seconds counts plus the fractional contribution of
    /// `frames + subframes/100` divided by the decoded
    /// frames-per-second. Returns the same `f64` regardless of
    /// drop-frame status — drop-frame numbering compensates the
    /// counter, not the wall clock, so a counter reading of
    /// `01:00:00:00` at 30-drop still corresponds to ~3600 s of real
    /// time. Callers that need exact NTSC-accurate timing should
    /// re-derive the seconds from the underlying tempo map.
    pub fn seconds_total(&self) -> f64 {
        let rate = self.frame_rate().frames_per_second() as f64;
        let frame_frac = self.frames as f64 + (self.subframes as f64) / 100.0;
        (self.hours_count() as f64) * 3600.0
            + (self.minutes as f64) * 60.0
            + (self.seconds as f64)
            + frame_frac / rate
    }
}

/// One sequencer-specific meta event pinned to the absolute tick
/// (relative to the start of its parent track) at which the
/// [`FF 7F len data`](MetaEvent::SequencerSpecific) meta event fires.
///
/// Returned by [`SmfFile::sequencer_specifics`] — see that method for
/// the merge semantics across multiple tracks.
///
/// `FF 7F` is the SMF escape hatch for sequencer-private or
/// manufacturer-private payloads carried inline with the music data.
/// The payload bytes are opaque to the SMF reader: by convention the
/// first byte (or first three bytes, when the first byte is `0x00`)
/// hold the SysEx-style manufacturer ID, and the rest is whatever the
/// originating sequencer chose to embed (project markers, plugin
/// state, automation hooks, …). The reader does not interpret the
/// payload — the raw bytes are surfaced verbatim so a caller that
/// knows the originating sequencer can decode them, while a generic
/// player can ignore them per spec.
///
/// This helper isolates the `FF 7F` stream so callers driving DAW
/// round-trip workflows (preserving the embedded blobs through a
/// load → save cycle) get a clean time-ordered list independent of
/// the surrounding text / rhythmic / cueing meta events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequencerSpecificEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files may place sequencer
    /// blobs on any track (typically the conductor / track 0 for
    /// project-wide metadata, or the affected music track for
    /// per-instrument state); format-2 files keep them per-track.
    pub track: usize,
    /// Raw payload bytes (the `data` portion of `FF 7F len data`).
    /// The SMF spec defines no further structure — by SysEx
    /// convention bytes `[0]` (or `[0..=2]` when `[0] == 0x00`) hold
    /// the manufacturer ID, but the parser does not interpret them.
    /// Stored as `Vec<u8>` so the caller can route by ID or treat
    /// the payload as opaque per the spec's ignore-if-unknown rule.
    pub data: Vec<u8>,
}

impl SequencerSpecificEvent {
    /// Borrow the raw payload bytes.
    pub fn data_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// One sequence-number meta event pinned to the absolute tick
/// (relative to the start of its parent track) at which the
/// [`FF 00 02 ssss`](MetaEvent::SequenceNumber) meta event fires.
///
/// Returned by [`SmfFile::sequence_numbers`] — see that method for
/// the merge semantics across multiple tracks.
///
/// `FF 00 02 ssss` declares a 16-bit identifier for the sequence the
/// track belongs to. The Standard MIDI File Specification 1.0 reserves
/// the event for the very first event of a track (delta time zero); on
/// a format-2 file it labels each pattern so the sequence can be cued
/// from a Song Select / MIDI Cue message, and on a format-0 / format-1
/// file it labels the file as a whole (the spec recommends placing it
/// on the conductor / track-0). The parser surfaces every occurrence
/// in time order so callers can recover labels from files that place
/// the event off the recommended tick or carry one per track.
///
/// This helper isolates the `FF 00` stream so callers cueing pattern
/// playback or building DAW track lists keyed on the sequence ID get
/// a clean time-ordered list independent of the surrounding text /
/// rhythmic / cueing / sequencer-private meta events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceNumberEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase. The Standard
    /// MIDI File Specification 1.0 reserves the event for delta-time
    /// zero (the first event of a track); the parser surfaces every
    /// occurrence rather than enforcing the placement rule so files
    /// that carry the event later in a track still round-trip.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files conventionally place the
    /// label on track 0 (conductor); format-2 files carry one per
    /// pattern track so each pattern has its own identifier.
    pub track: usize,
    /// The 16-bit sequence identifier (`ssss` from the `FF 00 02 ssss`
    /// byte sequence, big-endian). The Standard MIDI File
    /// Specification 1.0 leaves the value space unconstrained — any
    /// `0..=65535` is legal and round-tripped verbatim.
    pub number: u16,
}

impl SequenceNumberEvent {
    /// The 16-bit sequence identifier.
    pub fn number(&self) -> u16 {
        self.number
    }
}

/// One MIDI Port meta event pinned to the absolute tick (relative to
/// the start of its parent track) at which the
/// [`FF 21 01 pp`](MetaEvent::Port) meta event fires.
///
/// Returned by [`SmfFile::midi_ports`] — see that method for the merge
/// semantics across multiple tracks.
///
/// `FF 21 01 pp` carries an unofficial routing hint that was added to
/// the SMF spec ahead of the multi-port era: the single payload byte
/// names the **physical port** (`0..=127`) the surrounding channel
/// messages on this track should be dispatched through. The Standard
/// MIDI File Specification 1.0 keeps four channel-message status
/// nibbles (`8x..Ex` for `Note Off..Pitch Bend`) and four channel bits
/// (`x0..xF`), which caps a single output stream at 16 channels; the
/// `FF 21` hint multiplies that ceiling by however many physical ports
/// the receiving sequencer wires up, by labelling each track with the
/// port it routes to. The convention is one `FF 21` near the start of
/// the track (delta-time zero, before the first channel-voice event)
/// and at most one per track, but the parser surfaces every occurrence
/// in time order so files that re-route mid-track still round-trip.
///
/// This helper isolates the `FF 21` stream so callers driving a
/// multi-port DAW back-end (16 channels × N ports) get a clean
/// time-ordered list independent of the surrounding meta streams. The
/// payload byte is surfaced verbatim — values `> 127` cannot occur (the
/// parser rejects them at the `data.len() == 1` precondition) and the
/// spec leaves the mapping from port index to physical destination up
/// to the receiving application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MidiPortEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase. The
    /// pre-multi-port convention places the routing hint at delta-time
    /// zero; the parser surfaces every occurrence rather than enforcing
    /// the placement rule so files that re-route mid-track still
    /// round-trip.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). The port-routing convention pairs one
    /// `FF 21` with the channel-voice stream on that track, so this
    /// field tells a multi-port back-end which track's channel
    /// messages the port assignment applies to.
    pub track: usize,
    /// The physical port byte (the `pp` payload of `FF 21 01 pp`,
    /// `0..=127`). The Standard MIDI File Specification 1.0 leaves the
    /// mapping from this index to a physical 5-pin DIN output up to
    /// the receiving application — typically `0` is the first output
    /// port, `1` the second, and so on. Values `> 127` cannot occur:
    /// the spec reserves the high bit for status / running-status use
    /// and the parser rejects `FF 21` payloads that aren't exactly one
    /// byte.
    pub port: u8,
}

impl MidiPortEvent {
    /// The physical port index (`0..=127`).
    pub fn port(&self) -> u8 {
        self.port
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

    /// Collect every [`MetaEvent::SmpteOffset`] (`FF 54 05 hr mn se
    /// fr ff`) from every track, pinned to the absolute tick at which
    /// it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two offsets at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] /
    /// [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::track_names`] /
    /// [`SmfFile::instrument_names`] / [`SmfFile::texts`] /
    /// [`SmfFile::copyrights`] and the scheduler use (`scheduler.rs`
    /// §"merged event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries an SMPTE Offset
    /// meta event — the spec doesn't require one, and a sequencer
    /// that needs a default cue should treat absence as
    /// `00:00:00:00:00` at the file's nominal frame rate.
    ///
    /// `FF 54` declares the SMPTE wall-clock position at which the
    /// *parent track's first event* is meant to fire — the offset is
    /// rate-typed via the bits-5-6 packing on the `hr` byte (see
    /// [`FrameRate`] and the MIDI Time Code spec, RP-004/008
    /// §"HOURS COUNT"). The SMF specification places the event at
    /// the head of the track so a sequencer can autolocate without
    /// reading the rest of the track; later placement is uncommon
    /// but not forbidden, so this helper surfaces every occurrence.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn smpte_offsets(&self) -> Vec<SmpteOffsetEvent> {
        let mut out: Vec<SmpteOffsetEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::SmpteOffset {
                    hours,
                    minutes,
                    seconds,
                    frames,
                    subframes,
                }) = &ev.kind
                {
                    out.push(SmpteOffsetEvent {
                        tick: abs,
                        track: track_idx,
                        hours_raw: *hours,
                        minutes: *minutes,
                        seconds: *seconds,
                        frames: *frames,
                        subframes: *subframes,
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

    /// Collect every [`MetaEvent::SequencerSpecific`] (`FF 7F len
    /// data`) from every track, pinned to the absolute tick at which
    /// it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two payloads at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] /
    /// [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::track_names`] /
    /// [`SmfFile::instrument_names`] / [`SmfFile::texts`] /
    /// [`SmfFile::copyrights`] / [`SmfFile::smpte_offsets`] and the
    /// scheduler use (`scheduler.rs` §"merged event list, sorted by
    /// absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries an `FF 7F` meta
    /// event — every file is free of sequencer-private payloads
    /// unless the originating tool chose to embed one.
    ///
    /// Only `FF 7F` is selected — channel-message `F0` / `F7` SysEx
    /// events (which travel as [`Event::SysEx`]) are *not* surfaced
    /// here; those are part of the wire-event stream rather than the
    /// meta-event family, and downstream consumers route them through
    /// the scheduler's SysEx pump rather than reading them out as a
    /// list. The two channels coexist: a file may carry both an
    /// `F0 … F7` Universal Real-Time Master Volume on the conductor
    /// track and a private `FF 7F` plugin-state blob alongside it.
    ///
    /// The payload bytes are surfaced verbatim — the parser does not
    /// interpret the manufacturer-ID convention, so a caller routing
    /// by manufacturer should inspect [`SequencerSpecificEvent::data`]
    /// directly (typically `data[0]`, or `data[0..=2]` when
    /// `data[0] == 0x00`). Empty payloads (`FF 7F 00`) are surfaced
    /// as `data.is_empty()` rather than filtered out — the spec
    /// permits a zero-length blob.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn sequencer_specifics(&self) -> Vec<SequencerSpecificEvent> {
        let mut out: Vec<SequencerSpecificEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::SequencerSpecific(data)) = &ev.kind {
                    out.push(SequencerSpecificEvent {
                        tick: abs,
                        track: track_idx,
                        data: data.clone(),
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

    /// Collect every [`MetaEvent::SequenceNumber`] (`FF 00 02 ssss`)
    /// from every track, pinned to the absolute tick at which it
    /// fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two labels at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] /
    /// [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::track_names`] /
    /// [`SmfFile::instrument_names`] / [`SmfFile::texts`] /
    /// [`SmfFile::copyrights`] / [`SmfFile::smpte_offsets`] /
    /// [`SmfFile::sequencer_specifics`] and the scheduler use
    /// (`scheduler.rs` §"merged event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries an `FF 00` meta
    /// event. A file without explicit sequence numbers is legal; the
    /// Standard MIDI File Specification 1.0 leaves the identifier
    /// optional.
    ///
    /// Only `FF 00` is selected — neighbouring rhythmic, text, cueing,
    /// and sequencer-private meta events stay on their own helpers so
    /// callers cueing pattern playback from the sequence ID get a clean
    /// time-ordered list independent of the surrounding meta streams.
    ///
    /// Lifts the SMF meta-event iterator family from 12 to **13**
    /// total: `tempo_map`, `time_signatures`, `key_signatures`,
    /// `markers`, `lyrics`, `cue_points`, `track_names`,
    /// `instrument_names`, `texts`, `copyrights`, `smpte_offsets`,
    /// `sequencer_specifics`, and `sequence_numbers`.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn sequence_numbers(&self) -> Vec<SequenceNumberEvent> {
        let mut out: Vec<SequenceNumberEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::SequenceNumber(number)) = &ev.kind {
                    out.push(SequenceNumberEvent {
                        tick: abs,
                        track: track_idx,
                        number: *number,
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

    /// Collect every [`MetaEvent::Port`] (`FF 21 01 pp`) from every
    /// track, pinned to the absolute tick at which it fires, in time
    /// order.
    ///
    /// `FF 21 01 pp` carries an unofficial routing hint: the single
    /// payload byte (`pp`, `0..=127`) names the physical port the
    /// surrounding channel messages on this track should be dispatched
    /// through. The Standard MIDI File Specification 1.0 caps a single
    /// channel-voice stream at 16 channels (the four status nibbles
    /// `8x..Ex` combined with the four channel bits `x0..xF`); the
    /// port hint lets a multi-port DAW back-end multiply that ceiling
    /// by however many physical outputs it wires up, by labelling each
    /// track with the port it routes to.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two hints at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] /
    /// [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::track_names`] /
    /// [`SmfFile::instrument_names`] / [`SmfFile::texts`] /
    /// [`SmfFile::copyrights`] / [`SmfFile::smpte_offsets`] /
    /// [`SmfFile::sequencer_specifics`] / [`SmfFile::sequence_numbers`]
    /// and the scheduler use (`scheduler.rs` §"merged event list,
    /// sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries an `FF 21` meta
    /// event. A file without explicit port hints is legal; the
    /// pre-multi-port convention is one output port, so a receiver
    /// that finds no `FF 21` should dispatch every channel-voice
    /// event through port `0` (or whatever the application's default
    /// is).
    ///
    /// Only `FF 21` is selected — the neighbouring `FF 20`
    /// channel-prefix hint (deprecated, but seen in the wild) /
    /// rhythmic, text, cueing, and sequencer-private meta events stay
    /// on their own helpers so callers driving a port-routing layer
    /// get a clean time-ordered list independent of the surrounding
    /// meta streams.
    ///
    /// The convention is one `FF 21` near the start of a track (delta
    /// zero, before the first channel-voice event) and at most one
    /// per track, but the helper surfaces every occurrence rather
    /// than enforcing the placement rule so files that re-route
    /// mid-track still round-trip.
    ///
    /// Lifts the SMF meta-event iterator family from 13 to **14**
    /// total: `tempo_map`, `time_signatures`, `key_signatures`,
    /// `markers`, `lyrics`, `cue_points`, `track_names`,
    /// `instrument_names`, `texts`, `copyrights`, `smpte_offsets`,
    /// `sequencer_specifics`, `sequence_numbers`, and `midi_ports`.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn midi_ports(&self) -> Vec<MidiPortEvent> {
        let mut out: Vec<MidiPortEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::Port(port)) = &ev.kind {
                    out.push(MidiPortEvent {
                        tick: abs,
                        track: track_idx,
                        port: *port,
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

    /// Reconstruct the **on-the-wire channel state** for `channel`
    /// (0..=15) at absolute `tick`, by replaying every channel-voice
    /// event up to and including `tick` against the SMF spec's
    /// well-defined CC and program / pitch-bend defaults.
    ///
    /// Returns an [`SmfChannelSnapshot`] capturing the "what would a
    /// receiver be set to at this point" view of the channel: last
    /// Program Change, Bank Select MSB / LSB (CC 0 / CC 32), Channel
    /// Volume (CC 7), Pan (CC 10), Expression (CC 11), Modulation
    /// Wheel (CC 1), Sustain Pedal (CC 64, decoded as `value >= 64`),
    /// and the current 14-bit Pitch Bend value.
    ///
    /// This is the seek primitive: a player that wants to start
    /// playback at tick `T` calls
    /// `channel_snapshots_at(T)` once, pushes each non-default field
    /// into its synth as if a CC / Program Change / Pitch Bend had
    /// just arrived, and then begins emitting events at or after `T`.
    /// Without this initialisation the synth would render every note
    /// after `T` against GM defaults (volume 100, pan centre, modwheel
    /// 0, ...) rather than the state the file's earlier events had
    /// established.
    ///
    /// Events are replayed in scheduler order — every track is
    /// time-merged with a stable sort by absolute tick, track 0
    /// winning over track 1 at the same tick — exactly the merge
    /// rule used by [`SmfFile::tempo_map`] / `time_signatures` /
    /// `key_signatures` / the eight text-meta helpers /
    /// `smpte_offsets` and by `scheduler.rs` §"merged event list,
    /// sorted by absolute tick". Events at exactly `tick` are
    /// included in the replay — the snapshot reflects the channel
    /// state immediately *after* that tick fires.
    ///
    /// Format-2 files keep tracks independent (each is a separate
    /// pattern). The snapshot still pools every track's events into
    /// one timeline; callers playing one format-2 track in isolation
    /// should instead inspect [`SmfFile::tracks`] directly. (A
    /// format-2-aware variant could be added in a later round if the
    /// generic merge proves wrong for those callers.)
    ///
    /// Defaults follow the SMF spec + GM 1 *General MIDI System Level
    /// 1 Specification* (RP-003) recommended initial values:
    ///
    /// | field                    | default | source                       |
    /// |--------------------------|---------|------------------------------|
    /// | `program`                | `None`  | spec: no implicit program    |
    /// | `bank_msb`               | `None`  | spec: no implicit bank       |
    /// | `bank_lsb`               | `None`  | spec: no implicit bank       |
    /// | `volume` (CC 7)          | `100`   | GM 1 §"Default Channel Vol." |
    /// | `pan` (CC 10)            | `64`    | GM 1 §"Default Pan = centre" |
    /// | `expression` (CC 11)     | `127`   | GM 1 §"Default Expression"   |
    /// | `modulation` (CC 1)      | `0`     | GM 1 §"Default Modulation"   |
    /// | `sustain` (CC 64)        | `false` | GM 1 §"Default Sustain Off"  |
    /// | `pitch_bend`             | `0x2000`| SMF §"En lsb msb" centre     |
    ///
    /// The implementation is linear in the total event count and
    /// bounded above by [`MAX_EVENTS_PER_FILE`] — same as the
    /// existing iteration helpers. Note-on / note-off / aftertouch
    /// events are *not* replayed — they affect voice state, not
    /// channel state — so the snapshot is independent of the
    /// note timeline.
    pub fn channel_snapshot_at(&self, channel: u8, tick: u64) -> SmfChannelSnapshot {
        let mut snap = SmfChannelSnapshot::default();
        if channel >= 16 {
            return snap;
        }
        // Build a (abs_tick, track_idx, in_track_idx) ordering across
        // every channel-voice event on `channel`, then replay them
        // in (tick, track, in-track) order — stable sort preserves
        // the scheduler's merge convention.
        let mut events: Vec<(u64, usize, usize, ChannelBody)> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for (in_track_idx, ev) in track.events.iter().enumerate() {
                abs = abs.saturating_add(ev.delta as u64);
                if abs > tick {
                    break;
                }
                if let Event::Channel(ChannelMessage { channel: ch, body }) = &ev.kind {
                    if *ch == channel {
                        events.push((abs, track_idx, in_track_idx, *body));
                    }
                }
            }
        }
        events.sort_by_key(|(t, tr, ix, _)| (*t, *tr, *ix));
        for (_, _, _, body) in events {
            snap.apply(&body);
        }
        snap
    }

    /// Reconstruct an [`SmfChannelSnapshot`] for **every** MIDI
    /// channel (0..=15) at absolute `tick`. Equivalent to calling
    /// [`channel_snapshot_at`](Self::channel_snapshot_at) sixteen
    /// times, but walks the tracks only once — useful when seeking
    /// since a player typically initialises every channel at the
    /// seek target.
    ///
    /// Returns a `[SmfChannelSnapshot; 16]` indexed by channel: index
    /// 9 is the drum channel ("MIDI channel 10" in human-facing
    /// tools); index 0 is the first programmable channel ("MIDI
    /// channel 1"). All sixteen entries start from the SMF / GM 1
    /// defaults documented on
    /// [`channel_snapshot_at`](Self::channel_snapshot_at#defaults).
    pub fn channel_snapshots_at(&self, tick: u64) -> [SmfChannelSnapshot; 16] {
        let mut snaps: [SmfChannelSnapshot; 16] =
            std::array::from_fn(|_| SmfChannelSnapshot::default());
        // Walk every track once and dispatch each channel-voice event
        // into the matching snapshot in (tick, track, in-track) order.
        // We need a stable order, so accumulate into a single Vec
        // and sort, mirroring the merge convention the per-helper
        // accessors use.
        let mut events: Vec<(u64, usize, usize, u8, ChannelBody)> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for (in_track_idx, ev) in track.events.iter().enumerate() {
                abs = abs.saturating_add(ev.delta as u64);
                if abs > tick {
                    break;
                }
                if let Event::Channel(ChannelMessage { channel, body }) = &ev.kind {
                    events.push((abs, track_idx, in_track_idx, *channel, *body));
                }
            }
        }
        events.sort_by_key(|(t, tr, ix, _, _)| (*t, *tr, *ix));
        for (_, _, _, channel, body) in events {
            if (channel as usize) < snaps.len() {
                snaps[channel as usize].apply(&body);
            }
        }
        snaps
    }
}

/// Channel-level state of an SMF stream at one absolute tick, as
/// reconstructed by [`SmfFile::channel_snapshot_at`] /
/// [`SmfFile::channel_snapshots_at`].
///
/// Each field reflects the *most recent* matching channel-voice event
/// up to and including the snapshot tick. Fields that no event has
/// touched stay at the SMF / GM 1 recommended default (see the
/// defaults table on [`SmfFile::channel_snapshot_at`]).
///
/// The snapshot is intentionally a "wire-state" view: it captures
/// what a receiver would be set to, not what the *audible* state is.
/// Note-on / note-off / aftertouch events are not folded in (they
/// affect voice state, not channel state), so the snapshot can be
/// produced cheaply for any tick without enumerating sounding notes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmfChannelSnapshot {
    /// Last MIDI Program Change (`Cn pp`) seen on this channel before
    /// the snapshot tick, or `None` if no Program Change has fired.
    /// A player initialising from this snapshot at seek time should
    /// emit a `Cn pp` to the synth before the first note when this is
    /// `Some`, otherwise leave the program alone (it's already on
    /// power-up patch 0 by default).
    pub program: Option<u8>,
    /// Bank Select MSB (CC 0), or `None` if no `Bn 00 vv` has fired.
    /// Per the GM 2 / MMA Bank Select convention, a seek-time
    /// initialiser should emit CC 0 + CC 32 + Program Change in that
    /// order to switch banks. Surfacing `None` lets the caller skip
    /// the CC 0 emission when no bank select ever happened.
    pub bank_msb: Option<u8>,
    /// Bank Select LSB (CC 32), or `None` if no `Bn 20 vv` has fired.
    pub bank_lsb: Option<u8>,
    /// Channel Volume (CC 7), 0..=127. Default 100 per GM 1 §"Default
    /// Channel Volume".
    pub volume: u8,
    /// Pan (CC 10), 0..=127, 64 = centre. Default 64 per GM 1
    /// §"Default Pan = centre".
    pub pan: u8,
    /// Expression Controller (CC 11), 0..=127. Default 127 — GM 1
    /// §"Default Expression = full" specifies the controller is at
    /// full scale on power-up so that Channel Volume drives the
    /// final attenuation alone until an explicit `Bn 0B vv` arrives.
    pub expression: u8,
    /// Modulation Wheel (CC 1), 0..=127. Default 0.
    pub modulation: u8,
    /// Sustain Pedal (CC 64). `true` when the most recent
    /// `Bn 40 vv` had `vv >= 64`; the spec defines `vv < 64` as
    /// "off" and `vv >= 64` as "on". Default `false`.
    pub sustain: bool,
    /// Live Pitch Bend value as the raw 14-bit MIDI scalar
    /// (`0..=16383`). Centre is `0x2000`. Default `0x2000` (no bend).
    /// Decoded from `En lsb msb` as `(msb << 7) | lsb`.
    pub pitch_bend: u16,
}

impl Default for SmfChannelSnapshot {
    fn default() -> Self {
        Self {
            program: None,
            bank_msb: None,
            bank_lsb: None,
            volume: 100,
            pan: 64,
            expression: 127,
            modulation: 0,
            sustain: false,
            pitch_bend: 0x2000,
        }
    }
}

impl SmfChannelSnapshot {
    /// Fold one channel-voice event into the snapshot. Used by
    /// [`SmfFile::channel_snapshot_at`] /
    /// [`SmfFile::channel_snapshots_at`] during their replay loop;
    /// exposed publicly so callers running their own replay (e.g.
    /// against a custom track ordering) can reuse the same wire
    /// semantics.
    ///
    /// Note-on / note-off / poly-aftertouch / channel-aftertouch are
    /// **ignored** here — they don't modify channel state. Only CC,
    /// Program Change, and Pitch Bend update the snapshot.
    pub fn apply(&mut self, body: &ChannelBody) {
        match *body {
            ChannelBody::ControlChange { controller, value } => match controller {
                0 => self.bank_msb = Some(value),
                1 => self.modulation = value,
                7 => self.volume = value,
                10 => self.pan = value,
                11 => self.expression = value,
                32 => self.bank_lsb = Some(value),
                64 => self.sustain = value >= 64,
                _ => {} // other CCs aren't yet tracked on the snapshot
            },
            ChannelBody::ProgramChange { program } => self.program = Some(program),
            ChannelBody::PitchBend { value } => self.pitch_bend = value,
            ChannelBody::NoteOn { .. }
            | ChannelBody::NoteOff { .. }
            | ChannelBody::PolyAftertouch { .. }
            | ChannelBody::ChannelAftertouch { .. } => {
                // Voice / note state, not channel state — left alone.
            }
        }
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

// ───────────────────────── writer ─────────────────────────

/// Largest VLQ-encodable value (the 4-byte cap from the SMF spec).
/// Anything above this cannot be serialised as a delta-time, meta
/// payload length, or sysex payload length.
pub const MAX_VLQ_VALUE: u32 = 0x0FFF_FFFF;

impl SmfFile {
    /// Serialise this [`SmfFile`] to a complete SMF byte stream
    /// (`MThd` header + one `MTrk` chunk per [`Track`]), suitable to
    /// hand to [`parse`] for a structural round-trip.
    ///
    /// The emitted stream uses explicit status bytes throughout — the
    /// spec permits but does not require running-status compression on
    /// the wire, and the explicit form keeps the writer's output
    /// bit-identical regardless of the track's internal ordering. A
    /// reader that does not honour running status (very few do, the
    /// rule is conventional) can still consume the output unchanged.
    ///
    /// Each track must end with [`MetaEvent::EndOfTrack`]; the writer
    /// returns an error rather than auto-appending one so callers
    /// stay in control of which tick the end marker lands on (the
    /// scheduler keys final-tempo / final-CC events off the EOT tick,
    /// so silently moving it would change semantics). The header
    /// `ntrks` field must match `tracks.len()` for the same reason —
    /// the wire format pins the two together and a mismatch should
    /// surface at encode time rather than silently produce a file
    /// disagreeing with itself.
    ///
    /// Validation is strict so the producer fails fast on values that
    /// cannot fit the wire format:
    ///
    /// * VLQ-encoded values (delta-times, meta lengths, sysex lengths)
    ///   must be `<=` [`MAX_VLQ_VALUE`] (`0x0FFF_FFFF`, the 4-byte cap
    ///   from the SMF spec).
    /// * Channel-voice data bytes (`key`, `velocity`, `controller`,
    ///   `value`, `program`, `pressure`) must have the high bit clear
    ///   (`<= 0x7F`).
    /// * Pitch-bend values must fit in 14 bits (`<= 0x3FFF`).
    /// * The header division must be in legal range — `TicksPerQuarter`
    ///   is `1..=0x7FFF` (the high bit selects SMPTE), `Smpte`
    ///   `frames_per_second` must be one of `{24, 25, 29, 30}`.
    /// * Meta-event payload shapes that the parser fixed (e.g. the
    ///   2-byte SequenceNumber payload, the 3-byte Tempo payload, the
    ///   1-byte Port payload) are emitted at exactly the spec-mandated
    ///   length; the parser would reject any other shape on re-read.
    ///
    /// The output is a `Vec<u8>` rather than a `Write` sink so the
    /// writer can size each chunk in two passes (write the body, then
    /// patch the four-byte length prefix in place) without imposing a
    /// `Seek` bound on callers. SMF files are tiny by audio-codec
    /// standards (a megabyte is unusual) so a heap-resident buffer is
    /// the right shape.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if self.header.ntrks as usize != self.tracks.len() {
            return Err(Error::invalid(format!(
                "SMF: header.ntrks ({}) does not match tracks.len() ({}) — \
                 fix one side before encode so the wire format stays consistent",
                self.header.ntrks,
                self.tracks.len(),
            )));
        }
        let mut out: Vec<u8> = Vec::new();
        write_header_chunk(&mut out, &self.header)?;
        for (i, track) in self.tracks.iter().enumerate() {
            write_track_chunk(&mut out, track)
                .map_err(|e| Error::invalid(format!("SMF: track {i}: {e}", e = err_msg(&e))))?;
        }
        Ok(out)
    }
}

impl Track {
    /// Serialise this [`Track`] as a complete `MTrk` chunk (the `MTrk`
    /// FourCC, the four-byte big-endian chunk length, and the encoded
    /// event body). The track must end with
    /// [`MetaEvent::EndOfTrack`] — see [`SmfFile::to_bytes`] for the
    /// rationale on not auto-appending.
    pub fn to_bytes_chunk(&self) -> Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        write_track_chunk(&mut out, self)?;
        Ok(out)
    }
}

fn err_msg(e: &Error) -> String {
    match e {
        Error::InvalidData(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

fn write_header_chunk(out: &mut Vec<u8>, header: &SmfHeader) -> Result<()> {
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6u32.to_be_bytes());
    let format_code: u16 = match header.format {
        SmfFormat::SingleTrack => 0,
        SmfFormat::MultiTrackSimultaneous => 1,
        SmfFormat::MultiTrackIndependent => 2,
    };
    out.extend_from_slice(&format_code.to_be_bytes());
    out.extend_from_slice(&header.ntrks.to_be_bytes());
    let division_word: u16 = match header.division {
        Division::TicksPerQuarter(t) => {
            if t == 0 || t & 0x8000 != 0 {
                return Err(Error::invalid(format!(
                    "SMF: TicksPerQuarter division {t} out of range (1..=0x7FFF)",
                )));
            }
            t
        }
        Division::Smpte {
            frames_per_second,
            ticks_per_frame,
        } => {
            if !matches!(frames_per_second, 24 | 25 | 29 | 30) {
                return Err(Error::invalid(format!(
                    "SMF: SMPTE frame rate {frames_per_second} not in {{24, 25, 29, 30}}",
                )));
            }
            // The wire format stores `-fps` in the upper byte as i8.
            let upper = (-(frames_per_second as i16)) as i8 as u8;
            (u16::from(upper) << 8) | u16::from(ticks_per_frame)
        }
    };
    out.extend_from_slice(&division_word.to_be_bytes());
    Ok(())
}

fn write_track_chunk(out: &mut Vec<u8>, track: &Track) -> Result<()> {
    // Validate end-of-track placement: must be the final event.
    let eot_count = track
        .events
        .iter()
        .filter(|ev| matches!(&ev.kind, Event::Meta(MetaEvent::EndOfTrack)))
        .count();
    match eot_count {
        0 => {
            return Err(Error::invalid(
                "SMF: track has no MetaEvent::EndOfTrack — append one before encode",
            ));
        }
        1 => {
            if !matches!(
                track.events.last().map(|ev| &ev.kind),
                Some(Event::Meta(MetaEvent::EndOfTrack)),
            ) {
                return Err(Error::invalid(
                    "SMF: MetaEvent::EndOfTrack must be the last event in the track",
                ));
            }
        }
        n => {
            return Err(Error::invalid(format!(
                "SMF: track has {n} MetaEvent::EndOfTrack entries — exactly 1 (at the tail) is legal",
            )));
        }
    }

    out.extend_from_slice(b"MTrk");
    let len_pos = out.len();
    out.extend_from_slice(&[0u8; 4]); // placeholder; patched after body emit
    let body_start = out.len();
    for (i, ev) in track.events.iter().enumerate() {
        write_event(out, ev)
            .map_err(|e| Error::invalid(format!("SMF: event {i}: {msg}", msg = err_msg(&e))))?;
    }
    let body_len = out.len() - body_start;
    let len_be = (body_len as u32).to_be_bytes();
    out[len_pos..len_pos + 4].copy_from_slice(&len_be);
    Ok(())
}

fn write_event(out: &mut Vec<u8>, ev: &TrackEvent) -> Result<()> {
    write_vlq(out, ev.delta)?;
    match &ev.kind {
        Event::Channel(msg) => write_channel(out, msg),
        Event::Sysex { escape, data } => {
            out.push(if *escape { 0xF7 } else { 0xF0 });
            write_vlq(out, u32_len(data.len(), "sysex payload")?)?;
            out.extend_from_slice(data);
            Ok(())
        }
        Event::Meta(meta) => write_meta(out, meta),
    }
}

fn write_channel(out: &mut Vec<u8>, msg: &ChannelMessage) -> Result<()> {
    if msg.channel > 0x0F {
        return Err(Error::invalid(format!(
            "SMF: channel {} out of range 0..=15",
            msg.channel,
        )));
    }
    let chan = msg.channel & 0x0F;
    match msg.body {
        ChannelBody::NoteOff { key, velocity } => {
            check_data_byte(key, "NoteOff.key")?;
            check_data_byte(velocity, "NoteOff.velocity")?;
            out.extend_from_slice(&[0x80 | chan, key, velocity]);
        }
        ChannelBody::NoteOn { key, velocity } => {
            check_data_byte(key, "NoteOn.key")?;
            check_data_byte(velocity, "NoteOn.velocity")?;
            out.extend_from_slice(&[0x90 | chan, key, velocity]);
        }
        ChannelBody::PolyAftertouch { key, pressure } => {
            check_data_byte(key, "PolyAftertouch.key")?;
            check_data_byte(pressure, "PolyAftertouch.pressure")?;
            out.extend_from_slice(&[0xA0 | chan, key, pressure]);
        }
        ChannelBody::ControlChange { controller, value } => {
            check_data_byte(controller, "ControlChange.controller")?;
            check_data_byte(value, "ControlChange.value")?;
            out.extend_from_slice(&[0xB0 | chan, controller, value]);
        }
        ChannelBody::ProgramChange { program } => {
            check_data_byte(program, "ProgramChange.program")?;
            out.extend_from_slice(&[0xC0 | chan, program]);
        }
        ChannelBody::ChannelAftertouch { pressure } => {
            check_data_byte(pressure, "ChannelAftertouch.pressure")?;
            out.extend_from_slice(&[0xD0 | chan, pressure]);
        }
        ChannelBody::PitchBend { value } => {
            if value > 0x3FFF {
                return Err(Error::invalid(format!(
                    "SMF: PitchBend value {value:#06X} exceeds 14-bit range 0..=0x3FFF",
                )));
            }
            let lsb = (value & 0x7F) as u8;
            let msb = ((value >> 7) & 0x7F) as u8;
            out.extend_from_slice(&[0xE0 | chan, lsb, msb]);
        }
    }
    Ok(())
}

fn write_meta(out: &mut Vec<u8>, meta: &MetaEvent) -> Result<()> {
    out.push(0xFF);
    match meta {
        MetaEvent::SequenceNumber(n) => {
            out.push(0x00);
            out.push(0x02);
            out.extend_from_slice(&n.to_be_bytes());
        }
        MetaEvent::Text { kind, text } => {
            if !(0x01..=0x0F).contains(kind) {
                return Err(Error::invalid(format!(
                    "SMF: Text.kind 0x{kind:02X} out of range 0x01..=0x0F",
                )));
            }
            out.push(*kind);
            write_vlq(out, u32_len(text.len(), "Text payload")?)?;
            out.extend_from_slice(text);
        }
        MetaEvent::ChannelPrefix(c) => {
            check_data_byte(*c, "ChannelPrefix")?;
            out.extend_from_slice(&[0x20, 0x01, *c]);
        }
        MetaEvent::Port(p) => {
            check_data_byte(*p, "Port")?;
            out.extend_from_slice(&[0x21, 0x01, *p]);
        }
        MetaEvent::EndOfTrack => {
            out.extend_from_slice(&[0x2F, 0x00]);
        }
        MetaEvent::Tempo(us_per_qn) => {
            if *us_per_qn > 0x00FF_FFFF {
                return Err(Error::invalid(format!(
                    "SMF: Tempo {us_per_qn} exceeds 24-bit range",
                )));
            }
            out.push(0x51);
            out.push(0x03);
            out.push(((us_per_qn >> 16) & 0xFF) as u8);
            out.push(((us_per_qn >> 8) & 0xFF) as u8);
            out.push((us_per_qn & 0xFF) as u8);
        }
        MetaEvent::SmpteOffset {
            hours,
            minutes,
            seconds,
            frames,
            subframes,
        } => {
            out.push(0x54);
            out.push(0x05);
            out.extend_from_slice(&[*hours, *minutes, *seconds, *frames, *subframes]);
        }
        MetaEvent::TimeSignature {
            numerator,
            denominator_pow2,
            clocks_per_click,
            notated_32nd_per_quarter,
        } => {
            out.push(0x58);
            out.push(0x04);
            out.extend_from_slice(&[
                *numerator,
                *denominator_pow2,
                *clocks_per_click,
                *notated_32nd_per_quarter,
            ]);
        }
        MetaEvent::KeySignature { sharps_flats, mode } => {
            if !matches!(mode, 0 | 1) {
                return Err(Error::invalid(format!(
                    "SMF: KeySignature.mode {mode} not in {{0, 1}}",
                )));
            }
            out.push(0x59);
            out.push(0x02);
            out.push(*sharps_flats as u8);
            out.push(*mode);
        }
        MetaEvent::SequencerSpecific(data) => {
            out.push(0x7F);
            write_vlq(out, u32_len(data.len(), "SequencerSpecific payload")?)?;
            out.extend_from_slice(data);
        }
        MetaEvent::Unknown { type_byte, data } => {
            if *type_byte == 0x2F {
                return Err(Error::invalid(
                    "SMF: cannot emit Unknown { type_byte: 0x2F } — use MetaEvent::EndOfTrack",
                ));
            }
            out.push(*type_byte);
            write_vlq(out, u32_len(data.len(), "Unknown meta payload")?)?;
            out.extend_from_slice(data);
        }
    }
    Ok(())
}

fn check_data_byte(b: u8, field: &str) -> Result<()> {
    if b & 0x80 != 0 {
        return Err(Error::invalid(format!(
            "SMF: {field} value 0x{b:02X} has the MIDI status bit set (must be 0..=0x7F)",
        )));
    }
    Ok(())
}

fn u32_len(n: usize, what: &str) -> Result<u32> {
    if n > MAX_VLQ_VALUE as usize {
        return Err(Error::invalid(format!(
            "SMF: {what} length {n} exceeds VLQ cap {MAX_VLQ_VALUE}",
        )));
    }
    Ok(n as u32)
}

/// Append the SMF variable-length quantity for `value` to `out`. Bounded
/// to [`MAX_VLQ_VALUE`] (the 4-byte cap from the SMF spec); returns
/// [`Error::InvalidData`] for any larger value.
fn write_vlq(out: &mut Vec<u8>, value: u32) -> Result<()> {
    if value > MAX_VLQ_VALUE {
        return Err(Error::invalid(format!(
            "SMF: VLQ value {value:#X} exceeds 4-byte cap {MAX_VLQ_VALUE:#X}",
        )));
    }
    // Collect 7-bit groups LSB-first, then emit MSB-first with the
    // continuation bit set on every byte but the last.
    let mut groups: [u8; 4] = [0; 4];
    let mut n = 0usize;
    let mut v = value;
    loop {
        groups[n] = (v & 0x7F) as u8;
        n += 1;
        v >>= 7;
        if v == 0 {
            break;
        }
    }
    for i in (0..n).rev() {
        let last = i == 0;
        let byte = groups[i] | if last { 0x00 } else { 0x80 };
        out.push(byte);
    }
    Ok(())
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

    // ───────── SmpteOffsetEvent / SmfFile::smpte_offsets (FF 54) ─────────

    #[test]
    fn smpte_offsets_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.smpte_offsets().is_empty());
    }

    #[test]
    fn smpte_offsets_single_event_at_tick_zero_24fps() {
        // delta=0 FF 54 05 hr mn se fr ff
        // hr = 0x01 → yy=0 (24 fps), hours=1
        let events: Vec<u8> = vec![
            0x00, 0xFF, 0x54, 0x05, 0x01, 0x20, 0x10, 0x05, 0x32, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let so = smf.smpte_offsets();
        assert_eq!(so.len(), 1);
        assert_eq!(so[0].tick, 0);
        assert_eq!(so[0].track, 0);
        assert_eq!(so[0].hours_raw, 0x01);
        assert_eq!(so[0].hours_count(), 1);
        assert_eq!(so[0].frame_rate(), FrameRate::Fps24);
        assert_eq!(so[0].minutes, 0x20);
        assert_eq!(so[0].seconds, 0x10);
        assert_eq!(so[0].frames, 0x05);
        assert_eq!(so[0].subframes, 0x32);
    }

    #[test]
    fn smpte_offsets_decodes_all_four_frame_rates() {
        // Build four tracks, each carrying one FF 54 with a different
        // yy bit pattern in bits 5-6 of the hr byte. Hours stays 1 so
        // hours_count() is rate-independent.
        let mk_track = |hr: u8| -> Vec<u8> {
            vec![
                0x00, 0xFF, 0x54, 0x05, hr, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x2F, 0x00,
            ]
        };
        let mut blob = header_chunk(2, 4, 96);
        blob.extend(track_chunk(&mk_track(0b0000_0001))); // yy=0 → 24 fps, hr=1
        blob.extend(track_chunk(&mk_track(0b0010_0001))); // yy=1 → 25 fps, hr=1
        blob.extend(track_chunk(&mk_track(0b0100_0001))); // yy=2 → 30 drop, hr=1
        blob.extend(track_chunk(&mk_track(0b0110_0001))); // yy=3 → 30 non-drop, hr=1
        let smf = parse(&blob).unwrap();
        let so = smf.smpte_offsets();
        assert_eq!(so.len(), 4);
        // All four pinned to tick 0 → stable-sort preserves track
        // order: track 0 first, then 1, 2, 3.
        assert_eq!(so[0].track, 0);
        assert_eq!(so[0].frame_rate(), FrameRate::Fps24);
        assert_eq!(so[0].hours_count(), 1);
        assert_eq!(so[1].track, 1);
        assert_eq!(so[1].frame_rate(), FrameRate::Fps25);
        assert_eq!(so[1].hours_count(), 1);
        assert_eq!(so[2].track, 2);
        assert_eq!(so[2].frame_rate(), FrameRate::Fps30DropFrame);
        assert!(so[2].frame_rate().is_drop_frame());
        assert_eq!(so[2].hours_count(), 1);
        assert_eq!(so[3].track, 3);
        assert_eq!(so[3].frame_rate(), FrameRate::Fps30NonDrop);
        assert!(!so[3].frame_rate().is_drop_frame());
        assert_eq!(so[3].hours_count(), 1);
        // frames_per_second() reports the nominal rate.
        assert_eq!(so[0].frame_rate().frames_per_second(), 24);
        assert_eq!(so[1].frame_rate().frames_per_second(), 25);
        assert_eq!(so[2].frame_rate().frames_per_second(), 30);
        assert_eq!(so[3].frame_rate().frames_per_second(), 30);
    }

    #[test]
    fn smpte_offsets_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 FF 54 05 ...
        // track 1: delta=120 FF 54 05 ...
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x54, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x54, 0x05, 0x02, 0x00, 0x00, 0x00, 0x00]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let so = smf.smpte_offsets();
        assert_eq!(so.len(), 2);
        assert_eq!(so[0].tick, 120);
        assert_eq!(so[0].track, 1);
        assert_eq!(so[0].hours_raw, 0x02);
        assert_eq!(so[1].tick, 240);
        assert_eq!(so[1].track, 0);
        assert_eq!(so[1].hours_raw, 0x01);
    }

    #[test]
    fn smpte_offsets_stable_sort_keeps_track0_before_track1_at_same_tick() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x54, 0x05, 0x0A, 0x00, 0x00, 0x00, 0x00]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x54, 0x05, 0x0B, 0x00, 0x00, 0x00, 0x00]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let so = smf.smpte_offsets();
        assert_eq!(so.len(), 2);
        assert_eq!(so[0].track, 0);
        assert_eq!(so[0].hours_raw, 0x0A);
        assert_eq!(so[1].track, 1);
        assert_eq!(so[1].hours_raw, 0x0B);
    }

    #[test]
    fn smpte_offsets_filter_excludes_other_meta_kinds() {
        // FF 54 picked up; other meta events filtered out.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // tempo
        events.extend_from_slice(&[0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]); // time sig
        events.extend_from_slice(&[0x00, 0xFF, 0x59, 0x02, 0x00, 0x00]); // key sig
        events.extend_from_slice(&[0x00, 0xFF, 0x54, 0x05, 0x21, 0x1E, 0x2D, 0x10, 0x32]); // SMPTE: yy=1, hr=1
        events.extend_from_slice(&[0x00, 0xFF, 0x01, 0x04]);
        events.extend_from_slice(b"note");
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let so = smf.smpte_offsets();
        assert_eq!(so.len(), 1);
        assert_eq!(so[0].hours_raw, 0x21);
        assert_eq!(so[0].hours_count(), 1);
        assert_eq!(so[0].frame_rate(), FrameRate::Fps25);
        assert_eq!(so[0].minutes, 0x1E);
        assert_eq!(so[0].seconds, 0x2D);
        assert_eq!(so[0].frames, 0x10);
        assert_eq!(so[0].subframes, 0x32);
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.tempo_map().len(), 1);
        assert_eq!(smf.time_signatures().len(), 1);
        assert_eq!(smf.key_signatures().len(), 1);
        assert_eq!(smf.texts().len(), 1);
    }

    #[test]
    fn smpte_offsets_seconds_total_24fps() {
        // hr=0x01 (yy=0, hours=1), 30 min, 15 sec, 12 frames, 50 sub
        // → 3600 + 1800 + 15 + (12 + 0.50)/24 = 5415 + 12.5/24 = 5415.5208333…
        let ev = SmpteOffsetEvent {
            tick: 0,
            track: 0,
            hours_raw: 0x01,
            minutes: 30,
            seconds: 15,
            frames: 12,
            subframes: 50,
        };
        assert_eq!(ev.frame_rate(), FrameRate::Fps24);
        let expected = 1.0 * 3600.0 + 30.0 * 60.0 + 15.0 + 12.5 / 24.0;
        assert!((ev.seconds_total() - expected).abs() < 1e-9);
    }

    #[test]
    fn smpte_offsets_seconds_total_30fps_non_drop_at_origin() {
        // hr=0x60 (yy=3 non-drop, hours=0), all-zero rest → 0.0 s.
        let ev = SmpteOffsetEvent {
            tick: 0,
            track: 0,
            hours_raw: 0x60,
            minutes: 0,
            seconds: 0,
            frames: 0,
            subframes: 0,
        };
        assert_eq!(ev.frame_rate(), FrameRate::Fps30NonDrop);
        assert_eq!(ev.hours_count(), 0);
        assert_eq!(ev.seconds_total(), 0.0);
    }

    #[test]
    fn smpte_offsets_after_channel_events_tracks_absolute_tick() {
        // Running-status note-ons followed by a late-positioned FF 54.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x54, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let so = smf.smpte_offsets();
        assert_eq!(so.len(), 1);
        assert_eq!(so[0].tick, 240);
        assert_eq!(so[0].frame_rate(), FrameRate::Fps24);
        assert_eq!(so[0].hours_count(), 0);
    }

    #[test]
    fn frame_rate_from_hours_byte_only_uses_bits_5_and_6() {
        // Bit 7 is reserved per RP-004/008; bits 0-4 are hours. Only
        // bits 5-6 contribute to the rate decode.
        assert_eq!(FrameRate::from_hours_byte(0b0000_0000), FrameRate::Fps24);
        assert_eq!(FrameRate::from_hours_byte(0b1001_1111), FrameRate::Fps24); // bit 7 set + hours=31
        assert_eq!(FrameRate::from_hours_byte(0b0010_0000), FrameRate::Fps25);
        assert_eq!(
            FrameRate::from_hours_byte(0b0100_0000),
            FrameRate::Fps30DropFrame
        );
        assert_eq!(
            FrameRate::from_hours_byte(0b0110_0000),
            FrameRate::Fps30NonDrop
        );
    }

    // ───────── SequencerSpecificEvent / SmfFile::sequencer_specifics (FF 7F) ─────────

    #[test]
    fn sequencer_specifics_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.sequencer_specifics().is_empty());
    }

    #[test]
    fn sequencer_specifics_single_event_at_tick_zero() {
        // delta=0 FF 7F 04 41 00 01 02   (Roland-style: ID 0x41 + 3 bytes)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![
            0x00, 0xFF, 0x7F, 0x04, 0x41, 0x00, 0x01, 0x02, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sequencer_specifics();
        assert_eq!(sx.len(), 1);
        assert_eq!(sx[0].tick, 0);
        assert_eq!(sx[0].track, 0);
        assert_eq!(sx[0].data_bytes(), &[0x41, 0x00, 0x01, 0x02]);
    }

    #[test]
    fn sequencer_specifics_empty_payload_is_surfaced() {
        // FF 7F 00 — zero-length blob is legal per spec.
        let events: Vec<u8> = vec![0x00, 0xFF, 0x7F, 0x00, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sequencer_specifics();
        assert_eq!(sx.len(), 1);
        assert!(sx[0].data_bytes().is_empty());
    }

    #[test]
    fn sequencer_specifics_multiple_within_one_track_are_in_order() {
        // delta=0   FF 7F 02 41 10
        // delta=480 FF 7F 02 41 11
        // delta=0   FF 2F 00
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x7F, 0x02, 0x41, 0x10]);
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x7F, 0x02, 0x41, 0x11]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sequencer_specifics();
        assert_eq!(sx.len(), 2);
        assert_eq!(sx[0].tick, 0);
        assert_eq!(sx[0].data_bytes(), &[0x41, 0x10]);
        assert_eq!(sx[1].tick, 480);
        assert_eq!(sx[1].data_bytes(), &[0x41, 0x11]);
    }

    #[test]
    fn sequencer_specifics_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 FF 7F 01 AA
        // track 1: delta=120 FF 7F 01 BB; delta=240 FF 7F 01 CC
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x7F, 0x01, 0xAA]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x7F, 0x01, 0xBB]);
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x7F, 0x01, 0xCC]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sx = smf.sequencer_specifics();
        assert_eq!(sx.len(), 3);
        assert_eq!(sx[0].tick, 120);
        assert_eq!(sx[0].track, 1);
        assert_eq!(sx[0].data_bytes(), &[0xBB]);
        assert_eq!(sx[1].tick, 240);
        assert_eq!(sx[1].track, 0);
        assert_eq!(sx[1].data_bytes(), &[0xAA]);
        assert_eq!(sx[2].tick, 360);
        assert_eq!(sx[2].track, 1);
        assert_eq!(sx[2].data_bytes(), &[0xCC]);
    }

    #[test]
    fn sequencer_specifics_stable_sort_keeps_track0_before_track1_at_same_tick() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x7F, 0x01, 0x00]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x7F, 0x01, 0x01]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sx = smf.sequencer_specifics();
        assert_eq!(sx.len(), 2);
        assert_eq!(sx[0].track, 0);
        assert_eq!(sx[0].data_bytes(), &[0x00]);
        assert_eq!(sx[1].track, 1);
        assert_eq!(sx[1].data_bytes(), &[0x01]);
    }

    #[test]
    fn sequencer_specifics_filter_excludes_other_meta_kinds() {
        // FF 7F payload picked up; text + smpte + tempo + key/time
        // signatures all filtered out.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x7F, 0x03, 0x41, 0x10, 0x42];
        events.extend_from_slice(&[0x00, 0xFF, 0x01, 0x04]);
        events.extend_from_slice(b"Note");
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        events.extend_from_slice(&[0x00, 0xFF, 0x54, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        events.extend_from_slice(&[0x00, 0xFF, 0x59, 0x02, 0x00, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sequencer_specifics();
        assert_eq!(sx.len(), 1);
        assert_eq!(sx[0].data_bytes(), &[0x41, 0x10, 0x42]);
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.texts().len(), 1);
        assert_eq!(smf.track_names().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
        assert_eq!(smf.tempo_map().len(), 1);
        assert_eq!(smf.smpte_offsets().len(), 1);
        assert_eq!(smf.time_signatures().len(), 1);
        assert_eq!(smf.key_signatures().len(), 1);
    }

    #[test]
    fn sequencer_specifics_after_channel_events_track_absolute_tick() {
        // Running-status note-ons followed by a late-positioned FF 7F.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x40, 0x50]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0xFF, 0x7F, 0x02, 0x7D, 0x01]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sequencer_specifics();
        assert_eq!(sx.len(), 1);
        assert_eq!(sx[0].tick, 240);
        assert_eq!(sx[0].data_bytes(), &[0x7D, 0x01]);
    }

    // ───────── SequenceNumberEvent / SmfFile::sequence_numbers (FF 00) ─────────

    #[test]
    fn sequence_numbers_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.sequence_numbers().is_empty());
    }

    #[test]
    fn sequence_numbers_single_label_at_tick_zero() {
        // delta=0 FF 00 02 00 2A   (sequence ID = 42)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![0x00, 0xFF, 0x00, 0x02, 0x00, 0x2A, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sn = smf.sequence_numbers();
        assert_eq!(sn.len(), 1);
        assert_eq!(sn[0].tick, 0);
        assert_eq!(sn[0].track, 0);
        assert_eq!(sn[0].number(), 42);
    }

    #[test]
    fn sequence_numbers_big_endian_decode() {
        // FF 00 02 12 34 → 0x1234
        let events: Vec<u8> = vec![0x00, 0xFF, 0x00, 0x02, 0x12, 0x34, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sn = smf.sequence_numbers();
        assert_eq!(sn.len(), 1);
        assert_eq!(sn[0].number, 0x1234);
    }

    #[test]
    fn sequence_numbers_full_u16_range_round_trips() {
        // FF 00 02 FF FF → 0xFFFF
        let events: Vec<u8> = vec![0x00, 0xFF, 0x00, 0x02, 0xFF, 0xFF, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sn = smf.sequence_numbers();
        assert_eq!(sn.len(), 1);
        assert_eq!(sn[0].number, 0xFFFF);
    }

    #[test]
    fn sequence_numbers_format_2_per_pattern_labels() {
        // Format 2: one sequence number per track, each labelling its
        // pattern. The helper surfaces both in track order at tick 0.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x00, 0x02, 0x00, 0x01]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xFF, 0x00, 0x02, 0x00, 0x02]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(2, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sn = smf.sequence_numbers();
        assert_eq!(sn.len(), 2);
        assert_eq!(sn[0].tick, 0);
        assert_eq!(sn[0].track, 0);
        assert_eq!(sn[0].number, 1);
        assert_eq!(sn[1].tick, 0);
        assert_eq!(sn[1].track, 1);
        assert_eq!(sn[1].number, 2);
    }

    #[test]
    fn sequence_numbers_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two format-1 tracks both labelled at tick 0 — the stable
        // sort keeps track 0 before track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x00, 0x02, 0x00, 0x10]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xFF, 0x00, 0x02, 0x00, 0x11]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sn = smf.sequence_numbers();
        assert_eq!(sn.len(), 2);
        assert_eq!(sn[0].track, 0);
        assert_eq!(sn[0].number, 0x10);
        assert_eq!(sn[1].track, 1);
        assert_eq!(sn[1].number, 0x11);
    }

    #[test]
    fn sequence_numbers_late_position_tracks_absolute_tick() {
        // FF 00 is conventionally placed at delta-time zero, but the
        // helper surfaces every occurrence in time order so files
        // that place it later in a track still round-trip the label.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x00, 0x02, 0x07, 0xD0]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sn = smf.sequence_numbers();
        assert_eq!(sn.len(), 1);
        assert_eq!(sn[0].tick, 240);
        assert_eq!(sn[0].number, 2000);
    }

    #[test]
    fn sequence_numbers_filter_excludes_other_meta_kinds() {
        // FF 00 02 ssss picked up; text + smpte + tempo + key/time
        // signatures + sequencer-private all filtered out.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x00, 0x02, 0x00, 0x05];
        events.extend_from_slice(&[0x00, 0xFF, 0x01, 0x04]);
        events.extend_from_slice(b"Note");
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        events.extend_from_slice(&[0x00, 0xFF, 0x54, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        events.extend_from_slice(&[0x00, 0xFF, 0x59, 0x02, 0x00, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x7F, 0x02, 0x41, 0x10]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sn = smf.sequence_numbers();
        assert_eq!(sn.len(), 1);
        assert_eq!(sn[0].number, 5);
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.texts().len(), 1);
        assert_eq!(smf.track_names().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
        assert_eq!(smf.tempo_map().len(), 1);
        assert_eq!(smf.smpte_offsets().len(), 1);
        assert_eq!(smf.time_signatures().len(), 1);
        assert_eq!(smf.key_signatures().len(), 1);
        assert_eq!(smf.sequencer_specifics().len(), 1);
    }

    // ───────── MidiPortEvent / SmfFile::midi_ports (FF 21) ─────────

    #[test]
    fn midi_ports_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.midi_ports().is_empty());
    }

    #[test]
    fn midi_ports_single_hint_at_tick_zero() {
        // delta=0 FF 21 01 02   (route this track through port 2)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![0x00, 0xFF, 0x21, 0x01, 0x02, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ports = smf.midi_ports();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].tick, 0);
        assert_eq!(ports[0].track, 0);
        assert_eq!(ports[0].port(), 2);
    }

    #[test]
    fn midi_ports_full_seven_bit_range_round_trips() {
        // FF 21 01 7F → port 127 (max legal value, the high bit is
        // reserved for status / running-status use)
        let events: Vec<u8> = vec![0x00, 0xFF, 0x21, 0x01, 0x7F, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ports = smf.midi_ports();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, 0x7F);
    }

    #[test]
    fn midi_ports_per_track_routing_in_format_1() {
        // Multi-port format-1 file: each track names its own output
        // port at delta zero so the back-end can dispatch the
        // surrounding channel-voice stream through 16 × N channels.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x21, 0x01, 0x00]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xFF, 0x21, 0x01, 0x01]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t2: Vec<u8> = Vec::new();
        t2.extend_from_slice(&[0x00, 0xFF, 0x21, 0x01, 0x02]);
        t2.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 3, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        blob.extend(track_chunk(&t2));
        let smf = parse(&blob).unwrap();
        let ports = smf.midi_ports();
        assert_eq!(ports.len(), 3);
        assert_eq!(ports[0].track, 0);
        assert_eq!(ports[0].port, 0);
        assert_eq!(ports[1].track, 1);
        assert_eq!(ports[1].port, 1);
        assert_eq!(ports[2].track, 2);
        assert_eq!(ports[2].port, 2);
    }

    #[test]
    fn midi_ports_stable_sort_keeps_track0_before_track1_at_same_tick() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x21, 0x01, 0x05]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x21, 0x01, 0x06]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let ports = smf.midi_ports();
        assert_eq!(ports.len(), 2);
        assert_eq!(ports[0].track, 0);
        assert_eq!(ports[0].port, 5);
        assert_eq!(ports[1].track, 1);
        assert_eq!(ports[1].port, 6);
    }

    #[test]
    fn midi_ports_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 FF 21 01 0A
        // track 1: delta=120 FF 21 01 0B; delta=240 FF 21 01 0C
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x21, 0x01, 0x0A]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x21, 0x01, 0x0B]);
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x21, 0x01, 0x0C]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let ports = smf.midi_ports();
        assert_eq!(ports.len(), 3);
        assert_eq!(ports[0].tick, 120);
        assert_eq!(ports[0].track, 1);
        assert_eq!(ports[0].port, 0x0B);
        assert_eq!(ports[1].tick, 240);
        assert_eq!(ports[1].track, 0);
        assert_eq!(ports[1].port, 0x0A);
        assert_eq!(ports[2].tick, 360);
        assert_eq!(ports[2].track, 1);
        assert_eq!(ports[2].port, 0x0C);
    }

    #[test]
    fn midi_ports_late_position_tracks_absolute_tick() {
        // The convention places FF 21 at delta-time zero, but the
        // helper surfaces every occurrence in time order so files
        // that re-route mid-track still round-trip the hint.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xFF, 0x21, 0x01, 0x03]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ports = smf.midi_ports();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].tick, 240);
        assert_eq!(ports[0].port, 3);
    }

    #[test]
    fn midi_ports_filter_excludes_channel_prefix_and_other_meta_kinds() {
        // FF 21 picked up; FF 20 channel-prefix sibling stays
        // filtered out (different meta kind, different routing
        // semantics); text + smpte + tempo + key/time signatures +
        // sequencer-private + sequence-number all filtered out.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x21, 0x01, 0x04];
        events.extend_from_slice(&[0x00, 0xFF, 0x20, 0x01, 0x05]);
        events.extend_from_slice(&[0x00, 0xFF, 0x00, 0x02, 0x00, 0x42]);
        events.extend_from_slice(&[0x00, 0xFF, 0x01, 0x04]);
        events.extend_from_slice(b"Note");
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x04]);
        events.extend_from_slice(b"Lead");
        events.extend_from_slice(&[0x00, 0xFF, 0x05, 0x02]);
        events.extend_from_slice(b"la");
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        events.extend_from_slice(&[0x00, 0xFF, 0x54, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        events.extend_from_slice(&[0x00, 0xFF, 0x59, 0x02, 0x00, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x7F, 0x02, 0x41, 0x10]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ports = smf.midi_ports();
        assert_eq!(ports.len(), 1);
        assert_eq!(ports[0].port, 4);
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.texts().len(), 1);
        assert_eq!(smf.track_names().len(), 1);
        assert_eq!(smf.lyrics().len(), 1);
        assert_eq!(smf.tempo_map().len(), 1);
        assert_eq!(smf.smpte_offsets().len(), 1);
        assert_eq!(smf.time_signatures().len(), 1);
        assert_eq!(smf.key_signatures().len(), 1);
        assert_eq!(smf.sequencer_specifics().len(), 1);
        assert_eq!(smf.sequence_numbers().len(), 1);
    }

    // ───────── SmfChannelSnapshot / SmfFile::channel_snapshot_at ─────────

    #[test]
    fn channel_snapshot_default_when_no_events() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert_eq!(snap, SmfChannelSnapshot::default());
        // Spec-recommended defaults survive verbatim.
        assert_eq!(snap.program, None);
        assert_eq!(snap.bank_msb, None);
        assert_eq!(snap.bank_lsb, None);
        assert_eq!(snap.volume, 100);
        assert_eq!(snap.pan, 64);
        assert_eq!(snap.expression, 127);
        assert_eq!(snap.modulation, 0);
        assert!(!snap.sustain);
        assert_eq!(snap.pitch_bend, 0x2000);
    }

    #[test]
    fn channel_snapshot_program_change_at_tick_zero() {
        // delta=0 C0 05 (channel 0, program 5)
        let events: Vec<u8> = vec![0x00, 0xC0, 0x05, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert_eq!(snap.program, Some(5));
        // Other channels stay default.
        let other = smf.channel_snapshot_at(1, 0);
        assert_eq!(other.program, None);
    }

    #[test]
    fn channel_snapshot_cc_volume_pan_expression_mod() {
        // CC 7 = 80 (volume), CC 10 = 0 (pan hard left),
        // CC 11 = 100 (expression), CC 1 = 50 (modwheel)
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x07, 0x50, // CC 7 = 80
            0x00, 0xB0, 0x0A, 0x00, // CC 10 = 0
            0x00, 0xB0, 0x0B, 0x64, // CC 11 = 100
            0x00, 0xB0, 0x01, 0x32, // CC 1 = 50
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert_eq!(snap.volume, 80);
        assert_eq!(snap.pan, 0);
        assert_eq!(snap.expression, 100);
        assert_eq!(snap.modulation, 50);
    }

    #[test]
    fn channel_snapshot_sustain_pedal_threshold_64() {
        // CC 64 with value 63 → off; CC 64 with value 64 → on.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x40, 0x3F, // CC 64 = 63 → off
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert!(!snap.sustain);

        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x40, 0x40, // CC 64 = 64 → on
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert!(snap.sustain);
    }

    #[test]
    fn channel_snapshot_bank_select_msb_and_lsb_independent() {
        // CC 0 = 7 (bank MSB), CC 32 = 3 (bank LSB)
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x00, 0x07, // CC 0 = 7
            0x00, 0xB0, 0x20, 0x03, // CC 32 = 3
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert_eq!(snap.bank_msb, Some(7));
        assert_eq!(snap.bank_lsb, Some(3));
    }

    #[test]
    fn channel_snapshot_pitch_bend_14bit_decoded() {
        // En lsb msb. Bend = (msb << 7) | lsb. Use lsb=0x40 msb=0x20 →
        // value = 0x1040 = 4160.
        let events: Vec<u8> = vec![
            0x00, 0xE0, 0x40, 0x20, // pitch bend lsb=0x40 msb=0x20
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert_eq!(snap.pitch_bend, 0x1040);
        // Centre default on a channel that never bent.
        let other = smf.channel_snapshot_at(1, 0);
        assert_eq!(other.pitch_bend, 0x2000);
    }

    #[test]
    fn channel_snapshot_tick_filter_includes_events_at_tick_exactly() {
        // CC 7 = 50 at tick 100; CC 7 = 90 at tick 200.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xB0, 0x07, 0x32]); // CC 7 = 50 at tick 100
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xB0, 0x07, 0x5A]); // CC 7 = 90 at tick 200
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        // At tick 0 we haven't seen either CC yet.
        assert_eq!(smf.channel_snapshot_at(0, 0).volume, 100);
        // At tick 99 still nothing.
        assert_eq!(smf.channel_snapshot_at(0, 99).volume, 100);
        // At tick 100 the first CC fires.
        assert_eq!(smf.channel_snapshot_at(0, 100).volume, 50);
        // At tick 199 still the first CC.
        assert_eq!(smf.channel_snapshot_at(0, 199).volume, 50);
        // At tick 200 the second CC fires.
        assert_eq!(smf.channel_snapshot_at(0, 200).volume, 90);
        // At tick 99999 same as tick 200.
        assert_eq!(smf.channel_snapshot_at(0, 99_999).volume, 90);
    }

    #[test]
    fn channel_snapshot_running_status_program_changes_replayed() {
        // Three successive Program Change events (running status: C0
        // sticks across the trio). The snapshot at the end should
        // reflect the last program.
        let events: Vec<u8> = vec![
            0x00, 0xC0, 0x01, // PC 1
            0x00, 0x02, // running status PC 2
            0x00, 0x03, // running status PC 3
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert_eq!(snap.program, Some(3));
    }

    #[test]
    fn channel_snapshot_multi_channel_independence() {
        // Channel 0 → program 10. Channel 5 → program 50.
        let events: Vec<u8> = vec![
            0x00, 0xC0, 0x0A, // C0 0A
            0x00, 0xC5, 0x32, // C5 50
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap0 = smf.channel_snapshot_at(0, 0);
        let snap5 = smf.channel_snapshot_at(5, 0);
        assert_eq!(snap0.program, Some(10));
        assert_eq!(snap5.program, Some(50));
        assert_eq!(smf.channel_snapshot_at(1, 0).program, None);
    }

    #[test]
    fn channel_snapshot_notes_and_aftertouch_do_not_alter_state() {
        // Note-on / off / poly + channel aftertouch are pure voice
        // events. None of them should modify the snapshot.
        let events: Vec<u8> = vec![
            0x00, 0x90, 0x3C, 0x64, // Note on
            0x00, 0x80, 0x3C, 0x00, // Note off
            0x00, 0xA0, 0x3C, 0x40, // Poly aftertouch
            0x00, 0xD0, 0x55, // Channel aftertouch
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 999);
        assert_eq!(snap, SmfChannelSnapshot::default());
    }

    #[test]
    fn channel_snapshot_unknown_ccs_ignored() {
        // CC 99 + CC 100 are part of the NRPN / RPN-selector
        // machinery. We don't track them on the snapshot yet — they
        // should be ignored, leaving the rest of the snapshot at
        // defaults.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x63, 0x10, // CC 99 = 16
            0x00, 0xB0, 0x64, 0x20, // CC 100 = 32
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 999);
        assert_eq!(snap, SmfChannelSnapshot::default());
    }

    #[test]
    fn channel_snapshot_invalid_channel_returns_default() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        // Channels >= 16 are not legal MIDI channels and the helper
        // returns the default snapshot rather than indexing-out.
        assert_eq!(
            smf.channel_snapshot_at(16, 0),
            SmfChannelSnapshot::default()
        );
        assert_eq!(
            smf.channel_snapshot_at(255, 0),
            SmfChannelSnapshot::default()
        );
    }

    #[test]
    fn channel_snapshots_at_returns_sixteen_independent_states() {
        // Program change on every channel: channel N → program 2*N + 1.
        // This validates that the bulk method walks every channel in
        // one pass and yields a per-channel array indexed by channel.
        let mut events: Vec<u8> = Vec::new();
        for ch in 0u8..16 {
            let status = 0xC0 | ch;
            let program = 2 * ch + 1;
            events.extend_from_slice(&[0x00, status, program]);
        }
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let snaps = smf.channel_snapshots_at(0);
        for (ch, snap) in snaps.iter().enumerate() {
            assert_eq!(
                snap.program,
                Some((2 * ch + 1) as u8),
                "channel {ch} expected program {}",
                2 * ch + 1
            );
        }
        // Per-channel call must agree with the bulk call.
        for ch in 0u8..16 {
            assert_eq!(snaps[ch as usize], smf.channel_snapshot_at(ch, 0));
        }
    }

    #[test]
    fn channel_snapshots_at_merge_across_tracks_track0_wins_at_same_tick() {
        // Both tracks emit CC 7 on channel 0 at tick 0. Track 0
        // writes 30; track 1 writes 70. Per the stable-merge rule
        // (track 0 before track 1 at the same tick), track 1 fires
        // last → snapshot reflects 70 (last-writer-wins is the wire
        // semantics: the receiver overwrites volume each time).
        let t0: Vec<u8> = vec![0x00, 0xB0, 0x07, 0x1E, 0x00, 0xFF, 0x2F, 0x00];
        let t1: Vec<u8> = vec![0x00, 0xB0, 0x07, 0x46, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let snap = smf.channel_snapshot_at(0, 0);
        assert_eq!(snap.volume, 70);
    }

    #[test]
    fn channel_snapshot_apply_is_public_reusable() {
        // The fold operation is exposed so callers running custom
        // replay can reuse the same wire semantics.
        let mut snap = SmfChannelSnapshot::default();
        snap.apply(&ChannelBody::ProgramChange { program: 42 });
        snap.apply(&ChannelBody::ControlChange {
            controller: 7,
            value: 10,
        });
        snap.apply(&ChannelBody::ControlChange {
            controller: 64,
            value: 80,
        });
        snap.apply(&ChannelBody::PitchBend { value: 12345 });
        assert_eq!(snap.program, Some(42));
        assert_eq!(snap.volume, 10);
        assert!(snap.sustain);
        assert_eq!(snap.pitch_bend, 12345);
        // Note-on doesn't alter channel state.
        let before = snap;
        snap.apply(&ChannelBody::NoteOn {
            key: 60,
            velocity: 100,
        });
        assert_eq!(snap, before);
    }

    // ─────────────────── writer tests ───────────────────

    #[test]
    fn write_vlq_matches_test_helper_across_spec_examples() {
        // The SMF spec lists worked VLQ examples (see `vlq_multi_byte`
        // above); the writer must produce the same byte sequences.
        let cases: &[(u32, &[u8])] = &[
            (0, &[0x00]),
            (0x40, &[0x40]),
            (0x7F, &[0x7F]),
            (0x80, &[0x81, 0x00]),
            (0x2000, &[0xC0, 0x00]),
            (0x3FFF, &[0xFF, 0x7F]),
            (0x10_0000, &[0xC0, 0x80, 0x00]),
            (0x1F_FFFF, &[0xFF, 0xFF, 0x7F]),
            (0x20_0000, &[0x81, 0x80, 0x80, 0x00]),
            (0x0FFF_FFFF, &[0xFF, 0xFF, 0xFF, 0x7F]),
        ];
        for (v, bytes) in cases {
            let mut buf = Vec::new();
            write_vlq(&mut buf, *v).unwrap();
            assert_eq!(buf, bytes.to_vec(), "encode VLQ {v:#x}");
            // Round-trip back through read_vlq.
            let mut c = Cursor::new(&buf);
            assert_eq!(read_vlq(&mut c).unwrap(), *v);
        }
    }

    #[test]
    fn write_vlq_rejects_over_cap() {
        let mut buf = Vec::new();
        let err = write_vlq(&mut buf, MAX_VLQ_VALUE + 1).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn header_round_trip_format_0_tpqn() {
        let header = SmfHeader {
            format: SmfFormat::SingleTrack,
            ntrks: 1,
            division: Division::TicksPerQuarter(480),
        };
        let track = Track {
            events: vec![TrackEvent {
                delta: 0,
                kind: Event::Meta(MetaEvent::EndOfTrack),
            }],
        };
        let smf = SmfFile {
            header,
            tracks: vec![track],
        };
        let bytes = smf.to_bytes().unwrap();
        // Header is exactly 14 bytes; the EOT track is `MTrk` + 4-byte
        // length + 4-byte body (`00 FF 2F 00`) = 12 bytes.
        assert_eq!(bytes.len(), 14 + 12);
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn header_round_trip_smpte_minus_25_fps() {
        let header = SmfHeader {
            format: SmfFormat::MultiTrackSimultaneous,
            ntrks: 1,
            division: Division::Smpte {
                frames_per_second: 25,
                ticks_per_frame: 40,
            },
        };
        let track = Track {
            events: vec![TrackEvent {
                delta: 0,
                kind: Event::Meta(MetaEvent::EndOfTrack),
            }],
        };
        let smf = SmfFile {
            header,
            tracks: vec![track],
        };
        let bytes = smf.to_bytes().unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn round_trip_type_0_full_track() {
        // The fixture from `type_0_single_track_with_note_pair_and_tempo`,
        // but driven through `to_bytes` -> `parse` instead of a hand-built
        // byte blob.
        let track = Track {
            events: vec![
                TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::Tempo(500_000)),
                },
                TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::TimeSignature {
                        numerator: 4,
                        denominator_pow2: 2,
                        clocks_per_click: 24,
                        notated_32nd_per_quarter: 8,
                    }),
                },
                TrackEvent {
                    delta: 0,
                    kind: Event::Channel(ChannelMessage {
                        channel: 0,
                        body: ChannelBody::NoteOn {
                            key: 60,
                            velocity: 100,
                        },
                    }),
                },
                TrackEvent {
                    delta: 480,
                    kind: Event::Channel(ChannelMessage {
                        channel: 0,
                        body: ChannelBody::NoteOff {
                            key: 60,
                            velocity: 0x40,
                        },
                    }),
                },
                TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::EndOfTrack),
                },
            ],
        };
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(480),
            },
            tracks: vec![track],
        };
        let bytes = smf.to_bytes().unwrap();

        // First 14 bytes must be a well-formed MThd / format 0 / ntrks 1
        // / division 480.
        assert_eq!(&bytes[..4], b"MThd");
        assert_eq!(&bytes[4..8], &6u32.to_be_bytes());
        assert_eq!(&bytes[8..10], &0u16.to_be_bytes());
        assert_eq!(&bytes[10..12], &1u16.to_be_bytes());
        assert_eq!(&bytes[12..14], &480u16.to_be_bytes());

        // Re-parse and compare.
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn round_trip_all_meta_event_kinds() {
        // Cover every concrete meta variant the writer special-cases,
        // plus an `Unknown` passthrough.
        let metas = vec![
            MetaEvent::SequenceNumber(0xBEEF),
            MetaEvent::Text {
                kind: 0x01,
                text: b"hello".to_vec(),
            },
            MetaEvent::Text {
                kind: 0x05,
                text: b"la".to_vec(),
            },
            MetaEvent::ChannelPrefix(3),
            MetaEvent::Port(7),
            MetaEvent::Tempo(500_000),
            MetaEvent::SmpteOffset {
                hours: 0x21,
                minutes: 12,
                seconds: 34,
                frames: 5,
                subframes: 50,
            },
            MetaEvent::TimeSignature {
                numerator: 6,
                denominator_pow2: 3,
                clocks_per_click: 36,
                notated_32nd_per_quarter: 8,
            },
            MetaEvent::KeySignature {
                sharps_flats: -3,
                mode: 1,
            },
            MetaEvent::SequencerSpecific(vec![0x41, 0x10, 0x42]),
            MetaEvent::Unknown {
                type_byte: 0x60,
                data: vec![0xDE, 0xAD],
            },
        ];
        let mut events: Vec<TrackEvent> = metas
            .into_iter()
            .map(|m| TrackEvent {
                delta: 0,
                kind: Event::Meta(m),
            })
            .collect();
        events.push(TrackEvent {
            delta: 0,
            kind: Event::Meta(MetaEvent::EndOfTrack),
        });
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track { events }],
        };
        let bytes = smf.to_bytes().unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn round_trip_all_channel_voice_kinds() {
        let bodies = vec![
            ChannelBody::NoteOn {
                key: 60,
                velocity: 100,
            },
            ChannelBody::NoteOff {
                key: 60,
                velocity: 64,
            },
            ChannelBody::PolyAftertouch {
                key: 60,
                pressure: 80,
            },
            ChannelBody::ControlChange {
                controller: 7,
                value: 100,
            },
            ChannelBody::ProgramChange { program: 32 },
            ChannelBody::ChannelAftertouch { pressure: 90 },
            ChannelBody::PitchBend { value: 0x2000 },
            ChannelBody::PitchBend { value: 0x3FFF },
            ChannelBody::PitchBend { value: 0 },
        ];
        let mut events: Vec<TrackEvent> = bodies
            .into_iter()
            .map(|b| TrackEvent {
                delta: 10,
                kind: Event::Channel(ChannelMessage {
                    channel: 5,
                    body: b,
                }),
            })
            .collect();
        events.push(TrackEvent {
            delta: 0,
            kind: Event::Meta(MetaEvent::EndOfTrack),
        });
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(480),
            },
            tracks: vec![Track { events }],
        };
        let bytes = smf.to_bytes().unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn round_trip_sysex_f0_and_f7() {
        let events = vec![
            TrackEvent {
                delta: 0,
                kind: Event::Sysex {
                    escape: false,
                    data: vec![0x41, 0x10, 0x42, 0x12, 0x40, 0x00, 0x7F, 0x00, 0x41, 0xF7],
                },
            },
            TrackEvent {
                delta: 20,
                kind: Event::Sysex {
                    escape: true,
                    data: vec![0xFE], // active sensing escape, e.g.
                },
            },
            TrackEvent {
                delta: 0,
                kind: Event::Meta(MetaEvent::EndOfTrack),
            },
        ];
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track { events }],
        };
        let bytes = smf.to_bytes().unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn round_trip_format_1_multi_track() {
        // Two-track format-1 file: tempo on track 0, a note on track 1.
        let t0 = Track {
            events: vec![
                TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::Tempo(400_000)),
                },
                TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::EndOfTrack),
                },
            ],
        };
        let t1 = Track {
            events: vec![
                TrackEvent {
                    delta: 0,
                    kind: Event::Channel(ChannelMessage {
                        channel: 0,
                        body: ChannelBody::NoteOn {
                            key: 72,
                            velocity: 110,
                        },
                    }),
                },
                TrackEvent {
                    delta: 240,
                    kind: Event::Channel(ChannelMessage {
                        channel: 0,
                        body: ChannelBody::NoteOff {
                            key: 72,
                            velocity: 0,
                        },
                    }),
                },
                TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::EndOfTrack),
                },
            ],
        };
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::MultiTrackSimultaneous,
                ntrks: 2,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![t0, t1],
        };
        let bytes = smf.to_bytes().unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn write_rejects_track_missing_end_of_track() {
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track {
                events: vec![TrackEvent {
                    delta: 0,
                    kind: Event::Channel(ChannelMessage {
                        channel: 0,
                        body: ChannelBody::NoteOn {
                            key: 60,
                            velocity: 100,
                        },
                    }),
                }],
            }],
        };
        let err = smf.to_bytes().unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn write_rejects_eot_not_last() {
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track {
                events: vec![
                    TrackEvent {
                        delta: 0,
                        kind: Event::Meta(MetaEvent::EndOfTrack),
                    },
                    TrackEvent {
                        delta: 0,
                        kind: Event::Channel(ChannelMessage {
                            channel: 0,
                            body: ChannelBody::NoteOn {
                                key: 60,
                                velocity: 100,
                            },
                        }),
                    },
                ],
            }],
        };
        let err = smf.to_bytes().unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn write_rejects_ntrks_mismatch() {
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::MultiTrackSimultaneous,
                ntrks: 3,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track {
                events: vec![TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::EndOfTrack),
                }],
            }],
        };
        let err = smf.to_bytes().unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn write_rejects_pitch_bend_out_of_range() {
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track {
                events: vec![
                    TrackEvent {
                        delta: 0,
                        kind: Event::Channel(ChannelMessage {
                            channel: 0,
                            body: ChannelBody::PitchBend { value: 0x4000 },
                        }),
                    },
                    TrackEvent {
                        delta: 0,
                        kind: Event::Meta(MetaEvent::EndOfTrack),
                    },
                ],
            }],
        };
        let err = smf.to_bytes().unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn write_rejects_data_byte_with_status_bit() {
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track {
                events: vec![
                    TrackEvent {
                        delta: 0,
                        kind: Event::Channel(ChannelMessage {
                            channel: 0,
                            body: ChannelBody::NoteOn {
                                key: 0x80,
                                velocity: 100,
                            },
                        }),
                    },
                    TrackEvent {
                        delta: 0,
                        kind: Event::Meta(MetaEvent::EndOfTrack),
                    },
                ],
            }],
        };
        let err = smf.to_bytes().unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn write_rejects_invalid_smpte_fps() {
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::Smpte {
                    frames_per_second: 60,
                    ticks_per_frame: 40,
                },
            },
            tracks: vec![Track {
                events: vec![TrackEvent {
                    delta: 0,
                    kind: Event::Meta(MetaEvent::EndOfTrack),
                }],
            }],
        };
        let err = smf.to_bytes().unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn round_trip_large_delta_uses_multi_byte_vlq() {
        // 0x200000 takes 4 bytes to encode — exercise the long-VLQ
        // branch of the writer.
        let events = vec![
            TrackEvent {
                delta: 0,
                kind: Event::Channel(ChannelMessage {
                    channel: 0,
                    body: ChannelBody::NoteOn {
                        key: 60,
                        velocity: 100,
                    },
                }),
            },
            TrackEvent {
                delta: 0x20_0000,
                kind: Event::Channel(ChannelMessage {
                    channel: 0,
                    body: ChannelBody::NoteOff {
                        key: 60,
                        velocity: 64,
                    },
                }),
            },
            TrackEvent {
                delta: 0,
                kind: Event::Meta(MetaEvent::EndOfTrack),
            },
        ];
        let smf = SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(480),
            },
            tracks: vec![Track { events }],
        };
        let bytes = smf.to_bytes().unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn track_to_bytes_chunk_round_trips_inside_outer_file() {
        // The per-track helper should produce a self-contained MTrk
        // chunk that drops cleanly inside a hand-built header.
        let track = Track {
            events: vec![
                TrackEvent {
                    delta: 0,
                    kind: Event::Channel(ChannelMessage {
                        channel: 9, // drum channel by GM convention
                        body: ChannelBody::NoteOn {
                            key: 36,
                            velocity: 110,
                        },
                    }),
                },
                TrackEvent {
                    delta: 120,
                    kind: Event::Meta(MetaEvent::EndOfTrack),
                },
            ],
        };
        let chunk = track.to_bytes_chunk().unwrap();
        assert_eq!(&chunk[..4], b"MTrk");
        let body_len = u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as usize;
        assert_eq!(chunk.len(), 8 + body_len);

        // Sandwich into a complete file and round-trip through parse.
        let mut bytes = Vec::new();
        write_header_chunk(
            &mut bytes,
            &SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
        )
        .unwrap();
        bytes.extend_from_slice(&chunk);
        let smf = parse(&bytes).unwrap();
        assert_eq!(smf.tracks[0], track);
    }
}
