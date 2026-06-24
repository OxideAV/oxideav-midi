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

/// Incrementally assemble one [`Track`] from events placed at **absolute
/// ticks**, deferring the SMF delta-time arithmetic to [`build`].
///
/// The wire format stores each event's offset as a *delta* from the
/// previous event in the same track, which is awkward to author by hand
/// — every insertion shifts the deltas after it. `TrackBuilder` lets a
/// caller place events at absolute ticks in any order and computes the
/// deltas once at [`build`] time:
///
/// 1. Events are stably sorted by tick (insertion order breaks ties, so
///    two events at the same tick keep the order they were added — which
///    matters for, e.g., a Program Change that must precede the Note On
///    it patches).
/// 2. The delta of each event is `tick − previous_tick`.
/// 3. A trailing [`MetaEvent::EndOfTrack`] is appended automatically at
///    the last event's tick (delta 0) unless the caller already added
///    one, so the output is always a legal `MTrk`.
///
/// [`build`]: TrackBuilder::build
#[derive(Clone, Debug, Default)]
pub struct TrackBuilder {
    /// `(tick, insertion_index, event)` — the index preserves add order
    /// for the stable tie-break.
    events: Vec<(u64, usize, Event)>,
    next_index: usize,
}

impl TrackBuilder {
    /// A new, empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Place an arbitrary [`Event`] at `tick`. Returns `self` for
    /// chaining.
    pub fn push(&mut self, tick: u64, event: Event) -> &mut Self {
        self.events.push((tick, self.next_index, event));
        self.next_index += 1;
        self
    }

    /// Place a channel-voice message at `tick`.
    pub fn channel(&mut self, tick: u64, channel: u8, body: ChannelBody) -> &mut Self {
        self.push(tick, Event::Channel(ChannelMessage { channel, body }))
    }

    /// Place a meta event at `tick`.
    pub fn meta(&mut self, tick: u64, meta: MetaEvent) -> &mut Self {
        self.push(tick, Event::Meta(meta))
    }

    /// Convenience: a Note On (`9n`) at `tick` and a matching Note Off
    /// (`8n key 0`) at `tick + duration`.
    pub fn note(
        &mut self,
        tick: u64,
        duration: u64,
        channel: u8,
        key: u8,
        velocity: u8,
    ) -> &mut Self {
        self.channel(tick, channel, ChannelBody::NoteOn { key, velocity });
        self.channel(
            tick.saturating_add(duration),
            channel,
            ChannelBody::NoteOff { key, velocity: 0 },
        )
    }

    /// Resolve the absolute-tick events into a [`Track`] with computed
    /// delta-times.
    ///
    /// Events are stably sorted by tick (insertion order breaks ties).
    /// If the caller did not already add a final
    /// [`MetaEvent::EndOfTrack`], one is appended at the last event's
    /// tick. An empty builder yields a single-event track holding just
    /// the End-of-Track marker (the minimum legal `MTrk`).
    pub fn build(self) -> Track {
        let mut items = self.events;
        // Stable sort by tick; equal ticks keep insertion order via the
        // secondary index key.
        items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let has_trailing_eot = matches!(
            items.last().map(|(_, _, e)| e),
            Some(Event::Meta(MetaEvent::EndOfTrack))
        );
        let last_tick = items.last().map(|(t, _, _)| *t).unwrap_or(0);

        let mut events: Vec<TrackEvent> = Vec::with_capacity(items.len() + 1);
        let mut prev_tick: u64 = 0;
        for (tick, _, kind) in items {
            let delta = tick.saturating_sub(prev_tick);
            // SMF deltas are 28-bit VLQs; clamp pathological gaps to the
            // cap so the writer can always serialise (a gap this large
            // never occurs in real scores).
            let delta = delta.min(MAX_VLQ_VALUE as u64) as u32;
            events.push(TrackEvent { delta, kind });
            prev_tick = tick;
        }
        if !has_trailing_eot {
            // The auto-EOT lands at the last event's tick — `prev_tick`
            // already equals `last_tick` after the loop — so its delta
            // is 0. (`last_tick` is referenced only for documentation.)
            debug_assert_eq!(prev_tick, last_tick);
            events.push(TrackEvent {
                delta: 0,
                kind: Event::Meta(MetaEvent::EndOfTrack),
            });
        }
        Track { events }
    }
}

/// Incrementally assemble a complete [`SmfFile`] from one or more
/// [`TrackBuilder`]s, keeping the header `ntrks` in sync with the track
/// count so the result always satisfies [`SmfFile::to_bytes`]'s
/// header-consistency check.
///
/// The format defaults to `1` (multi-track simultaneous) and the
/// division to 480 ticks-per-quarter — the two most common choices —
/// but both are overridable.
#[derive(Clone, Debug)]
pub struct SmfBuilder {
    format: SmfFormat,
    division: Division,
    tracks: Vec<Track>,
}

impl Default for SmfBuilder {
    fn default() -> Self {
        Self {
            format: SmfFormat::MultiTrackSimultaneous,
            division: Division::TicksPerQuarter(480),
            tracks: Vec::new(),
        }
    }
}

impl SmfBuilder {
    /// A new builder: format 1, 480 ticks-per-quarter, no tracks.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the SMF format word (0 / 1 / 2).
    pub fn format(&mut self, format: SmfFormat) -> &mut Self {
        self.format = format;
        self
    }

    /// Set the time division (ticks-per-quarter or SMPTE).
    pub fn division(&mut self, division: Division) -> &mut Self {
        self.division = division;
        self
    }

    /// Append a finished [`Track`].
    pub fn add_track(&mut self, track: Track) -> &mut Self {
        self.tracks.push(track);
        self
    }

    /// Build a [`TrackBuilder`], resolve it, and append the result.
    pub fn track(&mut self, build: impl FnOnce(&mut TrackBuilder)) -> &mut Self {
        let mut tb = TrackBuilder::new();
        build(&mut tb);
        self.tracks.push(tb.build());
        self
    }

    /// Finalise into an [`SmfFile`]. `ntrks` is set from the track count.
    ///
    /// For format 0 the spec allows exactly one track; this is not
    /// enforced here (the writer / parser tolerate the count), but a
    /// caller targeting strict format-0 output should add a single
    /// track.
    pub fn build(self) -> SmfFile {
        SmfFile {
            header: SmfHeader {
                format: self.format,
                ntrks: self.tracks.len() as u16,
                division: self.division,
            },
            tracks: self.tracks,
        }
    }
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

/// Default tempo assumed before the first Set Tempo (`FF 51`) meta
/// event: 500 000 microseconds per quarter note = 120 BPM. The SMF
/// specification (M1 *MIDI 1.0 Detailed Specification*, the Standard
/// MIDI Files addendum) defines this as the playback default for any
/// file that omits an initial tempo.
pub const DEFAULT_TEMPO_US_PER_QUARTER: u32 = 500_000;

/// A precomputed mapping from absolute division ticks to wall-clock
/// seconds, folding a file's tempo map against its
/// [`Division`](crate::smf::Division).
///
/// Built once via [`SmfFile::tempo_timeline`], then queried repeatedly
/// with [`tick_to_seconds`](Self::tick_to_seconds) in `O(log n)` per
/// lookup (binary search over the tempo segments). This is the
/// read-side companion to the scheduler's tick→sample conversion — both
/// use the same arithmetic so a file's reported duration agrees with
/// the rendered output length.
///
/// ## Conversion math
///
/// For a [`Division::TicksPerQuarter`] `div` under a tempo of
/// `usec_per_quarter` microseconds per quarter note, the seconds spent
/// crossing one tick is constant within a tempo segment:
///
/// ```text
///   seconds_per_tick = (usec_per_quarter / 1_000_000) / div
/// ```
///
/// At each Set Tempo change the rate switches; the timeline accumulates
/// the elapsed seconds at each change boundary so an arbitrary tick is
/// resolved as `seconds_at_segment_start + ticks_into_segment *
/// seconds_per_tick`.
///
/// For a [`Division::Smpte`] file the timebase is already absolute
/// (frames × subdivisions per second), so tempo meta events do **not**
/// affect timing and the rate is constant:
///
/// ```text
///   seconds_per_tick = 1 / (frames_per_second * ticks_per_frame)
/// ```
///
/// matching the scheduler's SMPTE branch. Note the stored
/// `frames_per_second` byte is used verbatim — `29` is treated as a
/// literal 29 here (the same simplification the scheduler makes),
/// rather than the 29.97 drop-frame playback rate the SMPTE counter
/// nominally labels.
#[derive(Clone, Debug, PartialEq)]
pub struct TempoTimeline {
    /// Per-segment anchors. Each entry is
    /// `(start_tick, seconds_at_start_tick, seconds_per_tick)` and the
    /// vector is sorted by `start_tick` with a synthetic leading entry
    /// at tick 0 (the default tempo applies until the first change).
    segments: Vec<TimelineSegment>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TimelineSegment {
    start_tick: u64,
    seconds_at_start: f64,
    seconds_per_tick: f64,
}

impl TempoTimeline {
    /// Convert an absolute division tick to elapsed wall-clock seconds
    /// from the start of the file.
    ///
    /// Ticks before the first tempo segment boundary resolve against
    /// the default tempo (the segment list always begins at tick 0).
    /// The result is monotonically non-decreasing in `tick`.
    pub fn tick_to_seconds(&self, tick: u64) -> f64 {
        // Find the last segment whose start_tick <= tick. The list is
        // sorted and always non-empty (a tick-0 anchor is synthesised
        // at construction), so partition_point gives the segment index.
        let idx = self
            .segments
            .partition_point(|s| s.start_tick <= tick)
            .saturating_sub(1);
        let seg = self.segments[idx];
        let into_segment = tick.saturating_sub(seg.start_tick) as f64;
        seg.seconds_at_start + into_segment * seg.seconds_per_tick
    }

    /// The tempo segments backing this timeline, exposed for callers
    /// that want to walk boundaries directly. Each tuple is
    /// `(start_tick, seconds_at_start, seconds_per_tick)`; the first
    /// entry is always the tick-0 anchor.
    pub fn segments(&self) -> Vec<(u64, f64, f64)> {
        self.segments
            .iter()
            .map(|s| (s.start_tick, s.seconds_at_start, s.seconds_per_tick))
            .collect()
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

/// One Channel Prefix meta event pinned to the absolute tick (relative
/// to the start of its parent track) at which the
/// [`FF 20 01 cc`](MetaEvent::ChannelPrefix) meta event fires.
///
/// Returned by [`SmfFile::channel_prefixes`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `FF 20 01 cc` carries the **channel-binding hint** for meta and sysex
/// events that follow on the same track. The single payload byte names
/// the MIDI channel (`0..=15`) the following non-channel events should
/// be associated with — text, lyric, marker, cue point, sysex, and the
/// like — until either another `FF 20` arrives, the next channel-voice
/// event arrives (and supersedes the binding), or the track ends. The
/// Standard MIDI File Specification 1.0 lists the event as part of the
/// meta-event vocabulary and notes the pre-multi-port usage; it is
/// deprecated in modern authoring tools in favour of explicit per-track
/// channel-voice streams plus [`FF 21`](MetaEvent::Port) port hints, but
/// older files (pre-General-MIDI authoring suites, RPM/SMF converters)
/// still emit it and a round-trip workflow must preserve it.
///
/// This helper isolates the `FF 20` stream so callers reconstructing the
/// channel association of surrounding non-channel events get a clean
/// time-ordered list independent of the other meta streams. The payload
/// byte is surfaced verbatim — the spec scopes it at `0..=15` (one
/// nibble) but the parser accepts the full one-byte payload as written
/// so files with out-of-spec values still round-trip; the
/// [`channel`](Self::channel) accessor returns `Some(c)` only when
/// `c < 16` (an `Option` rather than a clamp because a value `>= 16`
/// is unambiguously non-spec and the receiver should likely fall back
/// to the most recent channel-voice channel rather than mask the byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelPrefixEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the meta event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase. The pre-multi-port
    /// convention places one `FF 20` near the start of a binding block
    /// (just before the first text / sysex it scopes), but the parser
    /// surfaces every occurrence rather than enforcing the placement
    /// rule so files that re-bind mid-track still round-trip.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). The channel-prefix binding scopes the
    /// surrounding non-channel events to a specific channel on this
    /// track; a multi-port player consults this together with the
    /// track's [`FF 21`](MetaEvent::Port) hint to derive the full
    /// `(port, channel)` routing for following meta / sysex events.
    pub track: usize,
    /// The channel byte (the `cc` payload of `FF 20 01 cc`). The
    /// Standard MIDI File Specification 1.0 reserves a single nibble
    /// (`0..=15`) for the channel index; the parser accepts the full
    /// one-byte payload as written so files with out-of-spec values
    /// (a single bit set in the high nibble, etc.) still round-trip.
    /// Use [`channel`](Self::channel) for the spec-clamped `Option<u8>`
    /// view, or read the raw byte directly.
    pub channel: u8,
}

impl ChannelPrefixEvent {
    /// The bound MIDI channel index, decoded into the spec's `0..=15`
    /// range. Returns `None` when the payload byte is out of range —
    /// the spec leaves the high four bits unspecified, so a value
    /// `>= 16` is unambiguously non-spec and a receiver should likely
    /// fall back to the channel of the most recent channel-voice event
    /// rather than mask the byte (the latter would silently route to
    /// an unintended channel).
    pub fn channel(&self) -> Option<u8> {
        if self.channel < 16 {
            Some(self.channel)
        } else {
            None
        }
    }
}

/// Classification of a *Registered* Parameter Number selected by the
/// CC 101 (RPN MSB) / CC 100 (RPN LSB) pair, per the MIDI 1.0 *Control
/// Change Messages — Data Bytes* document, Table 3a ("Registered
/// Parameter Numbers").
///
/// Registered Parameters are the MMA-defined subset of the Data Entry
/// pump's address space; the complementary Non-Registered Parameters
/// (CC 99 / CC 98) are manufacturer-private and carry no fixed
/// vocabulary, so they are surfaced only as a raw 14-bit number (see
/// [`SelectedParameter::NonRegistered`]).
///
/// The variant set mirrors Table 3a exactly:
///
/// * `0x0000` Pitch Bend Sensitivity
/// * `0x0001` Channel Fine Tuning (formerly Fine Tuning, MMA RP-022)
/// * `0x0002` Channel Coarse Tuning (formerly Coarse Tuning, RP-022)
/// * `0x0003` Tuning Program Change
/// * `0x0004` Tuning Bank Select
/// * `0x0005` Modulation Depth Range (MMA CA-26 / GM2)
/// * `0x0006` MPE Configuration Message (MPE specification)
/// * `0x3D00..=0x3D08` the nine Three-Dimensional Sound Controllers
///   (RP-049): Azimuth, Elevation, Gain, Distance Ratio, Maximum
///   Distance, Gain At Maximum Distance, Reference Distance Ratio, Pan
///   Spread Angle, Roll Angle
/// * `7FH:7FH` (packed `0x3FFF`) the Null Function Number (selecting it
///   disables the Data Entry / Increment / Decrement pump until a new
///   parameter is selected)
///
/// Any 14-bit number not named in Table 3a is "RESERVED for future MMA
/// Definition" and surfaces as [`RegisteredParameter::Reserved`] with
/// the raw number preserved, so a forward-compatible caller can still
/// route on the exact value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisteredParameter {
    /// `0x0000` — Pitch Bend Sensitivity. Data Entry MSB = ± semitones,
    /// LSB = ± cents.
    PitchBendSensitivity,
    /// `0x0001` — Channel Fine Tuning (formerly Fine Tuning, RP-022).
    /// The 14-bit Data Entry value spans ±100 cents around centre
    /// (`0x40 0x00` = A440).
    ChannelFineTuning,
    /// `0x0002` — Channel Coarse Tuning (formerly Coarse Tuning,
    /// RP-022). Only the MSB is used; resolution is 100 cents
    /// (`0x40` = A440).
    ChannelCoarseTuning,
    /// `0x0003` — Tuning Program Change. Data Entry selects a MIDI
    /// Tuning Standard tuning-program number.
    TuningProgramChange,
    /// `0x0004` — Tuning Bank Select. Data Entry selects a MIDI Tuning
    /// Standard tuning-bank number.
    TuningBankSelect,
    /// `0x0005` — Modulation Depth Range (CA-26 / GM2). Data Entry sets
    /// the mod-wheel depth range.
    ModulationDepthRange,
    /// `0x0006` — MPE Configuration Message (MPE specification §2.2.1).
    /// Data Entry MSB carries the Member-Channel count.
    MpeConfiguration,
    /// `0x3D00..=0x3D08` — one of the nine Three-Dimensional Sound
    /// Controllers (RP-049). The inner byte is the LSB (`0x00`..=`0x08`):
    /// `0` Azimuth Angle, `1` Elevation Angle, `2` Gain, `3` Distance
    /// Ratio, `4` Maximum Distance, `5` Gain At Maximum Distance, `6`
    /// Reference Distance Ratio, `7` Pan Spread Angle, `8` Roll Angle.
    ThreeDimensionalSound(u8),
    /// `7FH:7FH` (packed `0x3FFF`) — Null Function Number for RPN / NRPN. Selecting it
    /// disables the Data Entry / Increment / Decrement controllers until
    /// a new RPN or NRPN is selected.
    Null,
    /// Any 14-bit number not assigned in Table 3a — "RESERVED for future
    /// MMA Definition". The raw `(msb << 7) | lsb` value is preserved.
    Reserved(u16),
}

impl RegisteredParameter {
    /// Classify the 14-bit RPN value `(msb << 7) | lsb` against Table 3a.
    ///
    /// `number` is the *packed* 14-bit value in `0..=0x3FFF`, with the
    /// CC 101 (MSB) byte in the high seven bits and the CC 100 (LSB) byte
    /// in the low seven. Table 3a displays parameters as an "MSB:LSB"
    /// byte pair (e.g. `3DH:00H` for Azimuth, `7FH:7FH` for Null); those
    /// pairs map to the packed values `(MSB << 7) | LSB` — `0x3DH:00H`
    /// → `0x1E80`, `7FH:7FH` → `0x3FFF` — which is the representation
    /// matched here (the same packing the runtime mixer uses).
    pub fn from_number(number: u16) -> Self {
        // 3D Sound Controllers: MSB 0x3D, LSB 0x00..=0x08 → packed
        // 0x1E80..=0x1E88.
        const SOUND_3D_BASE: u16 = (0x3D << 7) as u16; // 0x1E80
        match number {
            0x0000 => RegisteredParameter::PitchBendSensitivity,
            0x0001 => RegisteredParameter::ChannelFineTuning,
            0x0002 => RegisteredParameter::ChannelCoarseTuning,
            0x0003 => RegisteredParameter::TuningProgramChange,
            0x0004 => RegisteredParameter::TuningBankSelect,
            0x0005 => RegisteredParameter::ModulationDepthRange,
            0x0006 => RegisteredParameter::MpeConfiguration,
            n if (SOUND_3D_BASE..=SOUND_3D_BASE + 8).contains(&n) => {
                RegisteredParameter::ThreeDimensionalSound((n - SOUND_3D_BASE) as u8)
            }
            0x3FFF => RegisteredParameter::Null,
            other => RegisteredParameter::Reserved(other),
        }
    }
}

/// The parameter a Data Entry / Increment / Decrement write addresses at
/// the moment it fires, resolved from the channel's running RPN / NRPN
/// selector state.
///
/// Per the MIDI 1.0 *Control Change Messages — Data Bytes* document, the
/// Data Entry pump (CC 6 / CC 38) and the Data Increment / Decrement
/// controllers (CC 96 / CC 97) act on whichever parameter was most
/// recently selected by an RPN pair (CC 101 MSB / CC 100 LSB) or an NRPN
/// pair (CC 99 MSB / CC 98 LSB). Selecting an RPN supersedes a prior
/// NRPN selection and vice versa — there is a single active parameter
/// per channel, not one of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectedParameter {
    /// A Registered Parameter (CC 101 / CC 100 selector). Carries both
    /// the raw 14-bit number and its Table-3a classification.
    Registered {
        /// The raw `(msb << 7) | lsb` parameter number, `0..=0x3FFF`.
        number: u16,
        /// The Table-3a classification of `number`.
        param: RegisteredParameter,
    },
    /// A Non-Registered Parameter (CC 99 / CC 98 selector). NRPNs are
    /// manufacturer-private, so only the raw 14-bit number is surfaced.
    /// The Null Function Number (packed `0x3FFF`, the `7FH:7FH` MSB:LSB
    /// pair) selected via the NRPN pair is reported here as a `number` of
    /// `0x3FFF` (it disables the pump just as the RPN-null does).
    NonRegistered {
        /// The raw `(msb << 7) | lsb` parameter number, `0..=0x3FFF`.
        number: u16,
    },
}

impl SelectedParameter {
    /// The raw 14-bit parameter number regardless of registered /
    /// non-registered flavour.
    pub fn number(&self) -> u16 {
        match self {
            SelectedParameter::Registered { number, .. } => *number,
            SelectedParameter::NonRegistered { number } => *number,
        }
    }

    /// `true` for a Registered Parameter (CC 101 / CC 100 selector).
    pub fn is_registered(&self) -> bool {
        matches!(self, SelectedParameter::Registered { .. })
    }

    /// `true` when the selected parameter is the Null Function Number
    /// (packed `0x3FFF`, the `7FH:7FH` MSB:LSB pair) — the sentinel that
    /// disables the Data Entry / Increment / Decrement pump until a fresh
    /// parameter is selected. A write addressing a null parameter is
    /// never emitted by [`SmfFile::parameter_data_entries`].
    pub fn is_null(&self) -> bool {
        self.number() == 0x3FFF
    }
}

/// The kind of Data Entry pump action a [`ParameterDataEntry`] records.
///
/// The MIDI 1.0 *Control Change Messages — Data Bytes* document defines
/// two ways to write a selected parameter: an absolute Data Entry
/// (CC 6 = MSB, CC 38 = LSB), and a relative Data Increment (CC 96) /
/// Data Decrement (CC 97) whose own value byte is "don't care" (the
/// step is always ±1 per MMA RP-018).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataEntryAction {
    /// CC 6 — Data Entry MSB. The `value` byte (`0..=127`) is the
    /// most-significant 7 bits of the parameter's 14-bit value.
    EntryMsb(u8),
    /// CC 38 — Data Entry LSB. The `value` byte (`0..=127`) is the
    /// least-significant 7 bits of the parameter's 14-bit value.
    EntryLsb(u8),
    /// CC 96 — Data Increment. Steps the selected parameter up by one
    /// (the controller's own value byte is ignored per RP-018).
    Increment,
    /// CC 97 — Data Decrement. Steps the selected parameter down by one
    /// (the controller's own value byte is ignored per RP-018).
    Decrement,
}

/// One Data Entry pump action (CC 6 / CC 38 / CC 96 / CC 97) resolved
/// against the channel's running RPN / NRPN selector state, pinned to
/// the absolute tick at which it fires.
///
/// Returned by [`SmfFile::parameter_data_entries`] — see that method for
/// the selector-folding rules and the multi-track merge semantics.
///
/// Where [`SmfFile::control_changes`] surfaces *raw* CC bytes and leaves
/// the RPN / NRPN Data Entry pump for the receiving application to
/// resolve, this event carries the *resolved* target: which parameter
/// the channel had selected (an RPN classified per Table 3a, or a raw
/// NRPN number) and what the pump did to it (an absolute MSB / LSB write
/// or a relative ±1 step). It is the read-side analogue of the runtime
/// RPN / NRPN state machine a synth maintains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterDataEntry {
    /// Cumulative delta-sum from the start of the track that carried the
    /// Data Entry controller, in division units. For format-1 SMFs this
    /// is also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the Data Entry controller came from.
    pub track: usize,
    /// The MIDI channel the write targets, in the spec's `0..=15` range.
    pub channel: u8,
    /// The parameter the channel had selected when the write fired —
    /// either a Table-3a-classified Registered Parameter or a raw NRPN
    /// number. Never the Null Function Number: writes addressing a null
    /// parameter are dropped (the spec disables the pump in that state).
    pub parameter: SelectedParameter,
    /// What the pump did to `parameter`.
    pub action: DataEntryAction,
}

impl ParameterDataEntry {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The parameter the write addressed.
    pub fn parameter(&self) -> SelectedParameter {
        self.parameter
    }

    /// The pump action the event records.
    pub fn action(&self) -> DataEntryAction {
        self.action
    }
}

/// One Program Change (`Cn pp`) channel-voice event pinned to the
/// absolute tick (relative to the start of its parent track) at which
/// the [`ChannelBody::ProgramChange`] wire event fires.
///
/// Returned by [`SmfFile::program_changes`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `Cn pp` is the Standard MIDI File §"Channel Voice Messages" §`Cn`
/// patch-select message: a single data byte `pp` (`0..=127`) names the
/// MIDI program number the receiving synth should switch the channel
/// to. In a General MIDI 1 receiver this resolves through the Bank
/// Select pair (CC 0 / CC 32) to one of the 128 GM melodic patches —
/// `0` Acoustic Grand Piano, `40` Violin, `73` Flute, … — but the
/// SMF parser doesn't bind any patch-list semantics here: it surfaces
/// the raw program byte and lets the player resolve the patch against
/// whatever bank is active at that tick.
///
/// This helper isolates the patch-select stream so callers driving an
/// instrument-list view (DAW track inspector, soft-synth voice
/// rebuilder, song-form patch-change timeline) get a clean
/// time-ordered list independent of the surrounding channel-voice and
/// meta-event streams. The companion wire-state primitive
/// [`SmfFile::channel_snapshot_at`] folds the *last* program change
/// per channel into [`SmfChannelSnapshot::program`] for seek
/// initialisation; this helper surfaces *every* change in chronological
/// order so callers building the full patch-change timeline (e.g. a
/// "song form" view that highlights the bar each instrument enters)
/// don't have to re-walk every track event manually.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramChangeEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the channel-voice event, in division units. For format-1 SMFs
    /// this is also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files commonly place all of a
    /// part's channel-voice events on one track; format-2 files keep
    /// patterns separate so each pattern's Program Change appears on
    /// its own track.
    pub track: usize,
    /// The MIDI channel the patch change targets, in the spec's
    /// `0..=15` range (channel "1" in human-facing tools is index `0`).
    /// Decoded from the low nibble of the `Cn` status byte at parse
    /// time so the parser's running-status bookkeeping is already
    /// resolved by the time the event reaches the helper.
    pub channel: u8,
    /// The program number (`pp` payload of `Cn pp`, `0..=127`). The
    /// Standard MIDI File Specification 1.0 reserves the high bit
    /// (`0x80`) for status / running-status framing; the parser
    /// rejects channel-voice data bytes with the high bit set so the
    /// value space here is always `0..=127`. Resolution against a
    /// patch list (General MIDI 1 / 2, GS, XG, …) is left to the
    /// receiving application.
    pub program: u8,
}

impl ProgramChangeEvent {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The MIDI program number (`0..=127`).
    pub fn program(&self) -> u8 {
        self.program
    }
}

/// One Control Change (`Bn cc vv`) channel-voice event pinned to the
/// absolute tick (relative to the start of its parent track) at which
/// the [`ChannelBody::ControlChange`] wire event fires.
///
/// Returned by [`SmfFile::control_changes`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `Bn cc vv` is the Standard MIDI File §"Channel Voice Messages" §`Bn`
/// continuous-controller / channel-mode message: a controller-number
/// byte `cc` (`0..=127`) selects one of the 128 indexes documented in
/// the MIDI 1.0 *Control Change Messages — Data Bytes* table — Bank
/// Select MSB (`0`), Modulation Wheel (`1`), Channel Volume (`7`), Pan
/// (`10`), Expression (`11`), Bank Select LSB (`32`), Damper / Sustain
/// (`64`), Data Entry MSB / LSB (`6` / `38`), RPN / NRPN MSB / LSB
/// (`100` / `101` and `98` / `99`), All Sound Off (`120`), Reset All
/// Controllers (`121`), All Notes Off (`123`), and the Channel Mode
/// family (`120..=127`) — and a value byte `vv` (`0..=127`) carries the
/// controller's new setting. The SMF parser surfaces both bytes raw so
/// the helper stays controller-agnostic: callers driving an RPN / NRPN
/// pair reassembler, a CC-7 / CC-11 volume / expression curve view, a
/// CC-64 pedal-on / pedal-off span renderer, or a channel-mode
/// (`120..=127`) reset-detector all read from the same `(controller,
/// value)` pair without re-walking the channel-voice stream.
///
/// This helper isolates the CC channel so callers building a
/// controller-automation view (DAW lane editor, soft-synth state
/// rebuilder, song-form CC-7 / CC-11 curve renderer, RPN / NRPN pair
/// joiner) get a clean time-ordered list independent of the
/// surrounding patch-select (`Cn`), pitch-bend (`En`), aftertouch
/// (`An` / `Dn`), note (`8n` / `9n`), and meta-event streams. The
/// companion wire-state primitive [`SmfFile::channel_snapshot_at`]
/// folds the *last* value of the six snapshot-tracked controllers
/// (Bank MSB / LSB, Modulation, Volume, Pan, Expression, Sustain) into
/// the snapshot at the seek point; this helper surfaces *every*
/// Control Change in chronological order so callers building an
/// automation timeline that includes every controller — not just the
/// six the snapshot tracks — don't have to re-walk the channel-voice
/// stream manually.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlChangeEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the channel-voice event, in division units. For format-1 SMFs
    /// this is also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files commonly place the bulk of
    /// a part's CC automation on the same track as its note-on /
    /// note-off events; format-2 files keep patterns separate so each
    /// pattern's CC stream appears on its own track.
    pub track: usize,
    /// The MIDI channel the controller change targets, in the spec's
    /// `0..=15` range (channel "1" in human-facing tools is index `0`).
    /// Decoded from the low nibble of the `Bn` status byte at parse
    /// time so the parser's running-status bookkeeping is already
    /// resolved by the time the event reaches the helper.
    pub channel: u8,
    /// The controller number (`cc` first data byte of `Bn cc vv`,
    /// `0..=127`). The Standard MIDI File Specification 1.0 reserves
    /// the high bit (`0x80`) for status / running-status framing; the
    /// parser rejects channel-voice data bytes with the high bit set
    /// so the value space here is always `0..=127`. The MIDI 1.0
    /// *Control Change Messages — Data Bytes* document assigns
    /// `120..=127` to the channel-mode family (All Sound Off, Reset
    /// All Controllers, Local Control, All Notes Off, Omni Off, Omni
    /// On, Mono / Poly Mode) — those values are *still* surfaced
    /// through this helper rather than diverted to a separate channel,
    /// so a player reset-detector can route on `controller == 123 &&
    /// value == 0` without consulting a second iterator.
    pub controller: u8,
    /// The controller value (`vv` second data byte of `Bn cc vv`,
    /// `0..=127`). Resolution against a controller-specific scale
    /// (the spec's "MSB" / "LSB" pairing for 14-bit controllers
    /// `0..=31` plus `32..=63`, the on-off threshold `value >= 64` for
    /// switch controllers `64..=69`, the Data Entry pump for RPN /
    /// NRPN parameter writes) is left to the receiving application:
    /// the helper stays controller-agnostic and surfaces the raw
    /// value byte so callers can pick their own controller-vocabulary
    /// policy.
    pub value: u8,
}

/// A typed Channel Mode Message (Control Change `120..=127`), classified
/// per the MIDI 1.0 Detailed Specification §"Channel Mode Messages".
///
/// These seven controllers share the `Bn` Control Change status but
/// carry mode semantics rather than continuous-controller values. Six of
/// them take a `0` value byte by convention; two carry an argument:
/// Local Control (on/off) and Mono Mode On (the channel count `m`).
///
/// Returned by [`ControlChangeEvent::channel_mode`] and collected by
/// [`SmfFile::channel_mode_messages`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelModeMessage {
    /// CC 120 — All Sound Off. Silences every sounding voice on the
    /// channel immediately (ignoring release envelopes), value `0`.
    AllSoundOff,
    /// CC 121 — Reset All Controllers. Returns the channel's controllers
    /// to their default values, value `0`.
    ResetAllControllers,
    /// CC 122 — Local Control. `on == true` for value `127` (keyboard
    /// drives the local sound engine), `false` for value `0` (keyboard
    /// and engine decoupled). Values `1..=126` are non-standard; the
    /// spec's switch rule treats `0..=63` as off and `64..=127` as on.
    LocalControl { on: bool },
    /// CC 123 — All Notes Off, value `0`.
    AllNotesOff,
    /// CC 124 — Omni Mode Off (also acts as All Notes Off), value `0`.
    OmniOff,
    /// CC 125 — Omni Mode On (also acts as All Notes Off), value `0`.
    OmniOn,
    /// CC 126 — Mono Mode On / Poly Mode Off (also acts as All Notes
    /// Off). `channels` is the value byte `m`: the number of channels
    /// over which Mono voices are assigned, or `0` for "the number of
    /// voices in the receiver" (assign one channel per voice across the
    /// receive range), per the spec.
    MonoOn { channels: u8 },
    /// CC 127 — Poly Mode On / Mono Mode Off (also acts as All Notes
    /// Off), value `0`.
    PolyOn,
}

impl ChannelModeMessage {
    /// `true` for messages 123–127, which the spec says also function as
    /// All Notes Off (`AllNotesOff`, `OmniOff`, `OmniOn`, `MonoOn`,
    /// `PolyOn`).
    pub fn is_all_notes_off(&self) -> bool {
        matches!(
            self,
            ChannelModeMessage::AllNotesOff
                | ChannelModeMessage::OmniOff
                | ChannelModeMessage::OmniOn
                | ChannelModeMessage::MonoOn { .. }
                | ChannelModeMessage::PolyOn
        )
    }
}

/// One of the five "Effects N Depth" controllers (CC 91–95), classified
/// by the MIDI 1.0 *Control Change Messages — Data Bytes* document
/// (Table 3). Each carries a `0..=127` depth/send level (`value`).
///
/// The two effects with a General MIDI default function — Reverb Send
/// (CC 91, "default: Reverb Send Level — see MMA RP-023") and Chorus
/// Send (CC 93, "default: Chorus Send Level") — are what the mixer's
/// system effects bus (CA-024) consumes; the other three preserve the
/// table's "formerly …" historical assignments (Tremolo / Celeste
/// [Detune] / Phaser depth) as their documented names. Returned by
/// [`ControlChangeEvent::effect_depth`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectDepth {
    /// CC 91 — Effects 1 Depth, GM default **Reverb Send Level**
    /// (formerly External Effects Depth).
    ReverbSend(u8),
    /// CC 92 — Effects 2 Depth (formerly Tremolo Depth).
    Tremolo(u8),
    /// CC 93 — Effects 3 Depth, GM default **Chorus Send Level**
    /// (formerly Chorus Depth).
    ChorusSend(u8),
    /// CC 94 — Effects 4 Depth (formerly Celeste / Detune Depth).
    Celeste(u8),
    /// CC 95 — Effects 5 Depth (formerly Phaser Depth).
    Phaser(u8),
}

impl EffectDepth {
    /// The raw `0..=127` depth / send-level value, regardless of which
    /// of the five effect slots this is.
    pub fn level(&self) -> u8 {
        match self {
            EffectDepth::ReverbSend(v)
            | EffectDepth::Tremolo(v)
            | EffectDepth::ChorusSend(v)
            | EffectDepth::Celeste(v)
            | EffectDepth::Phaser(v) => *v,
        }
    }

    /// The controller number (`91..=95`) this variant corresponds to.
    pub fn controller(&self) -> u8 {
        match self {
            EffectDepth::ReverbSend(_) => 91,
            EffectDepth::Tremolo(_) => 92,
            EffectDepth::ChorusSend(_) => 93,
            EffectDepth::Celeste(_) => 94,
            EffectDepth::Phaser(_) => 95,
        }
    }
}

impl ControlChangeEvent {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The controller number (`0..=127`).
    pub fn controller(&self) -> u8 {
        self.controller
    }

    /// The controller value byte (`0..=127`).
    pub fn value(&self) -> u8 {
        self.value
    }

    /// Returns `true` when the controller number falls in the
    /// channel-mode range (`120..=127`) defined by the MIDI 1.0
    /// *Control Change Messages — Data Bytes* document: All Sound Off
    /// (`120`), Reset All Controllers (`121`), Local Control (`122`),
    /// All Notes Off (`123`), Omni Mode Off (`124`), Omni Mode On
    /// (`125`), Mono Mode On (`126`), Poly Mode On (`127`).
    ///
    /// A reset-detector replaying a snapshot at a seek point can route
    /// on this predicate without re-checking the controller-number
    /// range manually; the spec assigns these values exclusively to
    /// the channel-mode family and never reuses them as continuous
    /// controllers.
    pub fn is_channel_mode(&self) -> bool {
        matches!(self.controller, 120..=127)
    }

    /// Classify a channel-mode controller (`120..=127`) into a typed
    /// [`ChannelModeMessage`], decoding the value-byte argument for
    /// Local Control (on/off) and Mono Mode On (channel count). Returns
    /// `None` for a continuous controller (`0..=119`).
    pub fn channel_mode(&self) -> Option<ChannelModeMessage> {
        Some(match self.controller {
            120 => ChannelModeMessage::AllSoundOff,
            121 => ChannelModeMessage::ResetAllControllers,
            122 => ChannelModeMessage::LocalControl {
                on: self.value >= 64,
            },
            123 => ChannelModeMessage::AllNotesOff,
            124 => ChannelModeMessage::OmniOff,
            125 => ChannelModeMessage::OmniOn,
            126 => ChannelModeMessage::MonoOn {
                channels: self.value,
            },
            127 => ChannelModeMessage::PolyOn,
            _ => return None,
        })
    }

    /// Classify an "Effects N Depth" controller (CC 91–95) into a typed
    /// [`EffectDepth`], carrying the `0..=127` depth / send-level value.
    /// Returns `None` for any other controller number.
    ///
    /// CC 91 (Reverb Send) and CC 93 (Chorus Send) are the GM-default
    /// sends the mixer's CA-024 effects bus consumes; the parse-side
    /// classifier lets a caller surface every effect-depth edit in a
    /// score without driving the synth.
    pub fn effect_depth(&self) -> Option<EffectDepth> {
        Some(match self.controller {
            91 => EffectDepth::ReverbSend(self.value),
            92 => EffectDepth::Tremolo(self.value),
            93 => EffectDepth::ChorusSend(self.value),
            94 => EffectDepth::Celeste(self.value),
            95 => EffectDepth::Phaser(self.value),
            _ => return None,
        })
    }
}

/// One Channel Mode Message (`Bn cc vv`, `cc` in `120..=127`) pinned to
/// the absolute tick at which it fires, with the controller already
/// classified into a typed [`ChannelModeMessage`].
///
/// Returned by [`SmfFile::channel_mode_messages`] — the typed
/// counterpart of the channel-mode subset of [`SmfFile::control_changes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelModeEvent {
    /// Cumulative delta-sum from the start of the track that carried the
    /// controller, in division units. For format-1 SMFs this is also the
    /// absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the controller came from.
    pub track: usize,
    /// The MIDI channel index in the spec's `0..=15` range.
    pub channel: u8,
    /// The classified channel-mode message.
    pub message: ChannelModeMessage,
}

impl ChannelModeEvent {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The classified channel-mode message.
    pub fn message(&self) -> ChannelModeMessage {
        self.message
    }
}

/// One **Effects Depth** controller (`Bn cc vv`, `cc` in `91..=95`)
/// pinned to the absolute tick at which it fires, with the controller
/// classified into a typed [`EffectDepth`].
///
/// Returned by [`SmfFile::effect_depths`] — the typed counterpart of the
/// effects-depth subset of [`SmfFile::control_changes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EffectDepthEvent {
    /// Cumulative delta-sum from the start of the track that carried the
    /// controller, in division units. For format-1 SMFs this is also the
    /// absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the controller came from.
    pub track: usize,
    /// The MIDI channel index in the spec's `0..=15` range.
    pub channel: u8,
    /// The classified effect-depth controller + its level.
    pub depth: EffectDepth,
}

impl EffectDepthEvent {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The classified effect-depth controller (Reverb Send / Tremolo /
    /// Chorus Send / Celeste / Phaser) with its `0..=127` level.
    pub fn depth(&self) -> EffectDepth {
        self.depth
    }
}

/// One Pitch Bend channel-voice event pinned to the absolute tick
/// (relative to the start of its parent track) at which the
/// [`ChannelBody::PitchBend`] wire event fires.
///
/// Returned by [`SmfFile::pitch_bends`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `En lsb msb` is the Standard MIDI File §"Channel Voice Messages"
/// §`En` pitch-bend message: two data bytes `lsb` / `msb` (each
/// `0..=127`) combine into a single 14-bit unsigned value
/// `(msb << 7) | lsb` spanning `0..=0x3FFF`, with the centre (no-bend)
/// position at `0x2000`. The parser combines the two bytes at decode
/// time so the value reaches this helper already assembled. The amount
/// of pitch displacement a given value produces is the receiver's
/// concern: it depends on the channel's Pitch Bend Sensitivity (RPN 0,
/// default ±2 semitones), so the helper stays sensitivity-agnostic and
/// surfaces the raw 14-bit code together with a signed-from-centre
/// convenience accessor.
///
/// This helper isolates the pitch-bend stream so callers driving an
/// expression-curve view (DAW bend-lane editor, soft-synth wheel-state
/// rebuilder, a glissando / vibrato curve renderer) get a clean
/// time-ordered list independent of the surrounding control-change
/// (`Bn`), patch-select (`Cn`), aftertouch (`An` / `Dn`), and note
/// (`8n` / `9n`) channel-voice streams. The companion wire-state
/// primitive [`SmfFile::channel_snapshot_at`] folds the *last* pitch
/// bend per channel into [`SmfChannelSnapshot::pitch_bend`] for seek
/// initialisation; this helper surfaces *every* bend in chronological
/// order so callers building the full bend timeline don't have to
/// re-walk every track event manually.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitchBendEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the channel-voice event, in division units. For format-1 SMFs
    /// this is also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files commonly place the bulk of
    /// a part's bend automation on the same track as its note-on /
    /// note-off events; format-2 files keep patterns separate so each
    /// pattern's bend stream appears on its own track.
    pub track: usize,
    /// The MIDI channel the pitch bend targets, in the spec's `0..=15`
    /// range (channel "1" in human-facing tools is index `0`). Decoded
    /// from the low nibble of the `En` status byte at parse time so
    /// the parser's running-status bookkeeping is already resolved by
    /// the time the event reaches the helper.
    pub channel: u8,
    /// The combined 14-bit pitch-bend value `(msb << 7) | lsb`,
    /// `0..=0x3FFF`, with the no-bend centre at `0x2000`. The two wire
    /// data bytes each carry 7 significant bits (the high bit is the
    /// MIDI status flag the parser already rejected on a data byte), so
    /// the assembled value never exceeds 14 bits. Use
    /// [`PitchBendEvent::signed_value`] for the displacement from
    /// centre as a signed `-8192..=8191`.
    pub value: u16,
}

impl PitchBendEvent {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The raw 14-bit pitch-bend value `(msb << 7) | lsb`,
    /// `0..=0x3FFF`, centre `0x2000`.
    pub fn value(&self) -> u16 {
        self.value
    }

    /// The pitch-bend displacement from centre as a signed value in
    /// `-8192..=8191`: `value as i32 - 0x2000`. The centre code
    /// `0x2000` maps to `0`, the minimum code `0x0000` to `-8192`, and
    /// the maximum code `0x3FFF` to `8191`. Resolving the signed code
    /// to an actual pitch displacement requires the channel's Pitch
    /// Bend Sensitivity (RPN 0, default ±2 semitones), which the helper
    /// leaves to the receiving application.
    pub fn signed_value(&self) -> i16 {
        // value is always 0..=0x3FFF so the subtraction stays within
        // i16 range (-8192..=8191).
        (self.value as i32 - 0x2000) as i16
    }

    /// Returns `true` when the bend sits at the no-bend centre
    /// (`value == 0x2000`). A DAW bend-lane editor can collapse a run
    /// of centre values, and a wheel-release detector can route on this
    /// predicate without re-checking the raw code.
    pub fn is_centre(&self) -> bool {
        self.value == 0x2000
    }
}

/// One Polyphonic Key Pressure (per-key aftertouch) channel-voice event
/// pinned to the absolute tick (relative to the start of its parent
/// track) at which the [`ChannelBody::PolyAftertouch`] wire event fires.
///
/// Returned by [`SmfFile::poly_aftertouches`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `An kk pp` is the Standard MIDI File §"Channel Voice Messages" §`An`
/// (status nibble `1010`) Polyphonic Key Pressure message: the low
/// nibble of the status byte is the channel index (`0..=15`), the first
/// data byte `kk` (`0..=127`) names the key the pressure targets, and
/// the second data byte `pp` (`0..=127`) carries the pressure value. Per
/// the MIDI 1.0 *Summary of MIDI Messages* Table 1 this is "the per-key
/// pressure value" — distinct from Channel Pressure (`Dn`, the single
/// greatest pressure over all depressed keys), which travels on its own
/// surface ([`SmfFile::channel_pressures`]). It is generated by
/// keyboards with per-key aftertouch sensors.
///
/// This helper isolates the polyphonic-aftertouch stream so callers
/// driving a per-key expression view (a DAW poly-pressure lane editor, a
/// soft-synth per-voice aftertouch state rebuilder) get a clean
/// time-ordered list independent of the surrounding control-change
/// (`Bn`), patch-select (`Cn`), pitch-bend (`En`), channel-pressure
/// (`Dn`), and note (`8n` / `9n`) channel-voice streams. The pressure
/// value's musical effect (typically routed per-voice to volume, vibrato
/// depth, or filter cutoff by the receiving instrument) is the
/// receiver's concern, so the helper stays routing-agnostic and surfaces
/// the raw `0..=127` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolyAftertouchEvent {
    /// Cumulative delta-sum from the start of the track that carried the
    /// channel-voice event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files commonly place a part's
    /// poly-aftertouch automation on the same track as its note-on /
    /// note-off events; format-2 files keep patterns separate so each
    /// pattern's pressure stream appears on its own track.
    pub track: usize,
    /// The MIDI channel the pressure targets, in the spec's `0..=15`
    /// range (channel "1" in human-facing tools is index `0`). Decoded
    /// from the low nibble of the `An` status byte at parse time so the
    /// parser's running-status bookkeeping is already resolved by the
    /// time the event reaches the helper.
    pub channel: u8,
    /// The key `kk` the pressure targets, `0..=127` (same numbering as a
    /// Note On/Off key, where `60` is Middle C). The single wire data
    /// byte carries 7 significant bits (the high bit is the MIDI status
    /// flag the parser already rejected on a data byte), so the value
    /// never exceeds `0x7F`.
    pub key: u8,
    /// The pressure value `pp`, `0..=127`. The single wire data byte
    /// carries 7 significant bits (the high bit is the MIDI status flag
    /// the parser already rejected on a data byte), so the value never
    /// exceeds `0x7F`.
    pub pressure: u8,
}

impl PolyAftertouchEvent {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The key `kk` the pressure targets, `0..=127`.
    pub fn key(&self) -> u8 {
        self.key
    }

    /// The raw pressure value `pp`, `0..=127`.
    pub fn pressure(&self) -> u8 {
        self.pressure
    }
}

/// One Channel Pressure (mono aftertouch) channel-voice event pinned
/// to the absolute tick (relative to the start of its parent track) at
/// which the [`ChannelBody::ChannelAftertouch`] wire event fires.
///
/// Returned by [`SmfFile::channel_pressures`] — see that method for the
/// merge semantics across multiple tracks.
///
/// `Dn pp` is the Standard MIDI File §"Channel Voice Messages" §`Dn`
/// (status nibble `1101`) Channel Pressure message: the low nibble of
/// the status byte is the channel index (`0..=15`) and the single data
/// byte `pp` (`0..=127`) is the pressure value. Per the MIDI 1.0
/// *Summary of MIDI Messages* Table 1 this carries "the single greatest
/// pressure value (of all the current depressed keys)" — distinct from
/// polyphonic key pressure (`An`, per-key), which travels on its own
/// surface. It is most often generated by pressing harder on a key
/// after it bottoms out.
///
/// This helper isolates the channel-pressure stream so callers driving
/// an expression-curve view (a DAW pressure-lane editor, a soft-synth
/// channel-pressure state rebuilder) get a clean time-ordered list
/// independent of the surrounding control-change (`Bn`), patch-select
/// (`Cn`), pitch-bend (`En`), polyphonic-aftertouch (`An`), and note
/// (`8n` / `9n`) channel-voice streams. The pressure value's musical
/// effect (typically routed to volume, vibrato depth, or filter cutoff
/// by the receiving instrument) is the receiver's concern, so the
/// helper stays routing-agnostic and surfaces the raw `0..=127` byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelPressureEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the channel-voice event, in division units. For format-1 SMFs
    /// this is also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files commonly place a part's
    /// channel-pressure automation on the same track as its note-on /
    /// note-off events; format-2 files keep patterns separate so each
    /// pattern's pressure stream appears on its own track.
    pub track: usize,
    /// The MIDI channel the pressure targets, in the spec's `0..=15`
    /// range (channel "1" in human-facing tools is index `0`). Decoded
    /// from the low nibble of the `Dn` status byte at parse time so the
    /// parser's running-status bookkeeping is already resolved by the
    /// time the event reaches the helper.
    pub channel: u8,
    /// The pressure value `pp`, `0..=127`. The single wire data byte
    /// carries 7 significant bits (the high bit is the MIDI status flag
    /// the parser already rejected on a data byte), so the value never
    /// exceeds `0x7F`.
    pub pressure: u8,
}

impl ChannelPressureEvent {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The raw pressure value `pp`, `0..=127`.
    pub fn pressure(&self) -> u8 {
        self.pressure
    }
}

/// One sounding-note span — a Note On (`9n key vel`, `vel > 0`) matched
/// with the Note Off that releases it — pinned to the absolute ticks at
/// which the key is struck and released.
///
/// Returned by [`SmfFile::notes`]. Where the channel-voice helpers
/// ([`SmfFile::program_changes`], [`SmfFile::control_changes`],
/// [`SmfFile::pitch_bends`], [`SmfFile::channel_pressures`]) surface one
/// value per *wire* event, this helper pairs the two wire events that
/// bracket a sounding note into a single span carrying the note's
/// duration — the primitive a piano-roll / DAW note-lane view consumes
/// directly without re-deriving on/off pairing.
///
/// Per the MIDI 1.0 *Summary of MIDI Messages* Table 1, a Note On is
/// status nibble `1001` with data bytes `kkkkkkk` (key) + `vvvvvvv`
/// (velocity), and a Note Off is status nibble `1000` with the same two
/// data bytes. By the long-standing spec convention a Note On with
/// velocity `0` is treated as a Note Off (the "running-status
/// optimisation" that lets a stream of notes share one `9n` status
/// byte); [`SmfFile::notes`] honours that convention when matching, so a
/// `9n key 0` releases an open note exactly as an `8n key xx` would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Note {
    /// Absolute tick (cumulative delta-sum on the global merged
    /// timebase) at which the Note On fires. For format-1 SMFs every
    /// track shares one timebase, so this is directly comparable across
    /// tracks; for format-0 there is a single track.
    pub start_tick: u64,
    /// Absolute tick at which the matching Note Off fires.
    /// `end_tick >= start_tick` always holds (a Note Off can land on the
    /// same tick as its Note On — a zero-duration note — but never
    /// before it, since the pairing walks events in non-decreasing tick
    /// order).
    pub end_tick: u64,
    /// Index of the [`Track`] that carried the Note On (within
    /// [`SmfFile::tracks`]). The matching Note Off is conventionally on
    /// the same track, but the pairing is keyed on `(channel, key)` over
    /// the *globally* merged stream so a Note Off on a different track
    /// still closes the note; this field records where the note *began*.
    pub track: usize,
    /// The MIDI channel the note sounds on, in the spec's `0..=15` range
    /// (channel "1" in human-facing tools is index `0`). Decoded from
    /// the low nibble of the `9n` / `8n` status byte.
    pub channel: u8,
    /// The key (note) number `0..=127`, where `60` is Middle C. The
    /// single wire data byte carries 7 significant bits.
    pub key: u8,
    /// The Note On velocity `1..=127` — the attack strength. A velocity
    /// of `0` never appears here: a `9n key 0` is the Note-Off form and
    /// closes an open note rather than opening one.
    pub velocity: u8,
    /// The Note Off velocity `0..=127` — the release strength. An `8n`
    /// Note Off carries an explicit release velocity in its second data
    /// byte; a `9n key 0` Note-Off form has no release velocity, so this
    /// is `0` for notes closed by the velocity-0 convention. Most
    /// receivers ignore release velocity, but it is preserved here for
    /// callers that drive a release-sensitive instrument.
    pub off_velocity: u8,
}

impl Note {
    /// The MIDI channel index in the spec's `0..=15` range.
    pub fn channel(&self) -> u8 {
        self.channel
    }

    /// The key (note) number `0..=127`.
    pub fn key(&self) -> u8 {
        self.key
    }

    /// The Note On (attack) velocity `1..=127`.
    pub fn velocity(&self) -> u8 {
        self.velocity
    }

    /// The Note Off (release) velocity `0..=127` (`0` when the note was
    /// closed by the `9n key 0` velocity-0 convention).
    pub fn off_velocity(&self) -> u8 {
        self.off_velocity
    }

    /// The note's duration in division ticks: `end_tick - start_tick`.
    /// Always non-negative (the pairing closes a note only with a
    /// release at or after its onset). A zero-duration note is possible
    /// when a Note Off lands on the same tick as the Note On.
    pub fn duration_ticks(&self) -> u64 {
        self.end_tick - self.start_tick
    }
}

/// One System Exclusive event pinned to the absolute tick (relative
/// to the start of its parent track) at which the [`F0`](Event::Sysex)
/// (start) or [`F7`](Event::Sysex) (continuation / escape) wire event
/// fires.
///
/// Returned by [`SmfFile::sysex_events`] — see that method for the
/// merge semantics across multiple tracks.
///
/// The Standard MIDI File Specification 1.0 §"System Exclusive
/// Events" defines two flavours:
///
/// * **`F0 <varlen> <payload>`** — a complete or starting SysEx
///   message. By the spec's convention the trailing `F7` (when one
///   is present in the wire message) is included as the final byte
///   of `<payload>`; a payload missing the trailing `F7` indicates
///   a multi-packet message split with one or more `F7`-continuation
///   events following.
/// * **`F7 <varlen> <payload>`** — a continuation packet for a
///   previously-started `F0` message *or* an arbitrary escape
///   sequence whose payload is shipped verbatim to the wire (the
///   spec's "escaped" form, used for non-SysEx System messages a
///   sequencer wants to preserve).
///
/// This helper isolates both wire forms so callers driving an
/// external synthesiser or recovering manufacturer-specific payloads
/// get a clean time-ordered list independent of the channel-voice
/// and meta-event streams. The Universal Real-Time and Universal
/// Non-Real-Time families (`F0 7E …` and `F0 7F …`, defined in the
/// MIDI 1.0 *Universal System Exclusive Messages* document) travel
/// through this list verbatim; a caller routing by the universal
/// vocabulary should inspect [`SysExEvent::data`] directly
/// (typically `data[0]` to distinguish `0x7E` versus `0x7F` and the
/// SubID bytes that follow).
///
/// The payload bytes are surfaced verbatim — the parser does not
/// strip a trailing `F7` from an `F0` payload, so the [`data`] field
/// reproduces the on-the-wire bytes faithfully and a writer can
/// round-trip the helper output through [`SmfFile::to_bytes`]
/// without re-synthesising the SysEx framing.
///
/// [`data`]: SysExEvent::data
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SysExEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the SysEx event, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files commonly place the
    /// universal SysEx setup payloads (GM-on, Master Volume, etc.)
    /// on the conductor track (track 0); format-2 files keep them
    /// per-track.
    pub track: usize,
    /// `true` for the [`F7`](Event::Sysex) continuation / escape
    /// form, `false` for the [`F0`](Event::Sysex) start form. A
    /// caller reconstructing a multi-packet SysEx assembly walks the
    /// list with `is_escape == false` opening a fresh packet and
    /// each subsequent `is_escape == true` appending until the
    /// payload terminates with an `0xF7` end marker.
    pub is_escape: bool,
    /// The SysEx payload bytes, verbatim. For an `F0` start packet
    /// the trailing `F7` end marker (when present in the wire
    /// message) is included as the final byte of `data`; for a
    /// continuation / escape packet the bytes are shipped to the
    /// MIDI wire unchanged. Empty payloads (`F0 00` / `F7 00`) are
    /// surfaced as `data.is_empty()` rather than filtered out — the
    /// spec permits a zero-length packet.
    pub data: Vec<u8>,
}

/// Realm bit of a Universal System Exclusive packet — Non-Real-Time
/// (`0x7E`) or Real-Time (`0x7F`).
///
/// The leading payload byte of a Universal SysEx `F0` packet is one of
/// these two reserved manufacturer-ID slots, per the MIDI 1.0
/// *Universal System Exclusive Messages* document (Table 4). The two
/// realms partition the universal vocabulary by whether the receiving
/// device is expected to act *immediately* on receipt (`0x7F` —
/// Master Volume, MTC Quarter-Frame, MMC transport, Notation
/// Information) or whether the packet describes a *setup* / *bulk*
/// operation the device may process at its leisure (`0x7E` — Sample
/// Dump, General MIDI System On / Off, Identity Request, File Dump).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalRealm {
    /// `0x7E` — Universal Non-Real-Time. The MMA's Universal SysEx
    /// document calls this realm "Non-Real Time"; the receiving device
    /// may queue the message and process it when convenient (e.g.
    /// finishing a sample-dump assembly).
    NonRealTime,
    /// `0x7F` — Universal Real-Time. The receiving device is expected
    /// to act on the message immediately on arrival (e.g. updating
    /// Master Volume between two consecutive note-on events).
    RealTime,
}

/// Sub-ID #1 / Sub-ID #2 classification of a Universal System Exclusive
/// packet — the parsed category byte pair from Table 4 of the MIDI 1.0
/// *Universal System Exclusive Messages* document.
///
/// Returned by [`SysExEvent::universal_classification`] when the packet
/// is a Universal SysEx `F0 7E …` (Non-Real-Time) or `F0 7F …` (Real-
/// Time) message. The classification captures the realm, the Sub-ID #1
/// category, the Sub-ID #2 sub-category (when one is present in the
/// payload), and the device-id byte that precedes Sub-ID #1.
///
/// The classifier reads the well-known Sub-ID #1 values defined by the
/// MMA at the time of the round-246 trace of the doc. Values outside
/// the published vocabulary surface through the
/// [`UniversalSubId1::Other`] variant carrying the raw byte so callers
/// with deeper, more recent vocabulary can still route the packet.
///
/// Wire shape of a Universal SysEx `F0` packet:
///
/// ```text
///   F0 <realm> <device_id> <sub_id1> [<sub_id2> [..payload..]] F7
///   ^^                                                         ^^
///   start byte (eaten by the SMF parser; the [`SysExEvent::data`]
///   buffer starts at <realm>; the trailing F7 is preserved when
///   present)
/// ```
///
/// The classifier returns `None` for an `F7` continuation / escape
/// packet, for an `F0` packet whose leading byte is neither `0x7E`
/// nor `0x7F` (a manufacturer-prefixed packet — Roland `0x41`,
/// Yamaha `0x43`, …), and for any Universal packet truncated before
/// the Sub-ID #1 byte (a payload shorter than three bytes — realm,
/// device-id, sub-id1 are the minimum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniversalSysEx {
    /// Non-Real-Time (`0x7E`) versus Real-Time (`0x7F`).
    pub realm: UniversalRealm,
    /// The device-id byte that precedes Sub-ID #1 in the wire packet.
    /// `0x7F` is the broadcast target (every receiver); other values
    /// `0x00..=0x7E` address a specific device.
    pub device_id: u8,
    /// The parsed Sub-ID #1 category.
    pub sub_id1: UniversalSubId1,
}

/// A Universal System Exclusive packet pinned to its absolute tick on
/// the SMF timeline, returned by [`SmfFile::universal_sysex_events`].
///
/// This is the typed, Table-4-classified counterpart of the verbatim
/// [`SysExEvent`] stream returned by [`SmfFile::sysex_events`]: the
/// helper walks every `F0` SysEx packet on every track, calls
/// [`SysExEvent::universal_classification`] on each, keeps only the
/// packets that classify as Universal Non-Real-Time (`0x7E`) or
/// Universal Real-Time (`0x7F`), and surfaces them with the parsed
/// [`UniversalSysEx`] alongside the verbatim payload bytes so a
/// caller can route by the category enum without re-classifying.
///
/// Manufacturer-prefixed `F0` packets (Roland `0x41`, Yamaha `0x43`,
/// any other manufacturer-ID byte) are filtered out — callers
/// interested in those route through [`SmfFile::sysex_events`] and
/// [`SysExEvent::manufacturer_id`] instead. `F7` continuation /
/// escape packets are also filtered out — they are opaque payloads
/// shipped to the wire as part of a manufacturer-specific multi-packet
/// assembly that the opening `F0` declared, and the universal-only
/// view stays at one entry per logical universal message.
///
/// Per-track sequences are stably merged by absolute tick — track 0's
/// universal packets fire before track 1's at the same tick — the
/// same convention used by [`SmfFile::sysex_events`] and every meta-
/// event iteration helper.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniversalSysExEvent {
    /// Cumulative delta-sum from the start of the track that carried
    /// the SysEx packet, in division units. For format-1 SMFs this is
    /// also the absolute tick on the shared timebase.
    pub tick: u64,
    /// Index of the [`Track`] the event came from (within
    /// [`SmfFile::tracks`]). Format-1 files conventionally place GM-On
    /// / Master Volume / Master Tuning on the conductor track (track 0).
    pub track: usize,
    /// The Table-4 classification of this packet — realm, device-id,
    /// Sub-ID #1, and the Sub-ID #2 sub-category when one is present
    /// in the payload.
    pub classification: UniversalSysEx,
    /// The verbatim SysEx payload bytes from the wire — the same
    /// bytes [`SysExEvent::data`] would surface for this packet. The
    /// leading `<realm>` byte (`0x7E` / `0x7F`) is included; the
    /// trailing `0xF7` end-of-exclusive marker is included when
    /// present in the source file. Callers reading Sub-ID #2-derived
    /// arguments (Master Volume's 14-bit value, MTC Full Message's
    /// `hr/mn/se/fr` quartet, MTS Single Note Tuning's note + tuning
    /// triple, …) index into this buffer starting at the byte after
    /// Sub-ID #2 — see Table 4 for each category's wire shape.
    pub data: Vec<u8>,
}

/// A decoded MIDI Time Code **Full Message** — the single-packet SMPTE
/// time an MTC generator sends when a system is located or cued to an
/// absolute position (fast-forward / rewind / autolocate), rather than
/// streamed two-frames-at-a-time as Quarter-Frame messages.
///
/// Wire shape (RP-004/008 §"Full Message", 10 bytes):
///
/// ```text
/// F0 7F <device ID> 01 01 hr mn sc fr F7
///   hr = 0 yy zzzzz   yy = frame-rate type (bits 5-6); zzzzz = hours 0..=23
///   mn = minutes 0..=59
///   sc = seconds 0..=59
///   fr = frames  0..=29 (per the type's nominal rate)
/// ```
///
/// The packet is a Real-Time (`0x7F`) Universal SysEx with Sub-ID #1
/// `0x01` (MIDI Time Code) and Sub-ID #2 `0x01` (Full Message). Decoded
/// via [`UniversalSysExEvent::mtc_full_message`].
///
/// The `hr` byte packs the frame-rate type in bits 5-6 exactly as the
/// `FF 54` SMPTE Offset meta event does, so the rate is decoded through
/// the shared [`FrameRate::from_hours_byte`]. Bit 7 of `hr` is reserved
/// and ignored by receivers per the spec; the raw byte is preserved in
/// [`hours_raw`](Self::hours_raw) for callers that need it.
///
/// Counter fields are surfaced verbatim from the wire — the decoder does
/// not clamp to the per-rate legal ranges, so a pathological generator
/// that emits an out-of-range frame number stays visible to the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtcFullMessage {
    /// Raw `hr` byte. Bits 5-6 hold the frame-rate type (`yy`); bits
    /// 0-4 hold the hours count (`0..=23`); bit 7 is reserved and
    /// ignored by receivers per RP-004/008.
    pub hours_raw: u8,
    /// Minutes count (`0..=59` per spec; preserved verbatim).
    pub minutes: u8,
    /// Seconds count (`0..=59` per spec; preserved verbatim).
    pub seconds: u8,
    /// Frames count (`0..=29` for 30 Hz; `0..=24` for 25 Hz; `0..=23`
    /// for 24 Hz — preserved verbatim).
    pub frames: u8,
}

impl MtcFullMessage {
    /// Hours count (`0..=23`), the low five bits of the `hr` byte.
    pub fn hours_count(&self) -> u8 {
        self.hours_raw & 0b0001_1111
    }

    /// Decoded SMPTE frame rate from bits 5-6 of the `hr` byte.
    pub fn frame_rate(&self) -> FrameRate {
        FrameRate::from_hours_byte(self.hours_raw)
    }
}

/// A decoded MIDI Time Code **User Bits Message** — the 32 SMPTE/EBU
/// user-bits an MTC generator transfers down the line, plus the two
/// Binary Group Flag Bits.
///
/// SMPTE provides 32 "user bits" for application-defined data (date
/// code, reel number, …) that persist throughout a run of time code.
/// The message ships them as eight 4-bit Binary Groups (each in the
/// low nibble of a payload byte) plus a ninth byte holding two flag
/// bits.
///
/// Wire shape (RP-004/008 §"User Bits", 15 bytes):
///
/// ```text
/// F0 7F <device ID> 01 02 u1 u2 u3 u4 u5 u6 u7 u8 u9 F7
///   u1 = 0000aaaa   Binary Group 1 (low nibble)
///   u2 = 0000bbbb   Binary Group 2
///   u3 = 0000cccc   Binary Group 3
///   u4 = 0000dddd   Binary Group 4
///   u5 = 0000eeee   Binary Group 5
///   u6 = 0000ffff   Binary Group 6
///   u7 = 0000gggg   Binary Group 7
///   u8 = 0000hhhh   Binary Group 8
///   u9 = 000000ji   Binary Group Flag Bits (j = SMPTE bit 59 / EBU
///                   bit 43; i = SMPTE bit 43 / EBU bit 27)
/// ```
///
/// The packet is a Real-Time (`0x7F`) Universal SysEx with Sub-ID #1
/// `0x01` (MIDI Time Code) and Sub-ID #2 `0x02` (User Bits Message).
/// Decoded via [`UniversalSysExEvent::mtc_user_bits`].
///
/// Per the November-1991 redefinition noted in RP-004/008, when the
/// eight Binary Group nibbles carry 8-bit characters they reassemble
/// as four bytes in the order `hhhhgggg ffffeeee ddddcccc bbbbaaaa`;
/// [`reassembled`](Self::reassembled) returns exactly that 32-bit
/// value. The flag bits are surfaced individually via [`flag_i`] /
/// [`flag_j`] and the raw `u9` byte is preserved in
/// [`flag_byte`](Self::flag_byte).
///
/// Group nibbles are surfaced verbatim in [`groups`](Self::groups) —
/// the decoder masks each to its low four bits but otherwise does not
/// interpret BCD vs. 8-bit-character usage, leaving that to the caller.
///
/// [`flag_i`]: Self::flag_i
/// [`flag_j`]: Self::flag_j
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtcUserBits {
    /// The eight Binary Groups 1..=8, each the low four bits of its
    /// payload byte (`a` = `groups[0]` … `h` = `groups[7]`).
    pub groups: [u8; 8],
    /// Raw `u9` byte holding the two Binary Group Flag Bits in bits
    /// 0-1 (`i` = bit 0, `j` = bit 1); bits 2-7 are reserved.
    pub flag_byte: u8,
}

impl MtcUserBits {
    /// The `i` Binary Group Flag Bit (bit 0 of `u9`) — SMPTE time code
    /// bit 43 / EBU bit 27.
    pub fn flag_i(&self) -> bool {
        self.flag_byte & 0b0000_0001 != 0
    }

    /// The `j` Binary Group Flag Bit (bit 1 of `u9`) — SMPTE time code
    /// bit 59 / EBU bit 43.
    pub fn flag_j(&self) -> bool {
        self.flag_byte & 0b0000_0010 != 0
    }

    /// The 32-bit value the eight Binary Groups reassemble into when
    /// they carry 8-bit characters, per the November-1991 RP-004/008
    /// redefinition: byte order `hhhhgggg ffffeeee ddddcccc bbbbaaaa`,
    /// i.e. Group 1 (`a`) occupies the least-significant nibble and
    /// Group 8 (`h`) the most-significant nibble.
    pub fn reassembled(&self) -> u32 {
        let mut value = 0u32;
        for (i, &g) in self.groups.iter().enumerate() {
            value |= ((g & 0x0F) as u32) << (4 * i);
        }
        value
    }
}

/// The semantic state a Notation **Bar Number** message conveys, decoded
/// from its signed 14-bit `aa aa` (lsb-first) field per the MIDI 1.0
/// Detailed Specification §"Notation Information — Bar Marker".
///
/// The numbering system reserves the two extreme values as flags and
/// uses negative numbers for the count-in that precedes bar 1:
///
/// ```text
///   raw 14-bit value (two's-complement over 14 bits)
///     0x2000 (-8192)            NotRunning
///     0x2001 (-8191) ..= 0x0000 CountIn(-8191 ..= 0)
///     0x0001 (1)     ..= 0x1F7E (8062) BarInSong(1 ..= 8062)
///     0x1F7F (8063)             RunningUnknown
/// ```
///
/// The "last bar of count-in" is bar `0`; bar `1` is the first bar of
/// the song proper. Systems with a single count-in bar can simply count
/// from `0` upward and never see a negative number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotationBarState {
    /// `0x2000` (raw −8192) — the transport is stopped / not running.
    NotRunning,
    /// `0x2001 ..= 0x0000` (raw −8191 ..= 0) — a count-in bar. The
    /// payload is the signed bar number (negative until it reaches `0`,
    /// the last bar of count-in).
    CountIn(i16),
    /// `0x0001 ..= 0x1F7E` (1 ..= 8062) — a numbered bar within the
    /// song. Bar `1` is the first bar of the song proper.
    BarInSong(i16),
    /// `0x1F7F` (8063) — running, but the bar number is unknown (or it
    /// has exceeded the representable 8K range).
    RunningUnknown,
}

/// A decoded Notation **Bar Number** message — the high-level
/// measure-boundary marker an MTC/synchronisation source emits so a
/// display device can show the current bar without counting MIDI clocks.
///
/// Wire shape (MIDI 1.0 Detailed Specification §"Notation Information —
/// Bar Marker", 8 bytes):
///
/// ```text
/// F0 7F <device ID> 03 01 aa aa F7
///   aa aa = bar number, lsb first, signed over 14 bits
/// ```
///
/// The packet is a Real-Time (`0x7F`) Universal SysEx with Sub-ID #1
/// `0x03` (Notation Information) and Sub-ID #2 `0x01` (Bar Number).
/// Decoded via [`UniversalSysExEvent::notation_bar_number`].
///
/// The two `aa` bytes are each 7-bit data bytes; they pack lsb-first
/// into a 14-bit field that is then interpreted as a signed value
/// (sign-extended from bit 13). [`raw14`](Self::raw14) preserves the
/// unsigned 14-bit value and [`value`](Self::value) the sign-extended
/// `i16`; [`state`](Self::state) classifies it into the four documented
/// regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotationBarNumber {
    /// The unsigned 14-bit field assembled lsb-first from `aa aa`
    /// (`0x0000 ..= 0x3FFF`).
    pub raw14: u16,
}

impl NotationBarNumber {
    /// The sign-extended signed bar number (`−8192 ..= 8191`). Bit 13 of
    /// the 14-bit field is the sign bit.
    pub fn value(&self) -> i16 {
        // Sign-extend bit 13 into a full i16. Work in i32 to avoid the
        // intermediate overflow that `(x as i16) << 2` would hit for
        // values whose bit 13 is set (e.g. raw 0x2000 = 8192).
        let v = (self.raw14 & 0x3FFF) as i32;
        ((v << 18) >> 18) as i16
    }

    /// Classify the bar number into its documented semantic region:
    /// `0x2000` → [`NotationBarState::NotRunning`], the
    /// negative-through-zero count-in range, the positive in-song range,
    /// and the `0x1F7F` running-but-unknown flag.
    pub fn state(&self) -> NotationBarState {
        match self.raw14 & 0x3FFF {
            0x2000 => NotationBarState::NotRunning,
            0x1F7F => NotationBarState::RunningUnknown,
            _ => {
                let v = self.value();
                if v <= 0 {
                    NotationBarState::CountIn(v)
                } else {
                    NotationBarState::BarInSong(v)
                }
            }
        }
    }
}

/// A single time-signature pair within a Notation **Time Signature**
/// message — the `nn dd cc bb` quartet whose first two bytes mirror the
/// `FF 58` Standard MIDI File Time Signature meta event.
///
/// In a compound time signature several pairs follow the leading
/// `cc bb` metronome bytes; only the first pair carries `cc` / `bb`, so
/// the additional pairs surface here with `numerator` / `denominator`
/// only (their `clocks_per_click` / `thirty_seconds_per_quarter` echo
/// the leading pair's values for convenience).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotationTimeSignaturePair {
    /// `nn` — number of beats (the numerator).
    pub numerator: u8,
    /// `dd` — beat value as a negative power of two (the denominator is
    /// `1 << dd`); decode via [`denominator`](Self::denominator).
    pub denominator_pow2: u8,
}

impl NotationTimeSignaturePair {
    /// The decoded denominator, `1 << denominator_pow2` (e.g.
    /// `denominator_pow2 = 3` → `8`, a `/8` signature). Saturates the
    /// shift amount at 31 to avoid overflow on a pathological byte.
    pub fn denominator(&self) -> u32 {
        1u32 << (self.denominator_pow2.min(31))
    }
}

/// A decoded Notation **Time Signature** message — Immediate (`0x02`,
/// effective on receipt) or Delayed (`0x42`, effective on the next Bar
/// Marker). The body mirrors the `FF 58` Standard MIDI File Time
/// Signature meta event, extended with optional compound-signature
/// pairs.
///
/// Wire shape (MIDI 1.0 Detailed Specification §"Notation Information —
/// Time Signature"):
///
/// ```text
/// F0 7F <device ID> 03 02 ln nn dd cc bb [nn dd ...] F7   (Immediate)
/// F0 7F <device ID> 03 42 ln nn dd cc bb [nn dd ...] F7   (Delayed)
///   ln       number of data bytes that follow (4 for a simple meter;
///            +2 per extra compound pair)
///   nn       numerator (beats)
///   dd       denominator as a negative power of two (1 << dd)
///   cc       MIDI clocks per metronome click
///   bb       notated 32nd notes per MIDI quarter note
///   [nn dd]  additional pairs for a compound meter within the bar
/// ```
///
/// The packet is a Real-Time (`0x7F`) Universal SysEx with Sub-ID #1
/// `0x03` (Notation Information) and Sub-ID #2 `0x02` / `0x42`. Decoded
/// via [`UniversalSysExEvent::notation_time_signature`]; the
/// Immediate-versus-Delayed distinction is read from the classified
/// Sub-ID #2 and surfaced through [`is_delayed`](Self::is_delayed).
///
/// `clocks_per_click` and `thirty_seconds_per_quarter` are the leading
/// pair's metronome bytes — the four-byte form duplicates the `FF 58`
/// meta-event layout exactly. The first signature pair is always
/// [`pairs`](Self::pairs)`[0]`; compound meters append further pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotationTimeSignature {
    /// `true` for the Delayed form (Sub-ID #2 `0x42`, effective on the
    /// next Bar Marker); `false` for the Immediate form (`0x02`).
    pub delayed: bool,
    /// The signature pairs, in wire order. Always at least one; a
    /// compound meter carries two or more.
    pub pairs: Vec<NotationTimeSignaturePair>,
    /// `cc` — MIDI clocks per metronome click (from the leading pair).
    pub clocks_per_click: u8,
    /// `bb` — notated 32nd notes per MIDI quarter note (from the leading
    /// pair).
    pub thirty_seconds_per_quarter: u8,
}

impl NotationTimeSignature {
    /// `true` for the Delayed form, `false` for the Immediate form.
    pub fn is_delayed(&self) -> bool {
        self.delayed
    }

    /// The leading (and, for a simple meter, only) signature pair.
    pub fn primary(&self) -> NotationTimeSignaturePair {
        self.pairs[0]
    }

    /// `true` when the message carries more than one signature pair (a
    /// compound meter such as `4/4 + 3/8` within the same bar).
    pub fn is_compound(&self) -> bool {
        self.pairs.len() > 1
    }
}

/// A decoded **Device Control** Universal Real-Time SysEx message — the
/// `0x7F <device ID> 04 <nn> …` family that sets a receiver's global
/// (non-channel) state. Decoded via [`UniversalSysExEvent::device_control`].
///
/// Table 4 (MIDI 1.0 *Universal System Exclusive Messages*) assigns
/// Sub-ID #1 `0x04` to Device Control and lists five Sub-ID #2 members:
///
/// ```text
/// F0 7F <dev> 04 01 lsb msb F7   Master Volume   (14-bit, lsb first)
/// F0 7F <dev> 04 02 lsb msb F7   Master Balance  (14-bit, lsb first)
/// F0 7F <dev> 04 03 lsb msb F7   Master Fine Tuning   (CA-025)
/// F0 7F <dev> 04 04 00  msb F7   Master Coarse Tuning (CA-025)
/// F0 7F <dev> 04 05 …       F7   Global Parameter Control (CA-024)
/// ```
///
/// Master Volume / Balance pack a 14-bit value lsb-first per the MIDI
/// 1.0 Detailed Specification §"DEVICE CONTROL — MASTER VOLUME AND
/// MASTER BALANCE": `0x0000` = minimum / hard-left, `0x3FFF` = maximum /
/// hard-right, `0x2000` = centre (balance). The two tuning messages
/// follow CA-025 (Master Fine/Coarse Tuning); their semantics are
/// decoded into signed cents / semitones by [`DeviceControl`]'s helper
/// methods. Global Parameter Control (`0x05`) carries a
/// variable-length slot-path + parameter body (CA-024) that this
/// classifier surfaces as the [`GlobalParameterControl`] variant with
/// the raw body for caller parsing — the scheduler's GM2 effect bus
/// owns the detailed CA-024 walk.
///
/// [`GlobalParameterControl`]: DeviceControl::GlobalParameterControl
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceControl {
    /// `04 01 lsb msb` — Master Volume. A 14-bit unsigned level
    /// (`0..=0x3FFF`); `0x0000` = silence, `0x3FFF` = full scale.
    MasterVolume {
        /// The reassembled 14-bit value (`(msb << 7) | lsb`).
        value14: u16,
    },
    /// `04 02 lsb msb` — Master Balance. A 14-bit value where
    /// `0x0000` = hard left, `0x2000` = centre, `0x3FFF` = hard right
    /// (M1 v4.2.1 §"DEVICE CONTROL — MASTER VOLUME AND MASTER BALANCE").
    MasterBalance {
        /// The reassembled 14-bit value (`(msb << 7) | lsb`).
        value14: u16,
    },
    /// `04 03 lsb msb` — Master Fine Tuning (CA-025). A 14-bit value,
    /// lsb-first; `0x2000` = no displacement. The displacement in cents
    /// from A440 is `100/8192 × (value14 − 8192)`, spanning roughly
    /// `−100 … +100` cents over the full `0x0000 … 0x3FFF` range. Decode
    /// via [`fine_tuning_cents`](DeviceControl::fine_tuning_cents).
    MasterFineTuning {
        /// The reassembled 14-bit value (`(msb << 7) | lsb`).
        value14: u16,
    },
    /// `04 04 00 msb` — Master Coarse Tuning (CA-025). The LSB is always
    /// `0`; the MSB carries the semitone displacement with `0x40` = no
    /// change, spanning `−64 … +63` semitones. Decode via
    /// [`coarse_tuning_semitones`](DeviceControl::coarse_tuning_semitones).
    MasterCoarseTuning {
        /// Raw `msb` (semitone) byte; `0x40` = no displacement.
        msb: u8,
        /// Raw `lsb` byte; the spec fixes this at `0x00`.
        lsb: u8,
    },
    /// `04 05 …` — Global Parameter Control (CA-024). The variable-length
    /// slot-path + parameter-value body is surfaced verbatim (the bytes
    /// after Sub-ID #2, trailing `F7` excluded) for callers that walk the
    /// CA-024 structure themselves.
    GlobalParameterControl {
        /// The CA-024 body bytes following Sub-ID #2 (slot-path-length,
        /// param-width, value-width, then the slot / parameter / value
        /// fields). Empty when the packet carried no body.
        body: Vec<u8>,
    },
}

impl DeviceControl {
    /// For [`MasterFineTuning`](DeviceControl::MasterFineTuning), the
    /// displacement in cents from A440: `100/8192 × (value14 − 8192)`
    /// per CA-025. Returns `None` for any other variant.
    ///
    /// `0x2000` → `0.0` cents; `0x0000` → `−100.0`; `0x3FFF` →
    /// `+99.987…` (the CA-025 table's `100/8192 × (+8191)`).
    pub fn fine_tuning_cents(&self) -> Option<f64> {
        match self {
            DeviceControl::MasterFineTuning { value14 } => {
                Some(100.0 / 8192.0 * (*value14 as f64 - 8192.0))
            }
            _ => None,
        }
    }

    /// For [`MasterCoarseTuning`](DeviceControl::MasterCoarseTuning), the
    /// signed semitone displacement from A440 per CA-025: `msb − 64`,
    /// spanning `−64 … +63`. Returns `None` for any other variant.
    pub fn coarse_tuning_semitones(&self) -> Option<i8> {
        match self {
            DeviceControl::MasterCoarseTuning { msb, .. } => Some((*msb & 0x7F) as i8 - 64),
            _ => None,
        }
    }
}

impl UniversalSysExEvent {
    /// Decode this packet as a Notation **Bar Number** message when it
    /// is one, returning the signed 14-bit bar number and its semantic
    /// classification (not-running / count-in / in-song / running-
    /// unknown).
    ///
    /// Returns `None` unless the packet classifies as a Real-Time
    /// (`0x7F`) Notation Information Bar Number
    /// ([`UniversalSubId2::RtNotationBarNumber`]) *and* the verbatim
    /// payload carries both `aa` bytes after Sub-ID #2. A correctly-
    /// classified packet truncated before the second `aa` byte yields
    /// `None` rather than a half-assembled value.
    ///
    /// Payload indexing follows the [`data`](Self::data) layout
    /// `<realm> <device_id> <sub_id1> <sub_id2> aa_lsb aa_msb [F7]`, so
    /// the two data bytes begin at index 4 and pack lsb-first into the
    /// 14-bit field. High bits beyond bit 6 of each `aa` byte are
    /// masked off (they are 7-bit data bytes on the wire).
    pub fn notation_bar_number(&self) -> Option<NotationBarNumber> {
        if !matches!(
            self.classification.sub_id1,
            UniversalSubId1::NotationInformation(UniversalSubId2::RtNotationBarNumber)
        ) {
            return None;
        }
        // <realm> <device_id> <sub_id1> <sub_id2> aa_lsb aa_msb ...
        let lsb = (*self.data.get(4)? & 0x7F) as u16;
        let msb = (*self.data.get(5)? & 0x7F) as u16;
        Some(NotationBarNumber {
            raw14: lsb | (msb << 7),
        })
    }

    /// Decode this packet as a Notation **Time Signature** message
    /// (Immediate or Delayed) when it is one, returning every
    /// `numerator / denominator` pair plus the leading metronome bytes.
    ///
    /// Returns `None` unless the packet classifies as a Real-Time
    /// (`0x7F`) Notation Information Time Signature — Immediate
    /// ([`UniversalSubId2::RtNotationTimeSignatureImmediate`]) or
    /// Delayed ([`UniversalSubId2::RtNotationTimeSignatureDelayed`]) —
    /// *and* the verbatim payload carries the declared `ln` data bytes
    /// after Sub-ID #2. A packet whose `ln` is not a valid pair-count
    /// (less than 4, or not of the form `4 + 2k`) or whose body is
    /// truncated before `ln` bytes arrive yields `None`.
    ///
    /// Payload indexing follows the [`data`](Self::data) layout
    /// `<realm> <device_id> <sub_id1> <sub_id2> ln nn dd cc bb [nn dd ...]
    /// [F7]`: `ln` is at index 4, the leading `nn dd cc bb` quartet at
    /// indices 5..9, and each additional compound pair `nn dd` follows.
    /// The four-byte form duplicates the `FF 58` meta-event layout.
    pub fn notation_time_signature(&self) -> Option<NotationTimeSignature> {
        let delayed = match self.classification.sub_id1 {
            UniversalSubId1::NotationInformation(
                UniversalSubId2::RtNotationTimeSignatureImmediate,
            ) => false,
            UniversalSubId1::NotationInformation(
                UniversalSubId2::RtNotationTimeSignatureDelayed,
            ) => true,
            _ => return None,
        };
        // <realm> <device_id> <sub_id1> <sub_id2> ln nn dd cc bb ...
        let ln = *self.data.get(4)? as usize;
        // The minimum legal body is the 4-byte simple meter; compound
        // meters add 2 bytes per extra pair, so ln must be 4, 6, 8, ...
        if ln < 4 || (ln - 4) % 2 != 0 {
            return None;
        }
        // The leading quartet is nn dd cc bb at indices 5..9.
        let nn = *self.data.get(5)?;
        let dd = *self.data.get(6)?;
        let cc = *self.data.get(7)?;
        let bb = *self.data.get(8)?;
        let mut pairs = vec![NotationTimeSignaturePair {
            numerator: nn,
            denominator_pow2: dd,
        }];
        // Additional compound pairs are nn dd from index 9 onward; ln - 4
        // is the extra byte count (always even per the check above).
        let extra_pairs = (ln - 4) / 2;
        for k in 0..extra_pairs {
            let base = 9 + k * 2;
            let nn = *self.data.get(base)?;
            let dd = *self.data.get(base + 1)?;
            pairs.push(NotationTimeSignaturePair {
                numerator: nn,
                denominator_pow2: dd,
            });
        }
        Some(NotationTimeSignature {
            delayed,
            pairs,
            clocks_per_click: cc,
            thirty_seconds_per_quarter: bb,
        })
    }

    /// Decode this packet as an MTC **Full Message** when it is one,
    /// returning the SMPTE `hr / mn / sc / fr` quartet (and, via
    /// [`MtcFullMessage`], the decoded frame rate and hours count).
    ///
    /// Returns `None` unless the packet classifies as a Real-Time
    /// (`0x7F`) MIDI Time Code Full Message
    /// ([`UniversalSubId2::RtMtcFullMessage`]) *and* the verbatim
    /// payload carries the full four-byte time quartet after Sub-ID #2.
    /// A correctly-classified packet that is truncated before all four
    /// time bytes arrive (a malformed stream) yields `None` rather than
    /// a partially-decoded time.
    ///
    /// Payload indexing follows the [`data`](Self::data) layout
    /// `<realm> <device_id> <sub_id1> <sub_id2> hr mn sc fr [F7]`, so
    /// the quartet begins at byte index 4.
    pub fn mtc_full_message(&self) -> Option<MtcFullMessage> {
        if !matches!(
            self.classification.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::RtMtcFullMessage)
        ) {
            return None;
        }
        // <realm> <device_id> <sub_id1> <sub_id2> hr mn sc fr ...
        let hr = *self.data.get(4)?;
        let mn = *self.data.get(5)?;
        let sc = *self.data.get(6)?;
        let fr = *self.data.get(7)?;
        Some(MtcFullMessage {
            hours_raw: hr,
            minutes: mn,
            seconds: sc,
            frames: fr,
        })
    }

    /// Decode this packet as an MTC **User Bits Message** when it is
    /// one, returning the eight SMPTE/EBU Binary Groups and the Binary
    /// Group Flag Bits (and, via [`MtcUserBits`], the reassembled
    /// 32-bit value and the individual flag bits).
    ///
    /// Returns `None` unless the packet classifies as a Real-Time
    /// (`0x7F`) MIDI Time Code User Bits Message
    /// ([`UniversalSubId2::RtMtcUserBits`]) *and* the verbatim payload
    /// carries all nine `u1..u9` bytes after Sub-ID #2. A correctly-
    /// classified packet that is truncated before all nine bytes arrive
    /// (a malformed stream) yields `None` rather than partially-decoded
    /// groups.
    ///
    /// Payload indexing follows the [`data`](Self::data) layout
    /// `<realm> <device_id> <sub_id1> <sub_id2> u1..u9 [F7]`, so the
    /// nine user-bits bytes begin at byte index 4. Each group nibble is
    /// masked to its low four bits; reserved high bits in `u1..u8` are
    /// discarded, matching the `0000xxxx` wire shape.
    pub fn mtc_user_bits(&self) -> Option<MtcUserBits> {
        if !matches!(
            self.classification.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::RtMtcUserBits)
        ) {
            return None;
        }
        // <realm> <device_id> <sub_id1> <sub_id2> u1..u9 ...
        let mut groups = [0u8; 8];
        for (i, slot) in groups.iter_mut().enumerate() {
            *slot = *self.data.get(4 + i)? & 0x0F;
        }
        let flag_byte = *self.data.get(12)?;
        Some(MtcUserBits { groups, flag_byte })
    }

    /// Decode this packet as a **Device Control** message when it is one,
    /// returning the typed [`DeviceControl`] variant for Master Volume,
    /// Master Balance, Master Fine / Coarse Tuning, or Global Parameter
    /// Control.
    ///
    /// Returns `None` unless the packet classifies as a Real-Time
    /// (`0x7F`) Device Control message ([`UniversalSubId1::DeviceControl`])
    /// *and* the verbatim payload carries the variant's declared bytes
    /// after Sub-ID #2. A correctly-classified packet truncated before
    /// the value pair arrives yields `None` rather than a partial value.
    ///
    /// Payload indexing follows the [`data`](Self::data) layout
    /// `<realm> <device_id> <sub_id1> <sub_id2> lsb msb [F7]`, so the two
    /// data bytes begin at index 4 (Volume / Balance / Fine: 14-bit
    /// lsb-first; Coarse: `00 msb`). Global Parameter Control surfaces
    /// the body bytes from index 4 onward, with a trailing `0xF7`
    /// end-of-exclusive marker stripped when present.
    pub fn device_control(&self) -> Option<DeviceControl> {
        let sub_id2 = match self.classification.sub_id1 {
            UniversalSubId1::DeviceControl(s) => s,
            _ => return None,
        };
        // <realm> <device_id> <sub_id1> <sub_id2> lsb msb ...
        let value14 = |this: &Self| -> Option<u16> {
            let lsb = (*this.data.get(4)? & 0x7F) as u16;
            let msb = (*this.data.get(5)? & 0x7F) as u16;
            Some(lsb | (msb << 7))
        };
        match sub_id2 {
            UniversalSubId2::DeviceControlMasterVolume => Some(DeviceControl::MasterVolume {
                value14: value14(self)?,
            }),
            UniversalSubId2::DeviceControlMasterBalance => Some(DeviceControl::MasterBalance {
                value14: value14(self)?,
            }),
            UniversalSubId2::DeviceControlMasterFineTuning => {
                Some(DeviceControl::MasterFineTuning {
                    value14: value14(self)?,
                })
            }
            UniversalSubId2::DeviceControlMasterCoarseTuning => {
                let lsb = *self.data.get(4)? & 0x7F;
                let msb = *self.data.get(5)? & 0x7F;
                Some(DeviceControl::MasterCoarseTuning { msb, lsb })
            }
            UniversalSubId2::DeviceControlGlobalParameterControl => {
                // Body is everything after Sub-ID #2 (index 4 onward),
                // minus a trailing EOX marker when the source file kept it.
                let mut body: Vec<u8> = self.data.get(4..).unwrap_or(&[]).to_vec();
                if body.last() == Some(&0xF7) {
                    body.pop();
                }
                Some(DeviceControl::GlobalParameterControl { body })
            }
            _ => None,
        }
    }
}

/// Sub-ID #1 category of a Universal System Exclusive packet — the
/// `<sub_id1>` byte in the `F0 <realm> <device_id> <sub_id1> …` wire
/// shape, decoded against Table 4 of the MIDI 1.0 *Universal System
/// Exclusive Messages* document.
///
/// Each variant pairs the category name (Sub-ID #1) with the parsed
/// sub-category (Sub-ID #2, when one is present). Categories that
/// carry a fixed singleton message (`End of File`, `Wait`, …) are
/// modelled as a unit variant; categories that branch on Sub-ID #2 are
/// modelled as a variant with a [`UniversalSubId2`] field.
///
/// Unknown / future Sub-ID #1 values surface through the [`Other`]
/// variant carrying the raw byte so callers can route the packet to a
/// fallback handler.
///
/// [`Other`]: UniversalSubId1::Other
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalSubId1 {
    /// `0x01` — Sample Dump Header (Non-Real-Time only). Sub-ID #2 is
    /// absent in the Table 4 listing for this category.
    SampleDumpHeader,
    /// `0x02` — Sample Data Packet (Non-Real-Time only).
    SampleDataPacket,
    /// `0x03` — Sample Dump Request (Non-Real-Time only).
    SampleDumpRequest,
    /// `0x04 <nn>` — MIDI Time Code. The Sub-ID #2 byte further
    /// classifies the message; in the Non-Real-Time realm `0x04 nn`
    /// is the MTC Setup family; in the Real-Time realm `0x04 nn` is
    /// the MTC Quarter-Frame / Full / User Bits family. The
    /// classifier surfaces the raw `nn` so callers can split the
    /// realms themselves.
    MidiTimeCode(UniversalSubId2),
    /// `0x05 <nn>` — Sample Dump Extensions (Non-Real-Time) or
    /// Real-Time MTC Cueing (Real-Time). See [`UniversalSubId2`] for
    /// the per-realm Sub-ID #2 mapping.
    SampleDumpExtensionsOrMtcCueing(UniversalSubId2),
    /// `0x06 <nn>` — General Information (Non-Real-Time) or MMC
    /// Commands (Real-Time).
    GeneralInformationOrMmcCommands(UniversalSubId2),
    /// `0x07 <nn>` — File Dump (Non-Real-Time) or MMC Responses
    /// (Real-Time).
    FileDumpOrMmcResponses(UniversalSubId2),
    /// `0x08 <nn>` — MIDI Tuning Standard. The same Sub-ID #1 appears
    /// in both realms; Sub-ID #2 distinguishes the Bulk Dump
    /// Request / Reply / Single-Note / Scale-Octave variants per
    /// Table 4.
    MidiTuningStandard(UniversalSubId2),
    /// `0x09 <nn>` — General MIDI (Non-Real-Time `01`/`02`/`03` =
    /// GM 1 On / GM Off / GM 2 On) or Controller Destination Setting
    /// (Real-Time `01`/`02`/`03` = Channel Pressure / Polyphonic Key
    /// Pressure / Control Change).
    GeneralMidiOrControllerDestination(UniversalSubId2),
    /// `0x0A <nn>` — Downloadable Sounds (Non-Real-Time) or Key-Based
    /// Instrument Control (Real-Time, sub-id2 `0x01`).
    DownloadableSoundsOrKeyBasedInstrumentControl(UniversalSubId2),
    /// `0x0B <nn>` — File Reference Message (Non-Real-Time) or
    /// Scalable Polyphony MIP Message (Real-Time, sub-id2 `0x01`).
    FileReferenceOrScalablePolyphonyMip(UniversalSubId2),
    /// `0x0C <nn>` — MIDI Visual Control (Non-Real-Time) or Mobile
    /// Phone Control Message (Real-Time, sub-id2 `0x00`).
    MidiVisualControlOrMobilePhoneControl(UniversalSubId2),
    /// `0x0D <nn>` — MIDI Capability Inquiry (Non-Real-Time only).
    MidiCapabilityInquiry(UniversalSubId2),
    /// `0x02 <nn>` Real-Time — MIDI Show Control (Real-Time only).
    /// Distinct from the Non-Real-Time `0x02` Sample Data Packet
    /// category; the classifier reports this variant when the realm
    /// is Real-Time.
    MidiShowControl(UniversalSubId2),
    /// `0x03 <nn>` Real-Time — Notation Information (Real-Time only).
    /// Sub-ID #2 = `0x01` Bar Number, `0x02` Time Signature
    /// (Immediate), `0x42` Time Signature (Delayed) per Table 4.
    NotationInformation(UniversalSubId2),
    /// `0x04 <nn>` Real-Time — Device Control (Real-Time only).
    /// Sub-ID #2 = `0x01` Master Volume, `0x02` Master Balance,
    /// `0x03` Master Fine Tuning, `0x04` Master Coarse Tuning,
    /// `0x05` Global Parameter Control per Table 4.
    DeviceControl(UniversalSubId2),
    /// `0x7B` — End of File. Singleton-shaped Sub-ID #1: the wire
    /// packet terminates at the Sub-ID #1 byte (Non-Real-Time only).
    EndOfFile,
    /// `0x7C` — Wait. Singleton-shaped Sub-ID #1 (Non-Real-Time only).
    Wait,
    /// `0x7D` — Cancel. Singleton-shaped Sub-ID #1 (Non-Real-Time
    /// only).
    Cancel,
    /// `0x7E` — NAK. Singleton-shaped Sub-ID #1 (Non-Real-Time only).
    Nak,
    /// `0x7F` — ACK. Singleton-shaped Sub-ID #1 (Non-Real-Time only).
    Ack,
    /// Any Sub-ID #1 value outside the Table 4 vocabulary the
    /// classifier knows about, carrying the raw byte for caller
    /// inspection.
    Other(u8),
}

/// Sub-ID #2 byte of a Universal System Exclusive packet — the
/// `<sub_id2>` byte (when present) in the
/// `F0 <realm> <device_id> <sub_id1> <sub_id2> …` wire shape, decoded
/// against Table 4 of the MIDI 1.0 *Universal System Exclusive
/// Messages* document.
///
/// The classifier returns this variant for every Sub-ID #1 category
/// that branches on Sub-ID #2 in Table 4. The vocabulary the
/// classifier knows about is the union of every named Sub-ID #2 in
/// the document; unknown / future values surface through the
/// [`Other`] variant carrying the raw byte.
///
/// Some Sub-ID #2 values share a byte across realms with different
/// semantics (e.g. `0x09 0x01` is "GM 1 System On" in Non-Real-Time
/// and "Channel Pressure (Aftertouch)" in Real-Time); the classifier
/// preserves the raw byte alongside the parsed name so callers can
/// resolve the realm-dependent meaning by inspecting
/// [`UniversalSysEx::realm`].
///
/// [`Other`]: UniversalSubId2::Other
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniversalSubId2 {
    // ---- Non-Real-Time, Sub-ID #1 = 0x04 (MIDI Time Code Setup) ----
    /// `0x04 0x00` Non-Real-Time — Special.
    NonRtMtcSpecial,
    /// `0x04 0x01` Non-Real-Time — Punch In Points.
    NonRtMtcPunchInPoints,
    /// `0x04 0x02` Non-Real-Time — Punch Out Points.
    NonRtMtcPunchOutPoints,
    /// `0x04 0x03` Non-Real-Time — Delete Punch In Point.
    NonRtMtcDeletePunchInPoint,
    /// `0x04 0x04` Non-Real-Time — Delete Punch Out Point.
    NonRtMtcDeletePunchOutPoint,
    /// `0x04 0x05` Non-Real-Time — Event Start Point.
    NonRtMtcEventStartPoint,
    /// `0x04 0x06` Non-Real-Time — Event Stop Point.
    NonRtMtcEventStopPoint,
    /// `0x04 0x07` Non-Real-Time — Event Start Points with additional
    /// info.
    NonRtMtcEventStartPointsWithInfo,
    /// `0x04 0x08` Non-Real-Time — Event Stop Points with additional
    /// info.
    NonRtMtcEventStopPointsWithInfo,
    /// `0x04 0x09` Non-Real-Time — Delete Event Start Point.
    NonRtMtcDeleteEventStartPoint,
    /// `0x04 0x0A` Non-Real-Time — Delete Event Stop Point.
    NonRtMtcDeleteEventStopPoint,
    /// `0x04 0x0B` Non-Real-Time — Cue Points.
    NonRtMtcCuePoints,
    /// `0x04 0x0C` Non-Real-Time — Cue Points with additional info.
    NonRtMtcCuePointsWithInfo,
    /// `0x04 0x0D` Non-Real-Time — Delete Cue Point.
    NonRtMtcDeleteCuePoint,
    /// `0x04 0x0E` Non-Real-Time — Event Name in additional info.
    NonRtMtcEventNameWithInfo,

    // ---- Non-Real-Time, Sub-ID #1 = 0x05 (Sample Dump Extensions) ----
    /// `0x05 0x01` Non-Real-Time — Loop Points Transmission.
    SampleDumpLoopPointsTransmission,
    /// `0x05 0x02` Non-Real-Time — Loop Points Request.
    SampleDumpLoopPointsRequest,
    /// `0x05 0x03` Non-Real-Time — Sample Name Transmission.
    SampleDumpSampleNameTransmission,
    /// `0x05 0x04` Non-Real-Time — Sample Name Request.
    SampleDumpSampleNameRequest,
    /// `0x05 0x05` Non-Real-Time — Extended Dump Header.
    SampleDumpExtendedDumpHeader,
    /// `0x05 0x06` Non-Real-Time — Extended Loop Points Transmission.
    SampleDumpExtendedLoopPointsTransmission,
    /// `0x05 0x07` Non-Real-Time — Extended Loop Points Request.
    SampleDumpExtendedLoopPointsRequest,

    // ---- Non-Real-Time, Sub-ID #1 = 0x06 (General Information) ----
    /// `0x06 0x01` Non-Real-Time — Identity Request.
    GeneralInformationIdentityRequest,
    /// `0x06 0x02` Non-Real-Time — Identity Reply.
    GeneralInformationIdentityReply,

    // ---- Non-Real-Time, Sub-ID #1 = 0x07 (File Dump) ----
    /// `0x07 0x01` Non-Real-Time — Header.
    FileDumpHeader,
    /// `0x07 0x02` Non-Real-Time — Data Packet.
    FileDumpDataPacket,
    /// `0x07 0x03` Non-Real-Time — Request.
    FileDumpRequest,

    // ---- Non-Real-Time, Sub-ID #1 = 0x08 (MIDI Tuning Standard) ----
    /// `0x08 0x00` Non-Real-Time — Bulk Dump Request.
    MtsBulkDumpRequest,
    /// `0x08 0x01` Non-Real-Time — Bulk Dump Reply.
    MtsBulkDumpReply,
    /// `0x08 0x03` Non-Real-Time — Tuning Dump Request.
    MtsTuningDumpRequest,
    /// `0x08 0x04` Non-Real-Time — Key-Based Tuning Dump.
    MtsKeyBasedTuningDump,
    /// `0x08 0x05` Non-Real-Time — Scale/Octave Tuning Dump, 1 byte
    /// format.
    MtsScaleOctaveTuningDump1Byte,
    /// `0x08 0x06` Non-Real-Time — Scale/Octave Tuning Dump, 2 byte
    /// format.
    MtsScaleOctaveTuningDump2Byte,
    /// `0x08 0x07` shared — Single Note Tuning Change with Bank Select.
    /// Listed under Non-Real-Time `0x08 0x07` in the Table 4 entry and
    /// also under Real-Time `0x08 0x07` in the same row.
    MtsSingleNoteTuningChangeWithBankSelect,
    /// `0x08 0x08` shared — Scale/Octave Tuning, 1 byte format.
    /// Appears in both realms under the same Sub-ID #1 = `0x08`.
    MtsScaleOctaveTuning1Byte,
    /// `0x08 0x09` shared — Scale/Octave Tuning, 2 byte format.
    /// Appears in both realms under the same Sub-ID #1 = `0x08`.
    MtsScaleOctaveTuning2Byte,
    /// `0x08 0x02` Real-Time — Single Note Tuning Change. (Real-Time
    /// realm only; the Non-Real-Time `0x08 0x02` slot is unassigned
    /// in Table 4.)
    MtsRtSingleNoteTuningChange,

    // ---- Non-Real-Time, Sub-ID #1 = 0x09 (General MIDI) ----
    /// `0x09 0x01` Non-Real-Time — General MIDI 1 System On.
    GeneralMidi1SystemOn,
    /// `0x09 0x02` Non-Real-Time — General MIDI System Off.
    GeneralMidiSystemOff,
    /// `0x09 0x03` Non-Real-Time — General MIDI 2 System On.
    GeneralMidi2SystemOn,

    // ---- Non-Real-Time, Sub-ID #1 = 0x0A (Downloadable Sounds) ----
    /// `0x0A 0x01` Non-Real-Time — Turn DLS On.
    DlsTurnOn,
    /// `0x0A 0x02` Non-Real-Time — Turn DLS Off.
    DlsTurnOff,
    /// `0x0A 0x03` Non-Real-Time — Turn DLS Voice Allocation Off.
    DlsTurnVoiceAllocationOff,
    /// `0x0A 0x04` Non-Real-Time — Turn DLS Voice Allocation On.
    DlsTurnVoiceAllocationOn,

    // ---- Non-Real-Time, Sub-ID #1 = 0x0B (File Reference Message) ----
    /// `0x0B 0x01` Non-Real-Time — Open File.
    FileReferenceOpenFile,
    /// `0x0B 0x02` Non-Real-Time — Select or Reselect Contents.
    FileReferenceSelectOrReselectContents,
    /// `0x0B 0x03` Non-Real-Time — Open File and Select Contents.
    FileReferenceOpenFileAndSelectContents,
    /// `0x0B 0x04` Non-Real-Time — Close File.
    FileReferenceCloseFile,

    // ---- Real-Time, Sub-ID #1 = 0x01 (MIDI Time Code) ----
    /// `0x01 0x01` Real-Time — Full Message.
    RtMtcFullMessage,
    /// `0x01 0x02` Real-Time — User Bits.
    RtMtcUserBits,

    // ---- Real-Time, Sub-ID #1 = 0x02 (MIDI Show Control) ----
    /// `0x02 0x00` Real-Time — MSC Extensions.
    RtMscExtensions,

    // ---- Real-Time, Sub-ID #1 = 0x03 (Notation Information) ----
    /// `0x03 0x01` Real-Time — Bar Number.
    RtNotationBarNumber,
    /// `0x03 0x02` Real-Time — Time Signature (Immediate).
    RtNotationTimeSignatureImmediate,
    /// `0x03 0x42` Real-Time — Time Signature (Delayed).
    RtNotationTimeSignatureDelayed,

    // ---- Real-Time, Sub-ID #1 = 0x04 (Device Control) ----
    /// `0x04 0x01` Real-Time — Master Volume.
    DeviceControlMasterVolume,
    /// `0x04 0x02` Real-Time — Master Balance.
    DeviceControlMasterBalance,
    /// `0x04 0x03` Real-Time — Master Fine Tuning.
    DeviceControlMasterFineTuning,
    /// `0x04 0x04` Real-Time — Master Coarse Tuning.
    DeviceControlMasterCoarseTuning,
    /// `0x04 0x05` Real-Time — Global Parameter Control.
    DeviceControlGlobalParameterControl,

    // ---- Real-Time, Sub-ID #1 = 0x05 (MTC Cueing) ----
    /// `0x05 0x00` Real-Time — Special.
    RtMtcCueingSpecial,
    /// `0x05 0x01` Real-Time — Punch In Points.
    RtMtcCueingPunchInPoints,
    /// `0x05 0x02` Real-Time — Punch Out Points.
    RtMtcCueingPunchOutPoints,
    /// `0x05 0x05` Real-Time — Event Start Points.
    RtMtcCueingEventStartPoints,
    /// `0x05 0x06` Real-Time — Event Stop Points.
    RtMtcCueingEventStopPoints,
    /// `0x05 0x07` Real-Time — Event Start Points with additional info.
    RtMtcCueingEventStartPointsWithInfo,
    /// `0x05 0x08` Real-Time — Event Stop Points with additional info.
    RtMtcCueingEventStopPointsWithInfo,
    /// `0x05 0x0B` Real-Time — Cue Points.
    RtMtcCueingCuePoints,
    /// `0x05 0x0C` Real-Time — Cue Points with additional info.
    RtMtcCueingCuePointsWithInfo,
    /// `0x05 0x0E` Real-Time — Event Name in additional info.
    RtMtcCueingEventNameWithInfo,

    // ---- Real-Time, Sub-ID #1 = 0x09 (Controller Destination) ----
    /// `0x09 0x01` Real-Time — Channel Pressure (Aftertouch).
    ControllerDestinationChannelPressure,
    /// `0x09 0x02` Real-Time — Polyphonic Key Pressure (Aftertouch).
    ControllerDestinationPolyphonicKeyPressure,
    /// `0x09 0x03` Real-Time — Controller (Control Change).
    ControllerDestinationControlChange,

    // ---- Real-Time, Sub-ID #1 = 0x0A (Key-Based Instrument Control) ----
    /// `0x0A 0x01` Real-Time — Key-Based Instrument Control.
    RtKeyBasedInstrumentControl,

    // ---- Real-Time, Sub-ID #1 = 0x0B (Scalable Polyphony MIP) ----
    /// `0x0B 0x01` Real-Time — Scalable Polyphony MIP Message.
    RtScalablePolyphonyMipMessage,

    // ---- Real-Time, Sub-ID #1 = 0x0C (Mobile Phone Control) ----
    /// `0x0C 0x00` Real-Time — Mobile Phone Control Message.
    RtMobilePhoneControlMessage,

    /// A Sub-ID #2 byte outside the Table 4 vocabulary the classifier
    /// knows about, surfaced for caller inspection.
    Other(u8),
}

impl SysExEvent {
    /// `true` when the payload ends with the SMF SysEx end marker
    /// (`0xF7`). For an `F0` start packet this marks a self-contained
    /// SysEx message; an `F0` packet whose payload does not terminate
    /// with `0xF7` indicates a multi-packet message split that
    /// continues in one or more following `F7`-continuation events.
    ///
    /// Returns `false` for an empty payload, since an empty packet
    /// has no terminator to inspect.
    pub fn ends_with_eox(&self) -> bool {
        matches!(self.data.last(), Some(&0xF7))
    }

    /// `true` when this packet is an `F0` start *and* the payload
    /// terminates with `0xF7` — a self-contained, complete SysEx
    /// message that needs no continuation packets. Equivalent to
    /// `!is_escape && ends_with_eox()`; sugar for the common
    /// universal-SysEx case (GM-on `F0 7E 7F 09 01 F7`, Master
    /// Volume, Master Tuning) where callers route the whole packet
    /// in one step.
    pub fn is_complete_message(&self) -> bool {
        !self.is_escape && self.ends_with_eox()
    }

    /// Returns the manufacturer-ID byte (or the leading byte of a
    /// universal-SysEx packet, `0x7E` non-real-time or `0x7F`
    /// real-time) of an `F0` start packet. Returns `None` for an
    /// `F7` continuation / escape packet (whose payload is not
    /// manufacturer-prefixed in the SMF framing) and for an empty
    /// payload.
    ///
    /// A real-world manufacturer ID is either a single byte (the
    /// historic IDs assigned by the MMA, including the
    /// pre-allocated Sequential / Moog / Yamaha / Roland / Korg
    /// ranges) or a three-byte sequence starting with `0x00` (the
    /// expanded-ID convention reserving room for new manufacturers);
    /// this accessor returns the leading byte only, leaving the
    /// expanded-ID disambiguation to the caller — inspect
    /// [`SysExEvent::data`] directly to read `data[0..=2]` when
    /// `data[0] == 0x00`.
    pub fn manufacturer_id(&self) -> Option<u8> {
        if self.is_escape {
            return None;
        }
        self.data.first().copied()
    }

    /// Classify a Universal System Exclusive packet against Table 4 of
    /// the MIDI 1.0 *Universal System Exclusive Messages* document.
    ///
    /// Returns `Some(UniversalSysEx { realm, device_id, sub_id1 })` when
    /// the event is an `F0` start packet whose leading byte is `0x7E`
    /// (Universal Non-Real-Time) or `0x7F` (Universal Real-Time) and the
    /// packet is long enough to carry the realm, device-id, and
    /// Sub-ID #1 bytes (3 bytes minimum, before any trailing `0xF7`).
    ///
    /// Returns `None` for:
    /// * An `F7` continuation / escape packet — the payload of an
    ///   `F7` packet is opaque arbitrary bytes shipped to the wire,
    ///   not a Universal SysEx in its own right; a manufacturer-
    ///   prefixed multi-packet message whose continuation arrives on
    ///   `F7` carries the manufacturer-specific format the opener
    ///   defined, not Table 4 vocabulary.
    /// * A manufacturer-prefixed `F0` packet (the leading payload byte
    ///   is anything other than `0x7E` / `0x7F`). The MMA assigns
    ///   single-byte and `0x00`-prefixed three-byte manufacturer IDs
    ///   in a separate document; the classifier reports such packets
    ///   through [`SysExEvent::manufacturer_id`] only.
    /// * An `F0` Universal packet truncated before Sub-ID #1 (a
    ///   payload shorter than 3 bytes). A receiver would treat the
    ///   packet as malformed; the classifier reports `None` rather
    ///   than fabricate a fallback Sub-ID #1.
    ///
    /// Sub-ID #2 decoding is realm-aware: the same `(sub_id1, sub_id2)`
    /// byte pair can name different messages in the two realms (e.g.
    /// `0x09 0x01` is "General MIDI 1 System On" in Non-Real-Time and
    /// "Channel Pressure (Aftertouch)" in Real-Time, per Table 4); the
    /// classifier uses [`UniversalSysEx::realm`] to disambiguate before
    /// matching. Singleton Sub-ID #1 categories (`0x7B` End of File
    /// through `0x7F` ACK) do not carry a Sub-ID #2 byte; the
    /// classifier reports them as unit variants of [`UniversalSubId1`]
    /// regardless of any trailing bytes in the payload.
    ///
    /// Sub-ID #1 values outside the Table 4 vocabulary surface through
    /// [`UniversalSubId1::Other`] carrying the raw byte. Sub-ID #2
    /// values outside the per-Sub-ID #1 vocabulary surface through
    /// [`UniversalSubId2::Other`] carrying the raw byte. The
    /// classifier preserves unknown values rather than rejecting them
    /// — Table 4 was last extended after the document publication date
    /// and a forward-looking classifier needs to surface the raw bytes
    /// for callers with deeper vocabulary.
    pub fn universal_classification(&self) -> Option<UniversalSysEx> {
        if self.is_escape {
            return None;
        }
        // Wire layout: <realm> <device_id> <sub_id1> [<sub_id2> ...] [F7]
        if self.data.len() < 3 {
            return None;
        }
        let realm = match self.data[0] {
            0x7E => UniversalRealm::NonRealTime,
            0x7F => UniversalRealm::RealTime,
            _ => return None,
        };
        let device_id = self.data[1];
        let sub1_byte = self.data[2];
        let sub2_byte = self.data.get(3).copied();
        let sub_id1 = classify_universal_sub_id1(realm, sub1_byte, sub2_byte);
        Some(UniversalSysEx {
            realm,
            device_id,
            sub_id1,
        })
    }
}

/// A logical System Exclusive message reassembled from one or more
/// on-disk SMF SysEx packets.
///
/// The Standard MIDI File format lets a single MIDI SysEx message be
/// **split across multiple track events**: an `F0` start packet whose
/// payload does *not* end with the `0xF7` End-of-Exclusive marker is
/// continued by one or more `F7` continuation packets, the last of
/// which ends with `0xF7`. (A producer splits a long dump so other
/// events — timing, notes — can be interleaved at the right ticks, the
/// SMF analogue of the wire-level rule that an EOX or any other status
/// byte terminates a SysEx; MIDI 1.0 Detailed Specification, "EOX (End
/// of Exclusive)".)
///
/// [`SmfFile::reassembled_sysex_messages`] folds those packet chains
/// into one `ReassembledSysEx` per logical message, exposing the
/// concatenated manufacturer payload via [`body`](Self::body) without
/// the caller re-implementing the continuation state machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReassembledSysEx {
    /// Absolute tick of the **opening** `F0` packet (relative to its
    /// parent track's start; on a format-1 shared timebase this is the
    /// absolute tick).
    pub tick: u64,
    /// Index of the [`Track`] the opening packet came from. All
    /// continuation packets folded into this message come from the
    /// same track.
    pub track: usize,
    /// The reassembled payload, **excluding** the leading `0xF0` (which
    /// is implied by the `F0`-start framing and never stored in a
    /// packet's `data`) and **excluding** the trailing `0xF7` EOX
    /// marker. For a single-packet message this is the opener's payload
    /// minus its trailing `0xF7`; for a multi-packet message it is the
    /// opener's payload followed by every continuation packet's payload,
    /// with only the final `0xF7` stripped.
    pub body: Vec<u8>,
    /// `true` when the chain terminated with an `0xF7` EOX marker;
    /// `false` when the file ended (or a new `F0` opener intervened)
    /// before an EOX was seen — a truncated / malformed message whose
    /// `body` holds whatever bytes did arrive.
    pub complete: bool,
    /// Number of on-disk packets folded into this logical message (1 for
    /// a self-contained `F0 … F7`, ≥2 for a split message).
    pub packet_count: usize,
}

impl ReassembledSysEx {
    /// The manufacturer / universal-realm ID byte that opens the
    /// message — `body[0]` (the byte immediately after the implied
    /// `0xF0`): a single-byte manufacturer ID, the `0x00` expanded-ID
    /// prefix, or `0x7E` / `0x7F` for a Universal packet. `None` when
    /// the reassembled body is empty.
    pub fn id_byte(&self) -> Option<u8> {
        self.body.first().copied()
    }

    /// `true` when the message opens with a Universal SysEx realm byte
    /// (`0x7E` Non-Real-Time or `0x7F` Real-Time).
    pub fn is_universal(&self) -> bool {
        matches!(self.id_byte(), Some(0x7E | 0x7F))
    }
}

/// Internal: decode the `<sub_id1>` byte into a [`UniversalSubId1`]
/// against Table 4 of the MIDI 1.0 *Universal System Exclusive
/// Messages* document, branching on realm where the same byte names
/// different categories in the two realms.
fn classify_universal_sub_id1(
    realm: UniversalRealm,
    sub1: u8,
    sub2: Option<u8>,
) -> UniversalSubId1 {
    // Singleton-shaped Sub-ID #1 bytes (Non-Real-Time only). The wire
    // packet terminates at the Sub-ID #1 byte; the match arm returns
    // the unit variant regardless of any trailing payload.
    match sub1 {
        0x7B => return UniversalSubId1::EndOfFile,
        0x7C => return UniversalSubId1::Wait,
        0x7D => return UniversalSubId1::Cancel,
        0x7E => return UniversalSubId1::Nak,
        0x7F => return UniversalSubId1::Ack,
        _ => {}
    }
    // Non-Real-Time–only Sub-ID #1 categories (Sub-ID #2 absent in
    // Table 4's row for these): Sample Dump Header / Data / Request.
    if matches!(realm, UniversalRealm::NonRealTime) {
        match sub1 {
            0x01 => return UniversalSubId1::SampleDumpHeader,
            0x02 => return UniversalSubId1::SampleDataPacket,
            0x03 => return UniversalSubId1::SampleDumpRequest,
            _ => {}
        }
    }
    // Branching categories — Sub-ID #2 may be present; surface it
    // through the [`UniversalSubId2`] enum (or `Other(raw)` when the
    // value is outside the Table 4 vocabulary the classifier knows
    // about).
    match (realm, sub1) {
        (_, 0x04) => match realm {
            UniversalRealm::NonRealTime => {
                UniversalSubId1::MidiTimeCode(classify_nonrt_mtc_setup_sub2(sub2))
            }
            UniversalRealm::RealTime => {
                UniversalSubId1::DeviceControl(classify_rt_device_control_sub2(sub2))
            }
        },
        (_, 0x05) => match realm {
            UniversalRealm::NonRealTime => UniversalSubId1::SampleDumpExtensionsOrMtcCueing(
                classify_nonrt_sample_dump_extensions_sub2(sub2),
            ),
            UniversalRealm::RealTime => {
                UniversalSubId1::SampleDumpExtensionsOrMtcCueing(classify_rt_mtc_cueing_sub2(sub2))
            }
        },
        (_, 0x06) => match realm {
            UniversalRealm::NonRealTime => UniversalSubId1::GeneralInformationOrMmcCommands(
                classify_nonrt_general_information_sub2(sub2),
            ),
            UniversalRealm::RealTime => {
                UniversalSubId1::GeneralInformationOrMmcCommands(passthrough_sub2(sub2))
            }
        },
        (_, 0x07) => match realm {
            UniversalRealm::NonRealTime => {
                UniversalSubId1::FileDumpOrMmcResponses(classify_nonrt_file_dump_sub2(sub2))
            }
            UniversalRealm::RealTime => {
                UniversalSubId1::FileDumpOrMmcResponses(passthrough_sub2(sub2))
            }
        },
        (_, 0x08) => UniversalSubId1::MidiTuningStandard(classify_mts_sub2(realm, sub2)),
        (_, 0x09) => UniversalSubId1::GeneralMidiOrControllerDestination(
            classify_realm_sub2_for_0x09(realm, sub2),
        ),
        (_, 0x0A) => UniversalSubId1::DownloadableSoundsOrKeyBasedInstrumentControl(
            classify_realm_sub2_for_0x0a(realm, sub2),
        ),
        (_, 0x0B) => UniversalSubId1::FileReferenceOrScalablePolyphonyMip(
            classify_realm_sub2_for_0x0b(realm, sub2),
        ),
        (_, 0x0C) => UniversalSubId1::MidiVisualControlOrMobilePhoneControl(
            classify_realm_sub2_for_0x0c(realm, sub2),
        ),
        (UniversalRealm::NonRealTime, 0x0D) => {
            UniversalSubId1::MidiCapabilityInquiry(passthrough_sub2(sub2))
        }
        (UniversalRealm::RealTime, 0x01) => {
            UniversalSubId1::MidiTimeCode(classify_rt_mtc_sub2(sub2))
        }
        (UniversalRealm::RealTime, 0x02) => {
            UniversalSubId1::MidiShowControl(classify_rt_msc_sub2(sub2))
        }
        (UniversalRealm::RealTime, 0x03) => {
            UniversalSubId1::NotationInformation(classify_rt_notation_sub2(sub2))
        }
        _ => UniversalSubId1::Other(sub1),
    }
}

fn passthrough_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    UniversalSubId2::Other(sub2.unwrap_or(0))
}

fn classify_nonrt_mtc_setup_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x00) => UniversalSubId2::NonRtMtcSpecial,
        Some(0x01) => UniversalSubId2::NonRtMtcPunchInPoints,
        Some(0x02) => UniversalSubId2::NonRtMtcPunchOutPoints,
        Some(0x03) => UniversalSubId2::NonRtMtcDeletePunchInPoint,
        Some(0x04) => UniversalSubId2::NonRtMtcDeletePunchOutPoint,
        Some(0x05) => UniversalSubId2::NonRtMtcEventStartPoint,
        Some(0x06) => UniversalSubId2::NonRtMtcEventStopPoint,
        Some(0x07) => UniversalSubId2::NonRtMtcEventStartPointsWithInfo,
        Some(0x08) => UniversalSubId2::NonRtMtcEventStopPointsWithInfo,
        Some(0x09) => UniversalSubId2::NonRtMtcDeleteEventStartPoint,
        Some(0x0A) => UniversalSubId2::NonRtMtcDeleteEventStopPoint,
        Some(0x0B) => UniversalSubId2::NonRtMtcCuePoints,
        Some(0x0C) => UniversalSubId2::NonRtMtcCuePointsWithInfo,
        Some(0x0D) => UniversalSubId2::NonRtMtcDeleteCuePoint,
        Some(0x0E) => UniversalSubId2::NonRtMtcEventNameWithInfo,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_nonrt_sample_dump_extensions_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x01) => UniversalSubId2::SampleDumpLoopPointsTransmission,
        Some(0x02) => UniversalSubId2::SampleDumpLoopPointsRequest,
        Some(0x03) => UniversalSubId2::SampleDumpSampleNameTransmission,
        Some(0x04) => UniversalSubId2::SampleDumpSampleNameRequest,
        Some(0x05) => UniversalSubId2::SampleDumpExtendedDumpHeader,
        Some(0x06) => UniversalSubId2::SampleDumpExtendedLoopPointsTransmission,
        Some(0x07) => UniversalSubId2::SampleDumpExtendedLoopPointsRequest,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_nonrt_general_information_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x01) => UniversalSubId2::GeneralInformationIdentityRequest,
        Some(0x02) => UniversalSubId2::GeneralInformationIdentityReply,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_nonrt_file_dump_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x01) => UniversalSubId2::FileDumpHeader,
        Some(0x02) => UniversalSubId2::FileDumpDataPacket,
        Some(0x03) => UniversalSubId2::FileDumpRequest,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_mts_sub2(realm: UniversalRealm, sub2: Option<u8>) -> UniversalSubId2 {
    // Shared bytes (both realms): 0x07 / 0x08 / 0x09.
    match (realm, sub2) {
        (_, Some(0x07)) => UniversalSubId2::MtsSingleNoteTuningChangeWithBankSelect,
        (_, Some(0x08)) => UniversalSubId2::MtsScaleOctaveTuning1Byte,
        (_, Some(0x09)) => UniversalSubId2::MtsScaleOctaveTuning2Byte,
        (UniversalRealm::NonRealTime, Some(0x00)) => UniversalSubId2::MtsBulkDumpRequest,
        (UniversalRealm::NonRealTime, Some(0x01)) => UniversalSubId2::MtsBulkDumpReply,
        (UniversalRealm::NonRealTime, Some(0x03)) => UniversalSubId2::MtsTuningDumpRequest,
        (UniversalRealm::NonRealTime, Some(0x04)) => UniversalSubId2::MtsKeyBasedTuningDump,
        (UniversalRealm::NonRealTime, Some(0x05)) => UniversalSubId2::MtsScaleOctaveTuningDump1Byte,
        (UniversalRealm::NonRealTime, Some(0x06)) => UniversalSubId2::MtsScaleOctaveTuningDump2Byte,
        (UniversalRealm::RealTime, Some(0x02)) => UniversalSubId2::MtsRtSingleNoteTuningChange,
        (_, other) => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_realm_sub2_for_0x09(realm: UniversalRealm, sub2: Option<u8>) -> UniversalSubId2 {
    match (realm, sub2) {
        (UniversalRealm::NonRealTime, Some(0x01)) => UniversalSubId2::GeneralMidi1SystemOn,
        (UniversalRealm::NonRealTime, Some(0x02)) => UniversalSubId2::GeneralMidiSystemOff,
        (UniversalRealm::NonRealTime, Some(0x03)) => UniversalSubId2::GeneralMidi2SystemOn,
        (UniversalRealm::RealTime, Some(0x01)) => {
            UniversalSubId2::ControllerDestinationChannelPressure
        }
        (UniversalRealm::RealTime, Some(0x02)) => {
            UniversalSubId2::ControllerDestinationPolyphonicKeyPressure
        }
        (UniversalRealm::RealTime, Some(0x03)) => {
            UniversalSubId2::ControllerDestinationControlChange
        }
        (_, other) => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_realm_sub2_for_0x0a(realm: UniversalRealm, sub2: Option<u8>) -> UniversalSubId2 {
    match (realm, sub2) {
        (UniversalRealm::NonRealTime, Some(0x01)) => UniversalSubId2::DlsTurnOn,
        (UniversalRealm::NonRealTime, Some(0x02)) => UniversalSubId2::DlsTurnOff,
        (UniversalRealm::NonRealTime, Some(0x03)) => UniversalSubId2::DlsTurnVoiceAllocationOff,
        (UniversalRealm::NonRealTime, Some(0x04)) => UniversalSubId2::DlsTurnVoiceAllocationOn,
        (UniversalRealm::RealTime, Some(0x01)) => UniversalSubId2::RtKeyBasedInstrumentControl,
        (_, other) => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_realm_sub2_for_0x0b(realm: UniversalRealm, sub2: Option<u8>) -> UniversalSubId2 {
    match (realm, sub2) {
        (UniversalRealm::NonRealTime, Some(0x01)) => UniversalSubId2::FileReferenceOpenFile,
        (UniversalRealm::NonRealTime, Some(0x02)) => {
            UniversalSubId2::FileReferenceSelectOrReselectContents
        }
        (UniversalRealm::NonRealTime, Some(0x03)) => {
            UniversalSubId2::FileReferenceOpenFileAndSelectContents
        }
        (UniversalRealm::NonRealTime, Some(0x04)) => UniversalSubId2::FileReferenceCloseFile,
        (UniversalRealm::RealTime, Some(0x01)) => UniversalSubId2::RtScalablePolyphonyMipMessage,
        (_, other) => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_realm_sub2_for_0x0c(realm: UniversalRealm, sub2: Option<u8>) -> UniversalSubId2 {
    match (realm, sub2) {
        (UniversalRealm::RealTime, Some(0x00)) => UniversalSubId2::RtMobilePhoneControlMessage,
        (_, other) => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_rt_mtc_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x01) => UniversalSubId2::RtMtcFullMessage,
        Some(0x02) => UniversalSubId2::RtMtcUserBits,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_rt_msc_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x00) => UniversalSubId2::RtMscExtensions,
        // 0x01..=0x7F = MSC Commands per the MSC specification — surface
        // the raw byte; callers consulting the MSC document decode the
        // command set.
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_rt_notation_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x01) => UniversalSubId2::RtNotationBarNumber,
        Some(0x02) => UniversalSubId2::RtNotationTimeSignatureImmediate,
        Some(0x42) => UniversalSubId2::RtNotationTimeSignatureDelayed,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_rt_device_control_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x01) => UniversalSubId2::DeviceControlMasterVolume,
        Some(0x02) => UniversalSubId2::DeviceControlMasterBalance,
        Some(0x03) => UniversalSubId2::DeviceControlMasterFineTuning,
        Some(0x04) => UniversalSubId2::DeviceControlMasterCoarseTuning,
        Some(0x05) => UniversalSubId2::DeviceControlGlobalParameterControl,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
    }
}

fn classify_rt_mtc_cueing_sub2(sub2: Option<u8>) -> UniversalSubId2 {
    match sub2 {
        Some(0x00) => UniversalSubId2::RtMtcCueingSpecial,
        Some(0x01) => UniversalSubId2::RtMtcCueingPunchInPoints,
        Some(0x02) => UniversalSubId2::RtMtcCueingPunchOutPoints,
        Some(0x05) => UniversalSubId2::RtMtcCueingEventStartPoints,
        Some(0x06) => UniversalSubId2::RtMtcCueingEventStopPoints,
        Some(0x07) => UniversalSubId2::RtMtcCueingEventStartPointsWithInfo,
        Some(0x08) => UniversalSubId2::RtMtcCueingEventStopPointsWithInfo,
        Some(0x0B) => UniversalSubId2::RtMtcCueingCuePoints,
        Some(0x0C) => UniversalSubId2::RtMtcCueingCuePointsWithInfo,
        Some(0x0E) => UniversalSubId2::RtMtcCueingEventNameWithInfo,
        other => UniversalSubId2::Other(other.unwrap_or(0)),
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

    /// Build a [`TempoTimeline`] that converts absolute division ticks
    /// to elapsed wall-clock seconds, folding this file's
    /// [`tempo_map`](Self::tempo_map) against its
    /// [`Division`](crate::smf::Division).
    ///
    /// The timeline starts at the SMF default tempo
    /// ([`DEFAULT_TEMPO_US_PER_QUARTER`] = 120 BPM) and switches at each
    /// Set Tempo (`FF 51`) change in merged `(tick, track)` order — the
    /// same ordering [`tempo_map`](Self::tempo_map) and the scheduler
    /// use. A tempo change repeated at the same tick keeps only its
    /// final value (the last write at a tick wins, matching a receiver
    /// processing both in order). For an [`Division::Smpte`] file the
    /// rate is fixed by the timebase and Set Tempo events are ignored,
    /// exactly as the scheduler treats them.
    ///
    /// Build once, then query [`TempoTimeline::tick_to_seconds`]
    /// repeatedly; each lookup is `O(log n)` over the tempo segments.
    /// For a single conversion prefer [`tick_to_seconds`](Self::tick_to_seconds).
    ///
    /// Cost is `O(n log n)` in the total event count (one tempo-map
    /// pass plus the sort), bounded above by [`MAX_EVENTS_PER_FILE`].
    pub fn tempo_timeline(&self) -> TempoTimeline {
        let seconds_per_tick_at = |us_per_quarter: u32| -> f64 {
            match self.header.division {
                Division::TicksPerQuarter(div) => {
                    // (usec_per_quarter / 1_000_000) / division — the
                    // scheduler's samples_per_tick math with sr factored
                    // out. div is clamped to >= 1 to mirror the
                    // scheduler's div.max(1) guard against a malformed
                    // zero division.
                    (us_per_quarter as f64 / 1_000_000.0) / (div.max(1) as f64)
                }
                Division::Smpte {
                    frames_per_second,
                    ticks_per_frame,
                } => {
                    // Absolute timebase: tempo events do not apply.
                    let denom = (frames_per_second.max(1) as f64) * (ticks_per_frame.max(1) as f64);
                    1.0 / denom
                }
            }
        };

        let mut segments: Vec<TimelineSegment> = Vec::new();
        // Leading anchor at tick 0 with the default tempo. For an SMPTE
        // division the rate is constant so this single segment suffices;
        // for a musical division it covers the span before the first
        // Set Tempo change.
        segments.push(TimelineSegment {
            start_tick: 0,
            seconds_at_start: 0.0,
            seconds_per_tick: seconds_per_tick_at(DEFAULT_TEMPO_US_PER_QUARTER),
        });

        if matches!(self.header.division, Division::TicksPerQuarter(_)) {
            for change in self.tempo_map() {
                let prev = *segments.last().expect("anchor always present");
                if change.tick == prev.start_tick {
                    // A tempo change at the same tick as the current
                    // anchor replaces its rate (last write wins); the
                    // elapsed seconds at this boundary are unchanged.
                    if let Some(last) = segments.last_mut() {
                        last.seconds_per_tick =
                            seconds_per_tick_at(change.microseconds_per_quarter_note);
                    }
                    continue;
                }
                let span_ticks = change.tick.saturating_sub(prev.start_tick) as f64;
                let seconds_at_start = prev.seconds_at_start + span_ticks * prev.seconds_per_tick;
                segments.push(TimelineSegment {
                    start_tick: change.tick,
                    seconds_at_start,
                    seconds_per_tick: seconds_per_tick_at(change.microseconds_per_quarter_note),
                });
            }
        }

        TempoTimeline { segments }
    }

    /// Convert a single absolute division tick to elapsed wall-clock
    /// seconds from the start of the file.
    ///
    /// Convenience wrapper that builds a [`TempoTimeline`] and performs
    /// one lookup. For many conversions build the timeline once with
    /// [`tempo_timeline`](Self::tempo_timeline) and reuse it instead of
    /// paying the `O(n log n)` build cost per call.
    pub fn tick_to_seconds(&self, tick: u64) -> f64 {
        self.tempo_timeline().tick_to_seconds(tick)
    }

    /// The total wall-clock duration of the file in seconds: the elapsed
    /// time at the file's last event tick, folding the tempo map.
    ///
    /// The end tick is the maximum absolute tick across every track
    /// (each track's deltas summed independently, then the max taken) —
    /// for a format-0/1 file this is the latest event on the shared
    /// timebase, including the terminating End of Track. Returns `0.0`
    /// for a file with no events.
    ///
    /// Note this measures the *scheduled event span*, not the audible
    /// tail: a synth voice with a long release envelope may still be
    /// sounding after the last event. It mirrors the scheduler's
    /// `estimated_total_samples` stop bound, in seconds.
    ///
    /// Cost is `O(n log n)` in the total event count, bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn duration_seconds(&self) -> f64 {
        let mut end_tick: u64 = 0;
        for track in &self.tracks {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
            }
            end_tick = end_tick.max(abs);
        }
        self.tempo_timeline().tick_to_seconds(end_tick)
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

    /// Collect every [`MetaEvent::ChannelPrefix`] (`FF 20 01 cc`) from
    /// every track, pinned to the absolute tick at which it fires, in
    /// time order.
    ///
    /// `FF 20 01 cc` carries the **channel-binding hint** for
    /// non-channel events that follow on the same track. The single
    /// payload byte names the MIDI channel (`0..=15`) the following
    /// meta / sysex events should be associated with — text, lyric,
    /// marker, cue point, sysex — until another `FF 20` arrives, the
    /// next channel-voice event arrives and supersedes the binding,
    /// or the track ends. The Standard MIDI File Specification 1.0
    /// lists the event as part of the meta-event vocabulary and notes
    /// the pre-multi-port usage; it is deprecated in modern authoring
    /// tools in favour of explicit per-track channel-voice streams
    /// plus [`FF 21`](MetaEvent::Port) port hints, but older files
    /// still emit it and a round-trip workflow must preserve it.
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
    /// [`SmfFile::sequencer_specifics`] /
    /// [`SmfFile::sequence_numbers`] / [`SmfFile::midi_ports`] and
    /// the scheduler use (`scheduler.rs` §"merged event list, sorted
    /// by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries an `FF 20` meta
    /// event. A file without channel-prefix hints is the common case
    /// for modern authoring tools; a receiver that finds no `FF 20`
    /// should bind surrounding non-channel events to the channel of
    /// the most recent channel-voice event (or to channel 0 when no
    /// such event has occurred yet) per the SMF spec's
    /// channel-binding rule.
    ///
    /// Only `FF 20` is selected — the neighbouring
    /// [`FF 21`](MetaEvent::Port) port hint, rhythmic, text, cueing,
    /// and sequencer-private meta events stay on their own helpers
    /// (different routing semantics: per-track physical port
    /// assignment versus per-message channel override) so callers
    /// reconstructing the channel association of surrounding
    /// non-channel events get a clean time-ordered list independent
    /// of the other meta streams.
    ///
    /// The payload byte is surfaced verbatim. The spec scopes it at
    /// `0..=15` (one nibble) but the parser accepts the full one-byte
    /// payload as written, so files with out-of-spec values still
    /// round-trip; [`ChannelPrefixEvent::channel`] returns the
    /// spec-clamped `Option<u8>` view.
    ///
    /// Lifts the SMF meta-event iterator family from 14 to **15**
    /// total: `tempo_map`, `time_signatures`, `key_signatures`,
    /// `markers`, `lyrics`, `cue_points`, `track_names`,
    /// `instrument_names`, `texts`, `copyrights`, `smpte_offsets`,
    /// `sequencer_specifics`, `sequence_numbers`, `midi_ports`, and
    /// `channel_prefixes`.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn channel_prefixes(&self) -> Vec<ChannelPrefixEvent> {
        let mut out: Vec<ChannelPrefixEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Meta(MetaEvent::ChannelPrefix(channel)) = &ev.kind {
                    out.push(ChannelPrefixEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
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

    /// Collect every [`ChannelBody::ProgramChange`] (`Cn pp`) channel-voice
    /// event from every track, pinned to the absolute tick at which it
    /// fires, in time order.
    ///
    /// `Cn pp` selects the MIDI program (patch) on a specific channel:
    /// the low nibble of the status byte is the channel index (`0..=15`)
    /// and the single data byte `pp` (`0..=127`) is the program number.
    /// In General MIDI 1 a player resolves the program against the
    /// 128-patch GM melodic table; in GM 2 / GS / XG it resolves through
    /// the active Bank Select pair (CC 0 / CC 32) first; this helper
    /// stays bank-agnostic and surfaces the raw program byte alongside
    /// the channel so the caller's patch-resolution policy stays its own.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two patch changes at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] /
    /// [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::track_names`] /
    /// [`SmfFile::instrument_names`] / [`SmfFile::texts`] /
    /// [`SmfFile::copyrights`] / [`SmfFile::smpte_offsets`] /
    /// [`SmfFile::sequencer_specifics`] /
    /// [`SmfFile::sequence_numbers`] / [`SmfFile::midi_ports`] /
    /// [`SmfFile::channel_prefixes`] and the scheduler use
    /// (`scheduler.rs` §"merged event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a `Cn` event — a
    /// player initialising a fresh receiver should leave the channel on
    /// power-up patch `0` (Acoustic Grand Piano under GM 1) until an
    /// explicit Program Change arrives, per the spec convention.
    ///
    /// Only `Cn` is selected — neighbouring CC (`Bn`), pitch-bend
    /// (`En`), aftertouch (`An` / `Dn`), and note (`8n` / `9n`)
    /// channel-voice events stay on their own surfaces. Companion
    /// primitive [`SmfFile::channel_snapshot_at`] folds the *last*
    /// Program Change per channel into
    /// [`SmfChannelSnapshot::program`] for seek initialisation; this
    /// helper surfaces *every* Program Change in chronological order
    /// so callers building a song-form patch-change timeline (e.g. a
    /// DAW track-inspector view that highlights the bar each
    /// instrument enters) get the full list without a manual track
    /// walk.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn program_changes(&self) -> Vec<ProgramChangeEvent> {
        let mut out: Vec<ProgramChangeEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::ProgramChange { program },
                }) = &ev.kind
                {
                    out.push(ProgramChangeEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
                        program: *program,
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

    /// Collect every [`ChannelBody::ControlChange`] (`Bn cc vv`)
    /// channel-voice event from every track, pinned to the absolute
    /// tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two events at the same tick keep the `(track,
    /// in-track-position)` order — track 0's events fire before track
    /// 1's at the same tick. This matches the same stable-merge rule
    /// [`SmfFile::tempo_map`] / [`SmfFile::time_signatures`] /
    /// [`SmfFile::key_signatures`] / [`SmfFile::markers`] /
    /// [`SmfFile::lyrics`] / [`SmfFile::cue_points`] /
    /// [`SmfFile::track_names`] / [`SmfFile::instrument_names`] /
    /// [`SmfFile::texts`] / [`SmfFile::copyrights`] /
    /// [`SmfFile::smpte_offsets`] / [`SmfFile::sequencer_specifics`] /
    /// [`SmfFile::sequence_numbers`] / [`SmfFile::midi_ports`] /
    /// [`SmfFile::channel_prefixes`] / [`SmfFile::program_changes`]
    /// and the scheduler use (`scheduler.rs` §"merged event list,
    /// sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a `Bn` event — a
    /// player initialising a fresh receiver should leave every
    /// controller at its power-up default until an explicit Control
    /// Change arrives, per the spec convention. The MIDI 1.0
    /// *Control Change Messages — Data Bytes* document spells out the
    /// defaults: Volume `100`, Pan `64`, Expression `127`, Modulation
    /// `0`, Sustain `0`, RPN / NRPN MSB / LSB `127 / 127` (the "Null
    /// RPN" sentinel that disables Data Entry pickup).
    ///
    /// Only `Bn` is selected — neighbouring Program Change (`Cn`),
    /// pitch-bend (`En`), aftertouch (`An` / `Dn`), and note (`8n` /
    /// `9n`) channel-voice events stay on their own surfaces. The
    /// channel-mode family (`controller == 120..=127`) is *not*
    /// diverted — All Sound Off (`120`), Reset All Controllers
    /// (`121`), Local Control (`122`), All Notes Off (`123`), Omni
    /// Mode Off (`124`), Omni Mode On (`125`), Mono Mode On (`126`),
    /// Poly Mode On (`127`) all surface through this list with their
    /// `value` byte preserved, and [`ControlChangeEvent::is_channel_mode`]
    /// gives callers a one-line predicate to route them. Companion
    /// primitive [`SmfFile::channel_snapshot_at`] folds the *last*
    /// value of the six snapshot-tracked controllers (Bank MSB / LSB,
    /// Modulation, Volume, Pan, Expression, Sustain) into the
    /// snapshot for seek initialisation; this helper surfaces *every*
    /// Control Change in chronological order so callers building a
    /// full automation timeline — including the controllers the
    /// snapshot doesn't track (CC-1 Modulation curves with sub-tick
    /// granularity, the CC-6 / CC-38 Data Entry pump that drives RPN
    /// / NRPN parameter writes, the CC-100 / CC-101 RPN pair and
    /// CC-98 / CC-99 NRPN pair, the CC-91 / CC-93 effects-send
    /// levels, the channel-mode reset family) — don't have to
    /// re-walk the channel-voice stream manually.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn control_changes(&self) -> Vec<ControlChangeEvent> {
        let mut out: Vec<ControlChangeEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::ControlChange { controller, value },
                }) = &ev.kind
                {
                    out.push(ControlChangeEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
                        controller: *controller,
                        value: *value,
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

    /// Collect every **Channel Mode Message** (`Bn cc vv` with `cc` in
    /// `120..=127`) from every track, pinned to the absolute tick at
    /// which it fires, with each controller already classified into a
    /// typed [`ChannelModeMessage`].
    ///
    /// The typed counterpart of the channel-mode subset of
    /// [`SmfFile::control_changes`]: this helper walks the same `Bn`
    /// stream, keeps only the `120..=127` controllers (All Sound Off,
    /// Reset All Controllers, Local Control, All Notes Off, Omni Off /
    /// On, Mono On, Poly On — MIDI 1.0 Detailed Specification §"Channel
    /// Mode Messages"), decodes the value-byte argument for Local
    /// Control (on/off) and Mono Mode On (channel count), and filters
    /// out the continuous controllers (`0..=119`).
    ///
    /// The cumulative delta is summed per-track, then the per-track
    /// sequences are merged. The sort is stable so two mode messages at
    /// the same tick keep the `(track, in-track-position)` order — track
    /// 0's events fire before track 1's at the same tick, matching the
    /// stable-merge rule the rest of the iterator family uses.
    ///
    /// Returns an empty `Vec` when no track carries a channel-mode
    /// controller.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn channel_mode_messages(&self) -> Vec<ChannelModeEvent> {
        let mut out: Vec<ChannelModeEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::ControlChange { controller, value },
                }) = &ev.kind
                {
                    let scratch = ControlChangeEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
                        controller: *controller,
                        value: *value,
                    };
                    if let Some(message) = scratch.channel_mode() {
                        out.push(ChannelModeEvent {
                            tick: abs,
                            track: track_idx,
                            channel: *channel,
                            message,
                        });
                    }
                }
            }
        }
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every **Effects Depth** controller (`Bn cc vv` with `cc`
    /// in `91..=95`) from every track, pinned to the absolute tick at
    /// which it fires, with each controller classified into a typed
    /// [`EffectDepth`].
    ///
    /// The typed counterpart of the effects-depth subset of
    /// [`SmfFile::control_changes`]: it keeps only CC 91–95 (Reverb Send
    /// / Tremolo / Chorus Send / Celeste / Phaser — MIDI 1.0 *Control
    /// Change Messages* Table 3) and filters out every other controller.
    /// CC 91 / CC 93 are the two GM-default sends the synth's CA-024
    /// effects bus consumes; this iterator lets a caller surface the
    /// send automation for a score without driving the synth.
    ///
    /// Stable-merged by absolute tick (track 0 before track 1 at the
    /// same tick), matching the rest of the iterator family. Returns an
    /// empty `Vec` when no track carries an effects-depth controller.
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn effect_depths(&self) -> Vec<EffectDepthEvent> {
        let mut out: Vec<EffectDepthEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::ControlChange { controller, value },
                }) = &ev.kind
                {
                    let scratch = ControlChangeEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
                        controller: *controller,
                        value: *value,
                    };
                    if let Some(depth) = scratch.effect_depth() {
                        out.push(EffectDepthEvent {
                            tick: abs,
                            track: track_idx,
                            channel: *channel,
                            depth,
                        });
                    }
                }
            }
        }
        out.sort_by_key(|c| c.tick);
        out
    }

    /// Collect every [`ChannelBody::PitchBend`] (`En lsb msb`)
    /// channel-voice event from every track, pinned to the absolute
    /// tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two events at the same tick keep the `(track,
    /// in-track-position)` order — track 0's events fire before track
    /// 1's at the same tick. This matches the same stable-merge rule
    /// [`SmfFile::tempo_map`] / [`SmfFile::time_signatures`] /
    /// [`SmfFile::key_signatures`] / [`SmfFile::markers`] /
    /// [`SmfFile::lyrics`] / [`SmfFile::cue_points`] /
    /// [`SmfFile::track_names`] / [`SmfFile::instrument_names`] /
    /// [`SmfFile::texts`] / [`SmfFile::copyrights`] /
    /// [`SmfFile::smpte_offsets`] / [`SmfFile::sequencer_specifics`] /
    /// [`SmfFile::sequence_numbers`] / [`SmfFile::midi_ports`] /
    /// [`SmfFile::channel_prefixes`] / [`SmfFile::program_changes`] /
    /// [`SmfFile::control_changes`] and the scheduler use
    /// (`scheduler.rs` §"merged event list, sorted by absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries an `En` event — a
    /// player initialising a fresh receiver should leave every channel
    /// at the no-bend centre (`0x2000`) until an explicit Pitch Bend
    /// arrives, per the spec convention.
    ///
    /// Only `En` is selected — neighbouring Control Change (`Bn`),
    /// Program Change (`Cn`), aftertouch (`An` / `Dn`), and note (`8n`
    /// / `9n`) channel-voice events stay on their own surfaces. Each
    /// entry's `value` field carries the combined 14-bit code
    /// `(msb << 7) | lsb`, `0..=0x3FFF`, with the no-bend centre at
    /// `0x2000`; [`PitchBendEvent::signed_value`] converts it to the
    /// signed `-8192..=8191` displacement from centre and
    /// [`PitchBendEvent::is_centre`] flags the no-bend position.
    /// Companion primitive [`SmfFile::channel_snapshot_at`] folds the
    /// *last* pitch bend per channel into
    /// [`SmfChannelSnapshot::pitch_bend`] for seek initialisation; this
    /// helper surfaces *every* bend in chronological order so callers
    /// building a full bend-automation timeline (a DAW bend-lane
    /// editor, a glissando / vibrato curve renderer) don't have to
    /// re-walk the channel-voice stream manually. Resolving the 14-bit
    /// code to an actual pitch displacement requires the channel's
    /// Pitch Bend Sensitivity (RPN 0, default ±2 semitones), which the
    /// helper leaves to the receiving application.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn pitch_bends(&self) -> Vec<PitchBendEvent> {
        let mut out: Vec<PitchBendEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::PitchBend { value },
                }) = &ev.kind
                {
                    out.push(PitchBendEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
                        value: *value,
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

    /// Collect every [`ChannelBody::PolyAftertouch`] (`An kk pp`)
    /// channel-voice event from every track, pinned to the absolute
    /// tick at which it fires, in time order.
    ///
    /// `An kk pp` is the Polyphonic Key Pressure (per-key aftertouch)
    /// message: the low nibble of the status byte is the channel index
    /// (`0..=15`), the first data byte `kk` (`0..=127`) names the key,
    /// and the second data byte `pp` (`0..=127`) carries, per the MIDI
    /// 1.0 *Summary of MIDI Messages* Table 1, the per-key pressure
    /// value. It is distinct from Channel Pressure (`Dn`, the single
    /// greatest pressure over all depressed keys), which travels on its
    /// own surface ([`SmfFile::channel_pressures`]).
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two events at the same tick keep the `(track,
    /// in-track-position)` order — track 0's events fire before track
    /// 1's at the same tick. This matches the same stable-merge rule
    /// [`SmfFile::channel_pressures`] / [`SmfFile::control_changes`] /
    /// [`SmfFile::pitch_bends`] / [`SmfFile::notes`] and the scheduler
    /// use (`scheduler.rs` §"merged event list, sorted by absolute
    /// tick").
    ///
    /// Returns an empty `Vec` when no track carries an `An` event — a
    /// player initialising a fresh receiver should leave every key at
    /// zero pressure until an explicit Polyphonic Key Pressure arrives,
    /// per the spec convention.
    ///
    /// Only `An` is selected — neighbouring Control Change (`Bn`),
    /// Program Change (`Cn`), pitch-bend (`En`), channel-pressure
    /// (`Dn`), and note (`8n` / `9n`) channel-voice events stay on their
    /// own surfaces. Each entry's `pressure` field carries the raw
    /// `0..=127` value; the pressure's musical effect (typically routed
    /// per-voice to volume, vibrato depth, or filter cutoff by the
    /// receiving instrument) is the receiver's concern, so the helper
    /// stays routing-agnostic. This helper surfaces *every* Polyphonic
    /// Key Pressure in chronological order so callers building a full
    /// per-key pressure-automation timeline (a DAW poly-pressure lane
    /// editor, a soft-synth per-voice aftertouch rebuilder) don't have to
    /// re-walk the channel-voice stream manually.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn poly_aftertouches(&self) -> Vec<PolyAftertouchEvent> {
        let mut out: Vec<PolyAftertouchEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::PolyAftertouch { key, pressure },
                }) = &ev.kind
                {
                    out.push(PolyAftertouchEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
                        key: *key,
                        pressure: *pressure,
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

    /// Collect every [`ChannelBody::ChannelAftertouch`] (`Dn pp`)
    /// channel-voice event from every track, pinned to the absolute
    /// tick at which it fires, in time order.
    ///
    /// `Dn pp` is the Channel Pressure (mono aftertouch) message: the
    /// low nibble of the status byte is the channel index (`0..=15`)
    /// and the single data byte `pp` (`0..=127`) carries, per the MIDI
    /// 1.0 *Summary of MIDI Messages* Table 1, "the single greatest
    /// pressure value (of all the current depressed keys)". It is
    /// distinct from polyphonic key pressure (`An`, per-key), which
    /// travels on its own surface.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in the
    /// same track), then the per-track sequences are merged. The sort
    /// is stable so two events at the same tick keep the `(track,
    /// in-track-position)` order — track 0's events fire before track
    /// 1's at the same tick. This matches the same stable-merge rule
    /// [`SmfFile::tempo_map`] / [`SmfFile::time_signatures`] /
    /// [`SmfFile::key_signatures`] / [`SmfFile::markers`] /
    /// [`SmfFile::lyrics`] / [`SmfFile::cue_points`] /
    /// [`SmfFile::track_names`] / [`SmfFile::instrument_names`] /
    /// [`SmfFile::texts`] / [`SmfFile::copyrights`] /
    /// [`SmfFile::smpte_offsets`] / [`SmfFile::sequencer_specifics`] /
    /// [`SmfFile::sequence_numbers`] / [`SmfFile::midi_ports`] /
    /// [`SmfFile::channel_prefixes`] / [`SmfFile::program_changes`] /
    /// [`SmfFile::control_changes`] / [`SmfFile::pitch_bends`] and the
    /// scheduler use (`scheduler.rs` §"merged event list, sorted by
    /// absolute tick").
    ///
    /// Returns an empty `Vec` when no track carries a `Dn` event — a
    /// player initialising a fresh receiver should leave every channel
    /// at zero pressure until an explicit Channel Pressure arrives, per
    /// the spec convention.
    ///
    /// Only `Dn` is selected — neighbouring Control Change (`Bn`),
    /// Program Change (`Cn`), pitch-bend (`En`), polyphonic-aftertouch
    /// (`An`), and note (`8n` / `9n`) channel-voice events stay on
    /// their own surfaces. Each entry's `pressure` field carries the
    /// raw `0..=127` value; the pressure's musical effect (typically
    /// routed to volume, vibrato depth, or filter cutoff by the
    /// receiving instrument) is the receiver's concern, so the helper
    /// stays routing-agnostic. This helper surfaces *every* Channel
    /// Pressure in chronological order so callers building a full
    /// pressure-automation timeline (a DAW pressure-lane editor, a
    /// soft-synth channel-pressure rebuilder) don't have to re-walk the
    /// channel-voice stream manually.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn channel_pressures(&self) -> Vec<ChannelPressureEvent> {
        let mut out: Vec<ChannelPressureEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::ChannelAftertouch { pressure },
                }) = &ev.kind
                {
                    out.push(ChannelPressureEvent {
                        tick: abs,
                        track: track_idx,
                        channel: *channel,
                        pressure: *pressure,
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

    /// Pair every Note On (`9n key vel`, `vel > 0`) with the Note Off
    /// that releases it, returning one [`Note`] span per sounding note,
    /// in onset order.
    ///
    /// Where the channel-voice helpers ([`SmfFile::program_changes`] /
    /// [`SmfFile::control_changes`] / [`SmfFile::pitch_bends`] /
    /// [`SmfFile::channel_pressures`]) surface one value per *wire*
    /// event, this helper joins the two wire events that bracket a
    /// sounding note — its Note On and its Note Off — into a single span
    /// carrying the note's `start_tick`, `end_tick`, and (via
    /// [`Note::duration_ticks`]) its length. That is the primitive a
    /// piano-roll / DAW note-lane view consumes directly.
    ///
    /// **Merge + tie-break.** Events are walked over the *globally*
    /// merged stream sorted by `(absolute tick, track, in-track
    /// position)` — the same stable-merge convention every other
    /// iteration helper and the scheduler use (`scheduler.rs` §"merged
    /// event list, sorted by absolute tick"). This matters because a
    /// note's Note Off can, in principle, land on a different track from
    /// its Note On; matching over the merged stream pairs them
    /// correctly. The returned `Vec` is ordered by `(start_tick, track,
    /// in-track position)` so two notes struck on the same tick keep
    /// track 0 before track 1, then on-disk order within a track.
    ///
    /// **On/off matching.** A note is opened on a `9n key vel` with
    /// `vel > 0` and closed by the *next* release for the same
    /// `(channel, key)`. A release is either an `8n key off_vel` (the
    /// explicit Note-Off form, whose second data byte becomes
    /// [`Note::off_velocity`]) or a `9n key 0` — the velocity-0
    /// convention from the MIDI 1.0 *Summary of MIDI Messages* Table 1,
    /// which closes the note with `off_velocity == 0`. When several
    /// notes of the same pitch on the same channel are held at once
    /// (re-struck before release), releases are matched **first-in,
    /// first-out**: the earliest still-open onset is the one closed.
    ///
    /// **Unmatched events.** A release with no open note of that
    /// `(channel, key)` is dropped (an interpreter has nothing to turn
    /// off). A Note On with no matching release before end-of-file is
    /// dropped from the returned list — a span needs both ends to carry
    /// a duration; callers needing the dangling onsets can read the raw
    /// note events through the channel stream. (Well-formed SMFs balance
    /// every Note On with a Note Off; a hanging note is a producer bug,
    /// and most receivers silence it at end-of-track.)
    ///
    /// Returns an empty `Vec` for a file with no note activity (e.g. a
    /// tempo-map-only conductor track). Cost is `O(n log n)` in the
    /// total event count for the merge sort, bounded above by
    /// [`MAX_EVENTS_PER_FILE`] (the same cap the parser enforces).
    pub fn notes(&self) -> Vec<Note> {
        // Walk the globally merged stream in (tick, track, order) so a
        // Note Off on a different track from its Note On still pairs.
        // Each entry records the channel-voice body plus its origin so
        // the resulting Note carries the onset track.
        struct AbsNote<'a> {
            tick: u64,
            track: usize,
            order: usize,
            body: &'a ChannelBody,
            channel: u8,
        }
        let mut merged: Vec<AbsNote> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for (order, ev) in track.events.iter().enumerate() {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage { channel, body }) = &ev.kind {
                    if matches!(
                        body,
                        ChannelBody::NoteOn { .. } | ChannelBody::NoteOff { .. }
                    ) {
                        merged.push(AbsNote {
                            tick: abs,
                            track: track_idx,
                            order,
                            body,
                            channel: *channel,
                        });
                    }
                }
            }
        }
        // Stable sort by (tick, track, order) — the scheduler convention.
        merged.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then_with(|| a.track.cmp(&b.track))
                .then_with(|| a.order.cmp(&b.order))
        });

        // FIFO of still-open onsets per (channel, key). Each pending
        // entry keeps the onset's tick / track / velocity so the closed
        // span can be emitted in full.
        struct Pending {
            start_tick: u64,
            track: usize,
            velocity: u8,
        }
        // 16 channels × 128 keys; lazily-grown FIFO per slot.
        let mut open: Vec<Vec<Pending>> = (0..(16 * 128)).map(|_| Vec::new()).collect();
        let slot = |channel: u8, key: u8| -> usize { (channel as usize) * 128 + key as usize };

        let mut out: Vec<Note> = Vec::new();
        for ev in &merged {
            match ev.body {
                ChannelBody::NoteOn { key, velocity } if *velocity > 0 => {
                    open[slot(ev.channel, *key)].push(Pending {
                        start_tick: ev.tick,
                        track: ev.track,
                        velocity: *velocity,
                    });
                }
                // Note On with velocity 0 is the running-status Note-Off
                // form: close the earliest open note of this pitch with a
                // zero release velocity.
                ChannelBody::NoteOn { key, velocity: _ } => {
                    let fifo = &mut open[slot(ev.channel, *key)];
                    if !fifo.is_empty() {
                        let p = fifo.remove(0);
                        out.push(Note {
                            start_tick: p.start_tick,
                            end_tick: ev.tick,
                            track: p.track,
                            channel: ev.channel,
                            key: *key,
                            velocity: p.velocity,
                            off_velocity: 0,
                        });
                    }
                }
                ChannelBody::NoteOff { key, velocity } => {
                    let fifo = &mut open[slot(ev.channel, *key)];
                    if !fifo.is_empty() {
                        let p = fifo.remove(0);
                        out.push(Note {
                            start_tick: p.start_tick,
                            end_tick: ev.tick,
                            track: p.track,
                            channel: ev.channel,
                            key: *key,
                            velocity: p.velocity,
                            off_velocity: *velocity,
                        });
                    }
                }
                _ => {}
            }
        }

        // Order the result by onset, then by the onset track + a stable
        // tie-break. Sorting by start_tick alone is unstable across
        // chord notes that started on the same tick on different tracks,
        // so include track and the (already in onset order) emission
        // order to keep track 0 before track 1.
        out.sort_by(|a, b| {
            a.start_tick
                .cmp(&b.start_tick)
                .then_with(|| a.track.cmp(&b.track))
        });
        out
    }

    /// Return every [`Note`] span sounding at the absolute tick `tick`,
    /// in onset order — the piano-roll / seek companion to
    /// [`SmfFile::notes`].
    ///
    /// A note is *sounding* at `tick` when it has been struck at or
    /// before `tick` and has not yet been released: `start_tick <= tick`
    /// **and** `end_tick > tick`. The interval is therefore half-open
    /// `[start_tick, end_tick)` — a note released at exactly `tick` is no
    /// longer sounding (its key has come up on that tick), and a note
    /// struck at exactly `tick` *is* sounding (the snapshot reflects the
    /// state immediately after that tick's events fire, the same "events
    /// at exactly `tick` are included" convention as
    /// [`SmfFile::channel_snapshot_at`]). A zero-duration note
    /// (`start_tick == end_tick`, a Note Off landing on its own onset
    /// tick) is sounding at *no* tick and never appears.
    ///
    /// This is the note-level analogue of the channel-state
    /// [`SmfFile::channel_snapshot_at`] seek primitive: where the
    /// snapshot answers "what controller / program / bend state does a
    /// channel carry at tick T?", this answers "which keys are held down
    /// at tick T?" — exactly the set a DAW must re-trigger (or a renderer
    /// must prime into the voice pool) when seeking into the middle of a
    /// file rather than playing from the top.
    ///
    /// The result reuses the matched spans from [`SmfFile::notes`]
    /// verbatim, so the same on/off pairing rules apply: the velocity-0
    /// Note-Off convention, FIFO matching of re-struck pitches, and the
    /// drop of unmatched releases / hanging onsets (a note with no
    /// matching Note Off before end-of-file carries no `end_tick`, so it
    /// cannot be reported as sounding at any tick). The returned `Vec`
    /// preserves the `notes()` order — `(start_tick, track)` — so chord
    /// notes struck together stay grouped and track 0 precedes track 1.
    ///
    /// Cost is `O(n log n)` in the total event count (it runs one
    /// [`SmfFile::notes`] pass then filters), bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn active_notes_at(&self, tick: u64) -> Vec<Note> {
        self.notes()
            .into_iter()
            .filter(|n| n.start_tick <= tick && n.end_tick > tick)
            .collect()
    }

    /// Fold the RPN / NRPN Data Entry pump into resolved parameter-write
    /// events: every CC 6 / CC 38 (Data Entry MSB / LSB) and CC 96 /
    /// CC 97 (Data Increment / Decrement) controller, paired with the
    /// parameter the channel had selected when it fired.
    ///
    /// This is the read-side analogue of the runtime RPN / NRPN state
    /// machine a synth maintains, and the resolving companion to
    /// [`SmfFile::control_changes`]: where that helper surfaces the *raw*
    /// CC bytes and explicitly leaves "the Data Entry pump for RPN /
    /// NRPN parameter writes … to the receiving application", this helper
    /// runs that state machine for the caller and emits one
    /// [`ParameterDataEntry`] per pump action, tagged with the resolved
    /// target.
    ///
    /// ## Selector state machine (MIDI 1.0 *Control Change Messages —
    /// Data Bytes*)
    ///
    /// Per the spec's "To set or change the value of a Registered
    /// Parameter" procedure and Table 3 / Table 3a, the Data Entry pump
    /// acts on whichever parameter was *most recently selected*:
    ///
    /// * **RPN select** — CC 101 (`0x65`) sets the parameter MSB, CC 100
    ///   (`0x64`) the LSB. The two combine to a 14-bit Registered
    ///   Parameter Number classified per Table 3a.
    /// * **NRPN select** — CC 99 (`0x63`) sets the MSB, CC 98 (`0x62`)
    ///   the LSB. The 14-bit number is manufacturer-private, surfaced
    ///   raw.
    /// * Selecting an RPN supersedes a prior NRPN selection and vice
    ///   versa: there is **one** active parameter per channel, not one of
    ///   each. The half-set high / low byte of the *current* flavour is
    ///   preserved while the other byte is rewritten (a lone CC 100 after
    ///   a full RPN selection changes only the LSB), matching the
    ///   bit-merge the runtime mixer performs.
    /// * **Null** — the Null Function Number (packed `0x3FFF`, the
    ///   `7FH:7FH` MSB:LSB pair, selected through either pair) disables
    ///   the pump. A Data Entry / Increment /
    ///   Decrement arriving while the active parameter is null — or
    ///   before *any* parameter has been selected — produces **no** event
    ///   (the spec: "Setting RPN to 7FH,7FH will disable the data entry,
    ///   data increment, and data decrement controllers until a new RPN
    ///   or NRPN is selected").
    ///
    /// Data Increment (CC 96) and Data Decrement (CC 97) carry a
    /// "don't care" value byte per MMA RP-018; the emitted
    /// [`DataEntryAction::Increment`] / [`DataEntryAction::Decrement`]
    /// therefore drop the byte and record only the ±1 direction.
    ///
    /// ## Merge order
    ///
    /// The selector state is per-channel and evolves along the *globally
    /// merged* event stream — every track's events are summed to absolute
    /// ticks and merged in `(tick, track, in-track-position)` order (the
    /// same stable convention [`SmfFile::notes`] and the scheduler use)
    /// before the fold runs, so a selector on the conductor track
    /// correctly governs a Data Entry on a later part track at a higher
    /// tick. Within a tick, track 0 precedes track 1. The returned `Vec`
    /// is in that merged order.
    ///
    /// Returns an empty `Vec` when no Data Entry pump controller is ever
    /// addressed to a non-null selected parameter — the common case for a
    /// note-only sequence.
    ///
    /// Cost is `O(n log n)` in the total event count (one merge sort then
    /// a linear fold), bounded above by [`MAX_EVENTS_PER_FILE`].
    pub fn parameter_data_entries(&self) -> Vec<ParameterDataEntry> {
        // Build the globally merged channel-voice CC stream in
        // (tick, track, order) so a selector on one track governs a Data
        // Entry on another, exactly as a receiver merging both tracks
        // onto one wire would see them.
        struct AbsCc {
            tick: u64,
            track: usize,
            order: usize,
            channel: u8,
            controller: u8,
            value: u8,
        }
        let mut merged: Vec<AbsCc> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for (order, ev) in track.events.iter().enumerate() {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Channel(ChannelMessage {
                    channel,
                    body: ChannelBody::ControlChange { controller, value },
                }) = &ev.kind
                {
                    merged.push(AbsCc {
                        tick: abs,
                        track: track_idx,
                        order,
                        channel: *channel,
                        controller: *controller,
                        value: *value,
                    });
                }
            }
        }
        merged.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then_with(|| a.track.cmp(&b.track))
                .then_with(|| a.order.cmp(&b.order))
        });

        // Per-channel running selector. `number` is the packed 14-bit
        // RPN/NRPN value `(msb << 7) | lsb`; `registered` distinguishes
        // the flavour. The power-up default is the Null Function Number
        // (packed `0x3FFF`, the `7FH:7FH` MSB:LSB pair) with the
        // registered flag set — the spec's "RPN / NRPN MSB / LSB 127 /
        // 127" power-up convention — so the pump is disabled until an
        // explicit selection arrives.
        #[derive(Clone, Copy)]
        struct Selector {
            number: u16,
            registered: bool,
        }
        let mut sel = [Selector {
            number: 0x3FFF,
            registered: true,
        }; 16];

        let mut out: Vec<ParameterDataEntry> = Vec::new();
        for cc in &merged {
            let ch = (cc.channel & 0x0F) as usize;
            match cc.controller {
                // RPN MSB / LSB (CC 101 / CC 100): switch to the
                // registered flavour, rewriting only the addressed byte.
                101 => {
                    let cur = if sel[ch].registered {
                        sel[ch].number
                    } else {
                        0
                    };
                    sel[ch] = Selector {
                        number: (cur & 0x007F) | ((cc.value as u16 & 0x7F) << 7),
                        registered: true,
                    };
                }
                100 => {
                    let cur = if sel[ch].registered {
                        sel[ch].number
                    } else {
                        0
                    };
                    sel[ch] = Selector {
                        number: (cur & 0x3F80) | (cc.value as u16 & 0x7F),
                        registered: true,
                    };
                }
                // NRPN MSB / LSB (CC 99 / CC 98): switch to the
                // non-registered flavour, rewriting only the addressed
                // byte.
                99 => {
                    let cur = if !sel[ch].registered {
                        sel[ch].number
                    } else {
                        0
                    };
                    sel[ch] = Selector {
                        number: (cur & 0x007F) | ((cc.value as u16 & 0x7F) << 7),
                        registered: false,
                    };
                }
                98 => {
                    let cur = if !sel[ch].registered {
                        sel[ch].number
                    } else {
                        0
                    };
                    sel[ch] = Selector {
                        number: (cur & 0x3F80) | (cc.value as u16 & 0x7F),
                        registered: false,
                    };
                }
                // Data Entry MSB / LSB and Data Inc / Dec — emit only
                // when a non-null parameter is selected.
                6 | 38 | 96 | 97 => {
                    let s = sel[ch];
                    if s.number == 0x3FFF {
                        continue; // null / power-up sentinel disables the pump
                    }
                    let parameter = if s.registered {
                        SelectedParameter::Registered {
                            number: s.number,
                            param: RegisteredParameter::from_number(s.number),
                        }
                    } else {
                        SelectedParameter::NonRegistered { number: s.number }
                    };
                    let action = match cc.controller {
                        6 => DataEntryAction::EntryMsb(cc.value),
                        38 => DataEntryAction::EntryLsb(cc.value),
                        96 => DataEntryAction::Increment,
                        _ => DataEntryAction::Decrement,
                    };
                    out.push(ParameterDataEntry {
                        tick: cc.tick,
                        track: cc.track,
                        channel: cc.channel & 0x0F,
                        parameter,
                        action,
                    });
                }
                _ => {}
            }
        }
        out
    }

    /// Collect every [`Event::Sysex`] from every track — both the
    /// `F0` start and the `F7` continuation / escape flavours —
    /// pinned to the absolute tick at which it fires, in time order.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in
    /// the same track), then the per-track sequences are merged.
    /// The sort is stable so two packets at the same tick keep the
    /// `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::tempo_map`] /
    /// [`SmfFile::time_signatures`] / [`SmfFile::key_signatures`] /
    /// [`SmfFile::markers`] / [`SmfFile::lyrics`] /
    /// [`SmfFile::cue_points`] / [`SmfFile::track_names`] /
    /// [`SmfFile::instrument_names`] / [`SmfFile::texts`] /
    /// [`SmfFile::copyrights`] / [`SmfFile::smpte_offsets`] /
    /// [`SmfFile::sequencer_specifics`] /
    /// [`SmfFile::sequence_numbers`] / [`SmfFile::midi_ports`] /
    /// [`SmfFile::channel_prefixes`] and the scheduler use
    /// (`scheduler.rs` §"merged event list, sorted by absolute
    /// tick").
    ///
    /// Returns an empty `Vec` when no track carries an `F0` or `F7`
    /// event — most music-only sequences omit SysEx entirely.
    ///
    /// Both wire forms are surfaced. Callers reconstructing a
    /// multi-packet SysEx assembly walk the list with
    /// `is_escape == false` opening a fresh packet and each
    /// subsequent `is_escape == true` appending until the payload
    /// terminates with an `0xF7` end marker. The
    /// [`MetaEvent::SequencerSpecific`](MetaEvent::SequencerSpecific)
    /// channel — `FF 7F` private payloads, surfaced through
    /// [`SmfFile::sequencer_specifics`] — is *not* selected here;
    /// the two channels carry different semantics (SysEx travels to
    /// the MIDI wire; `FF 7F` is file-private metadata that does not)
    /// and a file may carry both an `F0 7E 7F 09 01 F7` Universal
    /// Non-Real-Time GM-On packet on the conductor track and a
    /// private `FF 7F` plugin-state blob alongside it.
    ///
    /// The payload bytes are surfaced verbatim — the parser does
    /// not strip a trailing `F7` from an `F0` payload, so a writer
    /// can round-trip the helper output through
    /// [`SmfFile::to_bytes`] without re-synthesising the SysEx
    /// framing.
    ///
    /// Lifts the SMF event iterator family by surfacing the SysEx
    /// channel alongside the 15 meta-event helpers; the meta-event
    /// list itself stays at 15 (`SmfFile::{tempo_map,
    /// time_signatures, key_signatures, markers, lyrics,
    /// cue_points, track_names, instrument_names, texts,
    /// copyrights, smpte_offsets, sequencer_specifics,
    /// sequence_numbers, midi_ports, channel_prefixes}`).
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn sysex_events(&self) -> Vec<SysExEvent> {
        let mut out: Vec<SysExEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Sysex { escape, data } = &ev.kind {
                    out.push(SysExEvent {
                        tick: abs,
                        track: track_idx,
                        is_escape: *escape,
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

    /// Fold the per-track `F0` / `F7` SysEx packet stream into complete
    /// logical [`ReassembledSysEx`] messages.
    ///
    /// The SMF format permits a single SysEx message to be split across
    /// track events: an `F0` opener whose payload does not end in `0xF7`
    /// is continued by `F7` packets until one ends in `0xF7` (EOX). This
    /// helper runs that continuation state machine per track and emits
    /// one message per chain, in time order (stably sorted by the
    /// opener's absolute tick).
    ///
    /// Rules:
    /// * An `F0` packet ending in `0xF7` is a self-contained message
    ///   (`packet_count == 1`, `complete == true`).
    /// * An `F0` packet *not* ending in `0xF7` opens a chain; subsequent
    ///   `F7` packets on the same track append until one ends in `0xF7`.
    /// * An `F7` packet with no chain open is a standalone **escape**
    ///   sequence (the SMF "escaped" form for arbitrary wire bytes) and
    ///   is surfaced as its own message — `complete` reflects whether
    ///   its own payload ends in `0xF7`. Escapes are not manufacturer-
    ///   prefixed, so [`ReassembledSysEx::id_byte`] reports their raw
    ///   first byte without further meaning.
    /// * A new `F0` opener while a chain is still open closes the prior
    ///   chain as `complete == false` (the producer never sent its EOX)
    ///   before starting the new one. A track that ends mid-chain
    ///   likewise flushes the partial message as `complete == false`.
    ///
    /// The reassembled [`ReassembledSysEx::body`] strips the trailing
    /// `0xF7` (when present) but keeps every manufacturer byte, so a
    /// caller routing by manufacturer / universal ID reads
    /// `body[0]` directly.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn reassembled_sysex_messages(&self) -> Vec<ReassembledSysEx> {
        let mut out: Vec<ReassembledSysEx> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            // The currently-open chain, if any: (opener tick, body so
            // far, packet count). `body` accumulates payloads with the
            // EOX stripping deferred to flush time.
            let mut open: Option<(u64, Vec<u8>, usize)> = None;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                let Event::Sysex { escape, data } = &ev.kind else {
                    continue;
                };
                let ends_eox = matches!(data.last(), Some(&0xF7));
                if !*escape {
                    // F0 opener. Any in-progress chain is abandoned
                    // (no EOX arrived) before this opener starts.
                    if let Some((tick, body, count)) = open.take() {
                        out.push(finish_sysex(tick, track_idx, body, false, count));
                    }
                    if ends_eox {
                        // Self-contained single packet.
                        out.push(finish_sysex(abs, track_idx, data.clone(), true, 1));
                    } else {
                        open = Some((abs, data.clone(), 1));
                    }
                } else {
                    // F7 packet.
                    match open.as_mut() {
                        Some((_, body, count)) => {
                            body.extend_from_slice(data);
                            *count += 1;
                            if ends_eox {
                                let (tick, body, count) = open.take().unwrap();
                                out.push(finish_sysex(tick, track_idx, body, true, count));
                            }
                        }
                        None => {
                            // Standalone escape: its own one-packet
                            // message. `complete` mirrors whether the
                            // escape payload itself terminates in EOX.
                            out.push(finish_sysex(abs, track_idx, data.clone(), ends_eox, 1));
                        }
                    }
                }
            }
            // Track ended mid-chain: flush the partial as incomplete.
            if let Some((tick, body, count)) = open.take() {
                out.push(finish_sysex(tick, track_idx, body, false, count));
            }
        }
        out.sort_by_key(|m| m.tick);
        out
    }

    /// Collect every **Universal** System Exclusive packet from every
    /// track — `F0 7E …` (Non-Real-Time) and `F0 7F …` (Real-Time)
    /// only — pinned to the absolute tick at which it fires, in time
    /// order, with the Table-4 classification eagerly resolved.
    ///
    /// The universal-only view is the typed counterpart of
    /// [`SmfFile::sysex_events`]: this helper walks the same per-track
    /// `Event::Sysex` stream, calls
    /// [`SysExEvent::universal_classification`] on each, keeps only the
    /// packets that classify as `Some(_)`, and returns a
    /// [`UniversalSysExEvent`] for each. Manufacturer-prefixed `F0`
    /// packets (Roland `0x41`, Yamaha `0x43`, any leading byte other
    /// than `0x7E` / `0x7F`) and `F7` continuation / escape packets are
    /// filtered out — callers interested in those route through
    /// [`SmfFile::sysex_events`] directly. `F0` packets truncated
    /// before the Sub-ID #1 byte (a payload shorter than 3 bytes) are
    /// also filtered, matching the contract of the underlying
    /// classifier.
    ///
    /// The cumulative delta is summed per-track (each track's
    /// [`TrackEvent::delta`] is relative to the previous event in
    /// the same track), then the per-track sequences are merged. The
    /// sort is stable so two universal packets at the same tick keep
    /// the `(track, in-track-position)` order — track 0's events fire
    /// before track 1's at the same tick. This matches the same
    /// stable-merge rule [`SmfFile::sysex_events`] /
    /// [`SmfFile::tempo_map`] / [`SmfFile::time_signatures`] /
    /// [`SmfFile::key_signatures`] / [`SmfFile::markers`] /
    /// [`SmfFile::lyrics`] / [`SmfFile::cue_points`] /
    /// [`SmfFile::track_names`] / [`SmfFile::instrument_names`] /
    /// [`SmfFile::texts`] / [`SmfFile::copyrights`] /
    /// [`SmfFile::smpte_offsets`] /
    /// [`SmfFile::sequencer_specifics`] /
    /// [`SmfFile::sequence_numbers`] / [`SmfFile::midi_ports`] /
    /// [`SmfFile::channel_prefixes`] and the scheduler use
    /// (`scheduler.rs` §"merged event list, sorted by absolute
    /// tick").
    ///
    /// Returns an empty `Vec` when no track carries a universal SysEx
    /// packet — music-only sequences without GM-On / Master Volume /
    /// MTC / MMC produce the empty list.
    ///
    /// The verbatim payload bytes are surfaced on
    /// [`UniversalSysExEvent::data`] so callers reading Sub-ID #2-
    /// derived arguments (Master Volume LSB / MSB, MTC `hr mn se fr`,
    /// MTS note + tuning triple, …) don't have to re-walk
    /// [`SmfFile::sysex_events`] alongside this list.
    ///
    /// Cost is linear in the total event count and bounded above by
    /// [`MAX_EVENTS_PER_FILE`].
    pub fn universal_sysex_events(&self) -> Vec<UniversalSysExEvent> {
        let mut out: Vec<UniversalSysExEvent> = Vec::new();
        for (track_idx, track) in self.tracks.iter().enumerate() {
            let mut abs: u64 = 0;
            for ev in &track.events {
                abs = abs.saturating_add(ev.delta as u64);
                if let Event::Sysex { escape, data } = &ev.kind {
                    // Re-use SysExEvent's classifier so the universal-
                    // view stays in lock-step with the per-event API:
                    // any future tweak to the Table 4 vocabulary lands
                    // in one place.
                    let scratch = SysExEvent {
                        tick: abs,
                        track: track_idx,
                        is_escape: *escape,
                        data: data.clone(),
                    };
                    if let Some(classification) = scratch.universal_classification() {
                        out.push(UniversalSysExEvent {
                            tick: abs,
                            track: track_idx,
                            classification,
                            data: scratch.data,
                        });
                    }
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

/// Build a [`ReassembledSysEx`] from an accumulated payload, stripping
/// a single trailing `0xF7` EOX marker (when present) from `body`.
fn finish_sysex(
    tick: u64,
    track: usize,
    mut body: Vec<u8>,
    complete: bool,
    packet_count: usize,
) -> ReassembledSysEx {
    if matches!(body.last(), Some(&0xF7)) {
        body.pop();
    }
    ReassembledSysEx {
        tick,
        track,
        body,
        complete,
        packet_count,
    }
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
        self.encode(RunningStatus::Explicit)
    }

    /// Serialise this [`SmfFile`] using **running-status compression**.
    ///
    /// Identical to [`to_bytes`](Self::to_bytes) except that a
    /// channel-voice event whose status byte equals the one emitted by
    /// the immediately preceding event omits the redundant status byte,
    /// per the SMF running-status rule (MIDI 1.0 Detailed Specification,
    /// "Running Status"). The running-status buffer is cleared by every
    /// meta event and every sysex event (`F0` / `F7`) — exactly the
    /// reset points the parser observes — so the compressed stream still
    /// round-trips byte-for-byte through [`parse`] back to this
    /// [`SmfFile`].
    ///
    /// A Note On with velocity 0 is left as written (the encoder does
    /// not rewrite `8n key 0` into a running `9n` Note Off, nor vice
    /// versa — it compresses only status bytes that are already equal,
    /// preserving the caller's event shapes). Same validation as
    /// [`to_bytes`](Self::to_bytes).
    pub fn to_bytes_running_status(&self) -> Result<Vec<u8>> {
        self.encode(RunningStatus::Compress)
    }

    fn encode(&self, rs: RunningStatus) -> Result<Vec<u8>> {
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
            write_track_chunk(&mut out, track, rs)
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
        write_track_chunk(&mut out, self, RunningStatus::Explicit)?;
        Ok(out)
    }

    /// Serialise this [`Track`] as a single `MTrk` chunk using
    /// running-status compression. See
    /// [`SmfFile::to_bytes_running_status`] for the compression rule.
    pub fn to_bytes_chunk_running_status(&self) -> Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        write_track_chunk(&mut out, self, RunningStatus::Compress)?;
        Ok(out)
    }
}

/// Whether a track serialiser emits an explicit status byte on every
/// channel-voice event or compresses consecutive same-status events
/// with running status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunningStatus {
    /// Always emit the status byte (the default, bit-stable output).
    Explicit,
    /// Omit the status byte when it equals the previously-emitted one.
    Compress,
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

fn write_track_chunk(out: &mut Vec<u8>, track: &Track, rs: RunningStatus) -> Result<()> {
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
    // The running-status buffer is local to each track: it starts
    // cleared and is reset by every meta / sysex event the same way the
    // parser resets it. Unused when `rs == Explicit`.
    let mut running: Option<u8> = None;
    for (i, ev) in track.events.iter().enumerate() {
        write_event(out, ev, rs, &mut running)
            .map_err(|e| Error::invalid(format!("SMF: event {i}: {msg}", msg = err_msg(&e))))?;
    }
    let body_len = out.len() - body_start;
    let len_be = (body_len as u32).to_be_bytes();
    out[len_pos..len_pos + 4].copy_from_slice(&len_be);
    Ok(())
}

fn write_event(
    out: &mut Vec<u8>,
    ev: &TrackEvent,
    rs: RunningStatus,
    running: &mut Option<u8>,
) -> Result<()> {
    write_vlq(out, ev.delta)?;
    match &ev.kind {
        Event::Channel(msg) => write_channel(out, msg, rs, running),
        Event::Sysex { escape, data } => {
            out.push(if *escape { 0xF7 } else { 0xF0 });
            write_vlq(out, u32_len(data.len(), "sysex payload")?)?;
            out.extend_from_slice(data);
            // Sysex clears the running-status buffer.
            *running = None;
            Ok(())
        }
        Event::Meta(meta) => {
            write_meta(out, meta)?;
            // Meta events clear the running-status buffer.
            *running = None;
            Ok(())
        }
    }
}

/// Status byte (high nibble | channel) and the two-or-one data bytes
/// for a channel-voice message. Returns the status and the body bytes
/// so the caller can decide whether to emit the status or rely on
/// running status.
fn channel_wire_bytes(msg: &ChannelMessage) -> Result<(u8, [u8; 2], usize)> {
    if msg.channel > 0x0F {
        return Err(Error::invalid(format!(
            "SMF: channel {} out of range 0..=15",
            msg.channel,
        )));
    }
    let chan = msg.channel & 0x0F;
    let (status, b0, b1, n) = match msg.body {
        ChannelBody::NoteOff { key, velocity } => {
            check_data_byte(key, "NoteOff.key")?;
            check_data_byte(velocity, "NoteOff.velocity")?;
            (0x80 | chan, key, velocity, 2)
        }
        ChannelBody::NoteOn { key, velocity } => {
            check_data_byte(key, "NoteOn.key")?;
            check_data_byte(velocity, "NoteOn.velocity")?;
            (0x90 | chan, key, velocity, 2)
        }
        ChannelBody::PolyAftertouch { key, pressure } => {
            check_data_byte(key, "PolyAftertouch.key")?;
            check_data_byte(pressure, "PolyAftertouch.pressure")?;
            (0xA0 | chan, key, pressure, 2)
        }
        ChannelBody::ControlChange { controller, value } => {
            check_data_byte(controller, "ControlChange.controller")?;
            check_data_byte(value, "ControlChange.value")?;
            (0xB0 | chan, controller, value, 2)
        }
        ChannelBody::ProgramChange { program } => {
            check_data_byte(program, "ProgramChange.program")?;
            (0xC0 | chan, program, 0, 1)
        }
        ChannelBody::ChannelAftertouch { pressure } => {
            check_data_byte(pressure, "ChannelAftertouch.pressure")?;
            (0xD0 | chan, pressure, 0, 1)
        }
        ChannelBody::PitchBend { value } => {
            if value > 0x3FFF {
                return Err(Error::invalid(format!(
                    "SMF: PitchBend value {value:#06X} exceeds 14-bit range 0..=0x3FFF",
                )));
            }
            let lsb = (value & 0x7F) as u8;
            let msb = ((value >> 7) & 0x7F) as u8;
            (0xE0 | chan, lsb, msb, 2)
        }
    };
    Ok((status, [b0, b1], n))
}

fn write_channel(
    out: &mut Vec<u8>,
    msg: &ChannelMessage,
    rs: RunningStatus,
    running: &mut Option<u8>,
) -> Result<()> {
    let (status, data, n) = channel_wire_bytes(msg)?;
    // Emit the status byte unless running-status compression is on AND
    // the previously-emitted status already equals this one.
    let omit = rs == RunningStatus::Compress && *running == Some(status);
    if !omit {
        out.push(status);
    }
    out.extend_from_slice(&data[..n]);
    *running = Some(status);
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

    // ───────── TempoTimeline / SmfFile::tick_to_seconds ─────────

    #[test]
    fn tick_to_seconds_default_tempo_120bpm() {
        // No FF 51 → default 500_000 µs/qn = 120 BPM. At 480 ticks per
        // quarter, one quarter note (480 ticks) lasts 0.5 s.
        let events: &[u8] = &[0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        // 0.5 s/quarter / 480 ticks = 1/960 s per tick.
        assert!((smf.tick_to_seconds(0) - 0.0).abs() < 1e-12);
        assert!((smf.tick_to_seconds(480) - 0.5).abs() < 1e-9);
        assert!((smf.tick_to_seconds(960) - 1.0).abs() < 1e-9);
        assert!((smf.tick_to_seconds(240) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn tick_to_seconds_initial_tempo_change() {
        // delta=0 FF 51 03 03 D0 90 → 250_000 µs/qn = 240 BPM. One
        // quarter (480 ticks) now lasts 0.25 s.
        let events: &[u8] = &[
            0x00, 0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        assert!((smf.tick_to_seconds(480) - 0.25).abs() < 1e-9);
        assert!((smf.tick_to_seconds(960) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn tick_to_seconds_piecewise_across_tempo_changes() {
        // tick 0   : 120 BPM (500_000) → 1/960 s/tick
        // tick 480 : 240 BPM (250_000) → 1/1920 s/tick
        // tick 960 :  60 BPM (1_000_000) → 1/480 s/tick
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 500_000
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]); // 250_000
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0xFF, 0x51, 0x03, 0x0F, 0x42, 0x40]); // 1_000_000
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let tl = smf.tempo_timeline();
        // First segment: 480 ticks @ 0.5 s/quarter = 0.5 s elapsed at tick 480.
        assert!((tl.tick_to_seconds(480) - 0.5).abs() < 1e-9);
        // Mid first segment.
        assert!((tl.tick_to_seconds(240) - 0.25).abs() < 1e-9);
        // Second segment: 480 ticks @ 0.25 s/quarter = +0.25 → 0.75 s at tick 960.
        assert!((tl.tick_to_seconds(960) - 0.75).abs() < 1e-9);
        // Mid second segment (tick 720, 240 ticks in): 0.5 + 240/1920.
        assert!((tl.tick_to_seconds(720) - (0.5 + 240.0 / 1920.0)).abs() < 1e-9);
        // Third segment: 480 ticks @ 1.0 s/quarter = +1.0 → 1.75 s at tick 1440.
        assert!((tl.tick_to_seconds(1440) - 1.75).abs() < 1e-9);
        // Monotonic non-decreasing.
        let mut prev = 0.0;
        for t in [0u64, 100, 480, 720, 960, 1200, 1440, 5000] {
            let s = tl.tick_to_seconds(t);
            assert!(s >= prev - 1e-12, "non-monotonic at tick {t}: {s} < {prev}");
            prev = s;
        }
    }

    #[test]
    fn tick_to_seconds_smpte_ignores_tempo() {
        // 25 fps, 40 ticks/frame → 1000 ticks/second exactly. A Set
        // Tempo event must NOT change the rate for an SMPTE division.
        let div = u16::from_be_bytes([0xE7, 0x28]); // -25 fps, 40 tpf
        let mut events: Vec<u8> = Vec::new();
        // A tempo change that would alter a musical division — ignored here.
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, div);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        // 1 / (25 * 40) = 1/1000 s per tick.
        assert!((smf.tick_to_seconds(1000) - 1.0).abs() < 1e-9);
        assert!((smf.tick_to_seconds(500) - 0.5).abs() < 1e-9);
        // Only the tick-0 anchor segment exists; tempo events skipped.
        assert_eq!(smf.tempo_timeline().segments().len(), 1);
    }

    #[test]
    fn duration_seconds_uses_max_end_tick_across_tracks() {
        // Track 0: 120 BPM, EOT at tick 960 (two quarters = 1.0 s).
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 500_000
        t0.extend_from_slice(&encode_vlq(960));
        t0.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        // Track 1: a note that ends at tick 1440 (three quarters = 1.5 s).
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        t1.extend_from_slice(&encode_vlq(1440));
        t1.extend_from_slice(&[0x80, 0x3C, 0x40]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        // Max end tick = 1440 → 1.5 s at 120 BPM.
        assert!((smf.duration_seconds() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn duration_seconds_empty_file_is_zero() {
        let events: &[u8] = &[0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(events));
        let smf = parse(&blob).unwrap();
        // Only an EOT at tick 0 → zero elapsed.
        assert!((smf.duration_seconds() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn tempo_timeline_same_tick_change_last_wins() {
        // Two tempo changes both at tick 0: the later one (250_000)
        // must govern the segment rate. Place them on two tracks so the
        // merge collapses them to the same tick.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 500_000
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90]); // 250_000
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let tl = smf.tempo_timeline();
        // Single segment at tick 0; rate from the last write (track 1,
        // 250_000 = 240 BPM): 480 ticks → 0.25 s.
        assert_eq!(tl.segments().len(), 1);
        assert!((tl.tick_to_seconds(480) - 0.25).abs() < 1e-9);
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

    // ───────── SmfFile::channel_prefixes (FF 20 01 cc) ─────────

    #[test]
    fn channel_prefixes_empty_when_no_meta_event_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.channel_prefixes().is_empty());
    }

    #[test]
    fn channel_prefixes_single_binding_at_tick_zero() {
        // delta=0 FF 20 01 03   (bind surrounding meta to channel 3)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![0x00, 0xFF, 0x20, 0x01, 0x03, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_prefixes();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].tick, 0);
        assert_eq!(cps[0].track, 0);
        assert_eq!(cps[0].channel, 3);
        assert_eq!(cps[0].channel(), Some(3));
    }

    #[test]
    fn channel_prefixes_spec_nibble_range_round_trips() {
        // Every `cc` from 0..=15 round-trips and channel() returns Some.
        for cc in 0u8..=15 {
            let events: Vec<u8> = vec![0x00, 0xFF, 0x20, 0x01, cc, 0x00, 0xFF, 0x2F, 0x00];
            let mut blob = header_chunk(0, 1, 96);
            blob.extend(track_chunk(&events));
            let smf = parse(&blob).unwrap();
            let cps = smf.channel_prefixes();
            assert_eq!(cps.len(), 1);
            assert_eq!(cps[0].channel, cc);
            assert_eq!(cps[0].channel(), Some(cc));
        }
    }

    #[test]
    fn channel_prefixes_out_of_spec_byte_surfaces_raw_and_channel_returns_none() {
        // `cc = 0x20` is out-of-spec (high nibble set). Parser preserves
        // the byte verbatim so the file still round-trips, but
        // ChannelPrefixEvent::channel() declines to mask and returns
        // None so the receiver knows to fall back rather than route to
        // an unintended channel.
        let events: Vec<u8> = vec![0x00, 0xFF, 0x20, 0x01, 0x20, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_prefixes();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].channel, 0x20);
        assert_eq!(cps[0].channel(), None);
    }

    #[test]
    fn channel_prefixes_merge_across_tracks_sorted_by_tick() {
        // track 0: delta=240 FF 20 01 02
        // track 1: delta=120 FF 20 01 05; delta=240 FF 20 01 09
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x20, 0x01, 0x02]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0xFF, 0x20, 0x01, 0x05]);
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x20, 0x01, 0x09]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_prefixes();
        assert_eq!(cps.len(), 3);
        assert_eq!(cps[0].tick, 120);
        assert_eq!(cps[0].track, 1);
        assert_eq!(cps[0].channel, 0x05);
        assert_eq!(cps[1].tick, 240);
        assert_eq!(cps[1].track, 0);
        assert_eq!(cps[1].channel, 0x02);
        assert_eq!(cps[2].tick, 360);
        assert_eq!(cps[2].track, 1);
        assert_eq!(cps[2].channel, 0x09);
    }

    #[test]
    fn channel_prefixes_stable_sort_keeps_track0_before_track1_at_same_tick() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(240));
        t0.extend_from_slice(&[0xFF, 0x20, 0x01, 0x07]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(240));
        t1.extend_from_slice(&[0xFF, 0x20, 0x01, 0x08]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 480);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_prefixes();
        assert_eq!(cps.len(), 2);
        assert_eq!(cps[0].track, 0);
        assert_eq!(cps[0].channel, 0x07);
        assert_eq!(cps[1].track, 1);
        assert_eq!(cps[1].channel, 0x08);
    }

    #[test]
    fn channel_prefixes_filter_excludes_port_and_other_meta_kinds() {
        // FF 20 picked up; FF 21 port-hint sibling stays filtered out
        // (different meta kind, different routing semantics); text +
        // smpte + tempo + key/time signatures + sequencer-private +
        // sequence-number all filtered out.
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x20, 0x01, 0x06];
        events.extend_from_slice(&[0x00, 0xFF, 0x21, 0x01, 0x04]);
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
        let cps = smf.channel_prefixes();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].channel, 6);
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.midi_ports().len(), 1);
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

    #[test]
    fn channel_prefixes_to_bytes_round_trip() {
        // Round-trip through the writer: parse → to_bytes → parse must
        // surface the same ChannelPrefixEvent list. Exercises the
        // writer's FF 20 path (a guard against the helper drifting
        // out of sync with the mux).
        let mut events: Vec<u8> = vec![0x00, 0xFF, 0x20, 0x01, 0x0B];
        events.extend_from_slice(&encode_vlq(96));
        events.extend_from_slice(&[0xFF, 0x20, 0x01, 0x0C]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let rewritten = smf.to_bytes().unwrap();
        let reparsed = parse(&rewritten).unwrap();
        let cps = reparsed.channel_prefixes();
        assert_eq!(cps.len(), 2);
        assert_eq!(cps[0].tick, 0);
        assert_eq!(cps[0].channel, 0x0B);
        assert_eq!(cps[1].tick, 96);
        assert_eq!(cps[1].channel, 0x0C);
    }

    // ───────── SysExEvent / SmfFile::sysex_events (F0 / F7) ─────────

    #[test]
    fn sysex_events_empty_when_no_sysex_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.sysex_events().is_empty());
    }

    #[test]
    fn sysex_events_universal_gm_on_at_tick_zero() {
        // delta=0 F0 05 7E 7F 09 01 F7 — Universal Non-Real-Time GM-On,
        // self-contained: F0 start packet whose payload ends with F7.
        let events: Vec<u8> = vec![
            0x00, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7, // GM-on
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 1);
        assert_eq!(sx[0].tick, 0);
        assert_eq!(sx[0].track, 0);
        assert!(!sx[0].is_escape);
        assert_eq!(sx[0].data, vec![0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        assert!(sx[0].ends_with_eox());
        assert!(sx[0].is_complete_message());
        assert_eq!(sx[0].manufacturer_id(), Some(0x7E));
    }

    #[test]
    fn sysex_events_f0_without_trailing_f7_marks_multipacket_start() {
        // delta=0 F0 03 41 10 42 — Roland (0x41) Master Volume opener
        // without the closing F7, indicating a multi-packet message.
        let events: Vec<u8> = vec![
            0x00, 0xF0, 0x03, 0x41, 0x10, 0x42, //
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 1);
        assert!(!sx[0].is_escape);
        assert_eq!(sx[0].data, vec![0x41, 0x10, 0x42]);
        assert!(!sx[0].ends_with_eox());
        assert!(!sx[0].is_complete_message());
        assert_eq!(sx[0].manufacturer_id(), Some(0x41));
    }

    #[test]
    fn sysex_events_f7_continuation_pairs_after_f0() {
        // delta=0 F0 03 41 10 42 — opener (no trailing F7)
        // delta=8 F7 02 7B F7 — closing F7 packet (continuation + EOX)
        let mut events: Vec<u8> = vec![
            0x00, 0xF0, 0x03, 0x41, 0x10, 0x42, //
            0x08, 0xF7, 0x02, 0x7B, 0xF7, //
        ];
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 2);
        // Opener: F0, no EOX, manufacturer 0x41
        assert!(!sx[0].is_escape);
        assert_eq!(sx[0].tick, 0);
        assert_eq!(sx[0].manufacturer_id(), Some(0x41));
        assert!(!sx[0].is_complete_message());
        // Continuation: F7, EOX present, no manufacturer ID (escape form)
        assert!(sx[1].is_escape);
        assert_eq!(sx[1].tick, 8);
        assert_eq!(sx[1].data, vec![0x7B, 0xF7]);
        assert!(sx[1].ends_with_eox());
        assert!(!sx[1].is_complete_message()); // is_escape blocks "complete"
        assert_eq!(sx[1].manufacturer_id(), None); // F7 has no manufacturer prefix
    }

    #[test]
    fn sysex_events_empty_payload_surfaces_verbatim() {
        // delta=0 F0 00 — zero-length SysEx packet, spec-legal.
        let events: Vec<u8> = vec![
            0x00, 0xF0, 0x00, //
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 1);
        assert!(!sx[0].is_escape);
        assert!(sx[0].data.is_empty());
        assert!(!sx[0].ends_with_eox());
        assert!(!sx[0].is_complete_message());
        assert_eq!(sx[0].manufacturer_id(), None);
    }

    #[test]
    fn sysex_events_merge_across_tracks_sorted_by_tick() {
        // Track 0: tick 100 → F0 02 7E F7
        // Track 1: tick  50 → F0 02 41 F7
        // Merged order: track-1@50, then track-0@100.
        let t0: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(100));
            v.extend_from_slice(&[0xF0, 0x02, 0x7E, 0xF7]);
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let t1: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(50));
            v.extend_from_slice(&[0xF0, 0x02, 0x41, 0xF7]);
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 2);
        assert_eq!(sx[0].tick, 50);
        assert_eq!(sx[0].track, 1);
        assert_eq!(sx[0].manufacturer_id(), Some(0x41));
        assert_eq!(sx[1].tick, 100);
        assert_eq!(sx[1].track, 0);
        assert_eq!(sx[1].manufacturer_id(), Some(0x7E));
    }

    #[test]
    fn sysex_events_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Both tracks fire an F0 at absolute tick 64; stable sort by
        // tick keeps the per-track insertion order, so track 0's
        // packet precedes track 1's. Matches the merge convention
        // used by every existing iteration helper and the scheduler.
        let t0: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(64));
            v.extend_from_slice(&[0xF0, 0x02, 0x10, 0xF7]);
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let t1: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(64));
            v.extend_from_slice(&[0xF0, 0x02, 0x20, 0xF7]);
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 2);
        assert_eq!(sx[0].tick, 64);
        assert_eq!(sx[0].track, 0);
        assert_eq!(sx[0].manufacturer_id(), Some(0x10));
        assert_eq!(sx[1].tick, 64);
        assert_eq!(sx[1].track, 1);
        assert_eq!(sx[1].manufacturer_id(), Some(0x20));
    }

    #[test]
    fn sysex_events_filter_excludes_meta_and_channel_events() {
        // A track carrying one F0, one F7, plus several neighbouring
        // events (FF 7F sequencer-specific, FF 03 track name, FF 01
        // text, FF 21 port, FF 20 channel prefix, FF 51 tempo, B0 CC,
        // 90 note-on, FF 2F end-of-track). The helper must surface
        // exactly the F0 + F7 pair and ignore the rest. Cross-checks
        // against sequencer_specifics() — that helper should return
        // its single FF 7F entry unaffected by the SysEx selection.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x03, 0x02, b'A', b'B']); // FF 03 track name
        events.extend_from_slice(&[0x00, 0xFF, 0x01, 0x01, b'X']); // FF 01 text
        events.extend_from_slice(&[0x00, 0xFF, 0x21, 0x01, 0x02]); // FF 21 port
        events.extend_from_slice(&[0x00, 0xFF, 0x20, 0x01, 0x03]); // FF 20 channel prefix
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // FF 51 tempo
        events.extend_from_slice(&[0x00, 0xF0, 0x03, 0x7F, 0x10, 0xF7]); // F0 universal real-time
        events.extend_from_slice(&[0x00, 0xFF, 0x7F, 0x02, 0xAA, 0xBB]); // FF 7F sequencer-specific
        events.extend_from_slice(&[0x00, 0xF7, 0x02, 0xCC, 0xF7]); // F7 escape
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x64]); // CC 7 = 100
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x40]); // note on
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]); // end of track
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 2);
        assert!(!sx[0].is_escape);
        assert_eq!(sx[0].data, vec![0x7F, 0x10, 0xF7]);
        assert!(sx[1].is_escape);
        assert_eq!(sx[1].data, vec![0xCC, 0xF7]);
        // Cross-check: the sequencer_specifics helper still surfaces
        // its single FF 7F entry untouched.
        let ss = smf.sequencer_specifics();
        assert_eq!(ss.len(), 1);
        assert_eq!(ss[0].data, vec![0xAA, 0xBB]);
    }

    #[test]
    fn sysex_events_to_bytes_round_trip() {
        // Round-trip through the writer: parse → to_bytes → parse must
        // surface the same SysExEvent list. Exercises the writer's
        // F0 + F7 paths so the helper can't drift out of sync with
        // the mux.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        events.extend_from_slice(&encode_vlq(48));
        events.extend_from_slice(&[0xF0, 0x03, 0x41, 0x10, 0x42]); // opener
        events.extend_from_slice(&[0x08, 0xF7, 0x02, 0x7B, 0xF7]); // continuation
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let rewritten = smf.to_bytes().unwrap();
        let reparsed = parse(&rewritten).unwrap();
        let sx = reparsed.sysex_events();
        assert_eq!(sx.len(), 3);
        // GM-on at tick 0
        assert_eq!(sx[0].tick, 0);
        assert!(!sx[0].is_escape);
        assert_eq!(sx[0].data, vec![0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        assert!(sx[0].is_complete_message());
        // Roland opener at tick 48
        assert_eq!(sx[1].tick, 48);
        assert!(!sx[1].is_escape);
        assert_eq!(sx[1].data, vec![0x41, 0x10, 0x42]);
        assert_eq!(sx[1].manufacturer_id(), Some(0x41));
        // Continuation at tick 56
        assert_eq!(sx[2].tick, 56);
        assert!(sx[2].is_escape);
        assert_eq!(sx[2].data, vec![0x7B, 0xF7]);
        assert!(sx[2].ends_with_eox());
    }

    // ───────── SmfFile::universal_sysex_events ─────────

    #[test]
    fn universal_sysex_events_empty_when_no_sysex_present() {
        let events: Vec<u8> = vec![0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.universal_sysex_events().is_empty());
    }

    #[test]
    fn universal_sysex_events_filters_out_manufacturer_prefixed_and_escape_packets() {
        // delta=0 F0 06 41 10 42 01 02 F7 — Roland (0x41) manufacturer
        //                                   packet, filtered out.
        // delta=8 F0 05 7E 7F 09 01 F7   — Universal Non-Real-Time GM-On,
        //                                   retained.
        // delta=8 F7 02 CC F7            — escape / continuation packet,
        //                                   filtered out.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xF0, 0x06, 0x41, 0x10, 0x42, 0x01, 0x02, 0xF7]);
        events.extend_from_slice(&[0x08, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        events.extend_from_slice(&[0x08, 0xF7, 0x02, 0xCC, 0xF7]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        // Sanity: sysex_events still surfaces all three packets.
        assert_eq!(smf.sysex_events().len(), 3);
        // Universal-only view: GM-On only.
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 8);
        assert_eq!(u[0].track, 0);
        assert_eq!(u[0].classification.realm, UniversalRealm::NonRealTime);
        assert_eq!(u[0].classification.device_id, 0x7F);
        assert_eq!(
            u[0].classification.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidi1SystemOn,
            )
        );
        assert_eq!(u[0].data, vec![0x7E, 0x7F, 0x09, 0x01, 0xF7]);
    }

    #[test]
    fn universal_sysex_events_classifies_both_realms() {
        // delta=0  F0 05 7E 7F 09 01 F7 — GM 1 System On (Non-RT).
        // delta=64 F0 07 7F 7F 04 01 30 30 F7 — Master Volume (RT).
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        events.extend_from_slice(&[0x40, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x30, 0x30, 0xF7]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 2);
        // GM-On at tick 0
        assert_eq!(u[0].tick, 0);
        assert_eq!(u[0].classification.realm, UniversalRealm::NonRealTime);
        assert_eq!(
            u[0].classification.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidi1SystemOn,
            )
        );
        // Master Volume at tick 64
        assert_eq!(u[1].tick, 64);
        assert_eq!(u[1].classification.realm, UniversalRealm::RealTime);
        assert_eq!(u[1].classification.device_id, 0x7F);
        assert_eq!(
            u[1].classification.sub_id1,
            UniversalSubId1::DeviceControl(UniversalSubId2::DeviceControlMasterVolume),
        );
        // Data preserves verbatim payload including realm + trailing F7.
        assert_eq!(u[1].data, vec![0x7F, 0x7F, 0x04, 0x01, 0x30, 0x30, 0xF7]);
    }

    #[test]
    fn universal_sysex_events_merge_across_tracks_sorted_by_tick() {
        // Track 0: tick 100 → GM 1 System On.
        // Track 1: tick  50 → Master Volume.
        // Merged order: track-1@50, then track-0@100.
        let t0: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(100));
            v.extend_from_slice(&[0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let t1: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(50));
            v.extend_from_slice(&[0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x30, 0x30, 0xF7]);
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].tick, 50);
        assert_eq!(u[0].track, 1);
        assert_eq!(u[0].classification.realm, UniversalRealm::RealTime);
        assert_eq!(u[1].tick, 100);
        assert_eq!(u[1].track, 0);
        assert_eq!(u[1].classification.realm, UniversalRealm::NonRealTime);
    }

    #[test]
    fn universal_sysex_events_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Both tracks fire a Universal SysEx at absolute tick 64; stable
        // sort by tick keeps the per-track insertion order so track 0
        // precedes track 1. Matches the merge convention used by every
        // existing iteration helper and the scheduler.
        let t0: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(64));
            v.extend_from_slice(&[0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]); // GM-On
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let t1: Vec<u8> = {
            let mut v: Vec<u8> = Vec::new();
            v.extend_from_slice(&encode_vlq(64));
            v.extend_from_slice(&[0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x40, 0x40, 0xF7]); // Master Volume
            v.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
            v
        };
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].tick, 64);
        assert_eq!(u[0].track, 0);
        assert_eq!(u[0].classification.realm, UniversalRealm::NonRealTime);
        assert_eq!(u[1].tick, 64);
        assert_eq!(u[1].track, 1);
        assert_eq!(u[1].classification.realm, UniversalRealm::RealTime);
    }

    #[test]
    fn universal_sysex_events_drops_truncated_universal_packets() {
        // delta=0 F0 02 7E 7F — Universal realm + device-id only,
        //                       truncated before Sub-ID #1 (payload < 3
        //                       bytes pre-F7). Classifier returns None;
        //                       helper filters it out.
        // delta=8 F0 05 7E 7F 09 01 F7 — valid GM-On, retained.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xF0, 0x02, 0x7E, 0x7F]);
        events.extend_from_slice(&[0x08, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        // Sanity: sysex_events surfaces both.
        assert_eq!(smf.sysex_events().len(), 2);
        // Universal-only: just the GM-On.
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 8);
        assert_eq!(
            u[0].classification.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidi1SystemOn,
            )
        );
    }

    #[test]
    fn universal_sysex_events_classification_matches_per_event_call() {
        // The helper is a re-use of SysExEvent::universal_classification;
        // confirm the two views agree byte-for-byte on a mixed packet
        // mix so a future tweak to the per-event classifier can't drift
        // out of sync with this helper without the test catching it.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]); // GM 1 On
        events.extend_from_slice(&[0x00, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x02, 0xF7]); // GM Off
        events.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x30, 0x30, 0xF7]); // Master Volume
        events.extend_from_slice(&[
            0x00, 0xF0, 0x09, 0x7F, 0x00, 0x01, 0x01, 0x10, 0x20, 0x30, 0x40, 0xF7,
        ]); // RT MTC Full Message
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        // Walk sysex_events() and call universal_classification() to
        // build the parallel view, then compare.
        let parallel: Vec<(u64, usize, UniversalSysEx, Vec<u8>)> = smf
            .sysex_events()
            .into_iter()
            .filter_map(|ev| {
                ev.universal_classification()
                    .map(|c| (ev.tick, ev.track, c, ev.data.clone()))
            })
            .collect();
        assert_eq!(u.len(), parallel.len());
        assert_eq!(u.len(), 4);
        for (lhs, rhs) in u.iter().zip(parallel.iter()) {
            assert_eq!(lhs.tick, rhs.0);
            assert_eq!(lhs.track, rhs.1);
            assert_eq!(lhs.classification, rhs.2);
            assert_eq!(lhs.data, rhs.3);
        }
    }

    // ───────── UniversalSysExEvent::mtc_full_message ─────────

    /// Build a `UniversalSysExEvent` directly from a verbatim payload
    /// (the bytes `SysExEvent::data` would carry: `<realm> <device_id>
    /// <sub_id1> <sub_id2> …`). Mirrors `make_sysex` for the universal
    /// view.
    fn make_universal(payload: &[u8]) -> UniversalSysExEvent {
        let classification = make_sysex(payload).universal_classification().unwrap();
        UniversalSysExEvent {
            tick: 0,
            track: 0,
            classification,
            data: payload.to_vec(),
        }
    }

    #[test]
    fn mtc_full_message_decodes_smpte_quartet_and_rate() {
        // RP-004/008 worked example 01:37:52:16 @ 30 fps non-drop.
        // hr = 0 11 00001 = 0b0110_0001 = 0x61 (type 3 = 30 non-drop,
        // hours 1); mn = 37 = 0x25; sc = 52 = 0x34; fr = 16 = 0x10.
        // F0 7F 7F 01 01 61 25 34 10 F7
        let ev = make_universal(&[0x7F, 0x7F, 0x01, 0x01, 0x61, 0x25, 0x34, 0x10, 0xF7]);
        let full = ev.mtc_full_message().expect("classifies as Full Message");
        assert_eq!(full.hours_raw, 0x61);
        assert_eq!(full.hours_count(), 1);
        assert_eq!(full.minutes, 37);
        assert_eq!(full.seconds, 52);
        assert_eq!(full.frames, 16);
        assert_eq!(full.frame_rate(), FrameRate::Fps30NonDrop);
    }

    #[test]
    fn mtc_full_message_decodes_all_four_frame_rate_types() {
        // hr byte = 0 yy zzzzz; sweep the frame-rate type in bits 5-6
        // (yy = 00/01/10/11) with hours 0: 0x00 / 0x20 / 0x40 / 0x60.
        for (hr, rate) in [
            (0x00u8, FrameRate::Fps24),
            (0x20, FrameRate::Fps25),
            (0x40, FrameRate::Fps30DropFrame),
            (0x60, FrameRate::Fps30NonDrop),
        ] {
            let ev = make_universal(&[0x7F, 0x7F, 0x01, 0x01, hr, 0x00, 0x00, 0x00, 0xF7]);
            let full = ev.mtc_full_message().unwrap();
            assert_eq!(full.frame_rate(), rate, "hr={hr:#04x}");
            assert_eq!(full.hours_count(), 0);
        }
    }

    #[test]
    fn mtc_full_message_none_for_non_full_message_universal_packets() {
        // RT MTC User Bits (sub-id2 0x02) is MTC but not a Full Message.
        let user_bits = make_universal(&[
            0x7F, 0x7F, 0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF7,
        ]);
        assert_eq!(user_bits.mtc_full_message(), None);
        // GM 1 System On (Non-RT) — not MTC at all.
        let gm_on = make_universal(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        assert_eq!(gm_on.mtc_full_message(), None);
        // Master Volume (RT Device Control) — different sub-id1.
        let master_vol = make_universal(&[0x7F, 0x7F, 0x04, 0x01, 0x30, 0x30, 0xF7]);
        assert_eq!(master_vol.mtc_full_message(), None);
    }

    #[test]
    fn mtc_full_message_none_when_quartet_truncated() {
        // Correctly classified RT MTC Full Message but the stream ends
        // before all four time bytes arrive: hr + mn present, sc/fr
        // missing → None rather than a half-decoded time.
        let truncated = make_universal(&[0x7F, 0x7F, 0x01, 0x01, 0x61, 0x25]);
        assert_eq!(truncated.classification.realm, UniversalRealm::RealTime);
        assert_eq!(
            truncated.classification.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::RtMtcFullMessage)
        );
        assert_eq!(truncated.mtc_full_message(), None);
    }

    #[test]
    fn mtc_full_message_via_parsed_smf_at_absolute_tick() {
        // delta=128 F0 09 7F 7F 01 01 41 1E 0F 0C F7 — Full Message.
        // hr = 0 10 00001 = 0x41 → type 2 (30 drop), hours 1;
        // mn = 0x1E = 30; sc = 0x0F = 15; fr = 0x0C = 12.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(128));
        events.extend_from_slice(&[
            0xF0, 0x09, 0x7F, 0x7F, 0x01, 0x01, 0x41, 0x1E, 0x0F, 0x0C, 0xF7,
        ]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 128);
        let full = u[0].mtc_full_message().expect("Full Message decodes");
        assert_eq!(full.frame_rate(), FrameRate::Fps30DropFrame);
        assert!(full.frame_rate().is_drop_frame());
        assert_eq!(full.hours_count(), 1);
        assert_eq!(full.minutes, 30);
        assert_eq!(full.seconds, 15);
        assert_eq!(full.frames, 12);
    }

    // ───────── UniversalSysExEvent::mtc_user_bits ─────────

    #[test]
    fn mtc_user_bits_decodes_groups_and_flag_bits() {
        // RP-004/008 User Bits: F0 7F 7F 01 02 u1..u9 F7.
        // Groups a..h = 1..8; flag byte u9 = 0b11 (both i and j set).
        let ev = make_universal(&[
            0x7F, 0x7F, 0x01, 0x02, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x03, 0xF7,
        ]);
        let ub = ev.mtc_user_bits().expect("classifies as User Bits");
        assert_eq!(ub.groups, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(ub.flag_i());
        assert!(ub.flag_j());
        // Reassembly order hhhhgggg ffffeeee ddddcccc bbbbaaaa: Group 1
        // (a=1) is the least-significant nibble, Group 8 (h=8) the most.
        assert_eq!(ub.reassembled(), 0x8765_4321);
    }

    #[test]
    fn mtc_user_bits_masks_high_nibbles_and_isolates_flags() {
        // u1..u8 carry only their low nibble per the 0000xxxx wire
        // shape; high bits set on the wire are masked off.
        // u9 = 0b0000_0001 → i set, j clear.
        let ev = make_universal(&[
            0x7F, 0x7F, 0x01, 0x02, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF, 0xF0, 0xF1, 0x01, 0xF7,
        ]);
        let ub = ev.mtc_user_bits().unwrap();
        assert_eq!(ub.groups, [0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x0, 0x1]);
        assert!(ub.flag_i());
        assert!(!ub.flag_j());
        assert_eq!(ub.reassembled(), 0x10FE_DCBA);
    }

    #[test]
    fn mtc_user_bits_none_for_non_user_bits_universal_packets() {
        // RT MTC Full Message (sub-id2 0x01) is MTC but not User Bits.
        let full = make_universal(&[0x7F, 0x7F, 0x01, 0x01, 0x61, 0x25, 0x34, 0x10, 0xF7]);
        assert_eq!(full.mtc_user_bits(), None);
        // GM 1 System On (Non-RT) — not MTC at all.
        let gm_on = make_universal(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        assert_eq!(gm_on.mtc_user_bits(), None);
    }

    #[test]
    fn mtc_user_bits_none_when_payload_truncated() {
        // Correctly classified RT MTC User Bits but the stream ends
        // before all nine u1..u9 bytes arrive (u9 missing) → None.
        let truncated = make_universal(&[
            0x7F, 0x7F, 0x01, 0x02, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        ]);
        assert_eq!(truncated.classification.realm, UniversalRealm::RealTime);
        assert_eq!(
            truncated.classification.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::RtMtcUserBits)
        );
        assert_eq!(truncated.mtc_user_bits(), None);
    }

    #[test]
    fn mtc_user_bits_via_parsed_smf_at_absolute_tick() {
        // delta=64 F0 0D 7F 7F 01 02 u1..u9 F7 — User Bits message.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(64));
        events.extend_from_slice(&[
            0xF0, 0x0D, 0x7F, 0x7F, 0x01, 0x02, 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08,
            0x02, 0xF7,
        ]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 64);
        let ub = u[0].mtc_user_bits().expect("User Bits decodes");
        assert_eq!(ub.groups, [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8]);
        assert!(!ub.flag_i());
        assert!(ub.flag_j());
        assert_eq!(ub.reassembled(), 0x89AB_CDEF);
    }

    // ───────── UniversalSysExEvent::notation_bar_number ─────────

    #[test]
    fn notation_bar_number_decodes_in_song_bar() {
        // Bar 1: aa aa = 01 00 (lsb-first). F0 7F 7F 03 01 01 00 F7.
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x01, 0x00, 0xF7]);
        let bn = ev.notation_bar_number().expect("classifies as Bar Number");
        assert_eq!(bn.raw14, 1);
        assert_eq!(bn.value(), 1);
        assert_eq!(bn.state(), NotationBarState::BarInSong(1));
        // Bar 8062 (0x1F7E): lsb = 0x7E, msb = 0x3E (8062 = 0b01_1111_0111_1110).
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x7E, 0x3E, 0xF7]);
        let bn = ev.notation_bar_number().unwrap();
        assert_eq!(bn.raw14, 0x1F7E);
        assert_eq!(bn.value(), 8062);
        assert_eq!(bn.state(), NotationBarState::BarInSong(8062));
    }

    #[test]
    fn notation_bar_number_decodes_flag_and_count_in_regions() {
        // Not running: raw 0x2000 → lsb = 0x00, msb = 0x40.
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x00, 0x40, 0xF7]);
        let bn = ev.notation_bar_number().unwrap();
        assert_eq!(bn.raw14, 0x2000);
        assert_eq!(bn.value(), -8192);
        assert_eq!(bn.state(), NotationBarState::NotRunning);
        // Running, bar unknown: raw 0x1F7F → lsb = 0x7F, msb = 0x3E.
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x7F, 0x3E, 0xF7]);
        let bn = ev.notation_bar_number().unwrap();
        assert_eq!(bn.raw14, 0x1F7F);
        assert_eq!(bn.state(), NotationBarState::RunningUnknown);
        // Last bar of count-in is 0: raw 0x0000 → lsb 0x00 msb 0x00.
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x00, 0x00, 0xF7]);
        let bn = ev.notation_bar_number().unwrap();
        assert_eq!(bn.value(), 0);
        assert_eq!(bn.state(), NotationBarState::CountIn(0));
        // Count-in bar −1: raw 0x3FFF (two's complement of -1 over 14 bits)
        // → lsb 0x7F msb 0x7F.
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x7F, 0x7F, 0xF7]);
        let bn = ev.notation_bar_number().unwrap();
        assert_eq!(bn.value(), -1);
        assert_eq!(bn.state(), NotationBarState::CountIn(-1));
    }

    #[test]
    fn notation_bar_number_none_for_non_bar_number_packets() {
        // Time Signature (Immediate) is Notation but not Bar Number.
        let ts = make_universal(&[0x7F, 0x7F, 0x03, 0x02, 0x04, 0x04, 0x02, 0x18, 0x08, 0xF7]);
        assert_eq!(ts.notation_bar_number(), None);
        // GM 1 System On — not Notation at all.
        let gm_on = make_universal(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        assert_eq!(gm_on.notation_bar_number(), None);
    }

    #[test]
    fn notation_bar_number_none_when_truncated() {
        // Classified Bar Number but the second aa byte is missing.
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x01]);
        assert_eq!(
            ev.classification.sub_id1,
            UniversalSubId1::NotationInformation(UniversalSubId2::RtNotationBarNumber)
        );
        assert_eq!(ev.notation_bar_number(), None);
    }

    #[test]
    fn notation_bar_number_via_parsed_smf_at_absolute_tick() {
        // delta=48 F0 07 7F 7F 03 01 05 00 F7 — Bar Number = 5.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(48));
        events.extend_from_slice(&[0xF0, 0x07, 0x7F, 0x7F, 0x03, 0x01, 0x05, 0x00, 0xF7]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 48);
        let bn = u[0].notation_bar_number().expect("Bar Number decodes");
        assert_eq!(bn.state(), NotationBarState::BarInSong(5));
    }

    // ───────── UniversalSysExEvent::notation_time_signature ─────────

    #[test]
    fn notation_time_signature_decodes_simple_immediate_meter() {
        // 4/4, 24 clocks/click, 8 thirty-seconds/quarter — duplicates
        // the FF 58 meta event nn dd cc bb. ln = 4.
        // F0 7F 7F 03 02 04 04 02 18 08 F7
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x02, 0x04, 0x04, 0x02, 0x18, 0x08, 0xF7]);
        let ts = ev
            .notation_time_signature()
            .expect("classifies as Time Signature");
        assert!(!ts.is_delayed());
        assert!(!ts.is_compound());
        assert_eq!(ts.pairs.len(), 1);
        assert_eq!(ts.primary().numerator, 4);
        assert_eq!(ts.primary().denominator_pow2, 2);
        assert_eq!(ts.primary().denominator(), 4);
        assert_eq!(ts.clocks_per_click, 24);
        assert_eq!(ts.thirty_seconds_per_quarter, 8);
    }

    #[test]
    fn notation_time_signature_decodes_delayed_form() {
        // Delayed sub-id2 = 0x42. 3/8, 12 clocks/click, 8 32nds/quarter.
        // F0 7F 7F 03 42 04 03 03 0C 08 F7
        let ev = make_universal(&[0x7F, 0x7F, 0x03, 0x42, 0x04, 0x03, 0x03, 0x0C, 0x08, 0xF7]);
        let ts = ev
            .notation_time_signature()
            .expect("delayed Time Signature");
        assert!(ts.is_delayed());
        assert_eq!(ts.primary().numerator, 3);
        assert_eq!(ts.primary().denominator(), 8);
        assert_eq!(ts.clocks_per_click, 12);
    }

    #[test]
    fn notation_time_signature_decodes_compound_meter() {
        // Compound 4/4 + 3/8 within the same bar: ln = 6 (one extra pair).
        // F0 7F 7F 03 02 06 04 02 18 08 03 03 F7
        let ev = make_universal(&[
            0x7F, 0x7F, 0x03, 0x02, 0x06, 0x04, 0x02, 0x18, 0x08, 0x03, 0x03, 0xF7,
        ]);
        let ts = ev.notation_time_signature().expect("compound meter");
        assert!(ts.is_compound());
        assert_eq!(ts.pairs.len(), 2);
        assert_eq!(ts.pairs[0].numerator, 4);
        assert_eq!(ts.pairs[0].denominator(), 4);
        assert_eq!(ts.pairs[1].numerator, 3);
        assert_eq!(ts.pairs[1].denominator(), 8);
        // Metronome bytes come from the leading pair only.
        assert_eq!(ts.clocks_per_click, 24);
        assert_eq!(ts.thirty_seconds_per_quarter, 8);
    }

    #[test]
    fn notation_time_signature_none_for_invalid_or_truncated_body() {
        // ln = 3 is shorter than the 4-byte minimum → None.
        let short_ln = make_universal(&[0x7F, 0x7F, 0x03, 0x02, 0x03, 0x04, 0x02, 0x18, 0xF7]);
        assert_eq!(short_ln.notation_time_signature(), None);
        // ln = 5 is not of the 4 + 2k form → None.
        let odd_ln = make_universal(&[
            0x7F, 0x7F, 0x03, 0x02, 0x05, 0x04, 0x02, 0x18, 0x08, 0x03, 0xF7,
        ]);
        assert_eq!(odd_ln.notation_time_signature(), None);
        // ln = 6 declared but the second pair is truncated → None.
        let truncated = make_universal(&[0x7F, 0x7F, 0x03, 0x02, 0x06, 0x04, 0x02, 0x18, 0x08]);
        assert_eq!(truncated.notation_time_signature(), None);
        // Bar Number is Notation but not a Time Signature.
        let bar = make_universal(&[0x7F, 0x7F, 0x03, 0x01, 0x01, 0x00, 0xF7]);
        assert_eq!(bar.notation_time_signature(), None);
    }

    #[test]
    fn notation_time_signature_via_parsed_smf_at_absolute_tick() {
        // delta=96 F0 0A 7F 7F 03 02 04 06 03 0C 08 F7 — 6/8 immediate.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(96));
        events.extend_from_slice(&[
            0xF0, 0x0A, 0x7F, 0x7F, 0x03, 0x02, 0x04, 0x06, 0x03, 0x0C, 0x08, 0xF7,
        ]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 96);
        let ts = u[0]
            .notation_time_signature()
            .expect("Time Signature decodes");
        assert_eq!(ts.primary().numerator, 6);
        assert_eq!(ts.primary().denominator(), 8);
        assert!(!ts.is_delayed());
    }

    // ───────── UniversalSysExEvent::device_control (volume / balance) ─

    #[test]
    fn device_control_decodes_master_volume_14bit() {
        // F0 7F 7F 04 01 lsb msb F7 — 14-bit value (msb << 7) | lsb.
        // lsb=0x00 msb=0x40 → 0x2000 (half scale).
        let ev = make_universal(&[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x40, 0xF7]);
        assert_eq!(
            ev.device_control(),
            Some(DeviceControl::MasterVolume { value14: 0x2000 })
        );
        // Full scale: lsb=0x7F msb=0x7F → 0x3FFF.
        let full = make_universal(&[0x7F, 0x7F, 0x04, 0x01, 0x7F, 0x7F, 0xF7]);
        assert_eq!(
            full.device_control(),
            Some(DeviceControl::MasterVolume { value14: 0x3FFF })
        );
        // Silence: 0x0000.
        let zero = make_universal(&[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x00, 0xF7]);
        assert_eq!(
            zero.device_control(),
            Some(DeviceControl::MasterVolume { value14: 0 })
        );
    }

    #[test]
    fn device_control_decodes_master_balance_14bit() {
        // 0x2000 = centre per M1 v4.2.1 §"DEVICE CONTROL".
        let centre = make_universal(&[0x7F, 0x7F, 0x04, 0x02, 0x00, 0x40, 0xF7]);
        assert_eq!(
            centre.device_control(),
            Some(DeviceControl::MasterBalance { value14: 0x2000 })
        );
        // Hard left = 0x0000, hard right = 0x3FFF.
        let left = make_universal(&[0x7F, 0x7F, 0x04, 0x02, 0x00, 0x00, 0xF7]);
        assert_eq!(
            left.device_control(),
            Some(DeviceControl::MasterBalance { value14: 0 })
        );
        let right = make_universal(&[0x7F, 0x7F, 0x04, 0x02, 0x7F, 0x7F, 0xF7]);
        assert_eq!(
            right.device_control(),
            Some(DeviceControl::MasterBalance { value14: 0x3FFF })
        );
    }

    #[test]
    fn device_control_none_for_non_device_control_packets() {
        // GM 1 System On (Non-RT) — different sub-id1.
        let gm_on = make_universal(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        assert_eq!(gm_on.device_control(), None);
        // RT MTC Full Message — sub-id1 0x01, not Device Control.
        let mtc = make_universal(&[0x7F, 0x7F, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0xF7]);
        assert_eq!(mtc.device_control(), None);
    }

    #[test]
    fn device_control_none_when_value_pair_truncated() {
        // Classifies as Master Volume but the msb byte never arrives.
        let truncated = make_universal(&[0x7F, 0x7F, 0x04, 0x01, 0x40]);
        assert_eq!(
            truncated.classification.sub_id1,
            UniversalSubId1::DeviceControl(UniversalSubId2::DeviceControlMasterVolume)
        );
        assert_eq!(truncated.device_control(), None);
    }

    #[test]
    fn device_control_volume_via_parsed_smf_at_absolute_tick() {
        // delta=64 F0 07 7F 7F 04 01 00 20 F7 — Master Volume 0x1000.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(64));
        events.extend_from_slice(&[0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x00, 0x20, 0xF7]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 64);
        assert_eq!(
            u[0].device_control(),
            Some(DeviceControl::MasterVolume { value14: 0x1000 })
        );
    }

    // ───────── UniversalSysExEvent::device_control (CA-025 tuning) ────

    #[test]
    fn device_control_decodes_master_fine_tuning_cents() {
        // CA-025: F0 7F 7F 04 03 lsb msb F7 — 14-bit, 0x2000 = 0 cents,
        // displacement = 100/8192 × (value − 8192).
        // No change: lsb=0x00 msb=0x40 → 0x2000 → 0.0 cents.
        let centre = make_universal(&[0x7F, 0x7F, 0x04, 0x03, 0x00, 0x40, 0xF7]);
        let dc = centre.device_control().unwrap();
        assert_eq!(dc, DeviceControl::MasterFineTuning { value14: 0x2000 });
        assert_eq!(dc.fine_tuning_cents(), Some(0.0));
        // CA-025 table low end: 0x0000 → 100/8192 × (−8192) = −100 cents.
        let low = make_universal(&[0x7F, 0x7F, 0x04, 0x03, 0x00, 0x00, 0xF7]);
        let dc_low = low.device_control().unwrap();
        assert_eq!(dc_low, DeviceControl::MasterFineTuning { value14: 0 });
        assert!((dc_low.fine_tuning_cents().unwrap() - (-100.0)).abs() < 1e-9);
        // CA-025 table high end: 0x3FFF → 100/8192 × (+8191).
        let high = make_universal(&[0x7F, 0x7F, 0x04, 0x03, 0x7F, 0x7F, 0xF7]);
        let dc_high = high.device_control().unwrap();
        let expect = 100.0 / 8192.0 * 8191.0;
        assert!((dc_high.fine_tuning_cents().unwrap() - expect).abs() < 1e-9);
        // Helper is None for a non-fine-tuning variant.
        let vol = make_universal(&[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x40, 0xF7]);
        assert_eq!(vol.device_control().unwrap().fine_tuning_cents(), None);
    }

    #[test]
    fn device_control_decodes_master_coarse_tuning_semitones() {
        // CA-025: F0 7F 7F 04 04 00 msb F7 — LSB always 0, msb-64
        // semitones. 0x40 = no change.
        let centre = make_universal(&[0x7F, 0x7F, 0x04, 0x04, 0x00, 0x40, 0xF7]);
        let dc = centre.device_control().unwrap();
        assert_eq!(
            dc,
            DeviceControl::MasterCoarseTuning {
                msb: 0x40,
                lsb: 0x00
            }
        );
        assert_eq!(dc.coarse_tuning_semitones(), Some(0));
        // Low: msb=0x00 → −64 semitones.
        let low = make_universal(&[0x7F, 0x7F, 0x04, 0x04, 0x00, 0x00, 0xF7]);
        assert_eq!(
            low.device_control().unwrap().coarse_tuning_semitones(),
            Some(-64)
        );
        // High: msb=0x7F → +63 semitones.
        let high = make_universal(&[0x7F, 0x7F, 0x04, 0x04, 0x00, 0x7F, 0xF7]);
        assert_eq!(
            high.device_control().unwrap().coarse_tuning_semitones(),
            Some(63)
        );
        // Helper is None for a non-coarse-tuning variant.
        let vol = make_universal(&[0x7F, 0x7F, 0x04, 0x01, 0x00, 0x40, 0xF7]);
        assert_eq!(
            vol.device_control().unwrap().coarse_tuning_semitones(),
            None
        );
    }

    #[test]
    fn device_control_tuning_none_when_truncated() {
        // Fine tuning classified but msb missing.
        let fine = make_universal(&[0x7F, 0x7F, 0x04, 0x03, 0x10]);
        assert_eq!(fine.device_control(), None);
        // Coarse tuning classified but msb missing.
        let coarse = make_universal(&[0x7F, 0x7F, 0x04, 0x04, 0x00]);
        assert_eq!(coarse.device_control(), None);
    }

    #[test]
    fn device_control_fine_tuning_via_parsed_smf_at_absolute_tick() {
        // delta=48 F0 07 7F 7F 04 03 00 41 F7 — fine tuning 0x2080.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(48));
        events.extend_from_slice(&[0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x03, 0x00, 0x41, 0xF7]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let u = smf.universal_sysex_events();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].tick, 48);
        let dc = u[0].device_control().unwrap();
        assert_eq!(dc, DeviceControl::MasterFineTuning { value14: 0x2080 });
        // 0x2080 = 8320 → 100/8192 × (8320 − 8192) ≈ +1.5625 cents.
        let expect = 100.0 / 8192.0 * 128.0;
        assert!((dc.fine_tuning_cents().unwrap() - expect).abs() < 1e-9);
    }

    // ───────── SysExEvent::universal_classification ─────────

    fn make_sysex(payload: &[u8]) -> SysExEvent {
        SysExEvent {
            tick: 0,
            track: 0,
            is_escape: false,
            data: payload.to_vec(),
        }
    }

    fn make_sysex_escape(payload: &[u8]) -> SysExEvent {
        SysExEvent {
            tick: 0,
            track: 0,
            is_escape: true,
            data: payload.to_vec(),
        }
    }

    #[test]
    fn universal_classification_returns_none_for_f7_continuation_packet() {
        let ev = make_sysex_escape(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        assert!(ev.universal_classification().is_none());
    }

    #[test]
    fn universal_classification_returns_none_for_manufacturer_prefixed_packet() {
        // Roland (0x41) manufacturer prefix — not Universal.
        let ev = make_sysex(&[0x41, 0x10, 0x42, 0x12, 0x40, 0x00, 0x7F, 0x00, 0x41, 0xF7]);
        assert!(ev.universal_classification().is_none());
    }

    #[test]
    fn universal_classification_returns_none_for_three_byte_expanded_manufacturer_id() {
        // 0x00 leads a 3-byte expanded manufacturer ID; not Universal.
        let ev = make_sysex(&[0x00, 0x20, 0x33, 0x01, 0xF7]);
        assert!(ev.universal_classification().is_none());
    }

    #[test]
    fn universal_classification_returns_none_for_truncated_packet_below_three_bytes() {
        let ev = make_sysex(&[0x7E, 0x7F]); // realm + device-id only
        assert!(ev.universal_classification().is_none());
        let ev = make_sysex(&[0x7E]); // realm only
        assert!(ev.universal_classification().is_none());
        let ev = make_sysex(&[]); // empty
        assert!(ev.universal_classification().is_none());
    }

    #[test]
    fn universal_classification_gm1_system_on() {
        // F0 7E 7F 09 01 F7 — Universal Non-Real-Time GM 1 System On.
        let ev = make_sysex(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.realm, UniversalRealm::NonRealTime);
        assert_eq!(u.device_id, 0x7F);
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidi1SystemOn,
            )
        );
    }

    #[test]
    fn universal_classification_gm_system_off() {
        let ev = make_sysex(&[0x7E, 0x7F, 0x09, 0x02, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.realm, UniversalRealm::NonRealTime);
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidiSystemOff
            )
        );
    }

    #[test]
    fn universal_classification_gm2_system_on() {
        let ev = make_sysex(&[0x7E, 0x7F, 0x09, 0x03, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.realm, UniversalRealm::NonRealTime);
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidi2SystemOn,
            )
        );
    }

    #[test]
    fn universal_classification_master_volume_real_time() {
        // F0 7F 7F 04 01 lsb msb F7 — Universal Real-Time Master Volume.
        let ev = make_sysex(&[0x7F, 0x7F, 0x04, 0x01, 0x40, 0x60, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.realm, UniversalRealm::RealTime);
        assert_eq!(u.device_id, 0x7F);
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::DeviceControl(UniversalSubId2::DeviceControlMasterVolume)
        );
    }

    #[test]
    fn universal_classification_master_balance_fine_coarse_global() {
        let cases = [
            (0x02u8, UniversalSubId2::DeviceControlMasterBalance),
            (0x03, UniversalSubId2::DeviceControlMasterFineTuning),
            (0x04, UniversalSubId2::DeviceControlMasterCoarseTuning),
            (0x05, UniversalSubId2::DeviceControlGlobalParameterControl),
        ];
        for (sub2, expected) in cases {
            let ev = make_sysex(&[0x7F, 0x7F, 0x04, sub2, 0x00, 0x40, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(u.realm, UniversalRealm::RealTime);
            assert_eq!(u.sub_id1, UniversalSubId1::DeviceControl(expected));
        }
    }

    #[test]
    fn universal_classification_realm_disambiguates_shared_byte_pair() {
        // Sub-ID #1 = 0x09, Sub-ID #2 = 0x01:
        //   Non-Real-Time → GM 1 System On
        //   Real-Time     → Channel Pressure (Aftertouch) Destination
        let nrt = make_sysex(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        let rt = make_sysex(&[0x7F, 0x7F, 0x09, 0x01, 0xF7]);
        let unrt = nrt.universal_classification().unwrap();
        let urt = rt.universal_classification().unwrap();
        assert_eq!(
            unrt.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidi1SystemOn,
            )
        );
        assert_eq!(
            urt.sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::ControllerDestinationChannelPressure,
            )
        );
    }

    #[test]
    fn universal_classification_mtc_full_message_real_time() {
        // F0 7F <dev> 01 01 hr mn se fr F7
        let ev = make_sysex(&[0x7F, 0x00, 0x01, 0x01, 0x21, 0x18, 0x2D, 0x00, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.realm, UniversalRealm::RealTime);
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::RtMtcFullMessage)
        );
    }

    #[test]
    fn universal_classification_mtc_user_bits_real_time() {
        let ev = make_sysex(&[0x7F, 0x00, 0x01, 0x02, 0x00, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::RtMtcUserBits)
        );
    }

    #[test]
    fn universal_classification_nonrt_mtc_setup_punch_in_points() {
        // F0 7E <dev> 04 01 ... F7
        let ev = make_sysex(&[0x7E, 0x00, 0x04, 0x01, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.realm, UniversalRealm::NonRealTime);
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::NonRtMtcPunchInPoints)
        );
    }

    #[test]
    fn universal_classification_nonrt_mtc_full_setup_table() {
        let cases = [
            (0x00u8, UniversalSubId2::NonRtMtcSpecial),
            (0x01, UniversalSubId2::NonRtMtcPunchInPoints),
            (0x02, UniversalSubId2::NonRtMtcPunchOutPoints),
            (0x03, UniversalSubId2::NonRtMtcDeletePunchInPoint),
            (0x04, UniversalSubId2::NonRtMtcDeletePunchOutPoint),
            (0x05, UniversalSubId2::NonRtMtcEventStartPoint),
            (0x06, UniversalSubId2::NonRtMtcEventStopPoint),
            (0x07, UniversalSubId2::NonRtMtcEventStartPointsWithInfo),
            (0x08, UniversalSubId2::NonRtMtcEventStopPointsWithInfo),
            (0x09, UniversalSubId2::NonRtMtcDeleteEventStartPoint),
            (0x0A, UniversalSubId2::NonRtMtcDeleteEventStopPoint),
            (0x0B, UniversalSubId2::NonRtMtcCuePoints),
            (0x0C, UniversalSubId2::NonRtMtcCuePointsWithInfo),
            (0x0D, UniversalSubId2::NonRtMtcDeleteCuePoint),
            (0x0E, UniversalSubId2::NonRtMtcEventNameWithInfo),
        ];
        for (sub2, expected) in cases {
            let ev = make_sysex(&[0x7E, 0x7F, 0x04, sub2, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(u.sub_id1, UniversalSubId1::MidiTimeCode(expected));
        }
    }

    #[test]
    fn universal_classification_sample_dump_singletons_nonrt() {
        // Sub-ID #1 = 0x01/0x02/0x03 in Non-Real-Time → Sample Dump
        // Header / Data Packet / Request — these are Sub-ID #2-less
        // categories per Table 4.
        let cases = [
            (0x01u8, UniversalSubId1::SampleDumpHeader),
            (0x02, UniversalSubId1::SampleDataPacket),
            (0x03, UniversalSubId1::SampleDumpRequest),
        ];
        for (sub1, expected) in cases {
            let ev = make_sysex(&[0x7E, 0x00, sub1, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(u.realm, UniversalRealm::NonRealTime);
            assert_eq!(u.sub_id1, expected);
        }
    }

    #[test]
    fn universal_classification_general_information_identity_pair() {
        let req = make_sysex(&[0x7E, 0x7F, 0x06, 0x01, 0xF7]);
        let rep = make_sysex(&[0x7E, 0x00, 0x06, 0x02, 0xF7]);
        let ureq = req.universal_classification().unwrap();
        let urep = rep.universal_classification().unwrap();
        assert_eq!(
            ureq.sub_id1,
            UniversalSubId1::GeneralInformationOrMmcCommands(
                UniversalSubId2::GeneralInformationIdentityRequest,
            )
        );
        assert_eq!(
            urep.sub_id1,
            UniversalSubId1::GeneralInformationOrMmcCommands(
                UniversalSubId2::GeneralInformationIdentityReply,
            )
        );
    }

    #[test]
    fn universal_classification_file_dump_header_data_request() {
        let cases = [
            (0x01u8, UniversalSubId2::FileDumpHeader),
            (0x02, UniversalSubId2::FileDumpDataPacket),
            (0x03, UniversalSubId2::FileDumpRequest),
        ];
        for (sub2, expected) in cases {
            let ev = make_sysex(&[0x7E, 0x00, 0x07, sub2, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(u.sub_id1, UniversalSubId1::FileDumpOrMmcResponses(expected));
        }
    }

    #[test]
    fn universal_classification_mts_nonrt_full_table() {
        let cases = [
            (0x00u8, UniversalSubId2::MtsBulkDumpRequest),
            (0x01, UniversalSubId2::MtsBulkDumpReply),
            (0x03, UniversalSubId2::MtsTuningDumpRequest),
            (0x04, UniversalSubId2::MtsKeyBasedTuningDump),
            (0x05, UniversalSubId2::MtsScaleOctaveTuningDump1Byte),
            (0x06, UniversalSubId2::MtsScaleOctaveTuningDump2Byte),
            (
                0x07,
                UniversalSubId2::MtsSingleNoteTuningChangeWithBankSelect,
            ),
            (0x08, UniversalSubId2::MtsScaleOctaveTuning1Byte),
            (0x09, UniversalSubId2::MtsScaleOctaveTuning2Byte),
        ];
        for (sub2, expected) in cases {
            let ev = make_sysex(&[0x7E, 0x7F, 0x08, sub2, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(u.realm, UniversalRealm::NonRealTime);
            assert_eq!(u.sub_id1, UniversalSubId1::MidiTuningStandard(expected));
        }
    }

    #[test]
    fn universal_classification_mts_rt_full_table() {
        let cases = [
            (0x02u8, UniversalSubId2::MtsRtSingleNoteTuningChange),
            (
                0x07,
                UniversalSubId2::MtsSingleNoteTuningChangeWithBankSelect,
            ),
            (0x08, UniversalSubId2::MtsScaleOctaveTuning1Byte),
            (0x09, UniversalSubId2::MtsScaleOctaveTuning2Byte),
        ];
        for (sub2, expected) in cases {
            let ev = make_sysex(&[0x7F, 0x7F, 0x08, sub2, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(u.realm, UniversalRealm::RealTime);
            assert_eq!(u.sub_id1, UniversalSubId1::MidiTuningStandard(expected));
        }
    }

    #[test]
    fn universal_classification_dls_quartet_nonrt() {
        let cases = [
            (0x01u8, UniversalSubId2::DlsTurnOn),
            (0x02, UniversalSubId2::DlsTurnOff),
            (0x03, UniversalSubId2::DlsTurnVoiceAllocationOff),
            (0x04, UniversalSubId2::DlsTurnVoiceAllocationOn),
        ];
        for (sub2, expected) in cases {
            let ev = make_sysex(&[0x7E, 0x7F, 0x0A, sub2, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(
                u.sub_id1,
                UniversalSubId1::DownloadableSoundsOrKeyBasedInstrumentControl(expected)
            );
        }
    }

    #[test]
    fn universal_classification_file_reference_quartet_nonrt() {
        let cases = [
            (0x01u8, UniversalSubId2::FileReferenceOpenFile),
            (0x02, UniversalSubId2::FileReferenceSelectOrReselectContents),
            (
                0x03,
                UniversalSubId2::FileReferenceOpenFileAndSelectContents,
            ),
            (0x04, UniversalSubId2::FileReferenceCloseFile),
        ];
        for (sub2, expected) in cases {
            let ev = make_sysex(&[0x7E, 0x7F, 0x0B, sub2, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(
                u.sub_id1,
                UniversalSubId1::FileReferenceOrScalablePolyphonyMip(expected)
            );
        }
    }

    #[test]
    fn universal_classification_mvc_and_capability_inquiry_route_through() {
        // MIDI Visual Control 0x0C — Sub-ID #2 is opaque to the
        // classifier; the helper surfaces the raw byte through Other.
        let ev = make_sysex(&[0x7E, 0x7F, 0x0C, 0x12, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::MidiVisualControlOrMobilePhoneControl(UniversalSubId2::Other(0x12))
        );
        // MIDI Capability Inquiry 0x0D — also opaque per Table 4.
        let ev = make_sysex(&[0x7E, 0x7F, 0x0D, 0x34, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::MidiCapabilityInquiry(UniversalSubId2::Other(0x34))
        );
    }

    #[test]
    fn universal_classification_singletons_eof_wait_cancel_nak_ack() {
        let cases = [
            (0x7Bu8, UniversalSubId1::EndOfFile),
            (0x7C, UniversalSubId1::Wait),
            (0x7D, UniversalSubId1::Cancel),
            (0x7E, UniversalSubId1::Nak),
            (0x7F, UniversalSubId1::Ack),
        ];
        for (sub1, expected) in cases {
            let ev = make_sysex(&[0x7E, 0x00, sub1, 0xF7]);
            let u = ev.universal_classification().unwrap();
            assert_eq!(u.realm, UniversalRealm::NonRealTime);
            assert_eq!(u.sub_id1, expected);
        }
    }

    #[test]
    fn universal_classification_realm_zero_device_id_preserved() {
        // device-id = 0x00 (the specific-device target) is preserved,
        // not coerced to the broadcast (0x7F) target.
        let ev = make_sysex(&[0x7E, 0x00, 0x09, 0x01, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.device_id, 0x00);
        let ev = make_sysex(&[0x7E, 0x05, 0x09, 0x01, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.device_id, 0x05);
        let ev = make_sysex(&[0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.device_id, 0x7F);
    }

    #[test]
    fn universal_classification_notation_information_real_time() {
        let bar = make_sysex(&[0x7F, 0x7F, 0x03, 0x01, 0x00, 0x00, 0xF7]);
        let imm = make_sysex(&[0x7F, 0x7F, 0x03, 0x02, 0x04, 0x02, 0xF7]);
        let del = make_sysex(&[0x7F, 0x7F, 0x03, 0x42, 0x04, 0x02, 0xF7]);
        assert_eq!(
            bar.universal_classification().unwrap().sub_id1,
            UniversalSubId1::NotationInformation(UniversalSubId2::RtNotationBarNumber)
        );
        assert_eq!(
            imm.universal_classification().unwrap().sub_id1,
            UniversalSubId1::NotationInformation(UniversalSubId2::RtNotationTimeSignatureImmediate)
        );
        assert_eq!(
            del.universal_classification().unwrap().sub_id1,
            UniversalSubId1::NotationInformation(UniversalSubId2::RtNotationTimeSignatureDelayed)
        );
    }

    #[test]
    fn universal_classification_msc_extensions_distinct_from_msc_commands() {
        // MSC Extensions: F0 7F <dev> 02 00 ...
        let ext = make_sysex(&[0x7F, 0x00, 0x02, 0x00, 0xF7]);
        assert_eq!(
            ext.universal_classification().unwrap().sub_id1,
            UniversalSubId1::MidiShowControl(UniversalSubId2::RtMscExtensions)
        );
        // MSC Command: F0 7F <dev> 02 01 ... — opaque byte through Other.
        let cmd = make_sysex(&[0x7F, 0x00, 0x02, 0x01, 0xF7]);
        assert_eq!(
            cmd.universal_classification().unwrap().sub_id1,
            UniversalSubId1::MidiShowControl(UniversalSubId2::Other(0x01))
        );
    }

    #[test]
    fn universal_classification_realm_distinguishes_nonrt_02_from_rt_02() {
        // Sub-ID #1 = 0x02 means different things in the two realms:
        //   Non-Real-Time → Sample Data Packet (singleton-shaped Sub-ID #1)
        //   Real-Time     → MIDI Show Control (branches on Sub-ID #2)
        let nrt = make_sysex(&[0x7E, 0x00, 0x02, 0xF7]);
        let rt = make_sysex(&[0x7F, 0x00, 0x02, 0x00, 0xF7]);
        assert_eq!(
            nrt.universal_classification().unwrap().sub_id1,
            UniversalSubId1::SampleDataPacket
        );
        assert_eq!(
            rt.universal_classification().unwrap().sub_id1,
            UniversalSubId1::MidiShowControl(UniversalSubId2::RtMscExtensions)
        );
    }

    #[test]
    fn universal_classification_unknown_sub_id1_surfaces_through_other() {
        // Sub-ID #1 = 0x40 — outside Table 4's named vocabulary at the
        // round-246 trace of the doc. Surfaces through Other(0x40).
        let ev = make_sysex(&[0x7E, 0x7F, 0x40, 0x01, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(u.sub_id1, UniversalSubId1::Other(0x40));
    }

    #[test]
    fn universal_classification_unknown_sub_id2_surfaces_through_other() {
        // Sub-ID #1 = 0x04 (Non-Real-Time MTC Setup), Sub-ID #2 = 0x77
        // — outside Table 4's MTC Setup vocabulary. Surfaces through
        // Other(0x77).
        let ev = make_sysex(&[0x7E, 0x7F, 0x04, 0x77, 0xF7]);
        let u = ev.universal_classification().unwrap();
        assert_eq!(
            u.sub_id1,
            UniversalSubId1::MidiTimeCode(UniversalSubId2::Other(0x77))
        );
    }

    #[test]
    fn universal_classification_threaded_through_sysex_events_collection() {
        // End-to-end: parse an SMF carrying a GM-on, a Master Volume,
        // and a Master Fine Tuning, walk smf.sysex_events(), classify
        // each, and confirm the classifier finds the expected family.
        let mut events: Vec<u8> = Vec::new();
        // F0 05 7E 7F 09 01 F7 — GM 1 System On
        events.extend_from_slice(&[0x00, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7]);
        // F0 07 7F 7F 04 01 lsb msb F7 — Master Volume
        events.extend_from_slice(&[0x00, 0xF0, 0x07, 0x7F, 0x7F, 0x04, 0x01, 0x40, 0x60, 0xF7]);
        // F0 0A 7F 7F 04 03 nn ll mm rr ss F7 — Master Fine Tuning (CA-25)
        events.extend_from_slice(&[
            0x00, 0xF0, 0x0A, 0x7F, 0x7F, 0x04, 0x03, 0x00, 0x40, 0x00, 0x00, 0x00, 0xF7,
        ]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let sx = smf.sysex_events();
        assert_eq!(sx.len(), 3);
        let classes: Vec<_> = sx
            .iter()
            .map(|s| s.universal_classification().unwrap())
            .collect();
        assert_eq!(
            classes[0].sub_id1,
            UniversalSubId1::GeneralMidiOrControllerDestination(
                UniversalSubId2::GeneralMidi1SystemOn,
            )
        );
        assert_eq!(
            classes[1].sub_id1,
            UniversalSubId1::DeviceControl(UniversalSubId2::DeviceControlMasterVolume)
        );
        assert_eq!(
            classes[2].sub_id1,
            UniversalSubId1::DeviceControl(UniversalSubId2::DeviceControlMasterFineTuning)
        );
        // Realm split: GM-on is Non-Real-Time, the two Device Control
        // packets are Real-Time.
        assert_eq!(classes[0].realm, UniversalRealm::NonRealTime);
        assert_eq!(classes[1].realm, UniversalRealm::RealTime);
        assert_eq!(classes[2].realm, UniversalRealm::RealTime);
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

    // ───────── ProgramChangeEvent / SmfFile::program_changes (Cn pp) ─────────

    #[test]
    fn program_changes_empty_when_no_patch_select_present() {
        // Note On / Note Off only — no Program Change.
        let events: Vec<u8> = vec![
            0x00, 0x90, 0x3C, 0x64, 0x60, 0x80, 0x3C, 0x40, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.program_changes().is_empty());
    }

    #[test]
    fn program_changes_single_patch_at_tick_zero() {
        // delta=0 C0 28   (channel 0, program 40 = Violin under GM 1)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![0x00, 0xC0, 0x28, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 1);
        assert_eq!(pcs[0].tick, 0);
        assert_eq!(pcs[0].track, 0);
        assert_eq!(pcs[0].channel(), 0);
        assert_eq!(pcs[0].program(), 40);
    }

    #[test]
    fn program_changes_full_seven_bit_program_range_round_trips() {
        // C5 7F → channel 5, program 127 (max legal value).
        let events: Vec<u8> = vec![0x00, 0xC5, 0x7F, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 1);
        assert_eq!(pcs[0].channel, 5);
        assert_eq!(pcs[0].program, 127);
    }

    #[test]
    fn program_changes_low_nibble_decodes_channel_index() {
        // CF 01 → channel 15 (the high nibble of Cn is the status,
        // the low nibble is the channel index 0..=15).
        let events: Vec<u8> = vec![0x00, 0xCF, 0x01, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 1);
        assert_eq!(pcs[0].channel(), 15);
        assert_eq!(pcs[0].program(), 1);
    }

    #[test]
    fn program_changes_running_status_chain_decodes_each_change() {
        // C0 28 followed by a running-status reuse: C0 29 (no status byte
        // — running status repeats the previous Cn). Spec §"Running
        // Status" says any single-data-byte status (`Cn`, `Dn`) keeps a
        // single running-status data byte per event.
        // delta=0 C0 28  (set status; channel 0 / program 40)
        // delta=0   29   (running status — channel 0 / program 41)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![0x00, 0xC0, 0x28, 0x00, 0x29, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 2);
        assert_eq!(pcs[0].channel, 0);
        assert_eq!(pcs[0].program, 40);
        assert_eq!(pcs[1].channel, 0);
        assert_eq!(pcs[1].program, 41);
    }

    #[test]
    fn program_changes_late_position_tracks_absolute_tick() {
        // Patch change after a 240-tick rest — the helper surfaces it
        // at tick 240, not at zero.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xC2, 0x49]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 1);
        assert_eq!(pcs[0].tick, 240);
        assert_eq!(pcs[0].channel, 2);
        assert_eq!(pcs[0].program, 73);
    }

    #[test]
    fn program_changes_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two format-1 tracks both with a Program Change at tick 0 —
        // the stable sort keeps track 0 before track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xC0, 0x10]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xC1, 0x11]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 2);
        assert_eq!(pcs[0].track, 0);
        assert_eq!(pcs[0].channel, 0);
        assert_eq!(pcs[0].program, 0x10);
        assert_eq!(pcs[1].track, 1);
        assert_eq!(pcs[1].channel, 1);
        assert_eq!(pcs[1].program, 0x11);
    }

    #[test]
    fn program_changes_merge_across_tracks_sorted_by_tick() {
        // Track 0 at tick 100, track 1 at tick 50 — sort drops the
        // track-1 entry first.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(100));
        t0.extend_from_slice(&[0xC0, 0x20]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(50));
        t1.extend_from_slice(&[0xC1, 0x30]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 2);
        assert_eq!(pcs[0].tick, 50);
        assert_eq!(pcs[0].track, 1);
        assert_eq!(pcs[0].channel, 1);
        assert_eq!(pcs[0].program, 0x30);
        assert_eq!(pcs[1].tick, 100);
        assert_eq!(pcs[1].track, 0);
        assert_eq!(pcs[1].channel, 0);
        assert_eq!(pcs[1].program, 0x20);
    }

    #[test]
    fn program_changes_filter_excludes_other_channel_voice_kinds() {
        // Cn picked up; note-on / note-off / CC / pitch bend /
        // aftertouch all filtered out. Sibling helpers stay
        // uncontaminated.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x50]); // CC vol
        events.extend_from_slice(&[0x00, 0xC0, 0x05]); // program change
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x40]); // pitch bend
        events.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x20]); // poly AT
        events.extend_from_slice(&[0x00, 0xD0, 0x40]); // channel AT
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x40]); // note off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 1);
        assert_eq!(pcs[0].channel, 0);
        assert_eq!(pcs[0].program, 5);
    }

    #[test]
    fn program_changes_seek_initialisation_matches_channel_snapshot() {
        // The snapshot folds the *last* program change before the
        // seek tick into SmfChannelSnapshot::program; program_changes()
        // exposes the *full* timeline. Verify the two agree on what the
        // snapshot's program field would be at every change point.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xC0, 0x00]); // tick 0 — GM Piano
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xC0, 0x28]); // tick 100 — Violin
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xC0, 0x49]); // tick 200 — Flute
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pcs = smf.program_changes();
        assert_eq!(pcs.len(), 3);
        // Snapshot at each successive change point should match the
        // program byte the iterator surfaces.
        for (i, pc) in pcs.iter().enumerate() {
            let snap = smf.channel_snapshot_at(0, pc.tick);
            assert_eq!(
                snap.program,
                Some(pc.program),
                "snapshot at change {i} (tick {}) should resolve to program {}",
                pc.tick,
                pc.program
            );
        }
        // Mid-way between changes the snapshot still reports the most
        // recent program selected.
        let snap_mid = smf.channel_snapshot_at(0, 150);
        assert_eq!(snap_mid.program, Some(40));
    }

    // ───────── ControlChangeEvent / SmfFile::control_changes (Bn cc vv) ─────────

    #[test]
    fn control_changes_empty_when_no_cc_present() {
        // Note On / Note Off / Program Change only — no CC.
        let events: Vec<u8> = vec![
            0x00, 0xC0, 0x05, 0x00, 0x90, 0x3C, 0x64, 0x60, 0x80, 0x3C, 0x40, 0x00, 0xFF, 0x2F,
            0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.control_changes().is_empty());
    }

    #[test]
    fn control_changes_single_volume_at_tick_zero() {
        // delta=0 B0 07 64   (channel 0, controller 7 = Volume, value 100 = power-up default)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![0x00, 0xB0, 0x07, 0x64, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 1);
        assert_eq!(ccs[0].tick, 0);
        assert_eq!(ccs[0].track, 0);
        assert_eq!(ccs[0].channel(), 0);
        assert_eq!(ccs[0].controller(), 7);
        assert_eq!(ccs[0].value(), 100);
        assert!(!ccs[0].is_channel_mode());
    }

    #[test]
    fn control_changes_full_seven_bit_value_range_round_trips() {
        // B5 0A 7F → channel 5, controller 10 (Pan), value 127 (max legal).
        let events: Vec<u8> = vec![0x00, 0xB5, 0x0A, 0x7F, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 1);
        assert_eq!(ccs[0].channel, 5);
        assert_eq!(ccs[0].controller, 10);
        assert_eq!(ccs[0].value, 127);
    }

    #[test]
    fn control_changes_low_nibble_decodes_channel_index() {
        // BF 01 40 → channel 15 (the high nibble of Bn is the status,
        // the low nibble is the channel index 0..=15).
        let events: Vec<u8> = vec![0x00, 0xBF, 0x01, 0x40, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 1);
        assert_eq!(ccs[0].channel(), 15);
        assert_eq!(ccs[0].controller(), 1);
        assert_eq!(ccs[0].value(), 0x40);
    }

    #[test]
    fn control_changes_running_status_chain_decodes_each_change() {
        // Running-status reuse — Bn is a two-data-byte status so each
        // running-status frame supplies *two* bytes (`cc` + `vv`).
        // delta=0 B0 01 00  (set status; channel 0 / CC 1 Modulation / value 0)
        // delta=0    01 20  (running status — channel 0 / CC 1 / value 32)
        // delta=0    07 64  (running status — channel 0 / CC 7 Volume / value 100)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x01, 0x00, 0x00, 0x01, 0x20, 0x00, 0x07, 0x64, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 3);
        assert_eq!(ccs[0].channel, 0);
        assert_eq!(ccs[0].controller, 1);
        assert_eq!(ccs[0].value, 0);
        assert_eq!(ccs[1].channel, 0);
        assert_eq!(ccs[1].controller, 1);
        assert_eq!(ccs[1].value, 32);
        assert_eq!(ccs[2].channel, 0);
        assert_eq!(ccs[2].controller, 7);
        assert_eq!(ccs[2].value, 100);
    }

    #[test]
    fn control_changes_late_position_tracks_absolute_tick() {
        // CC after a 240-tick rest — the helper surfaces it at tick
        // 240, not at zero.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xB2, 0x0B, 0x7F]); // CC 11 Expression = 127 on ch 2
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 1);
        assert_eq!(ccs[0].tick, 240);
        assert_eq!(ccs[0].channel, 2);
        assert_eq!(ccs[0].controller, 11);
        assert_eq!(ccs[0].value, 127);
    }

    #[test]
    fn control_changes_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two format-1 tracks both with a CC at tick 0 — the stable
        // sort keeps track 0 before track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xB0, 0x07, 0x40]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xB1, 0x07, 0x60]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 2);
        assert_eq!(ccs[0].track, 0);
        assert_eq!(ccs[0].channel, 0);
        assert_eq!(ccs[0].value, 0x40);
        assert_eq!(ccs[1].track, 1);
        assert_eq!(ccs[1].channel, 1);
        assert_eq!(ccs[1].value, 0x60);
    }

    #[test]
    fn control_changes_merge_across_tracks_sorted_by_tick() {
        // Track 0 at tick 100, track 1 at tick 50 — sort drops the
        // track-1 entry first.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(100));
        t0.extend_from_slice(&[0xB0, 0x07, 0x20]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(50));
        t1.extend_from_slice(&[0xB1, 0x0A, 0x30]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 2);
        assert_eq!(ccs[0].tick, 50);
        assert_eq!(ccs[0].track, 1);
        assert_eq!(ccs[0].channel, 1);
        assert_eq!(ccs[0].controller, 10);
        assert_eq!(ccs[0].value, 0x30);
        assert_eq!(ccs[1].tick, 100);
        assert_eq!(ccs[1].track, 0);
        assert_eq!(ccs[1].channel, 0);
        assert_eq!(ccs[1].controller, 7);
        assert_eq!(ccs[1].value, 0x20);
    }

    #[test]
    fn control_changes_filter_excludes_other_channel_voice_kinds() {
        // Bn picked up; note-on / note-off / program / pitch bend /
        // aftertouch all filtered out. Sibling helpers stay
        // uncontaminated.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x50]); // CC vol
        events.extend_from_slice(&[0x00, 0xC0, 0x05]); // program change
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x40]); // pitch bend
        events.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x20]); // poly AT
        events.extend_from_slice(&[0x00, 0xD0, 0x40]); // channel AT
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x40]); // note off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 1);
        assert_eq!(ccs[0].channel, 0);
        assert_eq!(ccs[0].controller, 7);
        assert_eq!(ccs[0].value, 0x50);
    }

    #[test]
    fn control_changes_channel_mode_family_surfaces_with_predicate_set() {
        // Channel-mode controllers (120..=127) ride the same Bn lane
        // as continuous CCs — the helper surfaces them and the
        // is_channel_mode() predicate flags them.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xB0, 120, 0x00]); // All Sound Off
        events.extend_from_slice(&[0x00, 0xB0, 121, 0x00]); // Reset All Controllers
        events.extend_from_slice(&[0x00, 0xB0, 123, 0x00]); // All Notes Off
        events.extend_from_slice(&[0x00, 0xB0, 127, 0x00]); // Poly Mode On
        events.extend_from_slice(&[0x00, 0xB0, 7, 100]); // Volume — continuous, not mode
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        assert_eq!(ccs.len(), 5);
        assert!(ccs[0].is_channel_mode());
        assert!(ccs[1].is_channel_mode());
        assert!(ccs[2].is_channel_mode());
        assert!(ccs[3].is_channel_mode());
        assert!(!ccs[4].is_channel_mode()); // CC 7 Volume stays continuous
    }

    #[test]
    fn control_changes_seek_initialisation_matches_channel_snapshot() {
        // The snapshot folds the *last* CC-7 (Volume) before the seek
        // tick into SmfChannelSnapshot::volume; control_changes() exposes
        // the full automation timeline. Verify the two agree at every
        // change point and at a mid-way tick.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x10]); // tick 0 — vol 16
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xB0, 0x07, 0x40]); // tick 100 — vol 64
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xB0, 0x07, 0x7F]); // tick 200 — vol 127
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let ccs = smf.control_changes();
        // Filter to CC-7 only (the snapshot only tracks specific CCs).
        let vol: Vec<_> = ccs.iter().filter(|c| c.controller == 7).collect();
        assert_eq!(vol.len(), 3);
        for (i, cc) in vol.iter().enumerate() {
            let snap = smf.channel_snapshot_at(0, cc.tick);
            assert_eq!(
                snap.volume, cc.value,
                "snapshot at change {i} (tick {}) should resolve to volume {}",
                cc.tick, cc.value
            );
        }
        // Mid-way between changes the snapshot still reports the most
        // recent volume selected.
        let snap_mid = smf.channel_snapshot_at(0, 150);
        assert_eq!(snap_mid.volume, 0x40);
    }

    #[test]
    fn control_changes_to_bytes_round_trip() {
        // CC stream survives a to_bytes() / parse() round trip.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xB0, 0x01, 0x10]); // Modulation
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x55]); // Volume
        events.extend_from_slice(&[0x00, 0xB0, 0x0A, 0x40]); // Pan
        events.extend_from_slice(&[0x00, 0xB0, 0x40, 0x7F]); // Sustain on
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let original = smf.control_changes();
        assert_eq!(original.len(), 4);
        let muxed = smf.to_bytes().unwrap();
        let reparsed = parse(&muxed).unwrap();
        let after = reparsed.control_changes();
        assert_eq!(original, after);
    }

    // ───────── PitchBendEvent / SmfFile::pitch_bends (En lsb msb) ─────────

    #[test]
    fn pitch_bends_empty_when_no_bend_present() {
        // Note On / Note Off / CC / Program Change only — no En.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x07, 0x64, 0x00, 0x90, 0x3C, 0x64, 0x60, 0x80, 0x3C, 0x40, 0x00, 0xFF,
            0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.pitch_bends().is_empty());
    }

    #[test]
    fn pitch_bends_centre_at_tick_zero() {
        // delta=0 E0 00 40 → channel 0, lsb=0x00 msb=0x40 → (0x40<<7)|0 = 0x2000 centre.
        let events: Vec<u8> = vec![0x00, 0xE0, 0x00, 0x40, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 1);
        assert_eq!(pbs[0].tick, 0);
        assert_eq!(pbs[0].track, 0);
        assert_eq!(pbs[0].channel(), 0);
        assert_eq!(pbs[0].value(), 0x2000);
        assert_eq!(pbs[0].signed_value(), 0);
        assert!(pbs[0].is_centre());
    }

    #[test]
    fn pitch_bends_lsb_msb_combine_to_14bit_value() {
        // E0 7F 7F → max value (0x7F<<7)|0x7F = 0x3FFF; signed +8191.
        // E0 00 00 → min value 0; signed -8192.
        let events: Vec<u8> = vec![
            0x00, 0xE0, 0x7F, 0x7F, 0x00, 0xE0, 0x00, 0x00, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 2);
        assert_eq!(pbs[0].value(), 0x3FFF);
        assert_eq!(pbs[0].signed_value(), 8191);
        assert!(!pbs[0].is_centre());
        assert_eq!(pbs[1].value(), 0x0000);
        assert_eq!(pbs[1].signed_value(), -8192);
        assert!(!pbs[1].is_centre());
    }

    #[test]
    fn pitch_bends_low_nibble_decodes_channel_index() {
        // EF 40 60 → channel 15; value (0x60<<7)|0x40 = 0x3040.
        let events: Vec<u8> = vec![0x00, 0xEF, 0x40, 0x60, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 1);
        assert_eq!(pbs[0].channel(), 15);
        assert_eq!(pbs[0].value(), 0x3040);
        assert_eq!(pbs[0].signed_value(), 0x3040 - 0x2000);
    }

    #[test]
    fn pitch_bends_running_status_chain_decodes_each_bend() {
        // Running-status reuse — En is a two-data-byte status so each
        // running-status frame supplies *two* bytes (`lsb` + `msb`).
        // delta=0 E0 00 40  (set status; channel 0, centre 0x2000)
        // delta=0    00 20  (running status — 0x1000)
        // delta=0    7F 5F  (running status — 0x2FFF)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![
            0x00, 0xE0, 0x00, 0x40, 0x00, 0x00, 0x20, 0x00, 0x7F, 0x5F, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 3);
        assert_eq!(pbs[0].value, 0x2000);
        assert_eq!(pbs[1].value, 0x1000);
        assert_eq!(pbs[2].value, (0x5F << 7) | 0x7F);
        for pb in &pbs {
            assert_eq!(pb.channel, 0);
        }
    }

    #[test]
    fn pitch_bends_late_position_tracks_absolute_tick() {
        // Bend after a 240-tick rest — surfaced at tick 240, not zero.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xE2, 0x00, 0x30]); // bend on ch 2 → 0x1800
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 1);
        assert_eq!(pbs[0].tick, 240);
        assert_eq!(pbs[0].channel, 2);
        assert_eq!(pbs[0].value, 0x1800);
    }

    #[test]
    fn pitch_bends_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two format-1 tracks both with a bend at tick 0 — stable sort
        // keeps track 0 before track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xE0, 0x00, 0x10]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xE1, 0x00, 0x70]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 2);
        assert_eq!(pbs[0].track, 0);
        assert_eq!(pbs[0].channel, 0);
        assert_eq!(pbs[0].value, 0x0800);
        assert_eq!(pbs[1].track, 1);
        assert_eq!(pbs[1].channel, 1);
        assert_eq!(pbs[1].value, 0x3800);
    }

    #[test]
    fn pitch_bends_merge_across_tracks_sorted_by_tick() {
        // Track 0 at tick 100, track 1 at tick 50 — sort drops the
        // track-1 entry first.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(100));
        t0.extend_from_slice(&[0xE0, 0x00, 0x10]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(50));
        t1.extend_from_slice(&[0xE1, 0x00, 0x70]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 2);
        assert_eq!(pbs[0].tick, 50);
        assert_eq!(pbs[0].track, 1);
        assert_eq!(pbs[0].channel, 1);
        assert_eq!(pbs[0].value, 0x3800);
        assert_eq!(pbs[1].tick, 100);
        assert_eq!(pbs[1].track, 0);
        assert_eq!(pbs[1].channel, 0);
        assert_eq!(pbs[1].value, 0x0800);
    }

    #[test]
    fn pitch_bends_filter_excludes_other_channel_voice_kinds() {
        // En picked up; note-on / note-off / CC / program / aftertouch
        // all filtered out. Sibling helpers stay uncontaminated.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x50]); // CC vol
        events.extend_from_slice(&[0x00, 0xC0, 0x05]); // program change
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x50]); // pitch bend → 0x2800
        events.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x20]); // poly AT
        events.extend_from_slice(&[0x00, 0xD0, 0x40]); // channel AT
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x40]); // note off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 1);
        assert_eq!(pbs[0].channel, 0);
        assert_eq!(pbs[0].value, 0x2800);
    }

    #[test]
    fn pitch_bends_seek_initialisation_matches_channel_snapshot() {
        // The snapshot folds the *last* En before the seek tick into
        // SmfChannelSnapshot::pitch_bend; pitch_bends() exposes the full
        // automation timeline. Verify the two agree at every change
        // point and at a mid-way tick.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x10]); // tick 0 → 0x0800
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xE0, 0x00, 0x40]); // tick 100 → 0x2000 centre
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0xE0, 0x7F, 0x7F]); // tick 200 → 0x3FFF
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pbs = smf.pitch_bends();
        assert_eq!(pbs.len(), 3);
        for pb in &pbs {
            let snap = smf.channel_snapshot_at(0, pb.tick);
            assert_eq!(
                snap.pitch_bend, pb.value,
                "snapshot at tick {} should resolve to pitch_bend {:#06X}",
                pb.tick, pb.value
            );
        }
        // Mid-way between changes the snapshot still reports the most
        // recent bend selected.
        let snap_mid = smf.channel_snapshot_at(0, 150);
        assert_eq!(snap_mid.pitch_bend, 0x2000);
    }

    #[test]
    fn pitch_bends_to_bytes_round_trip() {
        // Bend stream survives a to_bytes() / parse() round trip.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x00]); // min
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x40]); // centre
        events.extend_from_slice(&[0x00, 0xE0, 0x7F, 0x7F]); // max
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let original = smf.pitch_bends();
        assert_eq!(original.len(), 3);
        let muxed = smf.to_bytes().unwrap();
        let reparsed = parse(&muxed).unwrap();
        let after = reparsed.pitch_bends();
        assert_eq!(original, after);
    }

    #[test]
    fn channel_pressures_single_byte_at_tick_zero() {
        // delta=0 D0 40 → channel 0, pressure 0x40.
        let events: Vec<u8> = vec![0x00, 0xD0, 0x40, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_pressures();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].tick, 0);
        assert_eq!(cps[0].track, 0);
        assert_eq!(cps[0].channel(), 0);
        assert_eq!(cps[0].pressure(), 0x40);
    }

    #[test]
    fn channel_pressures_low_nibble_decodes_channel_index() {
        // DF 7F → channel 15, max pressure 0x7F.
        let events: Vec<u8> = vec![0x00, 0xDF, 0x7F, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_pressures();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].channel(), 15);
        assert_eq!(cps[0].pressure(), 0x7F);
    }

    #[test]
    fn channel_pressures_running_status_chain_decodes_each_value() {
        // Dn is a one-data-byte status so each running-status frame
        // supplies a single pressure byte.
        // delta=0 D0 10  (set status; channel 0, pressure 0x10)
        // delta=0    40  (running status — 0x40)
        // delta=0    7F  (running status — 0x7F)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![
            0x00, 0xD0, 0x10, 0x00, 0x40, 0x00, 0x7F, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_pressures();
        assert_eq!(cps.len(), 3);
        assert_eq!(cps[0].pressure, 0x10);
        assert_eq!(cps[1].pressure, 0x40);
        assert_eq!(cps[2].pressure, 0x7F);
        for cp in &cps {
            assert_eq!(cp.channel, 0);
        }
    }

    #[test]
    fn channel_pressures_late_position_tracks_absolute_tick() {
        // Pressure after a 240-tick rest — surfaced at tick 240.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xD2, 0x55]); // channel 2, pressure 0x55
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_pressures();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].tick, 240);
        assert_eq!(cps[0].channel, 2);
        assert_eq!(cps[0].pressure, 0x55);
    }

    #[test]
    fn channel_pressures_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two format-1 tracks both with a pressure event at tick 0 —
        // stable sort keeps track 0 before track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xD0, 0x11]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xD1, 0x77]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_pressures();
        assert_eq!(cps.len(), 2);
        assert_eq!(cps[0].track, 0);
        assert_eq!(cps[0].channel, 0);
        assert_eq!(cps[0].pressure, 0x11);
        assert_eq!(cps[1].track, 1);
        assert_eq!(cps[1].channel, 1);
        assert_eq!(cps[1].pressure, 0x77);
    }

    #[test]
    fn channel_pressures_merge_across_tracks_sorted_by_tick() {
        // Track 0 at tick 100, track 1 at tick 50 — sort drops the
        // track-1 entry first.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(100));
        t0.extend_from_slice(&[0xD0, 0x11]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(50));
        t1.extend_from_slice(&[0xD1, 0x77]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_pressures();
        assert_eq!(cps.len(), 2);
        assert_eq!(cps[0].tick, 50);
        assert_eq!(cps[0].track, 1);
        assert_eq!(cps[0].channel, 1);
        assert_eq!(cps[0].pressure, 0x77);
        assert_eq!(cps[1].tick, 100);
        assert_eq!(cps[1].track, 0);
        assert_eq!(cps[1].channel, 0);
        assert_eq!(cps[1].pressure, 0x11);
    }

    #[test]
    fn channel_pressures_filter_excludes_other_channel_voice_kinds() {
        // Dn picked up; note-on / note-off / CC / program / pitch-bend /
        // poly-aftertouch all filtered out.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x50]); // CC vol
        events.extend_from_slice(&[0x00, 0xC0, 0x05]); // program change
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x50]); // pitch bend
        events.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x20]); // poly AT
        events.extend_from_slice(&[0x00, 0xD0, 0x40]); // channel AT
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x40]); // note off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let cps = smf.channel_pressures();
        assert_eq!(cps.len(), 1);
        assert_eq!(cps[0].channel, 0);
        assert_eq!(cps[0].pressure, 0x40);
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.pitch_bends().len(), 1);
        assert_eq!(smf.program_changes().len(), 1);
        assert_eq!(smf.control_changes().len(), 1);
    }

    #[test]
    fn channel_pressures_to_bytes_round_trip() {
        // Channel-pressure stream survives a to_bytes() / parse() round trip.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xD0, 0x00]); // min
        events.extend_from_slice(&[0x00, 0xD0, 0x40]); // mid
        events.extend_from_slice(&[0x00, 0xD0, 0x7F]); // max
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let original = smf.channel_pressures();
        assert_eq!(original.len(), 3);
        let muxed = smf.to_bytes().unwrap();
        let reparsed = parse(&muxed).unwrap();
        let after = reparsed.channel_pressures();
        assert_eq!(original, after);
    }

    // ----------------------------------------------------------------
    // poly_aftertouches() — An kk pp Polyphonic Key Pressure stream.
    // ----------------------------------------------------------------

    #[test]
    fn poly_aftertouches_single_event_at_tick_zero() {
        // delta=0 A0 3C 40 → channel 0, key 0x3C, pressure 0x40.
        let events: Vec<u8> = vec![0x00, 0xA0, 0x3C, 0x40, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pat = smf.poly_aftertouches();
        assert_eq!(pat.len(), 1);
        assert_eq!(pat[0].tick, 0);
        assert_eq!(pat[0].track, 0);
        assert_eq!(pat[0].channel(), 0);
        assert_eq!(pat[0].key(), 0x3C);
        assert_eq!(pat[0].pressure(), 0x40);
    }

    #[test]
    fn poly_aftertouches_low_nibble_decodes_channel_index() {
        // AF 7F 7F → channel 15, key 0x7F, max pressure 0x7F.
        let events: Vec<u8> = vec![0x00, 0xAF, 0x7F, 0x7F, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pat = smf.poly_aftertouches();
        assert_eq!(pat.len(), 1);
        assert_eq!(pat[0].channel(), 15);
        assert_eq!(pat[0].key(), 0x7F);
        assert_eq!(pat[0].pressure(), 0x7F);
    }

    #[test]
    fn poly_aftertouches_running_status_chain_decodes_each_pair() {
        // An is a two-data-byte status so each running-status frame
        // supplies a (key, pressure) pair.
        // delta=0 A0 3C 10  (set status; channel 0, key 0x3C, pressure 0x10)
        // delta=0    3E 40  (running status — key 0x3E, pressure 0x40)
        // delta=0    40 7F  (running status — key 0x40, pressure 0x7F)
        // delta=0 FF 2F 00
        let events: Vec<u8> = vec![
            0x00, 0xA0, 0x3C, 0x10, 0x00, 0x3E, 0x40, 0x00, 0x40, 0x7F, 0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pat = smf.poly_aftertouches();
        assert_eq!(pat.len(), 3);
        assert_eq!((pat[0].key, pat[0].pressure), (0x3C, 0x10));
        assert_eq!((pat[1].key, pat[1].pressure), (0x3E, 0x40));
        assert_eq!((pat[2].key, pat[2].pressure), (0x40, 0x7F));
        for p in &pat {
            assert_eq!(p.channel, 0);
        }
    }

    #[test]
    fn poly_aftertouches_late_position_tracks_absolute_tick() {
        // Poly-pressure after a 240-tick rest — surfaced at tick 240.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0xA2, 0x3C, 0x55]); // channel 2, key 0x3C, pressure 0x55
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pat = smf.poly_aftertouches();
        assert_eq!(pat.len(), 1);
        assert_eq!(pat[0].tick, 240);
        assert_eq!(pat[0].channel, 2);
        assert_eq!(pat[0].key, 0x3C);
        assert_eq!(pat[0].pressure, 0x55);
    }

    #[test]
    fn poly_aftertouches_stable_sort_keeps_track0_before_track1_at_same_tick() {
        // Two format-1 tracks both with a poly-pressure event at tick 0 —
        // stable sort keeps track 0 before track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x11]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0xA1, 0x40, 0x77]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let pat = smf.poly_aftertouches();
        assert_eq!(pat.len(), 2);
        assert_eq!(
            (pat[0].track, pat[0].channel, pat[0].key, pat[0].pressure),
            (0, 0, 0x3C, 0x11)
        );
        assert_eq!(
            (pat[1].track, pat[1].channel, pat[1].key, pat[1].pressure),
            (1, 1, 0x40, 0x77)
        );
    }

    #[test]
    fn poly_aftertouches_merge_across_tracks_sorted_by_tick() {
        // Track 0 at tick 100, track 1 at tick 50 — sort drops the
        // track-1 entry first.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(100));
        t0.extend_from_slice(&[0xA0, 0x3C, 0x11]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(50));
        t1.extend_from_slice(&[0xA1, 0x40, 0x77]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let pat = smf.poly_aftertouches();
        assert_eq!(pat.len(), 2);
        assert_eq!(
            (pat[0].tick, pat[0].track, pat[0].key, pat[0].pressure),
            (50, 1, 0x40, 0x77)
        );
        assert_eq!(
            (pat[1].tick, pat[1].track, pat[1].key, pat[1].pressure),
            (100, 0, 0x3C, 0x11)
        );
    }

    #[test]
    fn poly_aftertouches_filter_excludes_other_channel_voice_kinds() {
        // An picked up; note-on / note-off / CC / program / pitch-bend /
        // channel-aftertouch all filtered out.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x50]); // CC vol
        events.extend_from_slice(&[0x00, 0xC0, 0x05]); // program change
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x50]); // pitch bend
        events.extend_from_slice(&[0x00, 0xD0, 0x40]); // channel AT
        events.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x20]); // poly AT
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x40]); // note off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pat = smf.poly_aftertouches();
        assert_eq!(pat.len(), 1);
        assert_eq!(pat[0].channel, 0);
        assert_eq!(pat[0].key, 0x3C);
        assert_eq!(pat[0].pressure, 0x20);
        // Sibling helpers stay uncontaminated.
        assert_eq!(smf.channel_pressures().len(), 1);
        assert_eq!(smf.pitch_bends().len(), 1);
        assert_eq!(smf.program_changes().len(), 1);
        assert_eq!(smf.control_changes().len(), 1);
    }

    #[test]
    fn poly_aftertouches_to_bytes_round_trip() {
        // Poly-pressure stream survives a to_bytes() / parse() round trip.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x00]); // min pressure
        events.extend_from_slice(&[0x00, 0xA0, 0x40, 0x40]); // mid
        events.extend_from_slice(&[0x00, 0xA0, 0x43, 0x7F]); // max
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let original = smf.poly_aftertouches();
        assert_eq!(original.len(), 3);
        let muxed = smf.to_bytes().unwrap();
        let reparsed = parse(&muxed).unwrap();
        let after = reparsed.poly_aftertouches();
        assert_eq!(original, after);
    }

    #[test]
    fn poly_aftertouches_empty_when_none_present() {
        // A note-only track carries no An events.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x40]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.poly_aftertouches().is_empty());
    }

    // ----------------------------------------------------------------
    // notes() — Note On / Note Off pairing into sounding-note spans.
    // ----------------------------------------------------------------

    #[test]
    fn notes_empty_when_no_note_activity() {
        // A conductor-only track (tempo + EOT) has no note spans.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // tempo
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.notes().is_empty());
    }

    #[test]
    fn notes_single_note_pairs_on_and_off() {
        // 9n 3C 64 at tick 0, 8n 3C 40 at tick 480.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on, vel 100
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]); // note off, rel-vel 64
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 1);
        let n = notes[0];
        assert_eq!(n.start_tick, 0);
        assert_eq!(n.end_tick, 480);
        assert_eq!(n.duration_ticks(), 480);
        assert_eq!(n.track, 0);
        assert_eq!(n.channel(), 0);
        assert_eq!(n.key(), 0x3C);
        assert_eq!(n.velocity(), 100);
        assert_eq!(n.off_velocity(), 0x40);
    }

    #[test]
    fn notes_velocity_zero_note_on_closes_open_note() {
        // 9n 3C 64 (on) then 9n 3C 00 (the running-status Note-Off form).
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&encode_vlq(240));
        events.extend_from_slice(&[0x90, 0x3C, 0x00]); // vel-0 = note off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[0].end_tick, 240);
        assert_eq!(notes[0].velocity(), 100);
        // velocity-0 form carries no release velocity.
        assert_eq!(notes[0].off_velocity(), 0);
    }

    #[test]
    fn notes_low_nibble_decodes_channel_index() {
        // 9F .. on channel 15.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x9F, 0x40, 0x7F]);
        events.extend_from_slice(&encode_vlq(96));
        events.extend_from_slice(&[0x8F, 0x40, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].channel(), 15);
        assert_eq!(notes[0].key(), 0x40);
        assert_eq!(notes[0].velocity(), 0x7F);
    }

    #[test]
    fn notes_overlapping_same_pitch_pair_fifo() {
        // Two onsets of the same pitch before either releases — the
        // first release closes the first (earliest) onset (FIFO).
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x50]); // on #1 @0, vel 0x50
        events.extend_from_slice(&encode_vlq(10));
        events.extend_from_slice(&[0x90, 0x3C, 0x60]); // on #2 @10, vel 0x60
        events.extend_from_slice(&encode_vlq(10));
        events.extend_from_slice(&[0x80, 0x3C, 0x20]); // off @20 → closes #1
        events.extend_from_slice(&encode_vlq(10));
        events.extend_from_slice(&[0x80, 0x3C, 0x30]); // off @30 → closes #2
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 2);
        // Ordered by onset: #1 then #2.
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[0].end_tick, 20);
        assert_eq!(notes[0].velocity(), 0x50);
        assert_eq!(notes[1].start_tick, 10);
        assert_eq!(notes[1].end_tick, 30);
        assert_eq!(notes[1].velocity(), 0x60);
    }

    #[test]
    fn notes_chord_keeps_distinct_pitches() {
        // Three pitches struck together, released together — three spans.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&[0x00, 0x90, 0x40, 0x64]);
        events.extend_from_slice(&[0x00, 0x90, 0x43, 0x64]);
        events.extend_from_slice(&encode_vlq(192));
        events.extend_from_slice(&[0x80, 0x3C, 0x00]);
        events.extend_from_slice(&[0x00, 0x80, 0x40, 0x00]);
        events.extend_from_slice(&[0x00, 0x80, 0x43, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 192);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 3);
        // Onset tick identical → on-disk order preserved (stable).
        assert_eq!(notes[0].key(), 0x3C);
        assert_eq!(notes[1].key(), 0x40);
        assert_eq!(notes[2].key(), 0x43);
        for n in &notes {
            assert_eq!(n.start_tick, 0);
            assert_eq!(n.end_tick, 192);
        }
    }

    #[test]
    fn notes_off_on_different_track_still_pairs() {
        // Note On on track 0, Note Off on track 1 at a later tick. The
        // globally-merged walk pairs them across tracks.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // on @0 track 0
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(120));
        t1.extend_from_slice(&[0x80, 0x3C, 0x10]); // off @120 track 1
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[0].end_tick, 120);
        // Onset track is recorded (track 0).
        assert_eq!(notes[0].track, 0);
        assert_eq!(notes[0].off_velocity(), 0x10);
    }

    #[test]
    fn notes_chord_across_tracks_orders_track0_first() {
        // Same onset tick, two tracks — track 0 before track 1.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        t0.extend_from_slice(&encode_vlq(96));
        t0.extend_from_slice(&[0x80, 0x3C, 0x00]);
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&[0x00, 0x91, 0x40, 0x64]);
        t1.extend_from_slice(&encode_vlq(96));
        t1.extend_from_slice(&[0x81, 0x40, 0x00]);
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].track, 0);
        assert_eq!(notes[0].channel(), 0);
        assert_eq!(notes[1].track, 1);
        assert_eq!(notes[1].channel(), 1);
    }

    #[test]
    fn notes_hanging_on_without_off_is_dropped() {
        // A Note On with no matching Note Off yields no span.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.notes().is_empty());
    }

    #[test]
    fn notes_unmatched_off_is_dropped() {
        // A Note Off with no open note of that pitch is dropped.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x40]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.notes().is_empty());
    }

    #[test]
    fn notes_zero_duration_when_off_shares_onset_tick() {
        // Note Off on the same tick as the Note On → zero-length span.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[0].end_tick, 0);
        assert_eq!(notes[0].duration_ticks(), 0);
    }

    #[test]
    fn notes_filter_excludes_other_channel_voice_kinds() {
        // Surrounding CC / program / pitch-bend / aftertouch don't break
        // the pairing and don't appear as notes.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xB0, 0x07, 0x50]); // CC
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // note on
        events.extend_from_slice(&[0x00, 0xC0, 0x05]); // program
        events.extend_from_slice(&[0x00, 0xE0, 0x00, 0x50]); // pitch bend
        events.extend_from_slice(&[0x00, 0xD0, 0x40]); // chan AT
        events.extend_from_slice(&[0x00, 0xA0, 0x3C, 0x20]); // poly AT
        events.extend_from_slice(&encode_vlq(48));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]); // note off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let notes = smf.notes();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].start_tick, 0);
        assert_eq!(notes[0].end_tick, 48);
        // Sibling helpers stay populated and uncontaminated.
        assert_eq!(smf.control_changes().len(), 1);
        assert_eq!(smf.program_changes().len(), 1);
        assert_eq!(smf.pitch_bends().len(), 1);
        assert_eq!(smf.channel_pressures().len(), 1);
    }

    #[test]
    fn notes_survive_to_bytes_round_trip() {
        // The pairing is identical before and after a to_bytes()/parse()
        // structural round trip.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&[0x00, 0x90, 0x40, 0x50]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x80, 0x3C, 0x10]);
        events.extend_from_slice(&encode_vlq(120));
        events.extend_from_slice(&[0x80, 0x40, 0x20]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let original = smf.notes();
        assert_eq!(original.len(), 2);
        let muxed = smf.to_bytes().unwrap();
        let reparsed = parse(&muxed).unwrap();
        assert_eq!(original, reparsed.notes());
    }

    // ----------------------------------------------------------------
    // active_notes_at() — half-open [start_tick, end_tick) seek lens.
    // ----------------------------------------------------------------

    #[test]
    fn active_notes_at_empty_when_no_note_activity() {
        // A tempo-only conductor track has no notes at any tick.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.active_notes_at(0).is_empty());
        assert!(smf.active_notes_at(1000).is_empty());
    }

    #[test]
    fn active_notes_at_half_open_interval_boundaries() {
        // One note sounding over [0, 480): struck @0, released @480.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // on @0
        events.extend_from_slice(&encode_vlq(480));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]); // off @480
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        // Onset tick is inclusive: the note is sounding immediately after
        // the strike fires.
        assert_eq!(smf.active_notes_at(0).len(), 1);
        // Mid-span.
        assert_eq!(smf.active_notes_at(479).len(), 1);
        // Release tick is exclusive: the key has come up at exactly 480.
        assert!(smf.active_notes_at(480).is_empty());
        // After release.
        assert!(smf.active_notes_at(481).is_empty());
    }

    #[test]
    fn active_notes_at_before_onset_is_silent() {
        // Note struck @100; nothing sounds at an earlier tick.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0x90, 0x3C, 0x64]); // on @100
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0x80, 0x3C, 0x40]); // off @200
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 480);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.active_notes_at(99).is_empty());
        assert_eq!(smf.active_notes_at(100).len(), 1);
    }

    #[test]
    fn active_notes_at_zero_duration_note_never_sounds() {
        // Note Off on the same tick as the Note On — a zero-duration
        // span ([start == end]) is sounding at no tick.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // on @0
        events.extend_from_slice(&[0x00, 0x80, 0x3C, 0x00]); // off @0
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        // notes() still reports the zero-duration span...
        assert_eq!(smf.notes().len(), 1);
        assert_eq!(smf.notes()[0].duration_ticks(), 0);
        // ...but active_notes_at reports it at no tick (half-open).
        assert!(smf.active_notes_at(0).is_empty());
    }

    #[test]
    fn active_notes_at_chord_returns_all_held_keys_in_onset_order() {
        // Three pitches struck @0, released @192 — all three sound mid-span.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]);
        events.extend_from_slice(&[0x00, 0x90, 0x40, 0x64]);
        events.extend_from_slice(&[0x00, 0x90, 0x43, 0x64]);
        events.extend_from_slice(&encode_vlq(192));
        events.extend_from_slice(&[0x80, 0x3C, 0x00]);
        events.extend_from_slice(&[0x00, 0x80, 0x40, 0x00]);
        events.extend_from_slice(&[0x00, 0x80, 0x43, 0x00]);
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 192);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let held = smf.active_notes_at(96);
        assert_eq!(held.len(), 3);
        // notes() order preserved: ascending on-disk pitch order.
        assert_eq!(held[0].key(), 0x3C);
        assert_eq!(held[1].key(), 0x40);
        assert_eq!(held[2].key(), 0x43);
    }

    #[test]
    fn active_notes_at_staggered_notes_overlap_window() {
        // n1 [0, 200), n2 [100, 300). At 150 both sound; at 50 only n1;
        // at 250 only n2.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // n1 on @0
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0x90, 0x40, 0x64]); // n2 on @100
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0x80, 0x3C, 0x00]); // n1 off @200
        events.extend_from_slice(&encode_vlq(100));
        events.extend_from_slice(&[0x80, 0x40, 0x00]); // n2 off @300
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let at50 = smf.active_notes_at(50);
        assert_eq!(at50.len(), 1);
        assert_eq!(at50[0].key(), 0x3C);
        assert_eq!(smf.active_notes_at(150).len(), 2);
        let at250 = smf.active_notes_at(250);
        assert_eq!(at250.len(), 1);
        assert_eq!(at250[0].key(), 0x40);
    }

    #[test]
    fn active_notes_at_hanging_note_never_sounds() {
        // A Note On with no matching Note Off is dropped by notes(), so
        // it cannot be reported as sounding at any tick.
        let mut events: Vec<u8> = Vec::new();
        events.extend_from_slice(&[0x00, 0x90, 0x3C, 0x64]); // on @0, no off
        events.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.notes().is_empty());
        assert!(smf.active_notes_at(0).is_empty());
        assert!(smf.active_notes_at(50).is_empty());
    }

    // ---- parameter_data_entries (RPN / NRPN Data Entry pump) ----

    #[test]
    fn parameter_data_entries_empty_when_no_pump_controllers() {
        // A bare Volume CC carries no Data Entry pump action.
        let events: Vec<u8> = vec![0x00, 0xB0, 0x07, 0x64, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.parameter_data_entries().is_empty());
    }

    #[test]
    fn parameter_data_entries_data_entry_without_selection_is_dropped() {
        // CC 6 with no prior RPN/NRPN selection addresses the power-up
        // null parameter (packed 0x3FFF) and is suppressed per spec.
        let events: Vec<u8> = vec![0x00, 0xB0, 0x06, 0x0C, 0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.parameter_data_entries().is_empty());
    }

    #[test]
    fn parameter_data_entries_rpn0_pitch_bend_sensitivity() {
        // Select RPN 0 (CC 101=0 MSB, CC 100=0 LSB), then Data Entry
        // MSB=12 (CC 6) and LSB=50 (CC 38).
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x65, 0x00, // RPN MSB = 0
            0x00, 0xB0, 0x64, 0x00, // RPN LSB = 0
            0x00, 0xB0, 0x06, 0x0C, // Data Entry MSB = 12
            0x00, 0xB0, 0x26, 0x32, // Data Entry LSB = 50
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 2);
        assert_eq!(pd[0].channel(), 0);
        assert_eq!(
            pd[0].parameter(),
            SelectedParameter::Registered {
                number: 0x0000,
                param: RegisteredParameter::PitchBendSensitivity,
            }
        );
        assert!(pd[0].parameter().is_registered());
        assert!(!pd[0].parameter().is_null());
        assert_eq!(pd[0].action(), DataEntryAction::EntryMsb(12));
        assert_eq!(pd[1].action(), DataEntryAction::EntryLsb(50));
    }

    #[test]
    fn parameter_data_entries_nrpn_surfaces_raw_number() {
        // Select NRPN 0x0123 (CC 99=0x02 MSB, CC 98=0x23 LSB), then
        // Data Entry MSB.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x63, 0x02, // NRPN MSB = 2
            0x00, 0xB0, 0x62, 0x23, // NRPN LSB = 0x23
            0x00, 0xB0, 0x06, 0x40, // Data Entry MSB = 64
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 1);
        assert_eq!(
            pd[0].parameter(),
            SelectedParameter::NonRegistered { number: 0x0123 }
        );
        assert!(!pd[0].parameter().is_registered());
        assert_eq!(pd[0].action(), DataEntryAction::EntryMsb(64));
    }

    #[test]
    fn parameter_data_entries_increment_decrement_drop_value_byte() {
        // RPN 2 (Channel Coarse Tuning) then Data Inc (CC 96) and Data
        // Dec (CC 97) — the value byte is "don't care" per RP-018.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x65, 0x00, // RPN MSB = 0
            0x00, 0xB0, 0x64, 0x02, // RPN LSB = 2
            0x00, 0xB0, 0x60, 0x7F, // Data Increment (value ignored)
            0x00, 0xB0, 0x61, 0x00, // Data Decrement (value ignored)
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 2);
        assert_eq!(
            pd[0].parameter().number(),
            0x0002,
            "RPN 2 = Channel Coarse Tuning"
        );
        assert!(matches!(
            pd[0].parameter(),
            SelectedParameter::Registered {
                param: RegisteredParameter::ChannelCoarseTuning,
                ..
            }
        ));
        assert_eq!(pd[0].action(), DataEntryAction::Increment);
        assert_eq!(pd[1].action(), DataEntryAction::Decrement);
    }

    #[test]
    fn parameter_data_entries_null_disables_pump_until_reselect() {
        // Select RPN 0, do a Data Entry, then select the Null Function
        // Number (7FH:7FH, packed 0x3FFF) which must suppress the
        // following Data Entry.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x65, 0x00, // RPN MSB = 0
            0x00, 0xB0, 0x64, 0x00, // RPN LSB = 0  (RPN 0)
            0x00, 0xB0, 0x06, 0x02, // Data Entry MSB = 2  (emitted)
            0x00, 0xB0, 0x65, 0x7F, // RPN MSB = 127
            0x00, 0xB0, 0x64, 0x7F, // RPN LSB = 127 (Null)
            0x00, 0xB0, 0x06, 0x0A, // Data Entry MSB = 10 (suppressed)
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 1);
        assert_eq!(pd[0].action(), DataEntryAction::EntryMsb(2));
    }

    #[test]
    fn parameter_data_entries_rpn_supersedes_prior_nrpn() {
        // Select NRPN, then a full RPN selection — the RPN wins; a
        // Data Entry afterwards addresses the RPN, not the NRPN.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x63, 0x01, // NRPN MSB = 1
            0x00, 0xB0, 0x62, 0x02, // NRPN LSB = 2 (NRPN 0x0102)
            0x00, 0xB0, 0x65, 0x00, // RPN MSB = 0
            0x00, 0xB0, 0x64, 0x01, // RPN LSB = 1 (RPN 1)
            0x00, 0xB0, 0x06, 0x40, // Data Entry MSB
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 1);
        assert_eq!(
            pd[0].parameter(),
            SelectedParameter::Registered {
                number: 0x0001,
                param: RegisteredParameter::ChannelFineTuning,
            }
        );
    }

    #[test]
    fn parameter_data_entries_lone_lsb_rewrites_only_low_byte() {
        // Full RPN selection then a lone CC 100 (LSB) rewriting only the
        // low byte: 0x0006 (MSB 0, LSB 6) → MPE Configuration.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x65, 0x00, // RPN MSB = 0
            0x00, 0xB0, 0x64, 0x00, // RPN LSB = 0 (RPN 0)
            0x00, 0xB0, 0x64, 0x06, // RPN LSB = 6 only (now RPN 6)
            0x00, 0xB0, 0x06, 0x01, // Data Entry MSB
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 1);
        assert_eq!(pd[0].parameter().number(), 0x0006);
        assert!(matches!(
            pd[0].parameter(),
            SelectedParameter::Registered {
                param: RegisteredParameter::MpeConfiguration,
                ..
            }
        ));
    }

    #[test]
    fn parameter_data_entries_3d_sound_controller_classification() {
        // RPN 0x3D02 = GAIN (one of the RP-049 3D Sound Controllers).
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x65, 0x3D, // RPN MSB = 0x3D
            0x00, 0xB0, 0x64, 0x02, // RPN LSB = 0x02
            0x00, 0xB0, 0x06, 0x50, // Data Entry MSB
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 1);
        // MSB 0x3D, LSB 0x02 packs to (0x3D << 7) | 0x02 = 0x1E82.
        assert_eq!(
            pd[0].parameter(),
            SelectedParameter::Registered {
                number: 0x1E82,
                param: RegisteredParameter::ThreeDimensionalSound(2),
            }
        );
    }

    #[test]
    fn parameter_data_entries_reserved_rpn_preserves_raw_number() {
        // RPN 0x0010 is not in Table 3a → Reserved with the raw number.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x65, 0x00, // RPN MSB = 0
            0x00, 0xB0, 0x64, 0x10, // RPN LSB = 0x10 (RPN 0x0010)
            0x00, 0xB0, 0x06, 0x01, // Data Entry MSB
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 1);
        assert_eq!(
            pd[0].parameter(),
            SelectedParameter::Registered {
                number: 0x0010,
                param: RegisteredParameter::Reserved(0x0010),
            }
        );
    }

    #[test]
    fn parameter_data_entries_per_channel_selectors_independent() {
        // Channel 0 selects RPN 0, channel 1 selects RPN 2; each
        // channel's Data Entry resolves against its own selector.
        let events: Vec<u8> = vec![
            0x00, 0xB0, 0x65, 0x00, 0x00, 0xB0, 0x64, 0x00, // ch0 RPN 0
            0x00, 0xB1, 0x65, 0x00, 0x00, 0xB1, 0x64, 0x02, // ch1 RPN 2
            0x00, 0xB0, 0x06, 0x05, // ch0 Data Entry MSB
            0x00, 0xB1, 0x06, 0x07, // ch1 Data Entry MSB
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 2);
        assert_eq!(pd[0].channel(), 0);
        assert_eq!(pd[0].parameter().number(), 0x0000);
        assert_eq!(pd[1].channel(), 1);
        assert_eq!(pd[1].parameter().number(), 0x0002);
    }

    #[test]
    fn parameter_data_entries_selector_on_other_track_governs_entry() {
        // Track 0 selects RPN 0 at tick 0; track 1's Data Entry at a
        // later tick resolves against it through the global merge.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xB0, 0x65, 0x00]); // RPN MSB
        t0.extend_from_slice(&[0x00, 0xB0, 0x64, 0x00]); // RPN LSB (RPN 0)
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(10));
        t1.extend_from_slice(&[0xB0, 0x06, 0x03]); // ch0 Data Entry MSB @tick10
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let pd = smf.parameter_data_entries();
        assert_eq!(pd.len(), 1);
        assert_eq!(pd[0].tick, 10);
        assert_eq!(pd[0].track, 1);
        assert_eq!(pd[0].parameter().number(), 0x0000);
        assert_eq!(pd[0].action(), DataEntryAction::EntryMsb(3));
    }

    // ───────────── running-status writer ─────────────

    /// Build a single-track format-0 file from a list of channel/meta
    /// events (each at delta 0 except where noted), terminated with EOT.
    fn one_track_file(events: Vec<TrackEvent>) -> SmfFile {
        let mut events = events;
        events.push(TrackEvent {
            delta: 0,
            kind: Event::Meta(MetaEvent::EndOfTrack),
        });
        SmfFile {
            header: SmfHeader {
                format: SmfFormat::SingleTrack,
                ntrks: 1,
                division: Division::TicksPerQuarter(96),
            },
            tracks: vec![Track { events }],
        }
    }

    fn ch(channel: u8, body: ChannelBody, delta: u32) -> TrackEvent {
        TrackEvent {
            delta,
            kind: Event::Channel(ChannelMessage { channel, body }),
        }
    }

    #[test]
    fn running_status_compresses_consecutive_same_status_note_ons() {
        // Three Note On (same channel) → status emitted once.
        let smf = one_track_file(vec![
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 60,
                    velocity: 39,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 64,
                    velocity: 43,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 67,
                    velocity: 37,
                },
                0,
            ),
        ]);
        let explicit = smf.to_bytes().unwrap();
        let running = smf.to_bytes_running_status().unwrap();
        // Each omitted status saves exactly one byte; two are omitted.
        assert_eq!(running.len(), explicit.len() - 2);
        // The compressed body must contain exactly one 0x90 status byte
        // inside the MTrk payload (skip MThd=14 + MTrk tag/len=8).
        let body = &running[22..];
        assert_eq!(body.iter().filter(|&&b| b == 0x90).count(), 1);
        // And it must still round-trip.
        assert_eq!(parse(&running).unwrap(), smf);
    }

    #[test]
    fn running_status_does_not_cross_channel_change() {
        // Note On ch0 then Note On ch1 — different status, no omission.
        let smf = one_track_file(vec![
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 60,
                    velocity: 64,
                },
                0,
            ),
            ch(
                1,
                ChannelBody::NoteOn {
                    key: 62,
                    velocity: 64,
                },
                0,
            ),
        ]);
        let explicit = smf.to_bytes().unwrap();
        let running = smf.to_bytes_running_status().unwrap();
        assert_eq!(running, explicit);
        assert_eq!(parse(&running).unwrap(), smf);
    }

    #[test]
    fn running_status_cleared_by_meta_event() {
        // Note On, then a meta text, then Note On (same status) — the
        // meta resets the buffer so the second status must be re-emitted.
        let smf = one_track_file(vec![
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 60,
                    velocity: 64,
                },
                0,
            ),
            TrackEvent {
                delta: 0,
                kind: Event::Meta(MetaEvent::Text {
                    kind: 0x01,
                    text: b"x".to_vec(),
                }),
            },
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 62,
                    velocity: 64,
                },
                0,
            ),
        ]);
        let running = smf.to_bytes_running_status().unwrap();
        let body = &running[22..];
        // Two 0x90 status bytes survive (the meta interposed).
        assert_eq!(body.iter().filter(|&&b| b == 0x90).count(), 2);
        assert_eq!(parse(&running).unwrap(), smf);
    }

    #[test]
    fn running_status_cleared_by_sysex() {
        let smf = one_track_file(vec![
            ch(
                0,
                ChannelBody::ControlChange {
                    controller: 7,
                    value: 100,
                },
                0,
            ),
            TrackEvent {
                delta: 0,
                kind: Event::Sysex {
                    escape: false,
                    data: vec![0x7E, 0x7F, 0x09, 0x01, 0xF7],
                },
            },
            ch(
                0,
                ChannelBody::ControlChange {
                    controller: 7,
                    value: 90,
                },
                0,
            ),
        ]);
        let running = smf.to_bytes_running_status().unwrap();
        let body = &running[22..];
        assert_eq!(body.iter().filter(|&&b| b == 0xB0).count(), 2);
        assert_eq!(parse(&running).unwrap(), smf);
    }

    #[test]
    fn running_status_single_data_byte_messages_compress() {
        // Two consecutive Program Change on the same channel.
        let smf = one_track_file(vec![
            ch(0, ChannelBody::ProgramChange { program: 0 }, 0),
            ch(0, ChannelBody::ProgramChange { program: 1 }, 5),
        ]);
        let explicit = smf.to_bytes().unwrap();
        let running = smf.to_bytes_running_status().unwrap();
        assert_eq!(running.len(), explicit.len() - 1);
        assert_eq!(parse(&running).unwrap(), smf);
    }

    #[test]
    fn running_status_writer_matches_parser_compressed_input() {
        // A hand-built compressed stream (status once, then bare data
        // pairs) must parse to the same SmfFile our running-status
        // writer would emit.
        let events = [
            0x00, 0x90, 0x3C, 0x27, // Note On C
            0x00, 0x40, 0x2B, // running: Note On E
            0x00, 0x43, 0x25, // running: Note On G
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let parsed = parse(&blob).unwrap();
        let reemitted = parsed.to_bytes_running_status().unwrap();
        assert_eq!(reemitted, blob);
    }

    #[test]
    fn running_status_track_chunk_helper_matches_file_writer() {
        let smf = one_track_file(vec![
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 60,
                    velocity: 64,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::NoteOn {
                    key: 64,
                    velocity: 64,
                },
                0,
            ),
        ]);
        let chunk = smf.tracks[0].to_bytes_chunk_running_status().unwrap();
        let whole = smf.to_bytes_running_status().unwrap();
        // The file's MTrk chunk (after the 14-byte MThd) equals the
        // standalone chunk helper's output.
        assert_eq!(&whole[14..], &chunk[..]);
    }

    // ───────────── channel-mode classifier ─────────────

    #[test]
    fn channel_mode_messages_classify_full_120_127_family() {
        // CC 120..127 on channel 0, each at delta 0; CC 7 (continuous)
        // interposed to confirm it is filtered out.
        let events = [
            0x00, 0xB0, 120, 0x00, // All Sound Off
            0x00, 0xB0, 121, 0x00, // Reset All Controllers
            0x00, 0xB0, 122, 0x7F, // Local Control On (127)
            0x00, 0xB0, 122, 0x00, // Local Control Off (0)
            0x00, 0xB0, 123, 0x00, // All Notes Off
            0x00, 0xB0, 124, 0x00, // Omni Off
            0x00, 0xB0, 125, 0x00, // Omni On
            0x00, 0xB0, 126, 0x04, // Mono On, 4 channels
            0x00, 0xB0, 127, 0x00, // Poly On
            0x00, 0xB0, 0x07, 0x64, // CC 7 Volume — continuous, filtered
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let modes = smf.channel_mode_messages();
        assert_eq!(modes.len(), 9);
        assert_eq!(modes[0].message(), ChannelModeMessage::AllSoundOff);
        assert_eq!(modes[1].message(), ChannelModeMessage::ResetAllControllers);
        assert_eq!(
            modes[2].message(),
            ChannelModeMessage::LocalControl { on: true }
        );
        assert_eq!(
            modes[3].message(),
            ChannelModeMessage::LocalControl { on: false }
        );
        assert_eq!(modes[4].message(), ChannelModeMessage::AllNotesOff);
        assert_eq!(modes[5].message(), ChannelModeMessage::OmniOff);
        assert_eq!(modes[6].message(), ChannelModeMessage::OmniOn);
        assert_eq!(
            modes[7].message(),
            ChannelModeMessage::MonoOn { channels: 4 }
        );
        assert_eq!(modes[8].message(), ChannelModeMessage::PolyOn);
    }

    #[test]
    fn channel_mode_all_notes_off_predicate_covers_123_through_127() {
        assert!(!ChannelModeMessage::AllSoundOff.is_all_notes_off());
        assert!(!ChannelModeMessage::ResetAllControllers.is_all_notes_off());
        assert!(!ChannelModeMessage::LocalControl { on: true }.is_all_notes_off());
        assert!(ChannelModeMessage::AllNotesOff.is_all_notes_off());
        assert!(ChannelModeMessage::OmniOff.is_all_notes_off());
        assert!(ChannelModeMessage::OmniOn.is_all_notes_off());
        assert!(ChannelModeMessage::MonoOn { channels: 0 }.is_all_notes_off());
        assert!(ChannelModeMessage::PolyOn.is_all_notes_off());
    }

    #[test]
    fn channel_mode_local_control_switch_threshold_at_64() {
        // Spec switch rule: 0..=63 = off, 64..=127 = on.
        let cc63 = ControlChangeEvent {
            tick: 0,
            track: 0,
            channel: 0,
            controller: 122,
            value: 63,
        };
        let cc64 = ControlChangeEvent {
            tick: 0,
            track: 0,
            channel: 0,
            controller: 122,
            value: 64,
        };
        assert_eq!(
            cc63.channel_mode(),
            Some(ChannelModeMessage::LocalControl { on: false })
        );
        assert_eq!(
            cc64.channel_mode(),
            Some(ChannelModeMessage::LocalControl { on: true })
        );
    }

    #[test]
    fn channel_mode_returns_none_for_continuous_controller() {
        let cc = ControlChangeEvent {
            tick: 0,
            track: 0,
            channel: 0,
            controller: 7,
            value: 100,
        };
        assert_eq!(cc.channel_mode(), None);
    }

    #[test]
    fn channel_mode_messages_merge_across_tracks_by_tick() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(10));
        t0.extend_from_slice(&[0xB0, 123, 0x00]); // All Notes Off @10 track 0
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(5));
        t1.extend_from_slice(&[0xB1, 120, 0x00]); // All Sound Off @5 track 1
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let modes = smf.channel_mode_messages();
        assert_eq!(modes.len(), 2);
        assert_eq!(modes[0].tick, 5);
        assert_eq!(modes[0].channel(), 1);
        assert_eq!(modes[0].message(), ChannelModeMessage::AllSoundOff);
        assert_eq!(modes[1].tick, 10);
        assert_eq!(modes[1].channel(), 0);
        assert_eq!(modes[1].message(), ChannelModeMessage::AllNotesOff);
    }

    // ───────────── multi-packet sysex reassembly ─────────────

    #[test]
    fn reassemble_single_packet_complete_message() {
        // F0 7E 7F 09 01 F7 — GM System On, self-contained.
        let events = [
            0x00, 0xF0, 0x05, 0x7E, 0x7F, 0x09, 0x01, 0xF7, // F0, len 5
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let msgs = smf.reassembled_sysex_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].tick, 0);
        assert_eq!(msgs[0].track, 0);
        assert_eq!(msgs[0].body, vec![0x7E, 0x7F, 0x09, 0x01]);
        assert!(msgs[0].complete);
        assert_eq!(msgs[0].packet_count, 1);
        assert_eq!(msgs[0].id_byte(), Some(0x7E));
        assert!(msgs[0].is_universal());
    }

    #[test]
    fn reassemble_two_packet_split_message() {
        // F0 opener (no EOX) at tick 0, F7 continuation (with EOX) at +10.
        let events = [
            0x00, 0xF0, 0x03, 0x43, 0x12, 0x00, // F0 manufacturer 0x43, no EOX
            0x0A, 0xF7, 0x03, 0x01, 0x02, 0xF7, // F7 continuation ends EOX
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let msgs = smf.reassembled_sysex_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].tick, 0); // opener tick
                                     // body = opener payload (0x43 0x12 0x00) + continuation (0x01 0x02),
                                     // trailing F7 stripped.
        assert_eq!(msgs[0].body, vec![0x43, 0x12, 0x00, 0x01, 0x02]);
        assert!(msgs[0].complete);
        assert_eq!(msgs[0].packet_count, 2);
        assert_eq!(msgs[0].id_byte(), Some(0x43));
        assert!(!msgs[0].is_universal());
    }

    #[test]
    fn reassemble_three_packet_split_message() {
        let events = [
            0x00, 0xF0, 0x02, 0x41, 0x10, // F0 Roland, no EOX
            0x00, 0xF7, 0x02, 0x20, 0x21, // F7 middle, no EOX
            0x00, 0xF7, 0x02, 0x30, 0xF7, // F7 final, EOX
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let msgs = smf.reassembled_sysex_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, vec![0x41, 0x10, 0x20, 0x21, 0x30]);
        assert!(msgs[0].complete);
        assert_eq!(msgs[0].packet_count, 3);
    }

    #[test]
    fn reassemble_unterminated_chain_at_track_end_is_incomplete() {
        // F0 opener, F7 continuation without EOX, then track ends.
        let events = [
            0x00, 0xF0, 0x02, 0x43, 0x12, // F0 no EOX
            0x00, 0xF7, 0x02, 0x01, 0x02, // F7 no EOX
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let msgs = smf.reassembled_sysex_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, vec![0x43, 0x12, 0x01, 0x02]);
        assert!(!msgs[0].complete);
        assert_eq!(msgs[0].packet_count, 2);
    }

    #[test]
    fn reassemble_new_opener_abandons_prior_open_chain() {
        // F0 opener (no EOX), then a fresh F0 opener (complete) — the
        // first is flushed incomplete, the second is its own message.
        let events = [
            0x00, 0xF0, 0x02, 0x43, 0x12, // F0 no EOX (abandoned)
            0x00, 0xF0, 0x03, 0x7E, 0x7F, 0xF7, // F0 complete
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let msgs = smf.reassembled_sysex_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].body, vec![0x43, 0x12]);
        assert!(!msgs[0].complete);
        assert_eq!(msgs[1].body, vec![0x7E, 0x7F]);
        assert!(msgs[1].complete);
    }

    #[test]
    fn reassemble_standalone_escape_is_own_message() {
        // A lone F7 escape with no open chain.
        let events = [
            0x00, 0xF7, 0x03, 0xF8, 0xF8, 0xF8, // escaped real-time clocks
            0x00, 0xFF, 0x2F, 0x00,
        ];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        let msgs = smf.reassembled_sysex_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, vec![0xF8, 0xF8, 0xF8]);
        // Escape payload does not end in F7 → not complete.
        assert!(!msgs[0].complete);
        assert_eq!(msgs[0].packet_count, 1);
    }

    #[test]
    fn reassemble_merges_across_tracks_by_opener_tick() {
        // Track 0 opens a split at tick 20; track 1 has a complete
        // message at tick 5. Output is sorted by opener tick.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&encode_vlq(20));
        t0.extend_from_slice(&[0xF0, 0x02, 0x43, 0x01]); // opener, no EOX
        t0.extend_from_slice(&[0x00, 0xF7, 0x02, 0x02, 0xF7]); // EOX
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(5));
        t1.extend_from_slice(&[0xF0, 0x03, 0x7E, 0x7F, 0xF7]); // complete
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let msgs = smf.reassembled_sysex_messages();
        assert_eq!(msgs.len(), 2);
        // tick 5 (track 1) sorts before tick 20 (track 0).
        assert_eq!(msgs[0].tick, 5);
        assert_eq!(msgs[0].track, 1);
        assert_eq!(msgs[1].tick, 20);
        assert_eq!(msgs[1].track, 0);
        assert_eq!(msgs[1].body, vec![0x43, 0x01, 0x02]);
    }

    #[test]
    fn reassemble_empty_when_no_sysex() {
        let events = [0x00, 0xFF, 0x2F, 0x00];
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&events));
        let smf = parse(&blob).unwrap();
        assert!(smf.reassembled_sysex_messages().is_empty());
    }

    // ───────────── builder API ─────────────

    #[test]
    fn track_builder_computes_deltas_from_absolute_ticks() {
        let mut tb = TrackBuilder::new();
        tb.channel(
            0,
            0,
            ChannelBody::NoteOn {
                key: 60,
                velocity: 100,
            },
        );
        tb.channel(
            480,
            0,
            ChannelBody::NoteOff {
                key: 60,
                velocity: 0,
            },
        );
        let track = tb.build();
        // NoteOn@0 (delta 0), NoteOff@480 (delta 480), auto-EOT@480 (delta 0).
        assert_eq!(track.events.len(), 3);
        assert_eq!(track.events[0].delta, 0);
        assert_eq!(track.events[1].delta, 480);
        assert_eq!(track.events[2].delta, 0);
        assert!(matches!(
            track.events[2].kind,
            Event::Meta(MetaEvent::EndOfTrack)
        ));
    }

    #[test]
    fn track_builder_sorts_out_of_order_inserts_stably() {
        let mut tb = TrackBuilder::new();
        // Insert in non-monotonic tick order.
        tb.channel(
            480,
            0,
            ChannelBody::NoteOff {
                key: 60,
                velocity: 0,
            },
        );
        tb.channel(
            0,
            0,
            ChannelBody::NoteOn {
                key: 60,
                velocity: 100,
            },
        );
        // Two events at the SAME tick: insertion order must survive.
        tb.channel(0, 0, ChannelBody::ProgramChange { program: 5 });
        let track = tb.build();
        // After sort: NoteOn@0, ProgramChange@0 (insertion order at tie),
        // NoteOff@480, EOT.
        assert!(matches!(
            track.events[0].kind,
            Event::Channel(ChannelMessage {
                body: ChannelBody::NoteOn { .. },
                ..
            })
        ));
        assert!(matches!(
            track.events[1].kind,
            Event::Channel(ChannelMessage {
                body: ChannelBody::ProgramChange { program: 5 },
                ..
            })
        ));
        assert_eq!(track.events[1].delta, 0);
        assert!(matches!(
            track.events[2].kind,
            Event::Channel(ChannelMessage {
                body: ChannelBody::NoteOff { .. },
                ..
            })
        ));
        assert_eq!(track.events[2].delta, 480);
    }

    #[test]
    fn track_builder_note_helper_emits_on_off_pair() {
        let mut tb = TrackBuilder::new();
        tb.note(96, 192, 3, 64, 90);
        let track = tb.build();
        assert_eq!(track.events.len(), 3); // on, off, eot
        assert!(matches!(
            track.events[0].kind,
            Event::Channel(ChannelMessage {
                channel: 3,
                body: ChannelBody::NoteOn {
                    key: 64,
                    velocity: 90
                },
            })
        ));
        assert_eq!(track.events[0].delta, 96);
        assert!(matches!(
            track.events[1].kind,
            Event::Channel(ChannelMessage {
                channel: 3,
                body: ChannelBody::NoteOff {
                    key: 64,
                    velocity: 0
                },
            })
        ));
        assert_eq!(track.events[1].delta, 192);
    }

    #[test]
    fn track_builder_respects_explicit_trailing_eot() {
        let mut tb = TrackBuilder::new();
        tb.channel(
            0,
            0,
            ChannelBody::NoteOn {
                key: 60,
                velocity: 64,
            },
        );
        tb.meta(240, MetaEvent::EndOfTrack);
        let track = tb.build();
        // No second EOT appended.
        let eot_count = track
            .events
            .iter()
            .filter(|e| matches!(e.kind, Event::Meta(MetaEvent::EndOfTrack)))
            .count();
        assert_eq!(eot_count, 1);
        assert_eq!(track.events.last().unwrap().delta, 240);
    }

    #[test]
    fn track_builder_empty_yields_lone_end_of_track() {
        let track = TrackBuilder::new().build();
        assert_eq!(track.events.len(), 1);
        assert!(matches!(
            track.events[0].kind,
            Event::Meta(MetaEvent::EndOfTrack)
        ));
        assert_eq!(track.events[0].delta, 0);
    }

    #[test]
    fn smf_builder_sets_ntrks_and_round_trips() {
        let mut b = SmfBuilder::new();
        b.format(SmfFormat::MultiTrackSimultaneous)
            .division(Division::TicksPerQuarter(96));
        b.track(|t| {
            t.meta(0, MetaEvent::Tempo(500_000));
        });
        b.track(|t| {
            t.note(0, 96, 0, 60, 100);
        });
        let smf = b.build();
        assert_eq!(smf.header.ntrks, 2);
        assert_eq!(smf.tracks.len(), 2);
        // ntrks consistency means to_bytes accepts it; round-trip.
        let bytes = smf.to_bytes().unwrap();
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed, smf);
    }

    #[test]
    fn smf_builder_default_format1_480tpqn() {
        let mut b = SmfBuilder::new();
        b.track(|t| {
            t.note(0, 240, 0, 72, 64);
        });
        let smf = b.build();
        assert_eq!(smf.header.format, SmfFormat::MultiTrackSimultaneous);
        assert_eq!(smf.header.division, Division::TicksPerQuarter(480));
        assert_eq!(smf.header.ntrks, 1);
    }

    #[test]
    fn smf_builder_format0_single_track_round_trips_through_running_writer() {
        let mut b = SmfBuilder::new();
        b.format(SmfFormat::SingleTrack)
            .division(Division::TicksPerQuarter(96));
        b.track(|t| {
            t.channel(
                0,
                0,
                ChannelBody::NoteOn {
                    key: 60,
                    velocity: 64,
                },
            );
            t.channel(
                0,
                0,
                ChannelBody::NoteOn {
                    key: 64,
                    velocity: 64,
                },
            );
            t.channel(
                0,
                0,
                ChannelBody::NoteOn {
                    key: 67,
                    velocity: 64,
                },
            );
        });
        let smf = b.build();
        let bytes = smf.to_bytes_running_status().unwrap();
        assert_eq!(parse(&bytes).unwrap(), smf);
    }

    #[test]
    fn running_status_all_channel_voice_kinds_round_trip() {
        // Pairs of each two-data-byte kind compress; verify round-trip
        // across every channel-voice shape.
        let smf = one_track_file(vec![
            ch(
                0,
                ChannelBody::NoteOff {
                    key: 60,
                    velocity: 0,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::NoteOff {
                    key: 62,
                    velocity: 0,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::PolyAftertouch {
                    key: 60,
                    pressure: 10,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::PolyAftertouch {
                    key: 62,
                    pressure: 20,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::ControlChange {
                    controller: 1,
                    value: 5,
                },
                0,
            ),
            ch(
                0,
                ChannelBody::ControlChange {
                    controller: 1,
                    value: 6,
                },
                0,
            ),
            ch(0, ChannelBody::ChannelAftertouch { pressure: 33 }, 0),
            ch(0, ChannelBody::ChannelAftertouch { pressure: 44 }, 0),
            ch(0, ChannelBody::PitchBend { value: 0x2000 }, 0),
            ch(0, ChannelBody::PitchBend { value: 0x3000 }, 0),
        ]);
        let running = smf.to_bytes_running_status().unwrap();
        assert_eq!(parse(&running).unwrap(), smf);
    }

    // ───────────────────── effects-depth controllers (CC 91–95) ──────────────

    #[test]
    fn effect_depth_classifies_full_91_95_family() {
        // Map controller → expected typed variant + level.
        let cases = [
            (91u8, EffectDepth::ReverbSend(40)),
            (92, EffectDepth::Tremolo(40)),
            (93, EffectDepth::ChorusSend(40)),
            (94, EffectDepth::Celeste(40)),
            (95, EffectDepth::Phaser(40)),
        ];
        for (controller, expected) in cases {
            let cc = ControlChangeEvent {
                tick: 0,
                track: 0,
                channel: 0,
                controller,
                value: 40,
            };
            let depth = cc.effect_depth().expect("CC 91-95 must classify");
            assert_eq!(depth, expected);
            assert_eq!(depth.level(), 40);
            assert_eq!(depth.controller(), controller);
        }
    }

    #[test]
    fn effect_depth_returns_none_outside_91_95() {
        for controller in [7u8, 10, 90, 96, 11] {
            let cc = ControlChangeEvent {
                tick: 0,
                track: 0,
                channel: 0,
                controller,
                value: 64,
            };
            assert_eq!(
                cc.effect_depth(),
                None,
                "controller {controller} should not classify"
            );
        }
    }

    #[test]
    fn effect_depths_iterator_filters_and_merges() {
        // Track 0: CC 7 (volume, ignored) then CC 91 (reverb send) @20.
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xB0, 0x07, 0x64]); // CC 7 = 100 — filtered out
        t0.extend_from_slice(&encode_vlq(20));
        t0.extend_from_slice(&[0xB0, 93, 0x50]); // CC 93 Chorus Send = 80 @20
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        // Track 1: CC 91 reverb send @5.
        let mut t1: Vec<u8> = Vec::new();
        t1.extend_from_slice(&encode_vlq(5));
        t1.extend_from_slice(&[0xB1, 91, 0x28]); // CC 91 Reverb Send = 40 @5
        t1.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(1, 2, 96);
        blob.extend(track_chunk(&t0));
        blob.extend(track_chunk(&t1));
        let smf = parse(&blob).unwrap();
        let depths = smf.effect_depths();
        assert_eq!(depths.len(), 2, "CC 7 must be filtered out");
        // Stable-merged by tick: the @5 reverb send sorts first.
        assert_eq!(depths[0].tick, 5);
        assert_eq!(depths[0].channel(), 1);
        assert_eq!(depths[0].depth(), EffectDepth::ReverbSend(40));
        assert_eq!(depths[1].tick, 20);
        assert_eq!(depths[1].channel(), 0);
        assert_eq!(depths[1].depth(), EffectDepth::ChorusSend(80));
    }

    #[test]
    fn effect_depths_empty_when_no_effect_controllers() {
        let mut t0: Vec<u8> = Vec::new();
        t0.extend_from_slice(&[0x00, 0xB0, 0x07, 0x64]); // only CC 7
        t0.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        let mut blob = header_chunk(0, 1, 96);
        blob.extend(track_chunk(&t0));
        let smf = parse(&blob).unwrap();
        assert!(smf.effect_depths().is_empty());
    }
}
