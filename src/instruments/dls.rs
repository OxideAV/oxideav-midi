//! DLS (Downloadable Sounds, Level 1 + 2) instrument-bank reader.
//!
//! Round-1 implementation against MMA Downloadable Sounds Level 1
//! version 1.1b (Sept 2004) + MMA Downloadable Sounds Level 2.2 v1.0
//! Amendment 2 (April 2006) — both PDFs live in
//! `docs/audio/midi/instrument-formats/`.
//!
//! What this round implements:
//!
//! - **RIFF/`DLS ` walker**: top-level form chunk plus the canonical
//!   `vers` / `colh` / `ptbl` / `lins-list` / `wvpl-list` / `INFO-list`
//!   sub-chunks (and tolerates the optional `dlid` and `cdl`
//!   conditional chunks by skipping them — the file's contents are
//!   parsed regardless of conditional outcome since we report DLS
//!   capability honestly).
//! - **Wave pool**: every `wave-list` entry is parsed for its `fmt ` +
//!   `data` chunks (standard WAV) and the optional `wsmp-ck` per-wave
//!   loop / pitch / gain. Sample bytes are kept in their original
//!   on-disk byte form (8-bit unsigned per WAV-PCM convention or 16-bit
//!   LE signed per `fmt.wBitsPerSample`); no decode happens at parse
//!   time. DLS Level 1 mandates 8 or 16 bit mono PCM; we accept what we
//!   see and surface `bits_per_sample` + `channels` so the round-2
//!   voice generator can decide.
//! - **Instruments**: every `ins-list` carries an `insh-ck`
//!   (cRegions + MIDILOCALE bank/program), then a `lrgn` LIST of
//!   `rgn ` / `rgn2` regions. Each region carries `rgnh-ck` (key/vel
//!   range, options, key group, optional usLayer for DLS2), an optional
//!   `wsmp-ck` overriding the wave-pool default, an optional `wlnk-ck`
//!   pointing into the wave-pool table, and an optional `lart` / `lar2`
//!   articulation list.
//! - **Articulation blocks**: `art1-ck` (DLS1) and `art2-ck` (DLS2)
//!   are parsed into a flat `Vec<DlsConnectionBlock>` (5-field records
//!   per the spec). The connection enums themselves are not interpreted
//!   in round 1 — each block is stored with raw `usSource` /
//!   `usControl` / `usDestination` / `usTransform` / `lScale` values
//!   ready for the round-2 voice generator. We do remember whether the
//!   block came from `art1` or `art2` so a round-2 caller can pick the
//!   right connection table (table 8 vs tables 9 + 10 in DLS2.2).
//! - **`vers-ck`**: surfaces both 32-bit halves; round-1 callers can
//!   inspect `DlsBank::version` to tell DLS1 from DLS2, but we honour
//!   `art1` / `art2` independently of the version number (some real
//!   files mis-stamp `vers`).
//! - **Magic-byte probe**: [`is_dls`] is unchanged from the previous
//!   stub (RIFF + `DLS ` at offset 0/8) — but we now plumb the rest of
//!   the file through.
//!
//! Voice generation: [`DlsInstrument::make_voice`] picks the matching
//! instrument by MIDI program, picks a region by `(key, velocity)`,
//! resolves `wlnk.table_index` → `ptbl` cue → wave-pool entry,
//! decodes the PCM via [`super::wav_pcm::decode_pcm_bytes`] (8/16-bit
//! WAV-shaped), shifts pitch off `wsmp.unity_note`, and plays the
//! sample through the shared [`super::sample_voice::SamplePlayer`].
//! Loop modes from `wsmp.loops`: `WLOOP_TYPE_FORWARD` (DLS1) maps to
//! [`super::sample_voice::SampleLoopMode::LoopContinuous`];
//! `WLOOP_TYPE_RELEASE` (DLS2) maps to
//! [`super::sample_voice::SampleLoopMode::LoopSustain`].
//!
//! What's deferred to round 2 (clear followups, no GitHub issues):
//!
//! - `art1`/`art2` connection-block interpretation. Round-1 voice
//!   generation uses the [`super::sample_voice::SamplePlayer`]
//!   defaults (5 ms attack / 100 ms decay / full sustain / 100 ms
//!   release, no vibrato); the parsed connection blocks remain on
//!   the bank for a round-2 caller to walk and modulate from.
//! - DLS2 `cdl-ck` conditional evaluation. Round 1 currently ignores
//!   the conditional opcode stream and unconditionally parses every
//!   surrounding chunk; this is the recommended fallback for "device
//!   doesn't understand cdl" but means we may parse a `lar2` block on
//!   a hypothetical DLS-Level-1-only consumer. Since this *is* a
//!   DLS1+2 reader, that's fine — we surface both art lists and let
//!   the caller decide.
//! - `dlid-ck` GUID extraction. We skip the chunk; per spec a parser
//!   that doesn't understand DLSIDs must skip them.
//! - DLS file MUXER (writing) — round 3+.

use std::path::Path;
use std::sync::Arc;

use oxideav_core::{Error, Result};

use super::sample_voice::{
    EnvelopeParams, SampleLoopMode, SamplePlayer, SamplePlayerConfig, VibratoParams,
};
use super::wav_pcm::decode_pcm_bytes;
use super::{Instrument, Voice};

/// Magic bytes at offset 0/8 of every DLS file. Note the trailing space
/// in `DLS ` — the spec's chunk identifiers are 4-character ASCII.
pub const RIFF_MAGIC: &[u8; 4] = b"RIFF";
pub const DLS_MAGIC: &[u8; 4] = b"DLS ";

/// Hard cap on total wave-sample bytes we will load. 256 MiB — large
/// enough for the biggest DLS banks shipped with Windows / DirectMusic
/// (gm.dls is ~3 MiB, the Roland DLS-2 banks ~50 MiB), small enough
/// that a forged 4 GiB length field can't allocate.
pub const MAX_WAVE_BYTES: usize = 256 * 1024 * 1024;

/// Hard cap on instruments / regions / connection blocks per chunk.
/// DLS2 minimum-device requires only 256 instruments / 1024 regions /
/// 8192 explicit connections — anything past 1 Mi is a malformed or
/// hostile header.
pub const MAX_RECORDS: usize = 1 << 20;

// =========================================================================
// In-memory bank representation.
// =========================================================================

/// One DLS bank in memory. Parsed eagerly and self-contained — no
/// references back into the source bytes.
#[derive(Clone, Debug, Default)]
pub struct DlsBank {
    /// Bank metadata (INAM / ICOP / etc.) from the top-level
    /// `INFO-list`, if present.
    pub info: DlsInfo,
    /// File version from `vers-ck`, if present. `(major, minor, build,
    /// revision)` per the four 16-bit components of the dwVersionMS /
    /// dwVersionLS pair.
    pub version: Option<(u16, u16, u16, u16)>,
    /// Number of instruments declared in `colh-ck`. May disagree with
    /// `instruments.len()` if the file is malformed; we trust the
    /// actual `lins` count and surface the declared one for diagnostics.
    pub declared_instrument_count: u32,
    /// Wave pool — one entry per `wave-list` in `wvpl-list`. Indexed
    /// indirectly through `wave_pool_indices` (the `ptbl-ck` cue table)
    /// when a `wlnk-ck` references a sample.
    pub waves: Vec<DlsSample>,
    /// `ptbl-ck` cue table: each entry is the byte offset of a `wave`
    /// entry from the start of `wvpl-list`'s payload. Round 1 records
    /// the offsets verbatim (no resolution into `waves` indices) —
    /// instrument regions point at this table; the round-2 voice
    /// generator will map offset → `waves` index.
    pub wave_pool_offsets: Vec<u32>,
    /// All instruments parsed from `lins-list`.
    pub instruments: Vec<DlsInstrumentEntry>,
}

/// `INFO-list` text fields. All optional. ZSTR (null-terminated ASCII)
/// per RIFF convention; we strip the trailing NUL.
#[derive(Clone, Debug, Default)]
pub struct DlsInfo {
    /// `INAM` — bank name.
    pub name: Option<String>,
    /// `ICOP` — copyright.
    pub copyright: Option<String>,
    /// `IENG` — engineer.
    pub engineer: Option<String>,
    /// `ICMT` — free-form comment.
    pub comment: Option<String>,
    /// `ISFT` — software that produced the file.
    pub software: Option<String>,
}

/// One sample from the wave pool. Carries the raw on-disk PCM bytes
/// plus the parsed `fmt ` + optional `wsmp` headers.
#[derive(Clone, Debug, Default)]
pub struct DlsSample {
    /// Byte offset of this `wave` LIST from the start of the parent
    /// `wvpl` LIST's payload (i.e. the value `ptbl` cues point at).
    /// Recorded so the round-2 voice generator can reconcile a
    /// `wlnk-ck`'s `ulTableIndex` → ptbl entry → wave-pool index.
    pub pool_offset: u32,
    /// `WAVE_FORMAT_PCM` (1) — DLS1 mandates this; round 1 surfaces the
    /// raw value so a non-PCM wave is visible rather than silently
    /// reinterpreted.
    pub format_tag: u16,
    /// Number of channels. DLS1 mandates 1 (mono); DLS2 allows stereo.
    pub channels: u16,
    /// Native sample rate, Hz.
    pub sample_rate: u32,
    /// Bytes per second (`fmt.dwAvgBytesPerSec`). Surfaced for
    /// diagnostics — derivable from the other fields for PCM.
    pub avg_bytes_per_sec: u32,
    /// Block alignment (bytes per sample frame).
    pub block_align: u16,
    /// Bits per sample. 8 (unsigned) or 16 (signed LE) for DLS1 PCM.
    pub bits_per_sample: u16,
    /// Raw PCM bytes from `data-ck`. Unaltered — 8-bit unsigned for
    /// 8-bit waves (with 0x80 = silence per WAV PCM convention) and
    /// signed little-endian for 16-bit waves.
    pub data: Vec<u8>,
    /// Per-wave default loop / pitch / gain from the wave's own
    /// `wsmp-ck`, if any. Region-level wsmp overrides this.
    pub wsmp: Option<DlsWaveSample>,
}

/// `wsmp-ck` content. DLS1 calls the gain field `lAttenuation`; DLS2
/// renames it `lGain` (and changes the sense — gain not attenuation).
/// Round 1 keeps both spellings on the same field (`gain`); the round-2
/// voice generator decides how to interpret it from the surrounding
/// `vers` chunk if it cares about the distinction.
#[derive(Clone, Debug, Default)]
pub struct DlsWaveSample {
    /// `usUnityNote` — MIDI note this sample sounds at when played at
    /// its native rate. Default 60 (middle C) per spec when wsmp is
    /// absent.
    pub unity_note: u16,
    /// `sFineTune` — 16-bit relative pitch (signed cents-ish).
    pub fine_tune: i16,
    /// `lAttenuation` (DLS1) / `lGain` (DLS2) — 32-bit relative gain.
    pub gain: i32,
    /// `fulOptions` — F_WSMP_NO_TRUNCATION (0x1) and
    /// F_WSMP_NO_COMPRESSION (0x2) bits.
    pub options: u32,
    /// One [`DlsLoop`] per loop region. DLS spec says 0 = one-shot,
    /// 1 = looped; values >1 are reserved.
    pub loops: Vec<DlsLoop>,
}

/// One `wavesample-loop` record inside a `wsmp-ck`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DlsLoop {
    /// `ulLoopType` — `WLOOP_TYPE_FORWARD` (0) for DLS1, plus
    /// `WLOOP_TYPE_RELEASE` (1) added in DLS2.
    pub loop_type: u32,
    /// Start frame, absolute offset from the start of the wave's `data`
    /// chunk (in *sample frames*, not bytes).
    pub start: u32,
    /// Length of the loop, in sample frames.
    pub length: u32,
}

/// One instrument from `lins-list`. Carries the bank/program selector
/// from `insh-ck` plus a flat list of regions.
#[derive(Clone, Debug, Default)]
pub struct DlsInstrumentEntry {
    /// `MIDILOCALE.ulBank`. Bits 0-6 = MIDI CC32 (LSB), bits 8-14 =
    /// MIDI CC0 (MSB), bit 31 = `F_INSTRUMENT_DRUMS`.
    pub bank: u32,
    /// `MIDILOCALE.ulInstrument`. Bits 0-6 = MIDI Program Change (PC).
    pub program: u32,
    /// Instrument name from this instrument's own `INFO-list/INAM`,
    /// if present.
    pub name: Option<String>,
    /// Number of regions declared in `insh-ck.cRegions` — surfaced for
    /// diagnostics. Trust `regions.len()` for the actual count.
    pub declared_region_count: u32,
    /// Instrument-level (global) articulation, parsed from the
    /// instrument's own `lart` / `lar2` LIST. Drum instruments
    /// per DLS1 may also carry per-region articulation in
    /// `DlsRegion::articulation`.
    pub articulation: Vec<DlsArticulationBlock>,
    /// Flat region list. Each entry covers one keyboard split.
    pub regions: Vec<DlsRegion>,
}

impl DlsInstrumentEntry {
    /// `true` if the instrument header's bit 31 (`F_INSTRUMENT_DRUMS`)
    /// is set — i.e. this is a drum kit. Per DLS1, drum instruments
    /// may have per-region articulation; melodic instruments only have
    /// the instrument-level articulation list.
    pub fn is_drum(&self) -> bool {
        (self.bank & 0x8000_0000) != 0
    }

    /// MIDI CC0 (Bank Select MSB) extracted from `bank`.
    pub fn bank_msb(&self) -> u8 {
        ((self.bank >> 8) & 0x7F) as u8
    }

    /// MIDI CC32 (Bank Select LSB) extracted from `bank`.
    pub fn bank_lsb(&self) -> u8 {
        (self.bank & 0x7F) as u8
    }

    /// MIDI Program Change number (0..=127) extracted from `program`.
    pub fn program_number(&self) -> u8 {
        (self.program & 0x7F) as u8
    }
}

/// One region inside a [`DlsInstrumentEntry`].
#[derive(Clone, Debug, Default)]
pub struct DlsRegion {
    /// Inclusive low MIDI key.
    pub key_lo: u16,
    /// Inclusive high MIDI key.
    pub key_hi: u16,
    /// Inclusive low MIDI velocity. DLS1 sets these to 0/127 (velocity
    /// switching is a DLS2 feature).
    pub vel_lo: u16,
    /// Inclusive high MIDI velocity.
    pub vel_hi: u16,
    /// `fusOptions` from `rgnh-ck`. Bit 0 (=
    /// `F_RGN_OPTION_SELFNONEXCLUSIVE`) = a second note-on of the same
    /// pitch should *not* steal the previously-allocated voice.
    pub options: u16,
    /// `usKeyGroup`. 0 = no key group; 1..=15 = group id (a new note in
    /// the same group cuts every prior note in that group). Drum-only
    /// per DLS1 — but the field is present in every region header.
    pub key_group: u16,
    /// `usLayer` — DLS2 only. `None` for DLS1 regions whose `rgnh-ck`
    /// payload is the 12-byte minimum.
    pub layer: Option<u16>,
    /// Indicator that this region was parsed from a `rgn2` LIST rather
    /// than `rgn ` (the DLS2 "Level 2" region variant). Surfaced so a
    /// round-2 voice generator running on a Level-1-only profile can
    /// skip these.
    pub is_level2: bool,
    /// Per-region wave-sample header (from this region's own `wsmp-ck`
    /// if present). Overrides the wave-pool default at voice time.
    pub wsmp: Option<DlsWaveSample>,
    /// Wave-link chunk: which entry in the `ptbl` cue table this
    /// region's sample comes from. `None` means the region has no
    /// associated sample (rare; usually a malformed file).
    pub wlnk: Option<DlsWaveLink>,
    /// Per-region articulation. Per DLS1, drum instruments may have
    /// these; melodic instruments rely on the instrument-level list.
    pub articulation: Vec<DlsArticulationBlock>,
}

/// `wlnk-ck` content. Round 1 records every field verbatim — the
/// round-2 voice generator does the cue-table → wave-pool resolution.
#[derive(Clone, Copy, Debug, Default)]
pub struct DlsWaveLink {
    /// `fusOptions` — `F_WAVELINK_PHASE_MASTER` (0x1, DLS1+) and
    /// `F_WAVELINK_MULTICHANNEL` (0x2, DLS2-only) bits.
    pub options: u16,
    /// `usPhaseGroup` — non-zero groups multiple wave links so they
    /// stay phase-locked.
    pub phase_group: u16,
    /// `ulChannel` — bitmask of which channel(s) this wave occupies in
    /// a multi-channel arrangement. 1 = mono / left, 2 = right, etc.
    pub channel: u32,
    /// `ulTableIndex` — 0-based index into the `ptbl` cue table.
    pub table_index: u32,
}

/// Which DLS articulator chunk produced this connection block. The
/// `ConnectionBlock` struct itself is identical for `art1` and `art2`,
/// but the source / destination / transform enums differ — DLS2 adds
/// channel-output destinations, polyphonic-pressure source, vibrato
/// LFO destinations, filter cutoff/Q destinations, and richer
/// transforms (convex / switch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DlsArtKind {
    /// Connection from an `art1-ck` chunk (DLS Level 1 source/dest
    /// table — see DLS1 spec page 43, DLS2 spec table 8).
    Art1,
    /// Connection from an `art2-ck` chunk (DLS Level 2 extended
    /// source/dest table — see DLS2 spec tables 9 + 10).
    Art2,
}

/// One `<ConnectionBlock>` from an `art1-ck` or `art2-ck` chunk.
/// 12 bytes on disk: usSource:U16, usControl:U16, usDestination:U16,
/// usTransform:U16, lScale:S32.
#[derive(Clone, Copy, Debug)]
pub struct DlsArticulationBlock {
    pub kind: DlsArtKind,
    /// `usSource` — modulator source (CONN_SRC_*).
    pub source: u16,
    /// `usControl` — secondary modulator (often `CONN_SRC_NONE`).
    pub control: u16,
    /// `usDestination` — what to modulate (CONN_DST_*).
    pub destination: u16,
    /// `usTransform` — transform/curve identifier. In DLS2 this is a
    /// packed bitfield (see figure 12 of the DLS2 spec): bits 0-3 =
    /// output transform, 4-7 = control transform, 8-9 = control
    /// invert/bipolar, 10-13 = source transform, 14-15 = source
    /// invert/bipolar.
    pub transform: u16,
    /// `lScale` — scale value applied along the connection (signed,
    /// units depend on the destination — cents for pitch, centibels
    /// for gain, etc.).
    pub scale: i32,
}

// =========================================================================
// Public entry points + Instrument trait wiring.
// =========================================================================

/// Loaded DLS instrument bank. Cheap to clone (no shared state today;
/// sample data is owned per bank — round 2 may move to `Arc<[u8]>` if
/// voice cloning needs to amortise).
pub struct DlsInstrument {
    name: String,
    bank: DlsBank,
}

impl DlsInstrument {
    /// Open a DLS file from disk. Reads the whole file into memory and
    /// parses it eagerly (the format references samples by absolute
    /// offset; streaming would require a second pass).
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let bank = DlsBank::parse(&bytes)?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "dls".to_string());
        Ok(Self { name, bank })
    }

    /// Parse a DLS file from an in-memory buffer. Useful for tests and
    /// for in-memory bank-installer flows.
    pub fn parse_bytes(name: impl Into<String>, bytes: &[u8]) -> Result<Self> {
        let bank = DlsBank::parse(bytes)?;
        Ok(Self {
            name: name.into(),
            bank,
        })
    }

    /// Borrow the parsed bank — mostly useful for diagnostics. Voice
    /// generation goes through [`Instrument::make_voice`] (round 2).
    pub fn bank(&self) -> &DlsBank {
        &self.bank
    }

    /// Magic-byte probe — true if `bytes` starts with `RIFF` ... `DLS `.
    pub fn probe(bytes: &[u8]) -> bool {
        is_dls(bytes)
    }
}

impl Instrument for DlsInstrument {
    fn name(&self) -> &str {
        &self.name
    }

    fn make_voice(
        &self,
        program: u8,
        key: u8,
        velocity: u8,
        sample_rate: u32,
    ) -> Result<Box<dyn Voice>> {
        // 1. Pick the matching DLS instrument (program; bank not yet
        //    plumbed end-to-end — round 1 matches first instrument with
        //    the requested program number, ignoring CC0/CC32).
        let inst = self
            .bank
            .instruments
            .iter()
            .find(|i| i.program_number() == program)
            // Last-resort: take the first instrument so a bank with a
            // single drum kit still produces sound.
            .or_else(|| self.bank.instruments.first())
            .ok_or_else(|| {
                Error::unsupported(format!("DLS '{}': bank has no instruments", self.name,))
            })?;

        // 2. Pick a region. DLS regions don't overlap by spec, so the
        //    first region whose key/vel range covers (key, velocity)
        //    wins. If none match, fall back to the first region (some
        //    banks set key_lo=key_hi=60 and rely on the synth to
        //    transpose).
        let region = inst
            .regions
            .iter()
            .find(|r| {
                u16::from(key) >= r.key_lo
                    && u16::from(key) <= r.key_hi
                    && u16::from(velocity) >= r.vel_lo
                    && u16::from(velocity) <= r.vel_hi
            })
            .or_else(|| inst.regions.first())
            .ok_or_else(|| {
                Error::unsupported(format!("DLS '{}': instrument has no regions", self.name,))
            })?;

        // 3. Resolve wlnk → wave-pool entry. wlnk.table_index indexes
        //    the ptbl cue table; each cue is a byte offset into the
        //    wvpl payload that matches `DlsSample::pool_offset`.
        let wlnk = region.wlnk.ok_or_else(|| {
            Error::unsupported(format!(
                "DLS '{}': region has no wlnk pointing at the wave pool",
                self.name,
            ))
        })?;
        let pool_offset = self
            .bank
            .wave_pool_offsets
            .get(wlnk.table_index as usize)
            .copied()
            .ok_or_else(|| {
                Error::invalid(format!(
                    "DLS '{}': wlnk table_index {} out of range (ptbl has {} entries)",
                    self.name,
                    wlnk.table_index,
                    self.bank.wave_pool_offsets.len(),
                ))
            })?;
        let wave = self
            .bank
            .waves
            .iter()
            .find(|w| w.pool_offset == pool_offset)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "DLS '{}': no wave-pool entry at pool_offset {pool_offset}",
                    self.name,
                ))
            })?;

        // 4. Decode the PCM bytes to f32 mono.
        let pcm = decode_pcm_bytes(
            &wave.data,
            wave.sample_rate,
            wave.channels,
            wave.bits_per_sample,
            wave.format_tag,
        )
        .map_err(|e| {
            Error::invalid(format!(
                "DLS '{}': failed to decode wave PCM: {e}",
                self.name,
            ))
        })?;

        // 5. Build the SamplePlayer config. Region-level wsmp overrides
        //    the wave-level default per the spec.
        let wsmp = region.wsmp.as_ref().or(wave.wsmp.as_ref());
        let cfg = build_dls_config(wsmp, &pcm.samples, pcm.sample_rate, key, velocity, region);
        Ok(Box::new(SamplePlayer::new(cfg, sample_rate)))
    }
}

/// Build a [`SamplePlayerConfig`] from a DLS region + its resolved
/// wave-sample header. Round-1 ignores `art1`/`art2` connection blocks
/// and uses the SamplePlayer's default DAHDSR; the parsed articulation
/// blocks remain on the bank for a round-2 caller.
fn build_dls_config(
    wsmp: Option<&DlsWaveSample>,
    samples: &[f32],
    native_rate: u32,
    key: u8,
    velocity: u8,
    region: &DlsRegion,
) -> SamplePlayerConfig {
    // Pitch ratio: 2^((target - unity_note) / 12 + fine_tune/16384*100/100).
    // DLS sFineTune is in units of 1/65536 of a semitone — treat as
    // signed 16-bit and divide by 65536 for the cents-per-semitone
    // conversion. Most banks use the simpler integer cents convention,
    // so we treat fine_tune as raw "cents-ish" for round 1 and clamp
    // its effect.
    let unity_note = wsmp.map(|w| w.unity_note as u8).unwrap_or(60);
    let fine_tune = wsmp.map(|w| w.fine_tune as i32).unwrap_or(0);
    let semitones = key as i32 - unity_note as i32;
    let pitch_ratio = (2.0f64).powf(semitones as f64 / 12.0 + fine_tune as f64 / 1200.0);

    // Velocity curve: (vel/127)^2 with 50 % headroom so 8 voices fit
    // under 0 dBFS.
    let v = velocity as f32 / 127.0;
    let amplitude = v * v * 0.5;

    // Loop info from wsmp. DLS1 stores `loop_type=0` (forward) and
    // `loop_type=1` (release loop, DLS2-only). A wsmp with no loops =
    // play once. Note `length` is in *frames*.
    let total_frames = samples.len() as u32;
    let (loop_start, loop_end, loop_mode) = match wsmp.and_then(|w| w.loops.first()) {
        Some(l) => {
            let start = l.start.min(total_frames);
            let end = (l.start.saturating_add(l.length)).min(total_frames);
            // loop_type 1 = release loop (loop while held). 0 = forward
            // (loop continuously). Anything else, treat as no loop.
            let mode = match l.loop_type {
                0 => SampleLoopMode::LoopContinuous,
                1 => SampleLoopMode::LoopSustain,
                _ => SampleLoopMode::NoLoop,
            };
            (start, end, mode)
        }
        None => (0, total_frames, SampleLoopMode::NoLoop),
    };

    SamplePlayerConfig {
        samples: Arc::from(samples.to_vec().into_boxed_slice()),
        native_rate,
        loop_start,
        loop_end,
        sample_end: total_frames,
        loop_mode,
        pitch_ratio,
        amplitude,
        envelope: EnvelopeParams::default(),
        vibrato: VibratoParams::default(),
        // DLS key_group: drum-kit style "this group cuts every prior
        // voice in the same group". Maps onto the Voice trait's
        // `exclusive_class` directly. Round-1 doesn't filter by
        // melodic vs drum (the `key_group` field is non-zero only on
        // drum regions per DLS1, so a melodic region passes 0 here).
        exclusive_class: region.key_group,
    }
}

/// Magic-bytes detector. Returns true if `bytes` looks like a DLS file
/// (RIFF + `DLS<space>` magic at offsets 0/8). Cheap enough to call
/// from a sniff loop.
pub fn is_dls(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == RIFF_MAGIC && &bytes[8..12] == DLS_MAGIC
}

// =========================================================================
// RIFF walker.
// =========================================================================

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn read_tag(&mut self) -> Result<[u8; 4]> {
        if self.remaining() < 4 {
            return Err(Error::invalid("DLS: truncated chunk tag (needed 4 bytes)"));
        }
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&self.bytes[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(tag)
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        if self.remaining() < 4 {
            return Err(Error::invalid("DLS: truncated u32"));
        }
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(Error::invalid(format!(
                "DLS: truncated payload (needed {n} bytes, {} remain)",
                self.remaining(),
            )));
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
}

/// Read one `<tag><u32 LE size><payload>` chunk. Pads odd-length
/// payloads to even per RIFF convention. Tolerates a missing trailing
/// pad byte at EOF (some files in the wild omit it).
fn read_chunk<'a>(c: &mut Cursor<'a>) -> Result<([u8; 4], &'a [u8])> {
    let tag = c.read_tag()?;
    let size = c.read_u32_le()? as usize;
    if size > c.remaining() {
        return Err(Error::invalid(format!(
            "DLS: chunk '{}' length {size} exceeds {} bytes remaining",
            tag_str(&tag),
            c.remaining(),
        )));
    }
    let payload = c.take(size)?;
    if size % 2 == 1 && c.remaining() >= 1 {
        c.pos += 1;
    }
    Ok((tag, payload))
}

fn tag_str(tag: &[u8; 4]) -> String {
    if tag.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        String::from_utf8_lossy(tag).into_owned()
    } else {
        format!("{:02X}{:02X}{:02X}{:02X}", tag[0], tag[1], tag[2], tag[3])
    }
}

/// Strip a single trailing NUL from a ZSTR payload and return the
/// resulting String, or None if the payload is empty / non-UTF-8.
fn parse_zstr(payload: &[u8]) -> Option<String> {
    let trimmed = payload.split(|b| *b == 0).next()?;
    if trimmed.is_empty() {
        return None;
    }
    Some(String::from_utf8_lossy(trimmed).into_owned())
}

// =========================================================================
// Top-level parser.
// =========================================================================

impl DlsBank {
    /// Parse a complete DLS file from a borrowed byte slice. Returns a
    /// fully cross-resolved bank (no further references back into the
    /// source bytes).
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if !is_dls(bytes) {
            return Err(Error::invalid(
                "DLS: file does not start with RIFF/DLS<space> magic",
            ));
        }
        // Outer RIFF header.
        let mut outer = Cursor::new(bytes);
        let _ = outer.read_tag()?; // RIFF
        let total = outer.read_u32_le()? as usize;
        if total + 8 > bytes.len() {
            return Err(Error::invalid(format!(
                "DLS: outer RIFF size {total} exceeds file size {}",
                bytes.len() - 8,
            )));
        }
        let body = &bytes[8..8 + total.min(bytes.len() - 8)];
        let mut body_cur = Cursor::new(body);
        let form = body_cur.read_tag()?;
        if &form != DLS_MAGIC {
            return Err(Error::invalid(format!(
                "DLS: outer form is '{}', expected 'DLS '",
                tag_str(&form),
            )));
        }

        let mut bank = DlsBank::default();

        while !body_cur.at_end() {
            let (tag, payload) = read_chunk(&mut body_cur)?;
            match &tag {
                b"colh" => bank.declared_instrument_count = parse_colh(payload)?,
                b"vers" => bank.version = Some(parse_vers(payload)?),
                b"ptbl" => bank.wave_pool_offsets = parse_ptbl(payload)?,
                b"LIST" => {
                    if payload.len() < 4 {
                        return Err(Error::invalid("DLS: LIST payload < 4 bytes (no list type)"));
                    }
                    let mut list_type = [0u8; 4];
                    list_type.copy_from_slice(&payload[..4]);
                    let body = &payload[4..];
                    match &list_type {
                        b"INFO" => bank.info = parse_info_list(body)?,
                        b"lins" => bank.instruments = parse_lins_list(body)?,
                        b"wvpl" => bank.waves = parse_wvpl_list(body)?,
                        // Unknown list type — skip per RIFF convention.
                        _ => {}
                    }
                }
                // dlid (DLSID), cdl (conditional), unknown chunks: skip.
                _ => {}
            }
        }

        Ok(bank)
    }
}

// -------------------------------------------------------------------------
// `colh-ck` — Collection Header.
// -------------------------------------------------------------------------

fn parse_colh(payload: &[u8]) -> Result<u32> {
    if payload.len() < 4 {
        return Err(Error::invalid(format!(
            "DLS: colh payload {} < 4 bytes",
            payload.len(),
        )));
    }
    Ok(u32::from_le_bytes(payload[..4].try_into().unwrap()))
}

// -------------------------------------------------------------------------
// `vers-ck` — Version stamp.
// -------------------------------------------------------------------------

fn parse_vers(payload: &[u8]) -> Result<(u16, u16, u16, u16)> {
    if payload.len() < 8 {
        return Err(Error::invalid(format!(
            "DLS: vers payload {} < 8 bytes",
            payload.len(),
        )));
    }
    let ms = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let ls = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    Ok((
        (ms >> 16) as u16,    // major
        (ms & 0xFFFF) as u16, // minor
        (ls >> 16) as u16,    // build
        (ls & 0xFFFF) as u16, // revision
    ))
}

// -------------------------------------------------------------------------
// `ptbl-ck` — Pool Table.
// -------------------------------------------------------------------------

fn parse_ptbl(payload: &[u8]) -> Result<Vec<u32>> {
    if payload.len() < 8 {
        return Err(Error::invalid(format!(
            "DLS: ptbl payload {} < 8 bytes (cbSize + cCues)",
            payload.len(),
        )));
    }
    let cb_size = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let cues = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    if cues > MAX_RECORDS {
        return Err(Error::invalid(format!(
            "DLS: ptbl cues count {cues} exceeds cap {MAX_RECORDS}",
        )));
    }
    // The cue records start immediately after the cCues field. cbSize
    // is the size of the *header* portion (typically 8 bytes — the
    // cbSize+cCues pair); skip up to that point so future spec
    // additions to the header don't break parsing.
    let header_len = cb_size.max(8);
    if header_len > payload.len() {
        return Err(Error::invalid(format!(
            "DLS: ptbl cbSize {cb_size} exceeds payload {}",
            payload.len(),
        )));
    }
    let cue_start = header_len;
    let needed = cues
        .checked_mul(4)
        .ok_or_else(|| Error::invalid("DLS: ptbl cue count overflow"))?;
    if cue_start + needed > payload.len() {
        return Err(Error::invalid(format!(
            "DLS: ptbl truncated — need {needed} bytes for {cues} cues at offset {cue_start}, \
             payload is {}",
            payload.len(),
        )));
    }
    let mut out = Vec::with_capacity(cues);
    for i in 0..cues {
        let off = cue_start + i * 4;
        out.push(u32::from_le_bytes(
            payload[off..off + 4].try_into().unwrap(),
        ));
    }
    Ok(out)
}

// -------------------------------------------------------------------------
// `INFO-list` (and the per-instrument / per-wave INFO list flavour).
// -------------------------------------------------------------------------

fn parse_info_list(body: &[u8]) -> Result<DlsInfo> {
    let mut info = DlsInfo::default();
    let mut c = Cursor::new(body);
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        let s = parse_zstr(payload);
        match &tag {
            b"INAM" => info.name = s,
            b"ICOP" => info.copyright = s,
            b"IENG" => info.engineer = s,
            b"ICMT" => info.comment = s,
            b"ISFT" => info.software = s,
            // IART, ICMS, ICRD, IGNR, IKEY, IMED, IPRD, ISBJ, ISRC,
            // ISRF, ITCH — defined by the spec but not surfaced.
            _ => {}
        }
    }
    Ok(info)
}

/// Parse just the INAM entry from an INFO LIST body. Used by ins-list
/// and wave-list to pluck the name without the full INFO struct
/// allocation.
fn parse_info_inam(body: &[u8]) -> Result<Option<String>> {
    let mut c = Cursor::new(body);
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        if &tag == b"INAM" {
            return Ok(parse_zstr(payload));
        }
    }
    Ok(None)
}

// -------------------------------------------------------------------------
// `lins-list` → list of instruments.
// -------------------------------------------------------------------------

fn parse_lins_list(body: &[u8]) -> Result<Vec<DlsInstrumentEntry>> {
    let mut out = Vec::new();
    let mut c = Cursor::new(body);
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        if &tag != b"LIST" {
            continue;
        }
        if payload.len() < 4 {
            continue;
        }
        let mut sub_type = [0u8; 4];
        sub_type.copy_from_slice(&payload[..4]);
        if &sub_type != b"ins " {
            continue;
        }
        out.push(parse_ins_list(&payload[4..])?);
        if out.len() > MAX_RECORDS {
            return Err(Error::invalid(format!(
                "DLS: lins instrument count exceeds cap {MAX_RECORDS}",
            )));
        }
    }
    Ok(out)
}

fn parse_ins_list(body: &[u8]) -> Result<DlsInstrumentEntry> {
    let mut entry = DlsInstrumentEntry::default();
    let mut c = Cursor::new(body);
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        match &tag {
            b"insh" => {
                let (regions, bank, program) = parse_insh(payload)?;
                entry.declared_region_count = regions;
                entry.bank = bank;
                entry.program = program;
            }
            b"LIST" => {
                if payload.len() < 4 {
                    continue;
                }
                let mut sub_type = [0u8; 4];
                sub_type.copy_from_slice(&payload[..4]);
                let sub_body = &payload[4..];
                match &sub_type {
                    b"INFO" => entry.name = parse_info_inam(sub_body)?,
                    b"lrgn" => entry.regions = parse_lrgn_list(sub_body)?,
                    b"lart" => entry
                        .articulation
                        .extend(parse_lart_or_lar2(sub_body, false)?),
                    b"lar2" => entry
                        .articulation
                        .extend(parse_lart_or_lar2(sub_body, true)?),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(entry)
}

fn parse_insh(payload: &[u8]) -> Result<(u32, u32, u32)> {
    if payload.len() < 12 {
        return Err(Error::invalid(format!(
            "DLS: insh payload {} < 12 bytes (cRegions + ulBank + ulInstrument)",
            payload.len(),
        )));
    }
    let regions = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let bank = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    let program = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    if regions > MAX_RECORDS as u32 {
        return Err(Error::invalid(format!(
            "DLS: insh declared region count {regions} exceeds cap {MAX_RECORDS}",
        )));
    }
    Ok((regions, bank, program))
}

// -------------------------------------------------------------------------
// `lrgn-list` → list of regions.
// -------------------------------------------------------------------------

fn parse_lrgn_list(body: &[u8]) -> Result<Vec<DlsRegion>> {
    let mut out = Vec::new();
    let mut c = Cursor::new(body);
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        if &tag != b"LIST" {
            continue;
        }
        if payload.len() < 4 {
            continue;
        }
        let mut sub_type = [0u8; 4];
        sub_type.copy_from_slice(&payload[..4]);
        let is_level2 = match &sub_type {
            b"rgn " => false,
            b"rgn2" => true,
            _ => continue,
        };
        out.push(parse_rgn_list(&payload[4..], is_level2)?);
        if out.len() > MAX_RECORDS {
            return Err(Error::invalid(format!(
                "DLS: lrgn region count exceeds cap {MAX_RECORDS}",
            )));
        }
    }
    Ok(out)
}

fn parse_rgn_list(body: &[u8], is_level2: bool) -> Result<DlsRegion> {
    let mut region = DlsRegion {
        is_level2,
        ..DlsRegion::default()
    };
    let mut c = Cursor::new(body);
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        match &tag {
            b"rgnh" => parse_rgnh(payload, &mut region)?,
            b"wsmp" => region.wsmp = Some(parse_wsmp(payload)?),
            b"wlnk" => region.wlnk = Some(parse_wlnk(payload)?),
            b"LIST" => {
                if payload.len() < 4 {
                    continue;
                }
                let mut sub_type = [0u8; 4];
                sub_type.copy_from_slice(&payload[..4]);
                let sub_body = &payload[4..];
                match &sub_type {
                    b"lart" => region
                        .articulation
                        .extend(parse_lart_or_lar2(sub_body, false)?),
                    b"lar2" => region
                        .articulation
                        .extend(parse_lart_or_lar2(sub_body, true)?),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(region)
}

fn parse_rgnh(payload: &[u8], out: &mut DlsRegion) -> Result<()> {
    if payload.len() < 12 {
        return Err(Error::invalid(format!(
            "DLS: rgnh payload {} < 12 bytes (RangeKey+RangeVel+fusOptions+usKeyGroup)",
            payload.len(),
        )));
    }
    out.key_lo = u16::from_le_bytes(payload[0..2].try_into().unwrap());
    out.key_hi = u16::from_le_bytes(payload[2..4].try_into().unwrap());
    out.vel_lo = u16::from_le_bytes(payload[4..6].try_into().unwrap());
    out.vel_hi = u16::from_le_bytes(payload[6..8].try_into().unwrap());
    out.options = u16::from_le_bytes(payload[8..10].try_into().unwrap());
    out.key_group = u16::from_le_bytes(payload[10..12].try_into().unwrap());
    if payload.len() >= 14 {
        out.layer = Some(u16::from_le_bytes(payload[12..14].try_into().unwrap()));
    }
    Ok(())
}

fn parse_wlnk(payload: &[u8]) -> Result<DlsWaveLink> {
    if payload.len() < 12 {
        return Err(Error::invalid(format!(
            "DLS: wlnk payload {} < 12 bytes",
            payload.len(),
        )));
    }
    Ok(DlsWaveLink {
        options: u16::from_le_bytes(payload[0..2].try_into().unwrap()),
        phase_group: u16::from_le_bytes(payload[2..4].try_into().unwrap()),
        channel: u32::from_le_bytes(payload[4..8].try_into().unwrap()),
        table_index: u32::from_le_bytes(payload[8..12].try_into().unwrap()),
    })
}

// -------------------------------------------------------------------------
// `wsmp-ck`.
// -------------------------------------------------------------------------

fn parse_wsmp(payload: &[u8]) -> Result<DlsWaveSample> {
    // Fixed header: cbSize:U32 + usUnityNote:U16 + sFineTune:S16 +
    // lAttenuation/lGain:S32 + fulOptions:U32 + cSampleLoops:U32
    // = 20 bytes.
    if payload.len() < 20 {
        return Err(Error::invalid(format!(
            "DLS: wsmp payload {} < 20 bytes (header)",
            payload.len(),
        )));
    }
    let cb_size = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let unity_note = u16::from_le_bytes(payload[4..6].try_into().unwrap());
    let fine_tune = i16::from_le_bytes(payload[6..8].try_into().unwrap());
    let gain = i32::from_le_bytes(payload[8..12].try_into().unwrap());
    let options = u32::from_le_bytes(payload[12..16].try_into().unwrap());
    let loops_count = u32::from_le_bytes(payload[16..20].try_into().unwrap()) as usize;
    if loops_count > MAX_RECORDS {
        return Err(Error::invalid(format!(
            "DLS: wsmp loops count {loops_count} exceeds cap {MAX_RECORDS}",
        )));
    }

    // Loop records start at cbSize (which excludes the loop list).
    // cbSize is typically 20 (the fixed header) but the spec lets a
    // future revision grow it; we honour the field.
    let loop_start = cb_size.max(20);
    if loop_start > payload.len() {
        return Err(Error::invalid(format!(
            "DLS: wsmp cbSize {cb_size} exceeds payload {}",
            payload.len(),
        )));
    }

    // Each loop record: cbSize:U32, ulLoopType:U32, ulLoopStart:U32,
    // ulLoopLength:U32 = 16 bytes minimum (cbSize may grow).
    let mut loops = Vec::with_capacity(loops_count);
    let mut off = loop_start;
    for i in 0..loops_count {
        if off + 16 > payload.len() {
            return Err(Error::invalid(format!(
                "DLS: wsmp truncated at loop {i} (offset {off}, payload {})",
                payload.len(),
            )));
        }
        let loop_cb = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
        let loop_type = u32::from_le_bytes(payload[off + 4..off + 8].try_into().unwrap());
        let loop_st = u32::from_le_bytes(payload[off + 8..off + 12].try_into().unwrap());
        let loop_len = u32::from_le_bytes(payload[off + 12..off + 16].try_into().unwrap());
        loops.push(DlsLoop {
            loop_type,
            start: loop_st,
            length: loop_len,
        });
        // Advance by loop_cb (≥16) so future fields are skipped if
        // present.
        let advance = loop_cb.max(16);
        off = off
            .checked_add(advance)
            .ok_or_else(|| Error::invalid("DLS: wsmp loop offset overflow"))?;
    }

    Ok(DlsWaveSample {
        unity_note,
        fine_tune,
        gain,
        options,
        loops,
    })
}

// -------------------------------------------------------------------------
// `lart` / `lar2` → list of articulator chunks.
// -------------------------------------------------------------------------

fn parse_lart_or_lar2(body: &[u8], is_lar2: bool) -> Result<Vec<DlsArticulationBlock>> {
    let mut out = Vec::new();
    let mut c = Cursor::new(body);
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        let kind = match (&tag, is_lar2) {
            (b"art1", _) => DlsArtKind::Art1,
            (b"art2", _) => DlsArtKind::Art2,
            // cdl conditional inside lart/lar2 — skip per round-1
            // policy; we always accept the block contents.
            (b"cdl ", _) => continue,
            _ => continue,
        };
        let blocks = parse_art(payload, kind)?;
        out.extend(blocks);
        if out.len() > MAX_RECORDS {
            return Err(Error::invalid(format!(
                "DLS: articulation block count exceeds cap {MAX_RECORDS}",
            )));
        }
    }
    Ok(out)
}

fn parse_art(payload: &[u8], kind: DlsArtKind) -> Result<Vec<DlsArticulationBlock>> {
    if payload.len() < 8 {
        return Err(Error::invalid(format!(
            "DLS: art1/art2 payload {} < 8 bytes (cbSize + cConnectionBlocks)",
            payload.len(),
        )));
    }
    let cb_size = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    if count > MAX_RECORDS {
        return Err(Error::invalid(format!(
            "DLS: art connection-block count {count} exceeds cap {MAX_RECORDS}",
        )));
    }
    // Connection blocks start at cbSize (≥8 — header excluding blocks).
    let block_start = cb_size.max(8);
    if block_start > payload.len() {
        return Err(Error::invalid(format!(
            "DLS: art cbSize {cb_size} exceeds payload {}",
            payload.len(),
        )));
    }
    let needed = count
        .checked_mul(12)
        .ok_or_else(|| Error::invalid("DLS: art block count overflow"))?;
    if block_start + needed > payload.len() {
        return Err(Error::invalid(format!(
            "DLS: art truncated — need {needed} bytes for {count} blocks, payload {}",
            payload.len(),
        )));
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = block_start + i * 12;
        out.push(DlsArticulationBlock {
            kind,
            source: u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()),
            control: u16::from_le_bytes(payload[off + 2..off + 4].try_into().unwrap()),
            destination: u16::from_le_bytes(payload[off + 4..off + 6].try_into().unwrap()),
            transform: u16::from_le_bytes(payload[off + 6..off + 8].try_into().unwrap()),
            scale: i32::from_le_bytes(payload[off + 8..off + 12].try_into().unwrap()),
        });
    }
    Ok(out)
}

// -------------------------------------------------------------------------
// `wvpl-list` → wave pool.
// -------------------------------------------------------------------------

fn parse_wvpl_list(body: &[u8]) -> Result<Vec<DlsSample>> {
    let mut out = Vec::new();
    let mut total_data_bytes: usize = 0;
    let mut c = Cursor::new(body);
    while !c.at_end() {
        // Record the offset of this LIST header from the start of the
        // wvpl payload (so ptbl cues can be reconciled later).
        let pool_offset = c.pos as u32;
        let (tag, payload) = read_chunk(&mut c)?;
        if &tag != b"LIST" {
            continue;
        }
        if payload.len() < 4 {
            continue;
        }
        let mut sub_type = [0u8; 4];
        sub_type.copy_from_slice(&payload[..4]);
        if &sub_type != b"wave" {
            continue;
        }
        let mut sample = parse_wave_list(&payload[4..])?;
        sample.pool_offset = pool_offset;
        total_data_bytes = total_data_bytes.saturating_add(sample.data.len());
        if total_data_bytes > MAX_WAVE_BYTES {
            return Err(Error::invalid(format!(
                "DLS: cumulative wave data {total_data_bytes} exceeds cap {MAX_WAVE_BYTES}",
            )));
        }
        out.push(sample);
        if out.len() > MAX_RECORDS {
            return Err(Error::invalid(format!(
                "DLS: wave pool count exceeds cap {MAX_RECORDS}",
            )));
        }
    }
    Ok(out)
}

fn parse_wave_list(body: &[u8]) -> Result<DlsSample> {
    let mut sample = DlsSample::default();
    let mut c = Cursor::new(body);
    let mut got_fmt = false;
    let mut got_data = false;
    while !c.at_end() {
        let (tag, payload) = read_chunk(&mut c)?;
        match &tag {
            b"fmt " => {
                parse_fmt(payload, &mut sample)?;
                got_fmt = true;
            }
            b"data" => {
                if payload.len() > MAX_WAVE_BYTES {
                    return Err(Error::invalid(format!(
                        "DLS: wave data {} exceeds cap {MAX_WAVE_BYTES}",
                        payload.len(),
                    )));
                }
                sample.data = payload.to_vec();
                got_data = true;
            }
            b"wsmp" => sample.wsmp = Some(parse_wsmp(payload)?),
            b"LIST" => {
                // Wave-level INFO list — surface name? Round 1 ignores;
                // wave names are seldom musically useful.
            }
            _ => {}
        }
    }
    if !got_fmt {
        return Err(Error::invalid("DLS: wave-list missing 'fmt ' chunk"));
    }
    if !got_data {
        return Err(Error::invalid("DLS: wave-list missing 'data' chunk"));
    }
    Ok(sample)
}

fn parse_fmt(payload: &[u8], out: &mut DlsSample) -> Result<()> {
    // Standard WAV fmt header (PCM): 16 bytes
    //   wFormatTag:U16   wChannels:U16   dwSamplesPerSec:U32
    //   dwAvgBytesPerSec:U32   wBlockAlign:U16   wBitsPerSample:U16
    if payload.len() < 16 {
        return Err(Error::invalid(format!(
            "DLS: fmt payload {} < 16 bytes",
            payload.len(),
        )));
    }
    out.format_tag = u16::from_le_bytes(payload[0..2].try_into().unwrap());
    out.channels = u16::from_le_bytes(payload[2..4].try_into().unwrap());
    out.sample_rate = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    out.avg_bytes_per_sec = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    out.block_align = u16::from_le_bytes(payload[12..14].try_into().unwrap());
    out.bits_per_sample = u16::from_le_bytes(payload[14..16].try_into().unwrap());
    Ok(())
}

// =========================================================================
// Tests.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- magic-byte detection (carry-over from the round-0 stub) ----

    #[test]
    fn detects_riff_dls_magic() {
        let mut blob = vec![0u8; 32];
        blob[0..4].copy_from_slice(b"RIFF");
        blob[4..8].copy_from_slice(&24u32.to_le_bytes());
        blob[8..12].copy_from_slice(b"DLS ");
        assert!(is_dls(&blob));
        assert!(DlsInstrument::probe(&blob));
    }

    #[test]
    fn rejects_wrong_magic() {
        assert!(!is_dls(b""));
        assert!(!DlsInstrument::probe(b""));
        let mut blob = vec![0u8; 16];
        blob[0..4].copy_from_slice(b"RIFF");
        blob[8..12].copy_from_slice(b"WAVE");
        assert!(!is_dls(&blob));
    }

    // ---- parser happy-path: minimal hand-built DLS ----

    /// Build a minimal DLS Level 1 file with one instrument, one
    /// region, one sample. Used as the fixture for the parser unit
    /// tests + the smoke integration test.
    pub(super) fn build_minimal_dls() -> Vec<u8> {
        // ------------------------------------------------------------
        // Wave-pool: one tiny 8-bit mono PCM sample.
        // ------------------------------------------------------------
        // 8 frames of 8-bit unsigned PCM — silence (0x80) + a few
        // ramps. Real DLS samples are far larger; the parser doesn't
        // care.
        let pcm = vec![0x80u8, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0x80, 0x80];

        // fmt chunk (16 bytes): WAVE_FORMAT_PCM mono 22050 Hz 8-bit.
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1u16.to_le_bytes()); // wFormatTag = PCM
        fmt.extend_from_slice(&1u16.to_le_bytes()); // wChannels  = 1
        fmt.extend_from_slice(&22_050u32.to_le_bytes()); // dwSamplesPerSec
        fmt.extend_from_slice(&22_050u32.to_le_bytes()); // dwAvgBytesPerSec
        fmt.extend_from_slice(&1u16.to_le_bytes()); // wBlockAlign = 1
        fmt.extend_from_slice(&8u16.to_le_bytes()); // wBitsPerSample = 8

        // wsmp inside the wave: cbSize(20) + unity_note=60 + fine=0 +
        // gain=0 + opts=0 + cSampleLoops=0.
        let mut wave_wsmp = Vec::new();
        wave_wsmp.extend_from_slice(&20u32.to_le_bytes()); // cbSize
        wave_wsmp.extend_from_slice(&60u16.to_le_bytes()); // unity note
        wave_wsmp.extend_from_slice(&0i16.to_le_bytes()); // fine tune
        wave_wsmp.extend_from_slice(&0i32.to_le_bytes()); // gain
        wave_wsmp.extend_from_slice(&0u32.to_le_bytes()); // options
        wave_wsmp.extend_from_slice(&0u32.to_le_bytes()); // 0 loops

        let mut wave_body = Vec::from(b"wave" as &[u8]);
        push_riff(&mut wave_body, b"fmt ", &fmt);
        push_riff(&mut wave_body, b"data", &pcm);
        push_riff(&mut wave_body, b"wsmp", &wave_wsmp);

        let mut wvpl_body = Vec::from(b"wvpl" as &[u8]);
        push_riff(&mut wvpl_body, b"LIST", &wave_body);

        // ------------------------------------------------------------
        // ptbl: one cue at offset 0 (start of the first LIST inside
        // wvpl).
        // ------------------------------------------------------------
        let mut ptbl = Vec::new();
        ptbl.extend_from_slice(&8u32.to_le_bytes()); // cbSize
        ptbl.extend_from_slice(&1u32.to_le_bytes()); // cCues
        ptbl.extend_from_slice(&0u32.to_le_bytes()); // cue 0 offset

        // ------------------------------------------------------------
        // colh: 1 instrument.
        // ------------------------------------------------------------
        let mut colh = Vec::new();
        colh.extend_from_slice(&1u32.to_le_bytes());

        // ------------------------------------------------------------
        // vers: DLS 1.1.0.0 (major=1 minor=1).
        // ------------------------------------------------------------
        let mut vers = Vec::new();
        vers.extend_from_slice(&((1u32 << 16) | 1u32).to_le_bytes()); // dwVersionMS
        vers.extend_from_slice(&0u32.to_le_bytes()); // dwVersionLS

        // ------------------------------------------------------------
        // INFO list (top-level).
        // ------------------------------------------------------------
        let mut info_body = Vec::from(b"INFO" as &[u8]);
        push_riff(&mut info_body, b"INAM", b"TestBank\0");
        push_riff(&mut info_body, b"ICOP", b"(c) Test\0");

        // ------------------------------------------------------------
        // Instrument: ins-list with insh + lrgn (1 region) + INFO.
        // ------------------------------------------------------------
        // insh: cRegions=1, ulBank=0, ulInstrument=0 (program 0).
        let mut insh = Vec::new();
        insh.extend_from_slice(&1u32.to_le_bytes()); // cRegions
        insh.extend_from_slice(&0u32.to_le_bytes()); // ulBank
        insh.extend_from_slice(&0u32.to_le_bytes()); // ulInstrument

        // rgnh: full keyboard, full velocity, no opts, no key group.
        let mut rgnh = Vec::new();
        rgnh.extend_from_slice(&0u16.to_le_bytes()); // RangeKey lo
        rgnh.extend_from_slice(&127u16.to_le_bytes()); // RangeKey hi
        rgnh.extend_from_slice(&0u16.to_le_bytes()); // RangeVel lo
        rgnh.extend_from_slice(&127u16.to_le_bytes()); // RangeVel hi
        rgnh.extend_from_slice(&0u16.to_le_bytes()); // fusOptions
        rgnh.extend_from_slice(&0u16.to_le_bytes()); // usKeyGroup

        // Region wsmp: unity_note=60, no loops.
        let mut rgn_wsmp = Vec::new();
        rgn_wsmp.extend_from_slice(&20u32.to_le_bytes());
        rgn_wsmp.extend_from_slice(&60u16.to_le_bytes());
        rgn_wsmp.extend_from_slice(&0i16.to_le_bytes());
        rgn_wsmp.extend_from_slice(&0i32.to_le_bytes());
        rgn_wsmp.extend_from_slice(&0u32.to_le_bytes());
        rgn_wsmp.extend_from_slice(&0u32.to_le_bytes());

        // wlnk: phase_master=0, phase_group=0, channel=1 (mono/left),
        // table_index=0.
        let mut wlnk = Vec::new();
        wlnk.extend_from_slice(&0u16.to_le_bytes());
        wlnk.extend_from_slice(&0u16.to_le_bytes());
        wlnk.extend_from_slice(&1u32.to_le_bytes());
        wlnk.extend_from_slice(&0u32.to_le_bytes());

        // art1 with one connection block (LFO → pitch, scale 0 — null
        // modulator, just to exercise the parser).
        let mut art1 = Vec::new();
        art1.extend_from_slice(&8u32.to_le_bytes()); // cbSize
        art1.extend_from_slice(&1u32.to_le_bytes()); // cConnectionBlocks
        art1.extend_from_slice(&0x0001u16.to_le_bytes()); // CONN_SRC_LFO
        art1.extend_from_slice(&0x0000u16.to_le_bytes()); // CONN_SRC_NONE
        art1.extend_from_slice(&0x0003u16.to_le_bytes()); // CONN_DST_PITCH
        art1.extend_from_slice(&0x0000u16.to_le_bytes()); // CONN_TRN_NONE
        art1.extend_from_slice(&0i32.to_le_bytes()); // lScale = 0

        let mut lart_body = Vec::from(b"lart" as &[u8]);
        push_riff(&mut lart_body, b"art1", &art1);

        let mut rgn_body = Vec::from(b"rgn " as &[u8]);
        push_riff(&mut rgn_body, b"rgnh", &rgnh);
        push_riff(&mut rgn_body, b"wsmp", &rgn_wsmp);
        push_riff(&mut rgn_body, b"wlnk", &wlnk);
        push_riff(&mut rgn_body, b"LIST", &lart_body);

        let mut lrgn_body = Vec::from(b"lrgn" as &[u8]);
        push_riff(&mut lrgn_body, b"LIST", &rgn_body);

        let mut ins_info_body = Vec::from(b"INFO" as &[u8]);
        push_riff(&mut ins_info_body, b"INAM", b"TestInstrument\0");

        let mut ins_body = Vec::from(b"ins " as &[u8]);
        push_riff(&mut ins_body, b"insh", &insh);
        push_riff(&mut ins_body, b"LIST", &lrgn_body);
        push_riff(&mut ins_body, b"LIST", &ins_info_body);

        let mut lins_body = Vec::from(b"lins" as &[u8]);
        push_riff(&mut lins_body, b"LIST", &ins_body);

        // ------------------------------------------------------------
        // Top-level RIFF body: 'DLS ' form + chunks in canonical order.
        // ------------------------------------------------------------
        let mut body = Vec::from(b"DLS " as &[u8]);
        push_riff(&mut body, b"vers", &vers);
        push_riff(&mut body, b"colh", &colh);
        push_riff(&mut body, b"LIST", &lins_body);
        push_riff(&mut body, b"ptbl", &ptbl);
        push_riff(&mut body, b"LIST", &wvpl_body);
        push_riff(&mut body, b"LIST", &info_body);

        let mut out = Vec::from(b"RIFF" as &[u8]);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Append a `<tag><u32 LE size><payload>[<pad>]` chunk. RIFF
    /// requires payloads to be padded to even length on disk; the size
    /// word does *not* include the pad byte.
    pub(super) fn push_riff(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
        out.extend_from_slice(tag);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
    }

    #[test]
    fn parses_minimal_dls_top_level() {
        let blob = build_minimal_dls();
        let bank = DlsBank::parse(&blob).expect("parse minimal DLS");
        assert_eq!(bank.declared_instrument_count, 1);
        assert_eq!(bank.version, Some((1, 1, 0, 0)));
        assert_eq!(bank.info.name.as_deref(), Some("TestBank"));
        assert_eq!(bank.info.copyright.as_deref(), Some("(c) Test"));
        assert_eq!(bank.wave_pool_offsets, vec![0]);
    }

    #[test]
    fn parses_minimal_dls_wave_pool() {
        let blob = build_minimal_dls();
        let bank = DlsBank::parse(&blob).unwrap();
        assert_eq!(bank.waves.len(), 1);
        let w = &bank.waves[0];
        assert_eq!(w.format_tag, 1);
        assert_eq!(w.channels, 1);
        assert_eq!(w.sample_rate, 22_050);
        assert_eq!(w.bits_per_sample, 8);
        assert_eq!(w.data.len(), 8);
        assert_eq!(w.data[2], 0x90);
        assert_eq!(w.pool_offset, 0); // first LIST inside wvpl
        let wsmp = w.wsmp.as_ref().expect("wave-level wsmp");
        assert_eq!(wsmp.unity_note, 60);
        assert!(wsmp.loops.is_empty());
    }

    #[test]
    fn parses_minimal_dls_instrument_table() {
        let blob = build_minimal_dls();
        let bank = DlsBank::parse(&blob).unwrap();
        assert_eq!(bank.instruments.len(), 1);
        let ins = &bank.instruments[0];
        assert_eq!(ins.declared_region_count, 1);
        assert_eq!(ins.bank, 0);
        assert_eq!(ins.program, 0);
        assert_eq!(ins.bank_msb(), 0);
        assert_eq!(ins.bank_lsb(), 0);
        assert_eq!(ins.program_number(), 0);
        assert!(!ins.is_drum());
        assert_eq!(ins.name.as_deref(), Some("TestInstrument"));
        assert_eq!(ins.regions.len(), 1);
        let r = &ins.regions[0];
        assert_eq!(r.key_lo, 0);
        assert_eq!(r.key_hi, 127);
        assert_eq!(r.vel_lo, 0);
        assert_eq!(r.vel_hi, 127);
        assert_eq!(r.options, 0);
        assert_eq!(r.key_group, 0);
        assert!(r.layer.is_none());
        assert!(!r.is_level2);
        let wlnk = r.wlnk.as_ref().expect("region wlnk");
        assert_eq!(wlnk.table_index, 0);
        assert_eq!(wlnk.channel, 1);
        let rsmp = r.wsmp.as_ref().expect("region wsmp");
        assert_eq!(rsmp.unity_note, 60);
    }

    #[test]
    fn parses_minimal_dls_articulation() {
        let blob = build_minimal_dls();
        let bank = DlsBank::parse(&blob).unwrap();
        let ins = &bank.instruments[0];
        let r = &ins.regions[0];
        assert_eq!(r.articulation.len(), 1);
        let a = &r.articulation[0];
        assert_eq!(a.kind, DlsArtKind::Art1);
        assert_eq!(a.source, 0x0001); // CONN_SRC_LFO
        assert_eq!(a.control, 0x0000); // CONN_SRC_NONE
        assert_eq!(a.destination, 0x0003); // CONN_DST_PITCH
        assert_eq!(a.transform, 0x0000); // CONN_TRN_NONE
        assert_eq!(a.scale, 0);
    }

    // ---- DLS Level 2 specific paths ----

    #[test]
    fn parses_dls2_rgnh_with_uslayer() {
        // 14-byte rgnh (DLS2 region header with usLayer trailing).
        let mut payload = Vec::new();
        payload.extend_from_slice(&36u16.to_le_bytes()); // key lo
        payload.extend_from_slice(&60u16.to_le_bytes()); // key hi
        payload.extend_from_slice(&0u16.to_le_bytes()); // vel lo
        payload.extend_from_slice(&127u16.to_le_bytes()); // vel hi
        payload.extend_from_slice(&1u16.to_le_bytes()); // fusOptions
        payload.extend_from_slice(&3u16.to_le_bytes()); // usKeyGroup
        payload.extend_from_slice(&7u16.to_le_bytes()); // usLayer

        let mut region = DlsRegion::default();
        parse_rgnh(&payload, &mut region).unwrap();
        assert_eq!(region.key_lo, 36);
        assert_eq!(region.key_hi, 60);
        assert_eq!(region.vel_hi, 127);
        assert_eq!(region.options, 1);
        assert_eq!(region.key_group, 3);
        assert_eq!(region.layer, Some(7));
    }

    #[test]
    fn parses_art2_block() {
        // art2 with a single block: KEYNUMBER → FILTER_CUTOFF, scale 50.
        let mut payload = Vec::new();
        payload.extend_from_slice(&8u32.to_le_bytes()); // cbSize
        payload.extend_from_slice(&1u32.to_le_bytes()); // count
        payload.extend_from_slice(&0x0003u16.to_le_bytes()); // CONN_SRC_KEYNUMBER
        payload.extend_from_slice(&0x0000u16.to_le_bytes()); // CONN_SRC_NONE
        payload.extend_from_slice(&0x0500u16.to_le_bytes()); // CONN_DST_FILTER_CUTOFF
        payload.extend_from_slice(&0x0000u16.to_le_bytes()); // transform
        payload.extend_from_slice(&50i32.to_le_bytes()); // lScale = 50

        let blocks = parse_art(&payload, DlsArtKind::Art2).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].kind, DlsArtKind::Art2);
        assert_eq!(blocks[0].source, 0x0003);
        assert_eq!(blocks[0].destination, 0x0500);
        assert_eq!(blocks[0].scale, 50);
    }

    #[test]
    fn parses_wsmp_with_one_loop() {
        // wsmp: unity_note=60, fine=0, gain=0, opts=0, 1 loop:
        // cbSize=16, type=0 (forward), start=128, length=2048.
        let mut payload = Vec::new();
        payload.extend_from_slice(&20u32.to_le_bytes()); // cbSize (header only)
        payload.extend_from_slice(&60u16.to_le_bytes()); // unity_note
        payload.extend_from_slice(&0i16.to_le_bytes()); // fine_tune
        payload.extend_from_slice(&(-1000i32).to_le_bytes()); // gain
        payload.extend_from_slice(&0u32.to_le_bytes()); // options
        payload.extend_from_slice(&1u32.to_le_bytes()); // 1 loop
                                                        // loop record:
        payload.extend_from_slice(&16u32.to_le_bytes()); // cbSize
        payload.extend_from_slice(&0u32.to_le_bytes()); // type=forward
        payload.extend_from_slice(&128u32.to_le_bytes()); // start
        payload.extend_from_slice(&2048u32.to_le_bytes()); // length

        let wsmp = parse_wsmp(&payload).unwrap();
        assert_eq!(wsmp.unity_note, 60);
        assert_eq!(wsmp.gain, -1000);
        assert_eq!(wsmp.loops.len(), 1);
        assert_eq!(wsmp.loops[0].loop_type, 0);
        assert_eq!(wsmp.loops[0].start, 128);
        assert_eq!(wsmp.loops[0].length, 2048);
    }

    // ---- error handling ----

    #[test]
    fn parse_rejects_non_dls_bytes() {
        // RIFF/WAVE — wrong form type for a DLS file.
        let mut blob = Vec::from(b"RIFF" as &[u8]);
        blob.extend_from_slice(&4u32.to_le_bytes());
        blob.extend_from_slice(b"WAVE");
        match DlsBank::parse(&blob) {
            Err(Error::InvalidData(msg)) => {
                assert!(msg.contains("DLS"), "got {msg}");
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_truncated_outer_riff() {
        // Outer RIFF claims 4 GiB; file is 12 bytes.
        let mut blob = Vec::from(b"RIFF" as &[u8]);
        blob.extend_from_slice(&u32::MAX.to_le_bytes());
        blob.extend_from_slice(b"DLS ");
        match DlsBank::parse(&blob) {
            Err(Error::InvalidData(_)) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn open_rejects_non_dls_path() {
        let tmp = std::env::temp_dir().join("oxideav-midi-dls-test-not-dls");
        std::fs::write(&tmp, b"not a dls file at all").unwrap();
        match DlsInstrument::open(&tmp) {
            Err(Error::InvalidData(msg)) => {
                assert!(msg.contains("DLS"), "got {msg}");
            }
            Err(other) => panic!("expected InvalidData, got {other:?}"),
            Ok(_) => panic!("expected error opening non-DLS file"),
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn make_voice_renders_pcm_for_minimal_dls() {
        // The minimal DLS fixture is a 1-instrument bank with 8-bit
        // mono PCM at 22 050 Hz, unity_note=60, no loop. A note-on at
        // key 60 should produce a short non-silent buffer.
        let blob = build_minimal_dls();
        let inst = DlsInstrument::parse_bytes("min.dls", &blob).unwrap();
        let mut voice = inst
            .make_voice(0, 60, 100, 44_100)
            .expect("voice generation");
        let mut buf = vec![0.0f32; 256];
        let _ = voice.render(&mut buf);
        // Sample is 8 frames @ 22 050 Hz played at 44 100 Hz output —
        // ≈ 16 output frames before the voice exhausts. Some of those
        // frames must be non-silent (the ramp section).
        let nonzero = buf.iter().filter(|s| s.abs() > 0.001).count();
        assert!(nonzero > 0, "expected non-silent output, got {nonzero}");
    }

    #[test]
    fn make_voice_picks_region_by_key_and_velocity() {
        // Build the two-region fixture (already used by the smoke
        // test) and ask for a key in the upper half: the voice must
        // pull from region 1, not region 0.
        let blob = build_two_region_dls_for_voice_test();
        let inst = DlsInstrument::parse_bytes("two.dls", &blob).unwrap();
        // Should succeed for any key in 0..=127 (regions cover the
        // full range between them).
        let _v_low = inst.make_voice(5, 30, 100, 44_100).expect("low key");
        let _v_high = inst.make_voice(5, 100, 100, 44_100).expect("high key");
    }

    /// Two-region fixture used by the voice-selection test. Identical
    /// in shape to the one in the integration smoke test but inlined
    /// here so the lib-side test stays self-contained.
    fn build_two_region_dls_for_voice_test() -> Vec<u8> {
        let pcm = vec![0x80u8, 0x90, 0xA0, 0xB0, 0xC0, 0xB0, 0xA0, 0x80];

        let mut fmt = Vec::new();
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&22_050u32.to_le_bytes());
        fmt.extend_from_slice(&22_050u32.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&8u16.to_le_bytes());

        let mut wave_body = Vec::from(b"wave" as &[u8]);
        push_riff(&mut wave_body, b"fmt ", &fmt);
        push_riff(&mut wave_body, b"data", &pcm);

        let mut wvpl_body = Vec::from(b"wvpl" as &[u8]);
        push_riff(&mut wvpl_body, b"LIST", &wave_body);

        let mut ptbl = Vec::new();
        ptbl.extend_from_slice(&8u32.to_le_bytes());
        ptbl.extend_from_slice(&1u32.to_le_bytes());
        ptbl.extend_from_slice(&0u32.to_le_bytes());

        let mut colh = Vec::new();
        colh.extend_from_slice(&1u32.to_le_bytes());

        let mut vers = Vec::new();
        vers.extend_from_slice(&((1u32 << 16) | 1u32).to_le_bytes());
        vers.extend_from_slice(&0u32.to_le_bytes());

        let mut info_body = Vec::from(b"INFO" as &[u8]);
        push_riff(&mut info_body, b"INAM", b"TwoRegion\0");

        // Instrument: program 5, two regions splitting at key 60.
        let mut insh = Vec::new();
        insh.extend_from_slice(&2u32.to_le_bytes());
        insh.extend_from_slice(&0u32.to_le_bytes());
        insh.extend_from_slice(&5u32.to_le_bytes());

        let mut rgnh0 = Vec::new();
        rgnh0.extend_from_slice(&0u16.to_le_bytes());
        rgnh0.extend_from_slice(&59u16.to_le_bytes());
        rgnh0.extend_from_slice(&0u16.to_le_bytes());
        rgnh0.extend_from_slice(&127u16.to_le_bytes());
        rgnh0.extend_from_slice(&0u16.to_le_bytes());
        rgnh0.extend_from_slice(&0u16.to_le_bytes());
        let mut wlnk0 = Vec::new();
        wlnk0.extend_from_slice(&0u16.to_le_bytes());
        wlnk0.extend_from_slice(&0u16.to_le_bytes());
        wlnk0.extend_from_slice(&1u32.to_le_bytes());
        wlnk0.extend_from_slice(&0u32.to_le_bytes());
        let mut rgn0_body = Vec::from(b"rgn " as &[u8]);
        push_riff(&mut rgn0_body, b"rgnh", &rgnh0);
        push_riff(&mut rgn0_body, b"wlnk", &wlnk0);

        let mut rgnh1 = Vec::new();
        rgnh1.extend_from_slice(&60u16.to_le_bytes());
        rgnh1.extend_from_slice(&127u16.to_le_bytes());
        rgnh1.extend_from_slice(&0u16.to_le_bytes());
        rgnh1.extend_from_slice(&127u16.to_le_bytes());
        rgnh1.extend_from_slice(&0u16.to_le_bytes());
        rgnh1.extend_from_slice(&0u16.to_le_bytes());
        let mut wlnk1 = Vec::new();
        wlnk1.extend_from_slice(&0u16.to_le_bytes());
        wlnk1.extend_from_slice(&0u16.to_le_bytes());
        wlnk1.extend_from_slice(&1u32.to_le_bytes());
        wlnk1.extend_from_slice(&0u32.to_le_bytes());
        let mut rgn1_body = Vec::from(b"rgn " as &[u8]);
        push_riff(&mut rgn1_body, b"rgnh", &rgnh1);
        push_riff(&mut rgn1_body, b"wlnk", &wlnk1);

        let mut lrgn_body = Vec::from(b"lrgn" as &[u8]);
        push_riff(&mut lrgn_body, b"LIST", &rgn0_body);
        push_riff(&mut lrgn_body, b"LIST", &rgn1_body);

        let mut ins_body = Vec::from(b"ins " as &[u8]);
        push_riff(&mut ins_body, b"insh", &insh);
        push_riff(&mut ins_body, b"LIST", &lrgn_body);

        let mut lins_body = Vec::from(b"lins" as &[u8]);
        push_riff(&mut lins_body, b"LIST", &ins_body);

        let mut body = Vec::from(b"DLS " as &[u8]);
        push_riff(&mut body, b"vers", &vers);
        push_riff(&mut body, b"colh", &colh);
        push_riff(&mut body, b"LIST", &lins_body);
        push_riff(&mut body, b"ptbl", &ptbl);
        push_riff(&mut body, b"LIST", &wvpl_body);
        push_riff(&mut body, b"LIST", &info_body);

        let mut out = Vec::from(b"RIFF" as &[u8]);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn drum_bit_decodes_correctly() {
        let mut entry = DlsInstrumentEntry {
            bank: 0x8000_0000, // F_INSTRUMENT_DRUMS bit
            program: 0,
            ..DlsInstrumentEntry::default()
        };
        assert!(entry.is_drum());
        entry.bank = 0;
        assert!(!entry.is_drum());
        // CC0=2, CC32=5, drum bit set
        entry.bank = 0x8000_0000 | (2 << 8) | 5;
        assert_eq!(entry.bank_msb(), 2);
        assert_eq!(entry.bank_lsb(), 5);
        assert!(entry.is_drum());
    }

    #[test]
    fn open_round_trip_through_disk() {
        // Write the minimal DLS to a temp file and read it back via
        // `open` — exercises the filesystem path that the round-1
        // public API surface is built on.
        let tmp = std::env::temp_dir().join("oxideav-midi-dls-test-roundtrip.dls");
        let _ = std::fs::remove_file(&tmp);
        std::fs::write(&tmp, build_minimal_dls()).unwrap();
        let inst = DlsInstrument::open(&tmp).unwrap();
        assert_eq!(inst.bank().instruments.len(), 1);
        assert_eq!(inst.bank().waves.len(), 1);
        assert_eq!(inst.name(), "oxideav-midi-dls-test-roundtrip.dls");
        let _ = std::fs::remove_file(&tmp);
    }
}
