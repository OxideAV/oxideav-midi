//! SFZ (text patch format) instrument-bank reader.
//!
//! SFZ is a plain-text format: line-delimited `name=value` opcode pairs
//! grouped into `<region>` / `<group>` / `<global>` / `<control>` /
//! `<master>` sections, all referring to external sample files
//! (typically `.wav` or `.flac`). There is no fixed magic number;
//! convention is the file extension `.sfz` plus a comment, header, or
//! opcode in the first line.
//!
//! What round 1 (this revision) implements:
//!
//! - **Tokenizer + parser** for `<header>` / `opcode=value` syntax.
//!   Comments (`// ...` line and `/* ... */` block) are stripped before
//!   tokenisation. Values may contain spaces (sample paths most often)
//!   — they extend to the next `name=` pattern or the next `<header>`.
//! - **Inheritance**: `<global>` opcodes apply to every region; `<group>`
//!   opcodes apply to every region until the next `<group>` (or another
//!   `<global>`); `<master>` is treated as a higher-level group between
//!   `<global>` and `<group>` per ARIA convention; per-region opcodes
//!   override inherited ones. The flattened list of regions is the
//!   public API.
//! - **`<control>`**: only `default_path=` is honoured today; sample
//!   paths are resolved against `default_path` (relative to the SFZ
//!   file's directory). Other `<control>` opcodes are stored as raw
//!   strings on the [`SfzPatch::control`] map (accessible via
//!   [`SfzInstrument::patch`]) for round-2 use.
//! - **Sample loader**: [`SfzInstrument::open`] resolves every
//!   `sample=` path against (SFZ dir / default_path) and reads the
//!   bytes off disk into [`SfzRegion::sample_bytes`]. Unreadable files
//!   become a hard parse error so the caller learns about typos at
//!   load-time rather than note-on-time. Pass-through callers that
//!   want to keep paths unresolved use [`SfzInstrument::parse_str`].
//! - **`#include`**: not supported in round 1 (returns
//!   [`Error::Unsupported`]). The aggregate-syntax docs note that
//!   `#include "other.sfz"` may appear in `<global>` scope; we report
//!   the unsupported directive rather than silently dropping it so a
//!   user with an `#include`-heavy patch knows the file isn't being
//!   fully consumed.
//!
//! Voice generation: [`SfzInstrument::make_voice`] decodes the WAV
//! sample bytes (8/16/24/32-bit PCM and IEEE_FLOAT — see
//! [`super::wav_pcm`]), picks the highest-priority region matching
//! `(key, velocity)`, shifts pitch off `pitch_keycenter` + `tune` +
//! `transpose`, and instantiates a
//! [`super::sample_voice::SamplePlayer`] with a DAHDSR amplitude
//! envelope from `ampeg_*` opcodes plus a vibrato LFO from
//! `lfo01_freq` / `lfo01_pitch` / `lfo01_delay`. Patches loaded via
//! [`SfzInstrument::parse_str`] (no filesystem) report
//! [`Error::Unsupported`] at note-on time because there are no sample
//! bytes to render.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use oxideav_core::{Error, Result};

use super::sample_voice::{
    EnvelopeParams, SampleLoopMode, SamplePlayer, SamplePlayerConfig, VibratoParams,
};
use super::wav_pcm::decode_wav;
use super::{Instrument, Voice};

/// Loop-mode opcode values (`loop_mode=...`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LoopMode {
    /// `no_loop` — sample plays once and stops at sample end.
    #[default]
    NoLoop,
    /// `one_shot` — sample plays to end, ignoring note-off.
    OneShot,
    /// `loop_continuous` — sample loops forever between `loop_start`
    /// and `loop_end`, even after release.
    LoopContinuous,
    /// `loop_sustain` — sample loops while the note is held; on release
    /// the loop is broken and the sample continues to its end.
    LoopSustain,
}

impl LoopMode {
    /// Parse a `loop_mode=` value. Unknown values become `NoLoop` per
    /// SFZ tradition (engines typically ignore unrecognised modes).
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "no_loop" => LoopMode::NoLoop,
            "one_shot" => LoopMode::OneShot,
            "loop_continuous" => LoopMode::LoopContinuous,
            "loop_sustain" => LoopMode::LoopSustain,
            _ => LoopMode::NoLoop,
        }
    }

    /// Convert to the [`super::sample_voice::SampleLoopMode`] used by
    /// the shared sample-playback voice.
    pub fn to_sample_loop_mode(self) -> SampleLoopMode {
        match self {
            LoopMode::NoLoop => SampleLoopMode::NoLoop,
            LoopMode::OneShot => SampleLoopMode::OneShot,
            LoopMode::LoopContinuous => SampleLoopMode::LoopContinuous,
            LoopMode::LoopSustain => SampleLoopMode::LoopSustain,
        }
    }
}

/// One SFZ region — a sample plus its mapping (key range, velocity
/// range, loop, tuning, etc.). Opcodes inherited from the surrounding
/// `<global>` / `<master>` / `<group>` sections are flattened in;
/// per-region opcodes override.
#[derive(Clone, Debug, Default)]
pub struct SfzRegion {
    /// Path to the sample file, resolved against the SFZ file's
    /// directory + the active `default_path` from the most recent
    /// `<control>` block. `None` if no `sample=` opcode appeared in
    /// this region or any of its enclosing groups (which would make
    /// the region silent — likely a patch bug, but not invalid).
    pub sample_path: Option<PathBuf>,
    /// Loaded sample bytes — populated when [`SfzInstrument::open`]
    /// reads the patch from disk. `None` when constructed via
    /// [`SfzInstrument::parse_str`] (no filesystem hooks).
    pub sample_bytes: Option<Vec<u8>>,
    /// Lowest MIDI key this region responds to, inclusive (default 0).
    /// Set by `lokey=` or by `key=` (which also sets `hikey` +
    /// `pitch_keycenter`).
    pub lokey: u8,
    /// Highest MIDI key this region responds to, inclusive (default 127).
    pub hikey: u8,
    /// Lowest MIDI velocity this region responds to, inclusive (default 0).
    pub lovel: u8,
    /// Highest MIDI velocity this region responds to, inclusive (default 127).
    pub hivel: u8,
    /// MIDI key the sample sounds at when played at its native rate
    /// (default 60 = middle C). Set by `pitch_keycenter=` or `key=`.
    pub pitch_keycenter: u8,
    /// Loop start frame (default 0), set by `loop_start=`.
    pub loop_start: Option<u64>,
    /// Loop end frame (default = sample end), set by `loop_end=`.
    pub loop_end: Option<u64>,
    /// Loop mode (default `no_loop`), set by `loop_mode=`.
    pub loop_mode: LoopMode,
    /// Coarse tuning in semitones (default 0), set by `transpose=`.
    pub transpose: i32,
    /// Fine tuning in cents (-100..=+100, default 0), set by `tune=` or
    /// `pitch=` (legacy alias).
    pub tune: i32,
    /// Per-region volume in dB (default 0), set by `volume=`.
    pub volume: f32,
    /// Per-region pan (-100..=+100, default 0 = centre), set by `pan=`.
    pub pan: f32,
    /// Trigger condition: `attack` (default), `release`, `first`, `legato`.
    /// Stored as the raw string for round 2.
    pub trigger: String,
    /// Every `name=value` opcode that appeared anywhere in this
    /// region's inheritance chain, post-flattening — useful for
    /// round-2 voice generation, which will switch on a wider opcode
    /// set than the strongly-typed fields above.
    pub opcodes: BTreeMap<String, String>,
}

impl SfzRegion {
    /// Build a default region (lokey=0, hikey=127, lovel=0, hivel=127,
    /// pitch_keycenter=60, trigger="attack", everything else zero).
    /// Used as the base case for inheritance flattening.
    pub fn defaults() -> Self {
        Self {
            sample_path: None,
            sample_bytes: None,
            lokey: 0,
            hikey: 127,
            lovel: 0,
            hivel: 127,
            pitch_keycenter: 60,
            loop_start: None,
            loop_end: None,
            loop_mode: LoopMode::NoLoop,
            transpose: 0,
            tune: 0,
            volume: 0.0,
            pan: 0.0,
            trigger: "attack".to_string(),
            opcodes: BTreeMap::new(),
        }
    }
}

/// Parsed SFZ instrument: a flat list of regions plus the `<control>`
/// block as a raw opcode map. Held by [`SfzInstrument`].
#[derive(Clone, Debug, Default)]
pub struct SfzPatch {
    /// Flattened region list (`<global>` + `<master>` + `<group>` opcodes
    /// merged into each `<region>`).
    pub regions: Vec<SfzRegion>,
    /// Raw `<control>` opcodes (one per file, must precede everything
    /// else per SFZ v2). `default_path` is the only one consumed by
    /// the round-1 sample loader; others are surfaced for callers who
    /// want them.
    pub control: BTreeMap<String, String>,
}

/// SFZ instrument bank. Round 1 parses the patch and (when constructed
/// via [`open`](Self::open)) loads sample bytes off disk; voice
/// generation is round 2.
pub struct SfzInstrument {
    name: String,
    patch: SfzPatch,
}

impl SfzInstrument {
    /// Open a `.sfz` patch from disk. Reads the file, parses it, and
    /// resolves + loads every `sample=` path. A missing or unreadable
    /// sample is reported as an error so the caller learns at load
    /// time rather than note-on time. The SFZ file's directory plus
    /// any `<control> default_path=` opcode root sample paths.
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if !looks_like_sfz(path, &bytes) {
            return Err(Error::invalid(format!(
                "SFZ: '{}' does not look like an SFZ patch \
                 (no <region>/<group>/<global> header in first 4 KiB)",
                path.display(),
            )));
        }
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            Error::invalid(format!("SFZ: '{}' is not valid UTF-8: {e}", path.display(),))
        })?;

        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let mut patch = parse_str(text)?;
        let default_path = patch
            .control
            .get("default_path")
            .cloned()
            .unwrap_or_default();

        for region in &mut patch.regions {
            if let Some(sp) = region.sample_path.as_ref() {
                let resolved = resolve_sample_path(base, &default_path, sp);
                let data = std::fs::read(&resolved).map_err(|e| {
                    Error::invalid(format!(
                        "SFZ: '{}': cannot read sample '{}' (resolved to '{}'): {e}",
                        path.display(),
                        sp.display(),
                        resolved.display(),
                    ))
                })?;
                region.sample_bytes = Some(data);
                region.sample_path = Some(resolved);
            }
        }

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "sfz".to_string());
        Ok(Self { name, patch })
    }

    /// Parse an SFZ patch from an in-memory string without touching
    /// the filesystem. Sample paths are stored verbatim; no `default_path`
    /// resolution and no `sample_bytes` population.
    pub fn parse_str(name: impl Into<String>, text: &str) -> Result<Self> {
        Ok(Self {
            name: name.into(),
            patch: parse_str(text)?,
        })
    }

    /// Borrow the parsed patch (regions + control opcodes). Useful for
    /// diagnostics — voice generation goes through
    /// [`Instrument::make_voice`] (round 2).
    pub fn patch(&self) -> &SfzPatch {
        &self.patch
    }

    /// Borrow the flattened region table directly. Equivalent to
    /// `self.patch().regions.as_slice()`.
    pub fn regions(&self) -> &[SfzRegion] {
        &self.patch.regions
    }
}

impl Instrument for SfzInstrument {
    fn name(&self) -> &str {
        &self.name
    }

    fn make_voice(
        &self,
        _program: u8,
        key: u8,
        velocity: u8,
        sample_rate: u32,
    ) -> Result<Box<dyn Voice>> {
        // Region selection: pick the highest-priority region covering
        // (key, velocity). Per the SFZ tutorials, regions are evaluated
        // in declaration order and the *last* matching region wins
        // (later regions in a patch override earlier ones for the same
        // key/velocity slot). We walk in declaration order and keep the
        // last match — equivalent.
        let region = self
            .patch
            .regions
            .iter()
            .rfind(|r| {
                key >= r.lokey && key <= r.hikey && velocity >= r.lovel && velocity <= r.hivel
            })
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "SFZ '{}': no region matches key {key} velocity {velocity}",
                    self.name,
                ))
            })?;

        let bytes = region.sample_bytes.as_deref().ok_or_else(|| {
            Error::unsupported(format!(
                "SFZ '{}': region key=[{},{}] has no loaded sample bytes \
                 (was the patch parsed via parse_str() instead of open()?)",
                self.name, region.lokey, region.hikey,
            ))
        })?;

        let pcm = decode_wav(bytes).map_err(|e| {
            Error::invalid(format!(
                "SFZ '{}': failed to decode WAV sample for region key=[{},{}]: {e}",
                self.name, region.lokey, region.hikey,
            ))
        })?;

        let cfg = build_config_for_region(region, &pcm.samples, pcm.sample_rate, key, velocity);
        Ok(Box::new(SamplePlayer::new(cfg, sample_rate)))
    }
}

/// Build a [`SamplePlayerConfig`] for an SFZ region. Pulled out so it
/// can be re-used by tests that don't go through the full
/// `Instrument::make_voice` path.
fn build_config_for_region(
    region: &SfzRegion,
    samples: &[f32],
    native_rate: u32,
    key: u8,
    velocity: u8,
) -> SamplePlayerConfig {
    // Pitch ratio: 2^((target - center + transpose) / 12 + tune/1200).
    let semitones = key as i32 - region.pitch_keycenter as i32 + region.transpose;
    let cents = region.tune;
    let pitch_ratio = (2.0f64).powf(semitones as f64 / 12.0 + cents as f64 / 1200.0);

    // Velocity curve: SFZ default is (vel/127)^2 amplitude (per the
    // SFZ tutorials' "amp_velcurve_*" defaults). Combine with the
    // region's `volume` (in dB) and SFZ's amp_keytrack default
    // (which we leave at 0 = no keytrack here for simplicity).
    let v = velocity as f32 / 127.0;
    let vel_gain = v * v;
    // 0 dB volume → multiplier 1.0; +6 dB ≈ 2.0; -6 dB ≈ 0.5.
    let vol_gain = 10.0_f32.powf(region.volume / 20.0);
    // Headroom — target ~0.5 peak per voice so 8 simultaneous voices
    // don't clip.
    let amplitude = vel_gain * vol_gain * 0.5;

    let total_frames = samples.len() as u32;
    let sample_end = total_frames;
    let loop_start = region.loop_start.unwrap_or(0).min(u64::from(total_frames)) as u32;
    let loop_end = region
        .loop_end
        .unwrap_or(u64::from(total_frames))
        .min(u64::from(total_frames)) as u32;

    let envelope = build_envelope_from_opcodes(&region.opcodes);
    let vibrato = build_vibrato_from_opcodes(&region.opcodes);

    SamplePlayerConfig {
        samples: Arc::from(samples.to_vec().into_boxed_slice()),
        native_rate,
        loop_start,
        loop_end,
        sample_end,
        loop_mode: region.loop_mode.to_sample_loop_mode(),
        pitch_ratio,
        amplitude,
        envelope,
        vibrato,
        exclusive_class: 0,
    }
}

/// Pull `ampeg_*` opcodes off the region's flattened opcode map.
/// Defaults match the SFZ spec: 0 s delay/attack/hold, 0 s decay (i.e.
/// no decay phase, sustain at 100 %), 100 % sustain, 0 s release.
fn build_envelope_from_opcodes(opcodes: &BTreeMap<String, String>) -> EnvelopeParams {
    fn pf(map: &BTreeMap<String, String>, key: &str, default: f32) -> f32 {
        map.get(key)
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(default)
    }
    // SFZ ampeg_sustain is in percent (0..=100).
    let sustain_pct = pf(opcodes, "ampeg_sustain", 100.0).clamp(0.0, 100.0);
    EnvelopeParams {
        delay_s: pf(opcodes, "ampeg_delay", 0.0).max(0.0),
        // SFZ default attack is 0 — but a 0-sample attack on a sample-
        // playback voice produces a click. We keep 0 so the attack
        // matches the spec exactly; downstream the SamplePlayer caps
        // at 1 frame so the divide-by-zero is impossible.
        attack_s: pf(opcodes, "ampeg_attack", 0.0).max(0.0),
        hold_s: pf(opcodes, "ampeg_hold", 0.0).max(0.0),
        decay_s: pf(opcodes, "ampeg_decay", 0.0).max(0.0),
        sustain_level: sustain_pct / 100.0,
        // SFZ release default is 0; we set a tiny floor so a note-off
        // doesn't click. Spec-strict callers can override with
        // ampeg_release=0.
        release_s: pf(opcodes, "ampeg_release", 0.0).max(0.0),
    }
}

/// Pull SFZ vibrato LFO opcodes off the region map. SFZ v1 uses
/// `lfo01_freq` / `lfo01_pitch` / `lfo01_delay`; SFZ v2 also adds
/// `vibrato_freq` and `vibrato_depth` aliases. Round-1 honours both
/// spellings.
fn build_vibrato_from_opcodes(opcodes: &BTreeMap<String, String>) -> VibratoParams {
    fn pf(map: &BTreeMap<String, String>, keys: &[&str], default: f32) -> f32 {
        for k in keys {
            if let Some(s) = map.get(*k) {
                if let Ok(v) = s.trim().parse::<f32>() {
                    return v;
                }
            }
        }
        default
    }
    VibratoParams {
        freq_hz: pf(opcodes, &["lfo01_freq", "vibrato_freq"], 0.0).max(0.0),
        depth_cents: pf(opcodes, &["lfo01_pitch", "vibrato_depth"], 0.0),
        delay_s: pf(opcodes, &["lfo01_delay", "vibrato_delay"], 0.0).max(0.0),
    }
}

/// Resolve a `sample=` value against the SFZ file's directory + the
/// active `default_path`. If `sample` is absolute, return it unchanged;
/// otherwise prepend `base / default_path`.
fn resolve_sample_path(base: &Path, default_path: &str, sample: &Path) -> PathBuf {
    if sample.is_absolute() {
        return sample.to_path_buf();
    }
    let mut out = base.to_path_buf();
    if !default_path.is_empty() {
        // `default_path` may be "..", "samples/", "/abs/path", etc.
        let dp = Path::new(default_path);
        if dp.is_absolute() {
            out = dp.to_path_buf();
        } else {
            out.push(dp);
        }
    }
    out.push(sample);
    out
}

/// Heuristic detection: SFZ has no magic byte. We require either the
/// `.sfz` extension OR a recognisable section header (`<region>`,
/// `<group>`, `<global>`, `<control>`, `<master>`, `<curve>`,
/// `<effect>`, `<midi>`, `<sample>`) in the first 4 KiB of the file.
pub fn looks_like_sfz(path: &Path, bytes: &[u8]) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("sfz"))
        .unwrap_or(false)
    {
        return true;
    }
    let head = &bytes[..bytes.len().min(4096)];
    let Ok(head_str) = std::str::from_utf8(head) else {
        return false;
    };
    [
        "<region>",
        "<group>",
        "<global>",
        "<control>",
        "<master>",
        "<curve>",
        "<effect>",
        "<midi>",
        "<sample>",
    ]
    .iter()
    .any(|tag| head_str.contains(tag))
}

// =========================================================================
// Parser.
// =========================================================================

/// Parse an SFZ patch from text. The result has `<global>` / `<master>`
/// / `<group>` opcodes flattened into each `<region>` so callers see
/// one fully-resolved opcode map per region.
pub fn parse_str(text: &str) -> Result<SfzPatch> {
    let stripped = strip_comments(text);
    let tokens = tokenize(&stripped)?;

    let mut control: BTreeMap<String, String> = BTreeMap::new();
    let mut global: BTreeMap<String, String> = BTreeMap::new();
    let mut master: BTreeMap<String, String> = BTreeMap::new();
    let mut group: BTreeMap<String, String> = BTreeMap::new();
    let mut current: Option<BTreeMap<String, String>> = None;
    let mut current_kind: Option<HeaderKind> = None;
    let mut regions: Vec<SfzRegion> = Vec::new();

    for tok in tokens {
        match tok {
            Token::Header(name) => {
                let kind = HeaderKind::from(&name);
                // Close the previous block — commit its body into the
                // right destination *before* mutating any of the
                // global/master/group scopes (which a new
                // <global>/<master>/<group> header itself triggers).
                if let Some(b) = current.take() {
                    match current_kind {
                        Some(HeaderKind::Control) => {
                            control.extend(b);
                        }
                        Some(HeaderKind::Global) => {
                            global.clear();
                            global.extend(b);
                        }
                        Some(HeaderKind::Master) => {
                            master.clear();
                            master.extend(b);
                        }
                        Some(HeaderKind::Group) => {
                            group.clear();
                            group.extend(b);
                        }
                        Some(HeaderKind::Region) => {
                            regions.push(flatten_region(&global, &master, &group, &b));
                        }
                        Some(HeaderKind::Other) | None => {}
                    }
                }
                if matches!(kind, HeaderKind::Master) {
                    // A new <master> resets <group> per ARIA convention.
                    group.clear();
                }
                if matches!(kind, HeaderKind::Global) {
                    master.clear();
                    group.clear();
                }
                current_kind = Some(kind);
                current = Some(BTreeMap::new());
            }
            Token::Opcode(name, value) => {
                if name == "#include" {
                    return Err(Error::unsupported(format!(
                        "SFZ: '#include {value}' directive — round 1 reader does \
                         not follow includes (use parse_str on the concatenated text)",
                    )));
                }
                if name == "#define" {
                    // SFZ #define — round-1 stores the raw definition
                    // in the current scope but does no macro expansion.
                    if let Some(body) = current.as_mut() {
                        body.insert(format!("#define {value}"), String::new());
                    }
                    continue;
                }
                let body = current.get_or_insert_with(BTreeMap::new);
                body.insert(name, value);
            }
        }
    }

    // Flush the trailing block.
    if let Some(b) = current.take() {
        match current_kind {
            Some(HeaderKind::Control) => {
                control.extend(b);
            }
            Some(HeaderKind::Global) => {
                global.clear();
                global.extend(b);
            }
            Some(HeaderKind::Master) => {
                master.clear();
                master.extend(b);
            }
            Some(HeaderKind::Group) => {
                group.clear();
                group.extend(b);
            }
            Some(HeaderKind::Region) => {
                regions.push(flatten_region(&global, &master, &group, &b));
            }
            Some(HeaderKind::Other) | None => {}
        }
    }

    Ok(SfzPatch { regions, control })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeaderKind {
    Control,
    Global,
    Master,
    Group,
    Region,
    /// Curve / effect / midi / sample — recognised but the body is
    /// discarded by the round-1 region flattener.
    Other,
}

impl HeaderKind {
    fn from(name: &str) -> Self {
        match name {
            "control" => HeaderKind::Control,
            "global" => HeaderKind::Global,
            "master" => HeaderKind::Master,
            "group" => HeaderKind::Group,
            "region" => HeaderKind::Region,
            _ => HeaderKind::Other,
        }
    }
}

/// Build one fully-resolved region by merging global → master → group →
/// region opcodes (later overrides earlier).
fn flatten_region(
    global: &BTreeMap<String, String>,
    master: &BTreeMap<String, String>,
    group: &BTreeMap<String, String>,
    region: &BTreeMap<String, String>,
) -> SfzRegion {
    let mut merged: BTreeMap<String, String> = BTreeMap::new();
    for src in [global, master, group, region] {
        for (k, v) in src {
            merged.insert(k.clone(), v.clone());
        }
    }

    let mut out = SfzRegion::defaults();
    for (k, v) in &merged {
        apply_opcode(&mut out, k, v);
    }
    out.opcodes = merged;
    out
}

/// Strongly-type the small set of opcodes round-1 surfaces directly on
/// [`SfzRegion`] fields. Unknown opcodes stay in `out.opcodes` only —
/// they're not lost.
fn apply_opcode(out: &mut SfzRegion, key: &str, value: &str) {
    match key {
        "sample" => out.sample_path = Some(PathBuf::from(value)),
        "lokey" => {
            if let Some(k) = parse_key(value) {
                out.lokey = k;
            }
        }
        "hikey" => {
            if let Some(k) = parse_key(value) {
                out.hikey = k;
            }
        }
        "key" => {
            if let Some(k) = parse_key(value) {
                out.lokey = k;
                out.hikey = k;
                out.pitch_keycenter = k;
            }
        }
        "lovel" => {
            if let Ok(v) = value.trim().parse::<u16>() {
                out.lovel = v.min(127) as u8;
            }
        }
        "hivel" => {
            if let Ok(v) = value.trim().parse::<u16>() {
                out.hivel = v.min(127) as u8;
            }
        }
        "pitch_keycenter" => {
            if let Some(k) = parse_key(value) {
                out.pitch_keycenter = k;
            }
        }
        "loop_start" | "loopstart" => {
            if let Ok(v) = value.trim().parse::<u64>() {
                out.loop_start = Some(v);
            }
        }
        "loop_end" | "loopend" => {
            if let Ok(v) = value.trim().parse::<u64>() {
                out.loop_end = Some(v);
            }
        }
        "loop_mode" | "loopmode" => out.loop_mode = LoopMode::parse(value),
        "transpose" => {
            if let Ok(v) = value.trim().parse::<i32>() {
                out.transpose = v;
            }
        }
        "tune" | "pitch" => {
            if let Ok(v) = value.trim().parse::<i32>() {
                out.tune = v;
            }
        }
        "volume" => {
            if let Ok(v) = value.trim().parse::<f32>() {
                out.volume = v;
            }
        }
        "pan" => {
            if let Ok(v) = value.trim().parse::<f32>() {
                out.pan = v;
            }
        }
        "trigger" => out.trigger = value.trim().to_string(),
        _ => {}
    }
}

/// Parse a MIDI key — either a decimal integer (`60`) or a note name
/// (`c4`, `C#4`, `Db5`, …). SFZ tradition: `C4 = 60` (i.e. middle C is
/// MIDI 60, octave 4). Returns `None` on parse failure.
fn parse_key(value: &str) -> Option<u8> {
    let s = value.trim();
    if let Ok(n) = s.parse::<i32>() {
        if (0..=127).contains(&n) {
            return Some(n as u8);
        }
        return None;
    }
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let (note_idx, mut i) = match bytes[0].to_ascii_lowercase() {
        b'c' => (0i32, 1),
        b'd' => (2, 1),
        b'e' => (4, 1),
        b'f' => (5, 1),
        b'g' => (7, 1),
        b'a' => (9, 1),
        b'b' => (11, 1),
        _ => return None,
    };
    let mut accidental = 0i32;
    if i < bytes.len() {
        match bytes[i] {
            b'#' => {
                accidental = 1;
                i += 1;
            }
            b'b' | b'B' => {
                // Could be 'B' (note name) — but we already matched the
                // first char, so a 'b' here is a flat. Only treat 'b'
                // as flat when it follows a note letter.
                accidental = -1;
                i += 1;
            }
            _ => {}
        }
    }
    if i >= bytes.len() {
        return None;
    }
    // Octave can be negative ("c-1" = 0).
    let octave_str = &s[i..];
    let octave: i32 = octave_str.parse().ok()?;
    let midi = (octave + 1) * 12 + note_idx + accidental;
    if (0..=127).contains(&midi) {
        Some(midi as u8)
    } else {
        None
    }
}

// -------------------------------------------------------------------------
// Tokenizer.
// -------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Header(String),
    Opcode(String, String),
}

/// Strip `// ...` and `/* ... */` comments. Replaces the comment span
/// with a single space so adjacent tokens don't merge.
fn strip_comments(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        // Block comment.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            // Skip closing "*/" if present.
            if i + 1 < bytes.len() {
                i += 2;
            } else {
                // Unterminated block comment — bail with what we have.
                break;
            }
            out.push(' ');
            continue;
        }
        // Line comment.
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            out.push(' ');
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// Tokenise the comment-stripped text into headers + opcodes.
///
/// SFZ tokenisation is lenient — opcodes are whitespace-separated
/// `name=value` pairs but values can contain spaces (sample paths
/// most of the time). We resolve the ambiguity by greedily reading
/// the value until we hit either a `<` (next header) or something
/// that looks like the start of a new opcode (`name=`).
fn tokenize(text: &str) -> Result<Vec<Token>> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        // Skip whitespace.
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Header.
        if c == b'<' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end] != b'>' {
                end += 1;
            }
            if end >= bytes.len() {
                return Err(Error::invalid(
                    "SFZ: unterminated header (expected '>' to close '<...')",
                ));
            }
            let name = std::str::from_utf8(&bytes[start..end])
                .map_err(|_| Error::invalid("SFZ: non-UTF-8 header name"))?
                .trim()
                .to_ascii_lowercase();
            if name.is_empty() {
                return Err(Error::invalid("SFZ: empty header '<>'"));
            }
            tokens.push(Token::Header(name));
            i = end + 1;
            continue;
        }
        // Preprocessor directive (`#include "x"` or `#define $X 10`):
        // these have a name + space-delimited value rather than `name=value`.
        // We surface them as Opcode("#include"/"#define", rest-of-line) so
        // the parser can decide what to do.
        if c == b'#' {
            let line_start = i;
            let mut line_end = i;
            while line_end < bytes.len() && bytes[line_end] != b'\n' && bytes[line_end] != b'\r' {
                line_end += 1;
            }
            let line = std::str::from_utf8(&bytes[line_start..line_end])
                .map_err(|_| Error::invalid("SFZ: non-UTF-8 preprocessor directive"))?
                .trim();
            // Split into directive + remainder.
            let mut parts = line.splitn(2, char::is_whitespace);
            let directive = parts.next().unwrap_or("").to_ascii_lowercase();
            let value = parts.next().unwrap_or("").trim().to_string();
            tokens.push(Token::Opcode(directive, value));
            i = line_end;
            continue;
        }
        // Opcode `name=value`. Read name up to '=' or whitespace.
        let name_start = i;
        let mut name_end = i;
        while name_end < bytes.len()
            && bytes[name_end] != b'='
            && !bytes[name_end].is_ascii_whitespace()
            && bytes[name_end] != b'<'
        {
            name_end += 1;
        }
        if name_end == name_start {
            // Defensive — shouldn't happen given the whitespace skip
            // above, but bail rather than spin.
            i += 1;
            continue;
        }
        let name = std::str::from_utf8(&bytes[name_start..name_end])
            .map_err(|_| Error::invalid("SFZ: non-UTF-8 opcode name"))?
            .to_ascii_lowercase();
        // Skip whitespace between name and '='.
        let mut j = name_end;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() && bytes[j] != b'\n' {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'=' {
            // Stray bare token — could be a `#define` macro use, an
            // identifier left over from a weird patch, or trailing
            // garbage. SFZ tradition is to ignore.
            i = name_end;
            continue;
        }
        // Skip '=' and whitespace.
        j += 1;
        while j < bytes.len()
            && bytes[j].is_ascii_whitespace()
            && bytes[j] != b'\n'
            && bytes[j] != b'\r'
        {
            j += 1;
        }
        // Read value greedily until: header start, EOL break followed
        // by something that looks like a header / opcode, or another
        // `name=` pattern. Spaces within the value are kept.
        let value_start = j;
        let mut k = j;
        let mut last_non_ws = j;
        while k < bytes.len() {
            let b = bytes[k];
            if b == b'<' {
                break;
            }
            if b == b'\n' || b == b'\r' {
                // SFZ commonly has one opcode per line. Consume
                // newline and stop.
                break;
            }
            if b == b'=' {
                // We've hit `name=` for the *next* opcode. Walk back
                // to the last whitespace before this `=` — that
                // whitespace separates the previous value from the
                // next opcode's name.
                let mut back = k;
                while back > value_start && !bytes[back - 1].is_ascii_whitespace() {
                    back -= 1;
                }
                if back > value_start {
                    // back-1 is whitespace; the value ends at the
                    // last non-ws byte before that whitespace.
                    let mut end = back - 1;
                    while end > value_start && bytes[end - 1].is_ascii_whitespace() {
                        end -= 1;
                    }
                    last_non_ws = end;
                    k = back; // resume tokenisation at the next opcode name
                    break;
                }
                // No whitespace between value_start and `=` — degenerate
                // (`name==`). Treat the second `=` literally.
                k += 1;
                last_non_ws = k;
                continue;
            }
            if !b.is_ascii_whitespace() {
                last_non_ws = k + 1;
            }
            k += 1;
        }
        let value = std::str::from_utf8(&bytes[value_start..last_non_ws])
            .map_err(|_| Error::invalid("SFZ: non-UTF-8 opcode value"))?
            .trim()
            .to_string();
        tokens.push(Token::Opcode(name, value));
        i = k;
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_via_extension() {
        let path = Path::new("/tmp/foo.sfz");
        assert!(looks_like_sfz(path, b""));
    }

    #[test]
    fn detects_via_section_header() {
        let path = Path::new("/tmp/foo.txt");
        let body = b"// my patch\n<region>\nsample=kick.wav\n";
        assert!(looks_like_sfz(path, body));
    }

    #[test]
    fn rejects_random_text() {
        let path = Path::new("/tmp/foo.txt");
        assert!(!looks_like_sfz(path, b"hello world\n"));
    }

    #[test]
    fn parse_key_handles_decimal_and_note_names() {
        assert_eq!(parse_key("60"), Some(60));
        assert_eq!(parse_key("0"), Some(0));
        assert_eq!(parse_key("127"), Some(127));
        assert_eq!(parse_key("128"), None);
        assert_eq!(parse_key("c4"), Some(60));
        assert_eq!(parse_key("C4"), Some(60));
        assert_eq!(parse_key("c#4"), Some(61));
        assert_eq!(parse_key("Db4"), Some(61));
        assert_eq!(parse_key("a4"), Some(69));
        assert_eq!(parse_key("c-1"), Some(0));
        assert_eq!(parse_key("g9"), Some(127));
        assert_eq!(parse_key("xyz"), None);
    }

    #[test]
    fn strip_comments_handles_line_and_block() {
        let src = "// hello\nfoo=1 /* bar=2 */ baz=3\n";
        let stripped = strip_comments(src);
        assert!(!stripped.contains("hello"));
        assert!(!stripped.contains("bar"));
        assert!(stripped.contains("foo=1"));
        assert!(stripped.contains("baz=3"));
    }

    #[test]
    fn loop_mode_parse() {
        assert_eq!(LoopMode::parse("no_loop"), LoopMode::NoLoop);
        assert_eq!(LoopMode::parse("one_shot"), LoopMode::OneShot);
        assert_eq!(LoopMode::parse("loop_continuous"), LoopMode::LoopContinuous);
        assert_eq!(LoopMode::parse("loop_sustain"), LoopMode::LoopSustain);
        assert_eq!(LoopMode::parse("nonsense"), LoopMode::NoLoop);
    }

    #[test]
    fn parses_minimal_region() {
        let text = "<region> sample=kick.wav key=36";
        let patch = parse_str(text).unwrap();
        assert_eq!(patch.regions.len(), 1);
        let r = &patch.regions[0];
        assert_eq!(r.sample_path.as_ref().unwrap(), Path::new("kick.wav"));
        assert_eq!(r.lokey, 36);
        assert_eq!(r.hikey, 36);
        assert_eq!(r.pitch_keycenter, 36);
    }

    #[test]
    fn parses_multiple_regions_inheriting_global() {
        let text = "
            <global> volume=-3
            <region> sample=a.wav key=60
            <region> sample=b.wav key=62
        ";
        let patch = parse_str(text).unwrap();
        assert_eq!(patch.regions.len(), 2);
        assert!((patch.regions[0].volume - -3.0).abs() < 1e-6);
        assert!((patch.regions[1].volume - -3.0).abs() < 1e-6);
        assert_eq!(patch.regions[0].lokey, 60);
        assert_eq!(patch.regions[1].lokey, 62);
    }

    #[test]
    fn group_overrides_global_then_region_overrides_group() {
        let text = "
            <global> volume=-12 pan=10
            <group>  volume=-6
            <region> sample=a.wav key=60 pan=-50
        ";
        let patch = parse_str(text).unwrap();
        let r = &patch.regions[0];
        assert_eq!(r.volume, -6.0); // group beats global
        assert_eq!(r.pan, -50.0); // region beats global
    }

    #[test]
    fn group_resets_when_a_new_group_starts() {
        let text = "
            <group> volume=-6
            <region> sample=a.wav key=60
            <group> volume=-12
            <region> sample=b.wav key=62
        ";
        let patch = parse_str(text).unwrap();
        assert_eq!(patch.regions[0].volume, -6.0);
        assert_eq!(patch.regions[1].volume, -12.0);
    }

    #[test]
    fn parses_lokey_hikey_and_velocity_range() {
        let text = "
            <region> sample=a.wav lokey=36 hikey=48 lovel=64 hivel=127 pitch_keycenter=40
        ";
        let patch = parse_str(text).unwrap();
        let r = &patch.regions[0];
        assert_eq!(r.lokey, 36);
        assert_eq!(r.hikey, 48);
        assert_eq!(r.lovel, 64);
        assert_eq!(r.hivel, 127);
        assert_eq!(r.pitch_keycenter, 40);
    }

    #[test]
    fn parses_loop_opcodes() {
        let text = "
            <region> sample=a.wav loop_start=128 loop_end=2048 loop_mode=loop_continuous
        ";
        let patch = parse_str(text).unwrap();
        let r = &patch.regions[0];
        assert_eq!(r.loop_start, Some(128));
        assert_eq!(r.loop_end, Some(2048));
        assert_eq!(r.loop_mode, LoopMode::LoopContinuous);
    }

    #[test]
    fn parses_control_default_path() {
        let text = "
            <control> default_path=samples/
            <region> sample=kick.wav key=36
        ";
        let patch = parse_str(text).unwrap();
        assert_eq!(
            patch.control.get("default_path").map(String::as_str),
            Some("samples/"),
        );
    }

    #[test]
    fn handles_comments_around_opcodes() {
        let text = "
            // file header comment
            <region> /* inline */ sample=foo.wav // trailing
              key=60
        ";
        let patch = parse_str(text).unwrap();
        assert_eq!(patch.regions.len(), 1);
        assert_eq!(
            patch.regions[0].sample_path.as_ref().unwrap(),
            Path::new("foo.wav")
        );
        assert_eq!(patch.regions[0].lokey, 60);
    }

    #[test]
    fn unknown_opcodes_preserved_in_map() {
        let text = "<region> sample=a.wav weird_opcode=42";
        let patch = parse_str(text).unwrap();
        let r = &patch.regions[0];
        assert_eq!(
            r.opcodes.get("weird_opcode").map(String::as_str),
            Some("42")
        );
    }

    #[test]
    fn rejects_include() {
        let text = "#include \"other.sfz\"\n<region> sample=a.wav";
        let err = parse_str(text).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn make_voice_without_loaded_samples_returns_unsupported() {
        // parse_str doesn't load sample bytes off disk → the round-1
        // voice generator must report a clear error rather than crash.
        let inst = SfzInstrument::parse_str("test", "<region> sample=a.wav key=60").unwrap();
        match inst.make_voice(0, 60, 100, 44_100) {
            Err(Error::Unsupported(_)) => {}
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("expected Err"),
        }
    }

    #[test]
    fn make_voice_no_matching_region_returns_unsupported() {
        let inst = SfzInstrument::parse_str("test", "<region> sample=a.wav key=60").unwrap();
        // Key out of range (region only matches key 60).
        match inst.make_voice(0, 30, 100, 44_100) {
            Err(Error::Unsupported(msg)) => assert!(msg.contains("no region")),
            Err(other) => panic!("expected Unsupported, got {other:?}"),
            Ok(_) => panic!("expected Err"),
        }
    }

    #[test]
    fn open_loads_sample_bytes_from_disk() {
        // Build a tiny SFZ + matching sample under tempdir.
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-test-open");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let sample = tmp.join("kick.bin");
        std::fs::write(&sample, b"AAAA").unwrap();
        let sfz = tmp.join("patch.sfz");
        std::fs::write(&sfz, b"<region> sample=kick.bin key=36\n").unwrap();
        let inst = SfzInstrument::open(&sfz).unwrap();
        assert_eq!(inst.regions().len(), 1);
        let r = &inst.regions()[0];
        assert_eq!(r.sample_bytes.as_deref(), Some(b"AAAA" as &[u8]));
        assert_eq!(r.sample_path.as_ref().unwrap(), &sample);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn open_resolves_default_path_subdirectory() {
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-test-defaultpath");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("samples")).unwrap();
        std::fs::write(tmp.join("samples").join("kick.bin"), b"BBBB").unwrap();
        let sfz = tmp.join("patch.sfz");
        std::fs::write(
            &sfz,
            b"<control> default_path=samples\n<region> sample=kick.bin key=36\n",
        )
        .unwrap();
        let inst = SfzInstrument::open(&sfz).unwrap();
        assert_eq!(
            inst.regions()[0].sample_bytes.as_deref(),
            Some(b"BBBB" as &[u8])
        );
        assert_eq!(
            inst.regions()[0].sample_path.as_ref().unwrap(),
            &tmp.join("samples").join("kick.bin"),
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn open_reports_missing_sample() {
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-test-missing");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let sfz = tmp.join("patch.sfz");
        std::fs::write(&sfz, b"<region> sample=nope.bin key=36\n").unwrap();
        match SfzInstrument::open(&sfz) {
            Err(Error::InvalidData(msg)) => {
                assert!(msg.contains("nope.bin"), "got {msg}");
            }
            Err(other) => panic!("expected InvalidData, got {other:?}"),
            Ok(_) => panic!("expected error opening missing-sample patch"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parses_full_template_smoke() {
        // Tutorial-style template from the SFZ docs.
        let text = "
            //------------------------------------------------------------------------------
            // A basic sfz template
            //------------------------------------------------------------------------------
            <control>
            default_path=

            <global>

            <group>
              fil_type=lpf_2p
              cutoff=440
              resonance=2

              trigger=attack
              loop_mode=no_loop

            <region> sample=a.wav lokey=0 hikey=59 pitch_keycenter=60
            <region> sample=b.wav lokey=60 hikey=127 pitch_keycenter=72
        ";
        let patch = parse_str(text).unwrap();
        assert_eq!(patch.regions.len(), 2);
        for r in &patch.regions {
            assert_eq!(r.trigger, "attack");
            assert_eq!(r.loop_mode, LoopMode::NoLoop);
            assert_eq!(r.opcodes.get("cutoff").map(String::as_str), Some("440"));
        }
        assert_eq!(patch.regions[0].pitch_keycenter, 60);
        assert_eq!(patch.regions[1].pitch_keycenter, 72);
    }

    /// Build a tiny WAV: 8 frames of i16 PCM, mono, given sample rate.
    fn build_test_wav(samples: &[i16], rate: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        let data_size = (samples.len() * 2) as u32;
        bytes.extend_from_slice(&(36u32 + data_size).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn make_voice_renders_pcm_via_disk_loaded_sfz() {
        // End-to-end: write SFZ + WAV to a tempdir, open the patch,
        // make a voice for the matching key, render some samples.
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-makevoice");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // 64-frame ramp from -16384 to +16384.
        let samples: Vec<i16> = (0..64).map(|i| (i * 512 - 16384) as i16).collect();
        let wav = build_test_wav(&samples, 22_050);
        let wav_path = tmp.join("ramp.wav");
        std::fs::write(&wav_path, &wav).unwrap();
        let sfz_path = tmp.join("patch.sfz");
        std::fs::write(
            &sfz_path,
            b"<region> sample=ramp.wav key=60 loop_mode=loop_continuous loop_start=0 loop_end=64\n",
        )
        .unwrap();
        let inst = SfzInstrument::open(&sfz_path).unwrap();
        let mut voice = inst.make_voice(0, 60, 100, 44_100).expect("voice");
        let mut buf = vec![0.0f32; 4096];
        let n = voice.render(&mut buf);
        assert_eq!(n, 4096, "looping voice should fill the buffer");
        let nonzero = buf.iter().filter(|s| s.abs() > 0.001).count();
        assert!(nonzero > 100, "expected non-silent: {nonzero}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn make_voice_pitch_shifts_by_key_offset() {
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-pitchshift");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Long-enough ramp to render 1024 samples without looping.
        let samples: Vec<i16> = (0..16_384)
            .map(|i| ((i % 256) * 128) as i16 - 16384)
            .collect();
        let wav = build_test_wav(&samples, 44_100);
        let wav_path = tmp.join("ramp.wav");
        std::fs::write(&wav_path, &wav).unwrap();
        let sfz_path = tmp.join("patch.sfz");
        std::fs::write(
            &sfz_path,
            b"<region> sample=ramp.wav lokey=0 hikey=127 pitch_keycenter=60\n",
        )
        .unwrap();
        let inst = SfzInstrument::open(&sfz_path).unwrap();
        // Voice at key=60: native pitch.
        let mut a = inst.make_voice(0, 60, 100, 44_100).unwrap();
        // Voice at key=72: one octave up, samples advance 2x faster.
        let mut b = inst.make_voice(0, 72, 100, 44_100).unwrap();
        let mut buf_a = vec![0.0f32; 1024];
        let mut buf_b = vec![0.0f32; 1024];
        a.render(&mut buf_a);
        b.render(&mut buf_b);
        // Count zero crossings — `b` should have ~2x as many as `a`.
        let cross_a = buf_a.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let cross_b = buf_b.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            cross_b > cross_a,
            "+1 octave key offset should produce more zero crossings: a={cross_a} b={cross_b}",
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn sample_paths_with_spaces_are_kept_intact() {
        // Spaces inside a sample path are preserved up to the next
        // opcode boundary.
        let text = "<region> sample=My Drum Kit/kick.wav key=36";
        let patch = parse_str(text).unwrap();
        assert_eq!(
            patch.regions[0].sample_path.as_ref().unwrap(),
            Path::new("My Drum Kit/kick.wav"),
        );
        assert_eq!(patch.regions[0].lokey, 36);
    }
}
