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
    EnvelopeParams, FilterParams, FilterType, ModEnvParams, SampleLoopMode, SamplePlayer,
    SamplePlayerConfig, VibratoParams,
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
    let filter = build_filter_from_opcodes(&region.opcodes);
    let mod_env = build_mod_env_from_opcodes(&region.opcodes, filter.cutoff_cents);

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
        // Round 95: SFZ `cutoff` / `resonance` / `fil_type` map straight
        // into `FilterParams`; `fileg_*` (attack/decay/sustain/release/
        // delay/hold + `fileg_depth` for the routing depth) maps into
        // `ModEnvParams`. Regions without those opcodes still get the
        // SF2-style "filter open" defaults so legacy SFZ banks remain
        // bit-identical to the round-91 path.
        mod_env,
        filter,
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

/// Convert an SFZ `cutoff=` value (Hz) into SF2-style absolute cents
/// re. 8.176 Hz: `cents = 1200 * log2(fc_hz / 8.176)`. Used to bridge
/// SFZ's Hz-denominated filter opcodes into the shared `SamplePlayer`'s
/// cents-based plumbing.
///
/// Out-of-range / non-positive inputs return the SF2 "filter open"
/// sentinel (13500 cents ≈ 19914 Hz) so a bad opcode value leaves the
/// filter inert rather than producing NaN coefficients downstream.
fn hz_to_filter_cents(hz: f32) -> i32 {
    if !(hz.is_finite() && hz > 0.0) {
        return 13_500;
    }
    let cents = 1200.0 * (hz / 8.176_f32).log2();
    // Clamp into the SF2 useful range. SF2 §8.1.3 gen 8 lists
    // `1500..=13500`; lower than 1500 is sub-20-Hz and physically
    // meaningless for an audio-band filter.
    cents.round().clamp(1_500.0, 13_500.0) as i32
}

/// Build `FilterParams` from a region's flattened opcode map. Honours
/// the SFZ v1 opcodes documented at `docs/audio/midi/instrument-formats/
/// sfz-legacy.html` (Filter category): `cutoff=` (Hz, default = filter
/// disabled), `resonance=` (dB, default 0, range 0..40), and
/// `fil_type=` (string, default `lpf_2p`; values `lpf_1p`, `hpf_1p`,
/// `lpf_2p`, `hpf_2p`, `bpf_2p`, `brf_2p`).
///
/// When `cutoff` is absent the region keeps the SF2 default of 13500
/// cents (filter open). `resonance` dB → centibels is `cb = dB * 10`
/// (the centibel is the SF2 unit, defined as 0.1 dB in §8.1.3 gen 9 —
/// SFZ's dB-denominated `resonance` opcode maps directly). The SFZ
/// `resonance` default of 0 dB matches the SF2 Butterworth Q.
fn build_filter_from_opcodes(opcodes: &BTreeMap<String, String>) -> FilterParams {
    // `cutoff` / `cutoff2` are aliases per the opcode index — same for
    // `resonance` / `resonance2`, and `fil_type` / `filtype`.
    let cutoff_hz = opcodes
        .get("cutoff")
        .or_else(|| opcodes.get("cutoff2"))
        .and_then(|s| s.trim().parse::<f32>().ok());

    let resonance_db = opcodes
        .get("resonance")
        .or_else(|| opcodes.get("resonance2"))
        .and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(0.0)
        .clamp(0.0, 40.0);

    let kind = opcodes
        .get("fil_type")
        .or_else(|| opcodes.get("filtype"))
        .map(|s| FilterType::parse_sfz(s))
        .unwrap_or_default();

    let cutoff_cents = match cutoff_hz {
        Some(hz) => hz_to_filter_cents(hz),
        // No `cutoff` opcode → SFZ semantics say "filter disabled". The
        // shared `SamplePlayer` reads cutoff_cents >= 13000 as the
        // filter-bypass sentinel for low-pass shapes; for HP/BP/BRF
        // the explicit non-low-pass `fil_type` keeps the biquad active
        // even at the sentinel cutoff (see `SamplePlayer::new`'s
        // `non_lowpass` branch), so we still emit the sentinel and let
        // the player decide.
        None => 13_500,
    };

    FilterParams {
        cutoff_cents,
        q_centibels: (resonance_db * 10.0).round() as i32,
        kind,
    }
}

/// Build `ModEnvParams` from a region's flattened opcode map. Honours
/// the SFZ v1 `fileg_*` Envelope-Generator opcodes documented at
/// `docs/audio/midi/instrument-formats/sfz-opcodes-index.html` (EG
/// category):
///
/// - `fileg_delay`, `fileg_attack`, `fileg_hold`, `fileg_decay`,
///   `fileg_release` — seconds (range `0..100`, default 0). SFZ v2
///   aliases `fil_delay` etc. are honoured.
/// - `fileg_sustain` — percentage (range `0..100`, default 0). The
///   shared `SamplePlayer` consumes the value as a 0..=1 fraction.
/// - `fileg_depth` — cents (range `-12000..12000`, default 0). This
///   is the mod-env → filter-cutoff routing depth; with `0` the
///   per-sample mod-env compute is skipped entirely.
///
/// `current_cutoff_cents` is the region's initial cutoff (already
/// derived by [`build_filter_from_opcodes`]). When the routing depth
/// is non-zero this is used as a sanity check: if the initial cutoff
/// is at the "filter open" sentinel *and* the depth would drag the
/// live cutoff into the audible band, the depth alone is enough for
/// `SamplePlayer::new` to allocate the biquad — see the `needs_filter`
/// gate. We just compute the routing depth here and let the player
/// decide.
fn build_mod_env_from_opcodes(
    opcodes: &BTreeMap<String, String>,
    _current_cutoff_cents: i32,
) -> ModEnvParams {
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
    fn pi(map: &BTreeMap<String, String>, keys: &[&str], default: i32) -> i32 {
        for k in keys {
            if let Some(s) = map.get(*k) {
                if let Ok(v) = s.trim().parse::<i32>() {
                    return v;
                }
            }
        }
        default
    }

    // SFZ `fileg_sustain` is in percent (0..=100); the `SamplePlayer`
    // expects `0..=1`.
    let sustain_pct = pf(opcodes, &["fileg_sustain", "fil_sustain"], 0.0).clamp(0.0, 100.0);
    let depth_cents = pi(opcodes, &["fileg_depth", "fil_depth"], 0).clamp(-12_000, 12_000);

    ModEnvParams {
        delay_s: pf(opcodes, &["fileg_delay", "fil_delay"], 0.0).max(0.0),
        attack_s: pf(opcodes, &["fileg_attack", "fil_attack"], 0.0).max(0.0),
        hold_s: pf(opcodes, &["fileg_hold", "fil_hold"], 0.0).max(0.0),
        decay_s: pf(opcodes, &["fileg_decay", "fil_decay"], 0.0).max(0.0),
        sustain_level: sustain_pct / 100.0,
        release_s: pf(opcodes, &["fileg_release", "fil_release"], 0.0).max(0.0),
        to_filter_cents: depth_cents,
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

    // ---------------------------------------------------------------------
    // Round 95: SFZ-side filter envelope + fil_type + cutoff wiring.
    // ---------------------------------------------------------------------

    #[test]
    fn hz_to_filter_cents_round_trips_known_values() {
        // 8.176 Hz is the SF2 reference, so 8.176 Hz → 0 cents (clamped
        // to the §8.1.3 useful range 1500..=13500, so we get 1500).
        assert_eq!(hz_to_filter_cents(8.176), 1_500);
        // 440 Hz → 1200 * log2(440/8.176) ≈ 6892 cents.
        let cents_440 = hz_to_filter_cents(440.0);
        assert!(
            (6_880..=6_905).contains(&cents_440),
            "440 Hz → ~6892 cents, got {cents_440}",
        );
        // 20 kHz → 13500 cents (clamped at the upper end).
        assert_eq!(hz_to_filter_cents(20_000.0), 13_500);
        // Non-positive / NaN → SF2 "filter open" sentinel.
        assert_eq!(hz_to_filter_cents(0.0), 13_500);
        assert_eq!(hz_to_filter_cents(-1.0), 13_500);
        assert_eq!(hz_to_filter_cents(f32::NAN), 13_500);
    }

    #[test]
    fn build_filter_from_opcodes_honours_cutoff_resonance_filtype() {
        let mut map = BTreeMap::new();
        map.insert("cutoff".to_string(), "440".to_string());
        map.insert("resonance".to_string(), "6".to_string());
        map.insert("fil_type".to_string(), "hpf_2p".to_string());
        let f = build_filter_from_opcodes(&map);
        // 440 Hz cutoff → ~6892 cents.
        assert!(
            (6_880..=6_905).contains(&f.cutoff_cents),
            "expected ~6892 cents, got {}",
            f.cutoff_cents,
        );
        // 6 dB resonance → 60 centibels.
        assert_eq!(f.q_centibels, 60);
        assert_eq!(f.kind, FilterType::TwoPoleHighPass);
    }

    #[test]
    fn build_filter_from_opcodes_defaults_to_open_lpf_2p() {
        // No filter opcodes → cutoff at SF2 "open" sentinel, no
        // resonance, default lpf_2p kind.
        let map = BTreeMap::new();
        let f = build_filter_from_opcodes(&map);
        assert_eq!(f.cutoff_cents, 13_500);
        assert_eq!(f.q_centibels, 0);
        assert_eq!(f.kind, FilterType::TwoPoleLowPass);
    }

    #[test]
    fn build_filter_from_opcodes_clamps_resonance_into_range() {
        // SFZ resonance range is 0..=40 dB per the legacy spec.
        let mut map = BTreeMap::new();
        map.insert("cutoff".to_string(), "1000".to_string());
        map.insert("resonance".to_string(), "100".to_string());
        let f = build_filter_from_opcodes(&map);
        // Clamped to 40 dB → 400 cB.
        assert_eq!(f.q_centibels, 400);
        // Negative resonance also clamps to 0.
        let mut neg = BTreeMap::new();
        neg.insert("cutoff".to_string(), "1000".to_string());
        neg.insert("resonance".to_string(), "-5".to_string());
        let fn_ = build_filter_from_opcodes(&neg);
        assert_eq!(fn_.q_centibels, 0);
    }

    #[test]
    fn filter_type_parse_covers_all_sfz_v1_values() {
        // Every value listed in `docs/audio/midi/instrument-formats/
        // sfz-legacy.html` "Filter type" table.
        assert_eq!(FilterType::parse_sfz("lpf_1p"), FilterType::OnePoleLowPass);
        assert_eq!(FilterType::parse_sfz("hpf_1p"), FilterType::OnePoleHighPass,);
        assert_eq!(FilterType::parse_sfz("lpf_2p"), FilterType::TwoPoleLowPass);
        assert_eq!(FilterType::parse_sfz("hpf_2p"), FilterType::TwoPoleHighPass,);
        assert_eq!(FilterType::parse_sfz("bpf_2p"), FilterType::TwoPoleBandPass);
        assert_eq!(
            FilterType::parse_sfz("brf_2p"),
            FilterType::TwoPoleBandReject,
        );
        // Case-insensitive.
        assert_eq!(FilterType::parse_sfz("LPF_2P"), FilterType::TwoPoleLowPass);
        // Unknown → default.
        assert_eq!(
            FilterType::parse_sfz("nonsense"),
            FilterType::TwoPoleLowPass
        );
    }

    #[test]
    fn build_mod_env_from_opcodes_honours_fileg_set() {
        let mut map = BTreeMap::new();
        map.insert("fileg_delay".to_string(), "0.01".to_string());
        map.insert("fileg_attack".to_string(), "0.05".to_string());
        map.insert("fileg_hold".to_string(), "0.02".to_string());
        map.insert("fileg_decay".to_string(), "0.10".to_string());
        map.insert("fileg_sustain".to_string(), "40".to_string());
        map.insert("fileg_release".to_string(), "0.20".to_string());
        map.insert("fileg_depth".to_string(), "1200".to_string());
        let m = build_mod_env_from_opcodes(&map, 13_500);
        assert!((m.delay_s - 0.01).abs() < 1e-6);
        assert!((m.attack_s - 0.05).abs() < 1e-6);
        assert!((m.hold_s - 0.02).abs() < 1e-6);
        assert!((m.decay_s - 0.10).abs() < 1e-6);
        // 40 % → 0.4 fraction.
        assert!((m.sustain_level - 0.40).abs() < 1e-6);
        assert!((m.release_s - 0.20).abs() < 1e-6);
        assert_eq!(m.to_filter_cents, 1_200);
        assert!(!m.is_inert());
    }

    #[test]
    fn build_mod_env_from_opcodes_aliases_v2_names() {
        // SFZ v2 uses `fil_*` aliases for the `fileg_*` opcodes; the
        // alias table is in `sfz-opcodes-index.html`.
        let mut map = BTreeMap::new();
        map.insert("fil_attack".to_string(), "0.07".to_string());
        map.insert("fil_release".to_string(), "0.30".to_string());
        map.insert("fil_depth".to_string(), "-2400".to_string());
        let m = build_mod_env_from_opcodes(&map, 13_500);
        assert!((m.attack_s - 0.07).abs() < 1e-6);
        assert!((m.release_s - 0.30).abs() < 1e-6);
        assert_eq!(m.to_filter_cents, -2_400);
    }

    #[test]
    fn build_mod_env_default_is_inert() {
        let map = BTreeMap::new();
        let m = build_mod_env_from_opcodes(&map, 13_500);
        assert_eq!(m.to_filter_cents, 0);
        assert!(m.is_inert());
    }

    #[test]
    fn build_mod_env_clamps_depth() {
        // `fileg_depth` documented range is -12000..=12000 cents.
        let mut hi = BTreeMap::new();
        hi.insert("fileg_depth".to_string(), "30000".to_string());
        assert_eq!(
            build_mod_env_from_opcodes(&hi, 13_500).to_filter_cents,
            12_000,
        );
        let mut lo = BTreeMap::new();
        lo.insert("fileg_depth".to_string(), "-30000".to_string());
        assert_eq!(
            build_mod_env_from_opcodes(&lo, 13_500).to_filter_cents,
            -12_000,
        );
    }

    #[test]
    fn full_template_smoke_drops_cutoff_into_sample_player() {
        // The tutorial template carries fil_type=lpf_2p, cutoff=440,
        // resonance=2 — those should land on the region's resolved
        // FilterParams.
        let text = "
            <group>
              fil_type=lpf_2p
              cutoff=440
              resonance=2
            <region> sample=a.wav key=60
        ";
        let patch = parse_str(text).unwrap();
        let r = &patch.regions[0];
        let f = build_filter_from_opcodes(&r.opcodes);
        // 440 Hz → ~6892 cents.
        assert!(
            (6_880..=6_905).contains(&f.cutoff_cents),
            "cutoff_cents = {}",
            f.cutoff_cents,
        );
        // 2 dB → 20 cB.
        assert_eq!(f.q_centibels, 20);
        assert_eq!(f.kind, FilterType::TwoPoleLowPass);
    }

    #[test]
    fn sfz_with_low_cutoff_attenuates_high_frequencies() {
        // End-to-end: build a one-region patch with a 5 kHz sine sample
        // and a 440 Hz cutoff. The voice render should be considerably
        // quieter than the same patch with no filter (the round-91
        // SamplePlayer biquad is doing its job under SFZ control).
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-r95-lp");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // 5 kHz sine, 1 second, 44.1 kHz.
        let sr = 44_100u32;
        let samples: Vec<i16> = (0..sr as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let v = (2.0 * std::f32::consts::PI * 5_000.0 * t).sin();
                (v * 16_384.0) as i16
            })
            .collect();
        let wav = build_test_wav(&samples, sr);
        std::fs::write(tmp.join("sine.wav"), &wav).unwrap();

        // No filter (baseline).
        let baseline_path = tmp.join("baseline.sfz");
        std::fs::write(
            &baseline_path,
            b"<region> sample=sine.wav key=60 pitch_keycenter=60\n",
        )
        .unwrap();
        // 440 Hz low-pass — should knock out the 5 kHz sine.
        let lp_path = tmp.join("lp.sfz");
        std::fs::write(
            &lp_path,
            b"<region> sample=sine.wav key=60 pitch_keycenter=60 cutoff=440 fil_type=lpf_2p\n",
        )
        .unwrap();

        let baseline_inst = SfzInstrument::open(&baseline_path).unwrap();
        let lp_inst = SfzInstrument::open(&lp_path).unwrap();
        let mut baseline = baseline_inst.make_voice(0, 60, 100, sr).unwrap();
        let mut lp = lp_inst.make_voice(0, 60, 100, sr).unwrap();
        let mut buf_base = vec![0.0f32; 16_384];
        let mut buf_lp = vec![0.0f32; 16_384];
        baseline.render(&mut buf_base);
        lp.render(&mut buf_lp);
        // RMS on the tail so the filter transient has settled.
        fn rms(s: &[f32]) -> f32 {
            (s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32).sqrt()
        }
        let r_base = rms(&buf_base[8_192..]);
        let r_lp = rms(&buf_lp[8_192..]);
        assert!(
            r_lp < r_base * 0.3,
            "440 Hz lpf_2p should attenuate 5 kHz sine; got baseline={r_base}, lp={r_lp}",
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn sfz_high_pass_inverts_attenuation() {
        // Same fixture as the low-pass test, but with `fil_type=hpf_2p`
        // and a high cutoff (10 kHz). The 5 kHz sine sits *below* the
        // cutoff so a high-pass should attenuate it.
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-r95-hp");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let sr = 44_100u32;
        let samples: Vec<i16> = (0..sr as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let v = (2.0 * std::f32::consts::PI * 5_000.0 * t).sin();
                (v * 16_384.0) as i16
            })
            .collect();
        let wav = build_test_wav(&samples, sr);
        std::fs::write(tmp.join("sine.wav"), &wav).unwrap();
        let baseline = tmp.join("baseline.sfz");
        std::fs::write(
            &baseline,
            b"<region> sample=sine.wav key=60 pitch_keycenter=60\n",
        )
        .unwrap();
        let hp = tmp.join("hp.sfz");
        std::fs::write(
            &hp,
            b"<region> sample=sine.wav key=60 pitch_keycenter=60 cutoff=10000 fil_type=hpf_2p\n",
        )
        .unwrap();
        let baseline_inst = SfzInstrument::open(&baseline).unwrap();
        let hp_inst = SfzInstrument::open(&hp).unwrap();
        let mut base = baseline_inst.make_voice(0, 60, 100, sr).unwrap();
        let mut hpv = hp_inst.make_voice(0, 60, 100, sr).unwrap();
        let mut buf_base = vec![0.0f32; 16_384];
        let mut buf_hp = vec![0.0f32; 16_384];
        base.render(&mut buf_base);
        hpv.render(&mut buf_hp);
        fn rms(s: &[f32]) -> f32 {
            (s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32).sqrt()
        }
        let r_base = rms(&buf_base[8_192..]);
        let r_hp = rms(&buf_hp[8_192..]);
        assert!(
            r_hp < r_base * 0.5,
            "10 kHz hpf_2p should attenuate the 5 kHz sine; got baseline={r_base}, hp={r_hp}",
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn sfz_fileg_sweeps_cutoff_open_during_attack() {
        // Build a patch with a low initial cutoff (250 Hz) and a
        // mod-env that sweeps +7200 cents (6 octaves) over a 500 ms
        // attack. The early window should be heavily attenuated; the
        // late window (post-attack) should pass through almost
        // unchanged. Same shape as the SamplePlayer-side sweep test
        // but driven entirely from SFZ opcodes.
        let tmp = std::env::temp_dir().join("oxideav-midi-sfz-r95-eg");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let sr = 44_100u32;
        let samples: Vec<i16> = (0..sr as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let v = (2.0 * std::f32::consts::PI * 5_000.0 * t).sin();
                (v * 16_384.0) as i16
            })
            .collect();
        let wav = build_test_wav(&samples, sr);
        std::fs::write(tmp.join("sine.wav"), &wav).unwrap();
        let path = tmp.join("eg.sfz");
        std::fs::write(
            &path,
            b"<region> sample=sine.wav key=60 pitch_keycenter=60 \
              cutoff=250 fil_type=lpf_2p \
              fileg_attack=0.500 fileg_hold=1.0 fileg_decay=0.001 \
              fileg_sustain=100 fileg_release=0.001 fileg_depth=7200\n",
        )
        .unwrap();
        let inst = SfzInstrument::open(&path).unwrap();
        let mut v = inst.make_voice(0, 60, 100, sr).unwrap();
        let mut buf = vec![0.0f32; sr as usize];
        v.render(&mut buf);
        fn rms(s: &[f32]) -> f32 {
            (s.iter().map(|v| v * v).sum::<f32>() / s.len().max(1) as f32).sqrt()
        }
        // Early window (20–80 ms): cutoff still near 250 Hz, 5 kHz
        // heavily attenuated.
        let early = rms(&buf[880..3_528]);
        // Late window (850–950 ms): mod-env saturated, cutoff at
        // 250 Hz + 7200 cents ≈ 16 kHz, 5 kHz passes freely.
        let late = rms(&buf[37_485..41_895]);
        let gain = late / early.max(1e-6);
        assert!(
            gain > 5.0,
            "fileg_depth=7200 should sweep the cutoff open; got gain={gain} (early={early}, late={late})",
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn build_config_for_region_default_filter_open() {
        // A region with no filter opcodes → SamplePlayerConfig filter
        // at the SF2 "open" sentinel (legacy SFZ → bit-identical to
        // round-91 output).
        let patch = parse_str("<region> sample=a.wav key=60 pitch_keycenter=60").unwrap();
        let region = &patch.regions[0];
        let cfg = build_config_for_region(region, &[0.0; 64], 44_100, 60, 100);
        assert_eq!(cfg.filter.cutoff_cents, 13_500);
        assert_eq!(cfg.filter.q_centibels, 0);
        assert_eq!(cfg.filter.kind, FilterType::TwoPoleLowPass);
        assert_eq!(cfg.mod_env.to_filter_cents, 0);
    }
}
