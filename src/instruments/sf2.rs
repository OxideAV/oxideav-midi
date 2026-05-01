//! SoundFont 2 (`.sf2`) instrument-bank stub.
//!
//! Round-1: detects the RIFF/SFBK signature so callers can confirm a
//! file is a SoundFont before trying to load it, but actual sample
//! fetching returns `Error::Unsupported("SF2 sample fetch pending")`.
//!
//! Round-2 will fill in the RIFF chunk walker (LIST/INFO + sdta-smpl
//! plus pdta presets/instruments/samples) and produce voices that play
//! the matching sample for a given MIDI program + key + velocity.
//!
//! On-disk layout (informally — full spec is the SoundFont 2.04 PDF):
//!
//! ```text
//!   "RIFF" <u32 LE size> "sfbk"
//!     "LIST" <u32 LE size> "INFO" ...   -- bank metadata
//!     "LIST" <u32 LE size> "sdta" ...   -- sample data ("smpl" sub-chunk)
//!     "LIST" <u32 LE size> "pdta" ...   -- preset / instrument / sample headers
//! ```

use std::path::Path;

use oxideav_core::{Error, Result};

use super::{Instrument, Voice};

/// Magic bytes at offset 0/8 of every SoundFont 2 file.
pub const RIFF_MAGIC: &[u8; 4] = b"RIFF";
pub const SFBK_MAGIC: &[u8; 4] = b"sfbk";

/// Round-1 SF2 stub. Holds the path; loading happens in round-2.
pub struct Sf2Instrument {
    name: String,
}

impl Sf2Instrument {
    /// Open a `.sf2` file. Verifies the magic bytes and returns an
    /// `Sf2Instrument` without parsing the bank (round-2). Surfaces an
    /// `Error::Unsupported` from `make_voice` until that lands.
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if !is_sf2(&bytes) {
            return Err(Error::invalid(format!(
                "SF2: '{}' does not start with RIFF/sfbk magic",
                path.display(),
            )));
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "sf2".to_string());
        Ok(Self { name })
    }
}

impl Instrument for Sf2Instrument {
    fn name(&self) -> &str {
        &self.name
    }

    fn make_voice(
        &self,
        _program: u8,
        _key: u8,
        _velocity: u8,
        _sample_rate: u32,
    ) -> Result<Box<dyn Voice>> {
        Err(Error::unsupported(
            "SF2 sample fetch is pending — round-2 work. Use the pure-tone \
             fallback for now (instruments::tone::ToneInstrument).",
        ))
    }
}

/// Magic-bytes detector. Returns true if `bytes` looks like a SoundFont
/// 2 file (RIFF/sfbk header). Cheap enough to call on every candidate
/// before trying to parse.
pub fn is_sf2(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == RIFF_MAGIC && &bytes[8..12] == SFBK_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_riff_sfbk_magic() {
        let mut blob = vec![0u8; 32];
        blob[0..4].copy_from_slice(b"RIFF");
        blob[4..8].copy_from_slice(&24u32.to_le_bytes());
        blob[8..12].copy_from_slice(b"sfbk");
        assert!(is_sf2(&blob));
    }

    #[test]
    fn rejects_wrong_magic() {
        assert!(!is_sf2(b""));
        assert!(!is_sf2(b"RIFF\x00\x00\x00\x00WAVE...."));
    }
}
