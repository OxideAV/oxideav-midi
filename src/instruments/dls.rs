//! DLS (Downloadable Sounds, Level 1/2) instrument-bank stub.
//!
//! DLS is a RIFF-based bank format like SF2; the magic is `RIFF`...
//! `DLS ` (note trailing space) at offset 8. Round-1 detects the magic;
//! round-2 will walk the `lins`/`ptbl`/`wvpl` chunk hierarchy and emit
//! voices.

use std::path::Path;

use oxideav_core::{Error, Result};

use super::{Instrument, Voice};

pub const RIFF_MAGIC: &[u8; 4] = b"RIFF";
/// Note the trailing space — DLS chunk identifiers are 4-char ASCII.
pub const DLS_MAGIC: &[u8; 4] = b"DLS ";

pub struct DlsInstrument {
    name: String,
}

impl DlsInstrument {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if !is_dls(&bytes) {
            return Err(Error::invalid(format!(
                "DLS: '{}' does not start with RIFF/DLS<space> magic",
                path.display(),
            )));
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "dls".to_string());
        Ok(Self { name })
    }
}

impl Instrument for DlsInstrument {
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
            "DLS sample fetch is pending — round-2 work. Use the pure-tone \
             fallback for now (instruments::tone::ToneInstrument).",
        ))
    }
}

pub fn is_dls(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == RIFF_MAGIC && &bytes[8..12] == DLS_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_riff_dls_magic() {
        let mut blob = vec![0u8; 32];
        blob[0..4].copy_from_slice(b"RIFF");
        blob[4..8].copy_from_slice(&24u32.to_le_bytes());
        blob[8..12].copy_from_slice(b"DLS ");
        assert!(is_dls(&blob));
    }

    #[test]
    fn rejects_wrong_magic() {
        assert!(!is_dls(b""));
        let mut blob = vec![0u8; 16];
        blob[0..4].copy_from_slice(b"RIFF");
        blob[8..12].copy_from_slice(b"WAVE");
        assert!(!is_dls(&blob));
    }
}
