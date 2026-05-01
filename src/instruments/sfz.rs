//! SFZ (text patch format) instrument-bank stub.
//!
//! SFZ is a plain-text format: line-delimited opcode-value pairs grouped
//! into `<region>` / `<group>` / `<global>` sections, all referring to
//! external sample files (typically `.wav`). There is no fixed magic
//! number; convention is the file extension `.sfz` plus a comment or
//! `<region>` header in the first line.
//!
//! Round-1: detects "looks like SFZ" via the extension + first-line
//! probe; actual parsing returns `Error::Unsupported`.

use std::path::Path;

use oxideav_core::{Error, Result};

use super::{Instrument, Voice};

/// Round-1 SFZ stub. Like the SF2 variant, this stores the path and
/// fails on `make_voice` until round-2 lands the parser + sample
/// loader.
pub struct SfzInstrument {
    name: String,
}

impl SfzInstrument {
    /// Open a `.sfz` patch. Confirms the file is plausibly SFZ-shaped
    /// (extension match + UTF-8 decode) and returns the stub.
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if !looks_like_sfz(path, &bytes) {
            return Err(Error::invalid(format!(
                "SFZ: '{}' does not look like an SFZ patch \
                 (no <region>/<group>/<global> header in first 4 KiB)",
                path.display(),
            )));
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "sfz".to_string());
        Ok(Self { name })
    }
}

impl Instrument for SfzInstrument {
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
            "SFZ sample fetch is pending — round-2 work. Use the pure-tone \
             fallback for now (instruments::tone::ToneInstrument).",
        ))
    }
}

/// Heuristic detection: SFZ has no magic byte. We require either the
/// `.sfz` extension OR a recognisable section header (`<region>`,
/// `<group>`, `<global>`, `<control>`, `<master>`, `<curve>`,
/// `<effect>`, `<midi>`) in the first 4 KiB of the file.
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
    ]
    .iter()
    .any(|tag| head_str.contains(tag))
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
}
