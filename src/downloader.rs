//! Default-SoundFont downloader.
//!
//! Round-1: stubbed. The intended bank is **TimGM6mb** (Tim Brechbill's
//! GM-compatible SoundFont, ~6 MB, GPL-2.0-licensed) — small enough to
//! be a reasonable "first run" download for users who don't already
//! have a SoundFont installed. Pulling in `oxideav-http` as a hard
//! dependency for one call is heavier than this stage warrants, so the
//! actual fetch is deferred to a follow-up round; this module fixes
//! the URL + hash + file name so the integration is mechanical when
//! the time comes.
//!
//! Round-2 plan:
//!
//! 1. Optional dependency on `oxideav-http` (workspace dep, already
//!    pure-Rust + rustls).
//! 2. `download_default_soundfont(dest)` does:
//!    * If `dest` exists and its SHA-256 matches [`TIMGM6MB_SHA256`],
//!      return immediately.
//!    * Otherwise, GET [`TIMGM6MB_URL`] via `oxideav_http::open_http`,
//!      stream into `dest`, hash on-the-fly, and reject with
//!      `Error::InvalidData` if the hash doesn't match.
//! 3. Surface progress via a callback or, more likely, hook into
//!    `oxideav-source`'s `Read + Seek` plumbing so it Just Works.

use std::path::{Path, PathBuf};

use oxideav_core::{Error, Result};

/// URL hosting the canonical TimGM6mb SoundFont. Mirrored from the
/// upstream MIDI.org / Sourceforge page; we re-host on samples.oxideav.org
/// in round-2 to avoid availability flakiness from third-party hosts.
pub const TIMGM6MB_URL: &str = "https://samples.oxideav.org/instruments/sf2/TimGM6mb.sf2";

/// SHA-256 of the canonical TimGM6mb.sf2 (uppercase hex). Fixed at
/// round-2 once the mirrored file is finalised; the placeholder string
/// `"PENDING"` makes any premature use fail loudly.
pub const TIMGM6MB_SHA256: &str = "PENDING";

/// On-disk filename for the default SoundFont. Round-2 will write this
/// into `dest_dir.join(TIMGM6MB_FILENAME)`.
pub const TIMGM6MB_FILENAME: &str = "TimGM6mb.sf2";

/// Round-1 stub. Always returns `Error::Unsupported` until the
/// `oxideav-http` integration is wired up. Documented behaviour for
/// round-2 lives in the module-level rustdoc.
///
/// `dest` should be either a file path (full target name) or a directory
/// (we'll append [`TIMGM6MB_FILENAME`]); for now the parameter is
/// reserved.
pub fn download_default_soundfont(_dest: &Path) -> Result<PathBuf> {
    Err(Error::unsupported(format!(
        "oxideav-midi downloader pending; round-2 will fetch {TIMGM6MB_URL} \
         (sha256={TIMGM6MB_SHA256}) via oxideav-http",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_returns_unsupported() {
        let err = download_default_soundfont(Path::new("/tmp/x")).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn url_filename_round_trip() {
        // Sanity: the URL ends with the documented filename.
        assert!(TIMGM6MB_URL.ends_with(TIMGM6MB_FILENAME));
    }
}
