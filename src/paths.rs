//! Cross-platform search paths for soft-synth instrument banks.
//!
//! oxideav-midi never bundles instruments — the consumer (or the user's
//! OS) is expected to have a SoundFont 2 (`.sf2` / `.sf3`), an SFZ
//! patch (`.sfz`), or a DLS bank (`.dls`) installed somewhere on disk.
//! This module provides:
//!
//! * [`soundfont_search_paths`] — the per-OS list of directories the
//!   crate looks in by default.
//! * [`find_soundfonts`] — scan all of them and return every
//!   instrument-bank file found.
//! * [`find_first_soundfont`] — convenience wrapper that returns the
//!   first match (the typical "give me a SoundFont, any SoundFont" need).
//!
//! The user can override / extend the search list via the
//! `OXIDEAV_SOUNDFONT_PATH` environment variable, which is parsed
//! `PATH`-style: colon-separated on Unix, semicolon-separated on
//! Windows. Entries from this variable are searched **before** the
//! built-in defaults, so installed-on-the-system banks lose to a
//! user-curated explicit one.

use std::env;
use std::path::{Path, PathBuf};

/// Filename extensions we treat as instrument banks. Ordered by our
/// preference for round-2 implementation: SoundFont first, then SFZ,
/// then DLS.
pub const INSTRUMENT_EXTENSIONS: &[&str] = &["sf2", "sf3", "sfz", "dls"];

/// Environment variable that lets users prepend custom search
/// directories. Parsed `PATH`-style.
pub const ENV_VAR: &str = "OXIDEAV_SOUNDFONT_PATH";

/// Per-OS directories the crate checks when no explicit path is given.
///
/// The list is returned in lookup order — env-var overrides first,
/// then platform-default system dirs, then user-home dirs. Non-existent
/// paths are still returned (the caller filters when scanning).
pub fn soundfont_search_paths() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    // Env override always wins.
    if let Some(env_path) = env::var_os(ENV_VAR) {
        let sep = if cfg!(windows) { ';' } else { ':' };
        let env_str = env_path.to_string_lossy().into_owned();
        for entry in env_str.split(sep) {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                continue;
            }
            dirs.push(PathBuf::from(trimmed));
        }
    }

    // Platform-default system + user dirs.
    if cfg!(target_os = "macos") {
        dirs.push(PathBuf::from("/Library/Audio/Sounds/Banks"));
        if let Some(home) = env::var_os("HOME") {
            let mut p = PathBuf::from(home);
            p.push("Library/Audio/Sounds/Banks");
            dirs.push(p);
        }
    } else if cfg!(target_os = "windows") {
        dirs.push(PathBuf::from(r"C:\soundfonts"));
        if let Some(appdata) = env::var_os("APPDATA") {
            let mut p = PathBuf::from(appdata);
            p.push("soundfonts");
            dirs.push(p);
        }
    } else {
        // Treat everything else as a Linux/BSD/Unix-like layout.
        dirs.push(PathBuf::from("/usr/share/sounds/sf2"));
        dirs.push(PathBuf::from("/usr/share/sounds/sf3"));
        dirs.push(PathBuf::from("/usr/share/soundfonts"));
        if let Some(home) = env::var_os("HOME") {
            let mut p = PathBuf::from(home);
            p.push(".local/share/sounds/sf2");
            dirs.push(p);
        }
    }

    dirs
}

/// Walk every default search directory and collect every instrument-
/// bank file (`.sf2` / `.sf3` / `.sfz` / `.dls`) found in the top
/// level of each. Sub-directories are not recursed — instrument banks
/// are conventionally one-file-per-directory and recursive scans on
/// `~/Library` get expensive on a fresh macOS box.
///
/// Files are returned in `(directory, filename)` discovery order. Ties
/// inside a directory follow the OS's `read_dir` order, which is
/// platform-dependent — caller sort if you want determinism.
pub fn find_soundfonts() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for dir in soundfont_search_paths() {
        scan_dir(&dir, &mut found);
    }
    found
}

/// First match from [`find_soundfonts`], or `None` if no instrument
/// bank is installed. Callers needing a pure-tone fallback should
/// pivot to [`crate::instruments::tone`] when this returns `None`.
pub fn find_first_soundfont() -> Option<PathBuf> {
    find_soundfonts().into_iter().next()
}

fn scan_dir(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let ext_lc = ext.to_ascii_lowercase();
        if INSTRUMENT_EXTENSIONS.iter().any(|e| *e == ext_lc) {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn search_paths_are_nonempty() {
        let paths = soundfont_search_paths();
        assert!(!paths.is_empty(), "default search paths must not be empty");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macos_paths_include_library_banks() {
        let paths = soundfont_search_paths();
        assert!(
            paths
                .iter()
                .any(|p| p == Path::new("/Library/Audio/Sounds/Banks")),
            "expected /Library/Audio/Sounds/Banks in the search list, got {:?}",
            paths,
        );
    }

    #[test]
    fn env_var_entries_come_first() {
        // Save / restore env so we don't leak state across tests.
        let prev = env::var_os(ENV_VAR);
        let sep = if cfg!(windows) { ";" } else { ":" };
        let custom = format!("/tmp/oxideav-test-banks{sep}/var/empty/oxideav-test");
        // Safety: unsafe in 2024 edition; in 2021 it's still safe to call,
        // but we'd rather be explicit. Tests are single-threaded enough
        // for this not to matter in practice.
        env::set_var(ENV_VAR, &custom);
        let paths = soundfont_search_paths();
        assert!(paths.len() >= 2);
        assert_eq!(paths[0], PathBuf::from("/tmp/oxideav-test-banks"));
        assert_eq!(paths[1], PathBuf::from("/var/empty/oxideav-test"));
        // Restore.
        match prev {
            Some(v) => env::set_var(ENV_VAR, v),
            None => env::remove_var(ENV_VAR),
        }
    }

    #[test]
    fn scan_dir_finds_extensions() {
        // Build a temporary directory with one of each extension and a
        // distractor, then scan it.
        let mut dir = std::env::temp_dir();
        dir.push(format!("oxideav-midi-paths-test-{}", std::process::id(),));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for name in &["a.sf2", "b.SF3", "c.sfz", "d.dls", "e.txt", "f.mp3"] {
            fs::write(dir.join(name), b"").unwrap();
        }
        let mut out = Vec::new();
        scan_dir(&dir, &mut out);
        out.sort();
        let names: Vec<String> = out
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"a.sf2".to_string()));
        assert!(names.contains(&"b.SF3".to_string()));
        assert!(names.contains(&"c.sfz".to_string()));
        assert!(names.contains(&"d.dls".to_string()));
        assert!(!names.contains(&"e.txt".to_string()));
        assert!(!names.contains(&"f.mp3".to_string()));
        let _ = fs::remove_dir_all(&dir);
    }
}
