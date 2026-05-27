#![no_main]

//! Fuzz the Standard MIDI File parser against arbitrary bytes.
//!
//! The contract is: every input — well-formed `.mid` files, random
//! bytes, deliberately malformed chunk lengths or VLQs, oversized
//! sysex / meta varlen payloads, runaway `ntrks`, etc. — returns a
//! `Result`. Panics, OOMs, integer overflows in debug, or out-of-
//! bounds reads are treated as bugs.
//!
//! When the parse succeeds we additionally walk the three public
//! iteration helpers (`tempo_map`, `time_signatures`, `key_signatures`)
//! so the cumulative-tick accounting + meta-event extraction paths
//! cover fuzz-discovered shapes too.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(file) = oxideav_midi::smf::parse(data) {
        let _ = file.tempo_map();
        let _ = file.time_signatures();
        let _ = file.key_signatures();
    }
});
