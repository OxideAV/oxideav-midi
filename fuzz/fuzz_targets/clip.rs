#![no_main]

//! Fuzz the MIDI Clip File (`.midi2`, M2-116) parser against
//! arbitrary bytes.
//!
//! The contract matches the other targets: every input — well-formed
//! clips, random bytes, truncated UMP word streams, hostile Delta
//! Clockstamp runs, missing DCTPQ / Start / End of Clip framing —
//! returns a `Result`; panics, debug overflows, out-of-bounds reads,
//! or attacker-controlled allocations are bugs.
//!
//! When the parse succeeds we additionally run the Appendix-D
//! translation to SMF and re-serialise the clip, so the
//! clip → SMF converter and the canonical writer cover
//! fuzz-discovered shapes; a successfully re-written clip must parse
//! again.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(clip) = oxideav_midi::clip::parse(data) {
        let _ = clip.to_smf();
        if let Ok(bytes) = oxideav_midi::clip::write(&clip) {
            let _ = oxideav_midi::clip::parse(&bytes);
        }
    }
});
