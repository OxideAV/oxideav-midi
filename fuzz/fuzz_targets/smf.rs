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
//! cover fuzz-discovered shapes too, and require both SMF writers to
//! be fixed points of the parser: `parse(to_bytes(f)) == f`. The
//! writer may refuse only the models the reader tolerates but the
//! spec forbids on the wire — its documented preconditions: a header
//! `ntrks` that disagrees with the tracks actually present (the
//! reader skips unknown chunk types per the spec), a track whose End
//! of Track is missing, not last, or repeated (the reader accepts a
//! truncated track), and an `Unknown { 0x2F }` meta (a non-empty End
//! of Track) — and nothing else.

use libfuzzer_sys::fuzz_target;
use oxideav_midi::smf::{Event, MetaEvent};

fuzz_target!(|data: &[u8]| {
    if let Ok(file) = oxideav_midi::smf::parse(data) {
        let _ = file.tempo_map();
        let _ = file.time_signatures();
        let _ = file.key_signatures();
        let is_eot = |e: &oxideav_midi::smf::TrackEvent| {
            matches!(e.kind, Event::Meta(MetaEvent::EndOfTrack))
        };
        let unwritable = usize::from(file.header.ntrks) != file.tracks.len()
            || file.tracks.iter().any(|t| {
                t.events.iter().filter(|e| is_eot(e)).count() != 1
                    || !t.events.last().is_some_and(is_eot)
                    || t.events.iter().any(|e| {
                        matches!(
                            e.kind,
                            Event::Meta(MetaEvent::Unknown {
                                type_byte: 0x2F,
                                ..
                            })
                        )
                    })
            });
        for (label, written) in [
            ("explicit", file.to_bytes()),
            ("running-status", file.to_bytes_running_status()),
        ] {
            match written {
                Ok(bytes) => {
                    let again = oxideav_midi::smf::parse(&bytes)
                        .unwrap_or_else(|e| panic!("{label}: written SMF must parse: {e:?}"));
                    assert_eq!(again, file, "{label}: SMF writer must be a fixed point");
                }
                Err(_) => assert!(unwritable, "{label}: writer refused a parsed file"),
            }
        }
    }
});
