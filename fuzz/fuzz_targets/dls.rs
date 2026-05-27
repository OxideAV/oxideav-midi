#![no_main]

//! Fuzz the DLS Level 1 / Level 2 RIFF reader against arbitrary bytes.
//!
//! `DlsBank::parse` walks the outer `RIFF/DLS<space>` form, the
//! `colh` / `vers` / `ptbl` headers, the `lins-list` instrument list
//! (with per-instrument `insh` / `rgn ` / `rgn2` / `wsmp` / `wlnk` /
//! `lart-list` sub-chunks), the `wvpl-list` wave pool (each
//! `wave-list` entry with its own `fmt ` + `data` + optional `wsmp`
//! chunks), and the top-level `INFO-list`. `art1` / `art2`
//! connection blocks are parsed as fixed 12-byte records inside an
//! attacker-controlled `cConnectionBlocks` count.
//!
//! The contract is the same as the other targets: every input
//! returns a `Result`; no panic, OOM, integer overflow, or
//! out-of-bounds index. The cumulative wave-data byte cap
//! (`MAX_WAVE_BYTES = 256 MiB`) and the per-array record cap
//! (`MAX_RECORDS = 1 Mi`) keep allocations bounded by the spec
//! ceiling.

use libfuzzer_sys::fuzz_target;
use oxideav_midi::instruments::dls::DlsBank;

fuzz_target!(|data: &[u8]| {
    let _ = DlsBank::parse(data);
});
