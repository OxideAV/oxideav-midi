#![no_main]

//! Fuzz the SoundFont 2 RIFF reader against arbitrary bytes.
//!
//! `Sf2Bank::parse` walks the outer `RIFF/sfbk` form, the LIST INFO
//! metadata, LIST sdta (`smpl` + optional `sm24`), and LIST pdta
//! (`phdr` / `pbag` / `pgen` / `inst` / `ibag` / `igen` / `shdr`).
//! Each pdta sub-chunk is a fixed-stride record array whose count is
//! derived from `chunk_len / RECORD_SIZE`; the cross-link between
//! preset → bag → generator → instrument → bag → generator → sample
//! is attacker-controlled, and so are every key/velocity range and
//! sample-position field. The contract under test is just that any
//! input — valid, truncated, header-only, deliberately corrupted
//! cross-links, oversized declared lengths — returns a `Result` and
//! never panics, OOMs, or reads out of bounds.
//!
//! Round 91's safety caps (`MAX_SAMPLE_FRAMES = 256 Mi`,
//! `MAX_PDTA_RECORDS = 16 Mi`) keep the allocations bounded by the
//! spec ceiling rather than the attacker's declared length.

use libfuzzer_sys::fuzz_target;
use oxideav_midi::instruments::sf2::Sf2Bank;

fuzz_target!(|data: &[u8]| {
    let _ = Sf2Bank::parse(data);
});
