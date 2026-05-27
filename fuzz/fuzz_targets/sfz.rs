#![no_main]

//! Fuzz the SFZ text patch parser against arbitrary bytes.
//!
//! `sfz::parse_str` tokenises SFZ source (line `// …` and block
//! `/* … */` comments, `<header>` sections, opcode `name=value`
//! pairs with space-bearing values), walks `<control>` / `<global>` /
//! `<master>` / `<group>` / `<region>` headers, flattens inheritance
//! into one fully-resolved opcode map per region, and decodes the
//! strongly-typed fields (`lokey` / `hikey` / `pitch_keycenter`,
//! `loop_start` / `loop_end` / `loop_mode`, `tune` / `transpose` /
//! `volume` / `pan` / `trigger`, plus the round-95 filter and
//! envelope opcodes).
//!
//! The contract is the same as the other targets: any byte sequence
//! interpreted as UTF-8 returns a `Result`; non-UTF-8 byte sequences
//! are skipped (the parser's public API takes `&str`, and
//! libfuzzer-driven inputs do include valid UTF-8). No panic, OOM,
//! integer overflow, or out-of-bounds index on any input.

use libfuzzer_sys::fuzz_target;
use oxideav_midi::instruments::sfz;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = sfz::parse_str(text);
    }
});
