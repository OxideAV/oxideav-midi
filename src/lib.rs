//! MIDI — Standard MIDI File (SMF) parser + transport metadata + soft-synth scaffold.
//!
//! Round-1 scaffold. Modules and decoder registration are filled in by
//! the implementation pass. External instruments (SoundFont 2, SFZ, DLS)
//! are loaded from disk at runtime; nothing is bundled in the binary.

/// Public codec id string. Matches the aggregator feature name `midi`.
pub const CODEC_ID_STR: &str = "midi";
