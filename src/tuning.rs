//! MIDI Tuning Standard (MTS) — microtuning state + Universal SysEx
//! data-format decoders.
//!
//! Implements the retuning surface described in the MMA *MIDI Tuning
//! Messages* specification (`docs/audio/midi/extensions/
//! MIDI-Tuning-Updated-Specification.pdf`, incorporating CA-020 /
//! CA-021 / RP-020). The synth keeps two layers of microtuning state,
//! both expressed as a **signed cents offset** added to a key's
//! 12-tone-equal-temperament pitch before it reaches a voice:
//!
//! * **Key-based tuning** — a 128-entry table, one offset per MIDI key
//!   number, addressed globally (the "current" tuning program). Set by
//!   the Single-Note Tuning Change messages (real-time sub-ID#2 `02`
//!   and the bank form `07`). Per the spec these affect currently
//!   sounding notes immediately.
//! * **Scale/octave tuning** — a 12-entry table of pitch-class offsets
//!   (C, C#, … B) that repeats every octave, addressed per MIDI
//!   channel via the SysEx channel bitmap. Set by the Scale/Octave
//!   Tuning 1-byte (sub-ID#2 `08`) and 2-byte (`09`) forms.
//!
//! Both layers default to "equal temperament" (all offsets 0), so a
//! synth that never receives an MTS message renders bit-identically to
//! the pre-MTS path.
//!
//! ## Frequency data format (spec section 1)
//!
//! Key-based tuning carries a per-key target pitch as a 3-byte word
//! `xx yy zz`:
//!
//! ```text
//! 0xxxxxxx 0abcdefg 0hijklmn
//! xxxxxxx        = semitone (nearest equal-tempered semitone below)
//! abcdefghijklmn = fraction of 100 cents above that semitone (14 bits)
//! ```
//!
//! The reserved word `7F 7F 7F` means "no change" — the receiver leaves
//! the stored value for that key untouched.
//!
//! The target pitch, in fractional MIDI semitones, is
//! `semitone + fraction14 / 16384`. We store the **offset from the key
//! being addressed**: `(target_semitones - key) * 100` cents. So a
//! `3C 00 00` word on key 60 (the spec's middle-C example) yields a
//! zero offset (already equal temperament).
//!
//! ## Scale/octave format (CA-021 / RP-020)
//!
//! The 1-byte form encodes each pitch class as `00 = -64 cents`,
//! `40 = 0 cents`, `7F = +63 cents`, i.e. `value - 64` cents. The
//! 2-byte form is a 14-bit word with `0x0000 = -100 cents`,
//! `0x2000 = 0 cents`, `0x3FFF = +100 cents`, i.e.
//! `(raw - 8192) * 100 / 8192` cents.

use crate::mixer::NUM_CHANNELS;

/// Number of pitch classes in an octave.
const PITCH_CLASSES: usize = 12;

/// Reserved frequency-data word that means "leave this key's tuning
/// unchanged" (spec section 1).
pub const FREQ_NO_CHANGE: [u8; 3] = [0x7F, 0x7F, 0x7F];

/// Decode a 3-byte MTS frequency-data word into a signed cents offset
/// relative to the equal-tempered pitch of `key`.
///
/// Returns `None` for the reserved `7F 7F 7F` "no change" word. The
/// three bytes are masked to 7 bits each (SysEx data bytes have the
/// high bit clear); a stray set high bit is therefore ignored rather
/// than rejected, matching the spec's "discard unneeded bits" latitude.
///
/// The target pitch in fractional MIDI semitones is
/// `semitone + fraction14 / 16384`; the offset returned is
/// `(target - key) * 100` cents.
pub fn freq_word_to_cents_offset(word: [u8; 3], key: u8) -> Option<f32> {
    if word == FREQ_NO_CHANGE {
        return None;
    }
    let semitone = (word[0] & 0x7F) as i32;
    let fraction14 = (((word[1] & 0x7F) as u32) << 7) | ((word[2] & 0x7F) as u32);
    // semitones above the addressed key, including the fractional part.
    let target_semitones = semitone as f32 + (fraction14 as f32) / 16384.0;
    Some((target_semitones - key as f32) * 100.0)
}

/// Decode a 1-byte scale/octave offset (`00 = -64 c`, `40 = 0 c`,
/// `7F = +63 c`) into cents.
pub fn scale_octave_1byte_to_cents(value: u8) -> f32 {
    ((value & 0x7F) as i32 - 64) as f32
}

/// Decode a 2-byte scale/octave offset into cents. The 14-bit word maps
/// `0x0000 → -100 c`, `0x2000 → 0 c`, `0x3FFF → +100 c` linearly, i.e.
/// `(raw - 8192) * 200 / 16384`.
pub fn scale_octave_2byte_to_cents(msb: u8, lsb: u8) -> f32 {
    let raw = (((msb & 0x7F) as i32) << 7) | ((lsb & 0x7F) as i32);
    (raw - 0x2000) as f32 * 200.0 / 16384.0
}

/// Decode the 3-byte channel bitmap that prefixes the Scale/Octave
/// tuning messages (`ff gg hh`) into a 16-bit mask, bit `c` set ⇒
/// channel index `c` (0..=15) is targeted.
///
/// Per the spec:
/// * `ff` bits 0–1 = channels 15–16 (indices 14–15); bits 2–6 reserved.
/// * `gg` bits 0–6 = channels 8–14 (indices 7–13).
/// * `hh` bits 0–6 = channels 1–7 (indices 0–6).
pub fn scale_octave_channel_mask(ff: u8, gg: u8, hh: u8) -> u16 {
    let mut mask = 0u16;
    // hh: channels 1..=7 → indices 0..=6.
    for bit in 0..7 {
        if hh & (1 << bit) != 0 {
            mask |= 1 << bit;
        }
    }
    // gg: channels 8..=14 → indices 7..=13.
    for bit in 0..7 {
        if gg & (1 << bit) != 0 {
            mask |= 1 << (bit + 7);
        }
    }
    // ff: channels 15..=16 → indices 14..=15 (only bits 0–1 used).
    for bit in 0..2 {
        if ff & (1 << bit) != 0 {
            mask |= 1 << (bit + 14);
        }
    }
    mask
}

/// Microtuning state held by the [`Mixer`](crate::mixer::Mixer): a
/// global key-based table plus per-channel scale/octave tables.
///
/// All offsets are signed cents added to a key's equal-tempered pitch.
/// The default is equal temperament everywhere (all zero), so a synth
/// that never sees an MTS message produces output bit-identical to the
/// pre-MTS code path.
#[derive(Clone, Debug)]
pub struct TuningTable {
    /// Per-MIDI-key offset in cents (the "current" key-based tuning
    /// program). 128 entries; index = MIDI key number.
    key_offsets: [f32; 128],
    /// Per-channel scale/octave offsets in cents, one row of 12
    /// pitch-class offsets (C, C#, … B) per MIDI channel.
    scale_octave: [[f32; PITCH_CLASSES]; NUM_CHANNELS],
}

impl Default for TuningTable {
    fn default() -> Self {
        Self {
            key_offsets: [0.0; 128],
            scale_octave: [[0.0; PITCH_CLASSES]; NUM_CHANNELS],
        }
    }
}

impl TuningTable {
    /// A fresh equal-temperament table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total tuning offset (cents) applied to `(channel, key)`: the
    /// global key-based offset summed with the channel's scale/octave
    /// offset for that key's pitch class.
    pub fn offset_cents(&self, channel: u8, key: u8) -> f32 {
        let k = key as usize;
        let key_off = if k < 128 { self.key_offsets[k] } else { 0.0 };
        let ch = channel as usize % NUM_CHANNELS;
        let pc = (key % 12) as usize;
        key_off + self.scale_octave[ch][pc]
    }

    /// Apply one Single-Note Tuning Change entry: set `key`'s offset
    /// from a 3-byte frequency-data word. The reserved `7F 7F 7F`
    /// "no change" word leaves the stored offset untouched.
    pub fn set_key_freq_word(&mut self, key: u8, word: [u8; 3]) {
        if (key as usize) >= 128 {
            return;
        }
        if let Some(cents) = freq_word_to_cents_offset(word, key) {
            self.key_offsets[key as usize] = cents;
        }
    }

    /// Set a channel's scale/octave offset for one pitch class
    /// (`0 = C` … `11 = B`).
    pub fn set_scale_octave(&mut self, channel: u8, pitch_class: usize, cents: f32) {
        if pitch_class >= PITCH_CLASSES {
            return;
        }
        let ch = channel as usize % NUM_CHANNELS;
        self.scale_octave[ch][pitch_class] = cents;
    }

    /// Reset every layer to equal temperament. Used by GM System
    /// On/Off, which resets all controllers to defaults.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_word_equal_temperament_examples_are_zero_offset() {
        // Spec section 1 worked examples: each `nn 00 00` word names the
        // equal-tempered pitch of MIDI key nn, so addressed to that key
        // the offset is exactly zero.
        for key in [0u8, 12, 60, 61, 120, 127] {
            let off = freq_word_to_cents_offset([key, 0x00, 0x00], key).unwrap();
            assert!(off.abs() < 1e-3, "key {key} offset {off}");
        }
    }

    #[test]
    fn freq_word_no_change_is_none() {
        assert!(freq_word_to_cents_offset(FREQ_NO_CHANGE, 60).is_none());
    }

    #[test]
    fn freq_word_one_lsb_is_about_006_cents() {
        // `00 00 01` is the spec's smallest step: 100 cents / 16384 =
        // 0.0061 cents above 8.1758 Hz (key 0).
        let off = freq_word_to_cents_offset([0x00, 0x00, 0x01], 0).unwrap();
        assert!((off - 100.0 / 16384.0).abs() < 1e-6, "off {off}");
    }

    #[test]
    fn freq_word_addressed_to_lower_key_is_positive_semitones() {
        // `3D 00 00` is C#5 (key 61). Addressed to key 60 it should
        // read +100 cents (one semitone up).
        let off = freq_word_to_cents_offset([0x3D, 0x00, 0x00], 60).unwrap();
        assert!((off - 100.0).abs() < 1e-3, "off {off}");
    }

    #[test]
    fn freq_word_half_fraction_is_50_cents() {
        // fraction14 = 0x2000 = 8192 = half of 16384 ⇒ +50 cents.
        // 0x2000 packs as msb=0x40, lsb=0x00.
        let off = freq_word_to_cents_offset([0x3C, 0x40, 0x00], 60).unwrap();
        assert!((off - 50.0).abs() < 1e-3, "off {off}");
    }

    #[test]
    fn scale_octave_1byte_endpoints() {
        assert_eq!(scale_octave_1byte_to_cents(0x00), -64.0);
        assert_eq!(scale_octave_1byte_to_cents(0x40), 0.0);
        assert_eq!(scale_octave_1byte_to_cents(0x7F), 63.0);
    }

    #[test]
    fn scale_octave_2byte_endpoints() {
        assert!((scale_octave_2byte_to_cents(0x00, 0x00) - -100.0).abs() < 1e-3);
        assert!(scale_octave_2byte_to_cents(0x40, 0x00).abs() < 1e-3);
        // 0x3FFF maps to +100 * (16383-8192)/8192 ≈ +99.988 cents
        // (the +63/64-style off-by-one-step the spec calls "8191 steps").
        let top = scale_octave_2byte_to_cents(0x7F, 0x7F);
        assert!((top - 99.988).abs() < 0.05, "top {top}");
    }

    #[test]
    fn channel_mask_low_high_bits() {
        // hh bit0 ⇒ channel 1 (index 0).
        assert_eq!(scale_octave_channel_mask(0, 0, 0x01) & 1, 1);
        // gg bit0 ⇒ channel 8 (index 7).
        assert_eq!(scale_octave_channel_mask(0, 0x01, 0) >> 7 & 1, 1);
        // ff bit1 ⇒ channel 16 (index 15).
        assert_eq!(scale_octave_channel_mask(0x02, 0, 0) >> 15 & 1, 1);
        // ff bits 2–6 are reserved and must not light any channel.
        assert_eq!(scale_octave_channel_mask(0x7C, 0, 0), 0);
        // all 16 channels.
        assert_eq!(scale_octave_channel_mask(0x03, 0x7F, 0x7F), 0xFFFF);
    }

    #[test]
    fn table_default_is_zero_offset_everywhere() {
        let t = TuningTable::new();
        for ch in 0..16u8 {
            for key in 0..128u8 {
                assert_eq!(t.offset_cents(ch, key), 0.0);
            }
        }
    }

    #[test]
    fn table_key_offset_sums_with_scale_octave() {
        let mut t = TuningTable::new();
        // Retune key 60 up 25 cents (key-based) and pitch class C
        // (= key 60's class) down 10 cents on channel 3 (scale/octave).
        t.set_key_freq_word(60, [0x3C, 0x10, 0x00]); // +25 cents
        let expect_key = 100.0 * (0x800 as f32) / 16384.0; // 0x10<<7 = 2048
        assert!((t.offset_cents(0, 60) - expect_key).abs() < 1e-3);
        t.set_scale_octave(3, 0, -10.0);
        assert!((t.offset_cents(3, 60) - (expect_key - 10.0)).abs() < 1e-3);
        // Channel 0 is unaffected by channel 3's scale/octave row.
        assert!((t.offset_cents(0, 60) - expect_key).abs() < 1e-3);
        // Key 72 (also pitch class C) on channel 3 gets the -10 c but
        // not the key-60-only key-based offset.
        assert!((t.offset_cents(3, 72) - -10.0).abs() < 1e-3);
    }

    #[test]
    fn table_reset_restores_equal_temperament() {
        let mut t = TuningTable::new();
        t.set_key_freq_word(64, [0x41, 0x00, 0x00]);
        t.set_scale_octave(5, 4, 33.0);
        t.reset();
        assert_eq!(t.offset_cents(0, 64), 0.0);
        assert_eq!(t.offset_cents(5, 64), 0.0);
    }

    #[test]
    fn set_key_no_change_word_preserves_prior() {
        let mut t = TuningTable::new();
        t.set_key_freq_word(64, [0x41, 0x00, 0x00]); // +100 cents
        let before = t.offset_cents(0, 64);
        t.set_key_freq_word(64, FREQ_NO_CHANGE); // must not clobber
        assert_eq!(t.offset_cents(0, 64), before);
    }
}
