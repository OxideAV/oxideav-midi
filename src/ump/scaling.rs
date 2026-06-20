//! Bit-scaling primitives for MIDI 1.0 ⇄ MIDI 2.0 value translation.
//!
//! Implements the data-value translation rules of spec Appendix D:
//!
//! * §D.1.2 Core Rules — minimum→minimum, maximum→maximum, and the
//!   center value (`TRUNC((Highest + 1) / 2)`) always maps to the
//!   destination center.
//! * §D.1.3 Min-Center-Max upscaling — smooth bit-shift below center, an
//!   "expanded bit-repeat scheme" above center, so a low-resolution
//!   value spreads evenly across the full high-resolution range.
//! * §D.1.4 downscaling — a simple truncating right-shift.
//!
//! The functions are pure integer arithmetic and mirror the reference
//! pseudo-code in the specification (translated to Rust idioms, not
//! transcribed from any implementation).

/// Min-Center-Max upscaling of a low-resolution value to a higher
/// resolution (spec §D.1.3).
///
/// `src_bits` is the source field width (`> 1`), `dst_bits` the
/// destination width (`<= 32`, and `> src_bits`). Below the source
/// center the value is a plain left shift; above the center the
/// remaining low bits are filled by repeating the source's significant
/// bits (excluding the top bit), so that the maximum source value maps
/// to the destination maximum and the center maps to the center.
///
/// # Panics
/// Never panics for in-range inputs; out-of-range `src_bits`/`dst_bits`
/// (violating `1 < src_bits < dst_bits <= 32`) are clamped by saturating
/// the shift, which keeps the function total.
#[must_use]
pub fn scale_up(src_val: u32, src_bits: u8, dst_bits: u8) -> u32 {
    if src_bits >= dst_bits || src_bits < 2 || dst_bits > 32 {
        // Degenerate request: nothing meaningful to upscale. Return the
        // value bounded into the destination width.
        return if dst_bits >= 32 {
            src_val
        } else {
            src_val & ((1u32 << dst_bits) - 1)
        };
    }
    let scale_bits = dst_bits - src_bits;
    let src_center = 1u32 << (src_bits - 1);
    let bit_shifted = src_val << scale_bits;
    if src_val <= src_center {
        return bit_shifted;
    }
    // Expanded bit-repeat scheme: repeat all but the highest source bit.
    let repeat_bits = src_bits - 1;
    let repeat_mask = (1u32 << repeat_bits) - 1;
    let mut repeat_value = src_val & repeat_mask;
    if scale_bits > repeat_bits {
        repeat_value <<= scale_bits - repeat_bits;
    } else {
        repeat_value >>= repeat_bits - scale_bits;
    }
    let mut result = bit_shifted;
    while repeat_value != 0 {
        result |= repeat_value;
        repeat_value >>= repeat_bits;
    }
    result
}

/// Truncating downscaling of a high-resolution value to a lower
/// resolution (spec §D.1.4): a simple right shift, discarding the low
/// `src_bits - dst_bits` bits.
///
/// For `src_bits <= dst_bits` the value is returned unchanged (no bits
/// to drop).
#[must_use]
pub fn scale_down(src_val: u32, src_bits: u8, dst_bits: u8) -> u32 {
    if src_bits <= dst_bits {
        return src_val;
    }
    src_val >> (src_bits - dst_bits)
}

/// Upscale a MIDI 1.0 7-bit value to a MIDI 2.0 16-bit value
/// (§D.1.3 worked example: `scaleUp7to16`).
#[must_use]
pub fn scale_7_to_16(value7: u8) -> u16 {
    scale_up(u32::from(value7 & 0x7F), 7, 16) as u16
}

/// Upscale a MIDI 1.0 7-bit value to a MIDI 2.0 32-bit value.
#[must_use]
pub fn scale_7_to_32(value7: u8) -> u32 {
    scale_up(u32::from(value7 & 0x7F), 7, 32)
}

/// Upscale a MIDI 1.0 14-bit value to a MIDI 2.0 32-bit value.
#[must_use]
pub fn scale_14_to_32(value14: u16) -> u32 {
    scale_up(u32::from(value14 & 0x3FFF), 14, 32)
}

/// Downscale a MIDI 2.0 16-bit value to a MIDI 1.0 7-bit value.
#[must_use]
pub fn scale_16_to_7(value16: u16) -> u8 {
    scale_down(u32::from(value16), 16, 7) as u8
}

/// Downscale a MIDI 2.0 32-bit value to a MIDI 1.0 7-bit value.
#[must_use]
pub fn scale_32_to_7(value32: u32) -> u8 {
    scale_down(value32, 32, 7) as u8
}

/// Downscale a MIDI 2.0 32-bit value to a MIDI 1.0 14-bit value.
#[must_use]
pub fn scale_32_to_14(value32: u32) -> u16 {
    scale_down(value32, 32, 14) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_d13_numerical_examples_7_to_16() {
        // Spec §D.1.3 "Numerical Examples".
        assert_eq!(scale_7_to_16(0x0a), 0x1400);
        assert_eq!(scale_7_to_16(0x40), 0x8000); // center → center
        assert_eq!(scale_7_to_16(0x57), 0xaeba);
        assert_eq!(scale_7_to_16(0x7f), 0xffff); // max → max
    }

    #[test]
    fn spec_d12_min_and_max_endpoints() {
        // §D.1.2 Core Rules: min→min, max→max.
        assert_eq!(scale_7_to_16(0), 0);
        assert_eq!(scale_7_to_16(127), 0xffff);
        assert_eq!(scale_7_to_32(0), 0);
        assert_eq!(scale_7_to_32(127), 0xffff_ffff);
        assert_eq!(scale_14_to_32(0), 0);
        assert_eq!(scale_14_to_32(0x3fff), 0xffff_ffff);
    }

    #[test]
    fn center_values_map_to_center_table_23() {
        // §D.1.2 Table 23 Center Value Examples.
        assert_eq!(scale_up(0x40, 7, 8), 0x80);
        assert_eq!(scale_up(0x40, 7, 16), 0x8000);
        assert_eq!(scale_up(0x40, 7, 32), 0x8000_0000);
        assert_eq!(scale_up(0x2000, 14, 32), 0x8000_0000);
    }

    #[test]
    fn spec_d14_downscale_examples() {
        // §D.1.4 Numerical Examples (inverse of the upscale examples).
        assert_eq!(scale_16_to_7(0x1400), 0x0a);
        assert_eq!(scale_16_to_7(0x8000), 0x40);
        assert_eq!(scale_16_to_7(0xaeba), 0x57);
        assert_eq!(scale_16_to_7(0xffff), 0x7f);
    }

    #[test]
    fn round_trip_7_16_7_is_lossless() {
        // §D.1.1: MIDI 1.0 → 2.0 → 1.0 must yield the original value.
        for v in 0u8..=0x7f {
            assert_eq!(scale_16_to_7(scale_7_to_16(v)), v, "v={v}");
        }
    }

    #[test]
    fn round_trip_7_32_7_is_lossless() {
        for v in 0u8..=0x7f {
            assert_eq!(scale_32_to_7(scale_7_to_32(v)), v, "v={v}");
        }
    }

    #[test]
    fn round_trip_14_32_14_is_lossless() {
        for v in 0u16..=0x3fff {
            assert_eq!(scale_32_to_14(scale_14_to_32(v)), v, "v={v}");
        }
    }

    #[test]
    fn upscale_is_monotonic_nondecreasing() {
        // A smooth spread must never decrease as the source increases.
        let mut prev = 0u32;
        for v in 0u8..=0x7f {
            let up = scale_7_to_32(v);
            assert!(up >= prev, "non-monotone at v={v}");
            prev = up;
        }
    }
}
