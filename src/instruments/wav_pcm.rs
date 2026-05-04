//! Minimal RIFF/WAVE PCM decoder used by the SFZ and DLS sample
//! loaders. Decodes 8-bit unsigned, 16-bit signed LE, 24-bit signed LE,
//! and 32-bit float WAV files into a single mono f32 buffer.
//!
//! Why not depend on `oxideav-basic`? Two reasons:
//!
//!  1. `oxideav-midi` would gain a hard cross-crate dependency that
//!     would prevent it from building standalone (see the
//!     standalone-friendly project pattern in MEMORY).
//!  2. We only need ~100 lines of decoding for the WAV shapes that
//!     instrument-format samples actually use. The full WAV demuxer in
//!     `oxideav-basic` does much more (extensible header, IEEE float,
//!     multi-channel arrangements).
//!
//! Stereo and >2-channel files are mixed down to mono by averaging
//! channels — round-2 voice generation will keep the stereo intact and
//! pan the channels separately.

use oxideav_core::{Error, Result};

/// Decoded PCM in mono f32 form (`-1.0..=1.0`), plus the sample rate.
#[derive(Clone, Debug)]
pub struct DecodedPcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Original number of channels in the WAV — informational; the
    /// `samples` buffer is always mono.
    pub channels: u16,
    /// Original bits-per-sample. Informational.
    pub bits_per_sample: u16,
}

/// Decode a WAV file from its raw bytes. Returns the mono f32 sample
/// stream + the original sample rate. Hard caps on file size + sample
/// count keep a forged header from allocating unbounded memory.
pub fn decode_wav(bytes: &[u8]) -> Result<DecodedPcm> {
    if bytes.len() < 12 {
        return Err(Error::invalid(
            "WAV: file shorter than 12 bytes (no RIFF header)",
        ));
    }
    if &bytes[0..4] != b"RIFF" {
        return Err(Error::invalid("WAV: missing RIFF magic at offset 0"));
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(Error::invalid("WAV: missing WAVE form-type at offset 8"));
    }

    let mut fmt: Option<Fmt> = None;
    let mut data_bytes: Option<&[u8]> = None;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        if body_start.saturating_add(size) > bytes.len() {
            return Err(Error::invalid(format!(
                "WAV: chunk '{}' length {} exceeds remaining bytes",
                tag_str(tag),
                size,
            )));
        }
        let body = &bytes[body_start..body_start + size];
        match tag {
            b"fmt " => fmt = Some(parse_fmt(body)?),
            b"data" => data_bytes = Some(body),
            _ => {}
        }
        // Pad to even.
        pos = body_start + size + (size & 1);
    }

    let fmt = fmt.ok_or_else(|| Error::invalid("WAV: missing 'fmt ' chunk"))?;
    let data = data_bytes.ok_or_else(|| Error::invalid("WAV: missing 'data' chunk"))?;
    decode_data(&fmt, data)
}

/// Decode raw PCM bytes given an explicit format descriptor. Used by
/// the DLS path, which already has the `fmt ` fields parsed and the
/// data bytes in hand.
pub fn decode_pcm_bytes(
    data: &[u8],
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    format_tag: u16,
) -> Result<DecodedPcm> {
    let fmt = Fmt {
        format_tag,
        channels,
        sample_rate,
        bits_per_sample,
    };
    decode_data(&fmt, data)
}

#[derive(Clone, Copy, Debug)]
struct Fmt {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn parse_fmt(body: &[u8]) -> Result<Fmt> {
    if body.len() < 16 {
        return Err(Error::invalid(format!(
            "WAV: 'fmt ' chunk {} bytes < 16 minimum",
            body.len(),
        )));
    }
    Ok(Fmt {
        format_tag: u16::from_le_bytes(body[0..2].try_into().unwrap()),
        channels: u16::from_le_bytes(body[2..4].try_into().unwrap()),
        sample_rate: u32::from_le_bytes(body[4..8].try_into().unwrap()),
        bits_per_sample: u16::from_le_bytes(body[14..16].try_into().unwrap()),
    })
}

fn decode_data(fmt: &Fmt, data: &[u8]) -> Result<DecodedPcm> {
    if fmt.channels == 0 {
        return Err(Error::invalid("WAV: zero channels"));
    }
    if fmt.sample_rate == 0 {
        return Err(Error::invalid("WAV: zero sample rate"));
    }
    let channels = fmt.channels as usize;
    let bps = fmt.bits_per_sample as usize;
    let bytes_per_sample = bps.div_ceil(8);
    let bytes_per_frame = bytes_per_sample * channels;
    if bytes_per_frame == 0 {
        return Err(Error::invalid("WAV: zero bytes per frame"));
    }

    // WAVE_FORMAT_PCM = 1, WAVE_FORMAT_IEEE_FLOAT = 3, EXTENSIBLE = 0xFFFE.
    let is_float = fmt.format_tag == 3;
    let is_pcm = fmt.format_tag == 1 || fmt.format_tag == 0xFFFE;
    if !is_float && !is_pcm {
        return Err(Error::invalid(format!(
            "WAV: unsupported format tag 0x{:04X} (only PCM and IEEE_FLOAT)",
            fmt.format_tag,
        )));
    }
    if is_float && bps != 32 {
        return Err(Error::invalid(format!(
            "WAV: IEEE_FLOAT requires 32 bits/sample, got {bps}",
        )));
    }
    if !is_float && !matches!(bps, 8 | 16 | 24 | 32) {
        return Err(Error::invalid(format!(
            "WAV: unsupported PCM bit depth {bps} (need 8/16/24/32)",
        )));
    }

    let frame_count = data.len() / bytes_per_frame;
    // Hard cap: 256 MiB / 4 bytes per f32 = 64 Mi mono frames.
    const MAX_FRAMES: usize = 64 * 1024 * 1024;
    if frame_count > MAX_FRAMES {
        return Err(Error::invalid(format!(
            "WAV: {frame_count} frames exceeds {MAX_FRAMES} cap",
        )));
    }

    let mut out = Vec::with_capacity(frame_count);
    for frame in data.chunks_exact(bytes_per_frame) {
        let mut acc: f32 = 0.0;
        for ch in 0..channels {
            let s = &frame[ch * bytes_per_sample..(ch + 1) * bytes_per_sample];
            let v = if is_float {
                f32::from_le_bytes(s.try_into().unwrap())
            } else {
                match bps {
                    // WAV PCM convention: 8-bit is unsigned (0x80 = silence).
                    8 => (s[0] as f32 - 128.0) / 128.0,
                    // 16-bit signed LE.
                    16 => i16::from_le_bytes(s.try_into().unwrap()) as f32 / 32_768.0,
                    // 24-bit signed LE — sign-extend into i32.
                    24 => {
                        let raw = (s[0] as i32) | ((s[1] as i32) << 8) | ((s[2] as i32) << 16);
                        let signed = if raw & 0x80_0000 != 0 {
                            raw | !0xFF_FFFF
                        } else {
                            raw
                        };
                        signed as f32 / 8_388_608.0
                    }
                    // 32-bit signed LE PCM.
                    32 => i32::from_le_bytes(s.try_into().unwrap()) as f32 / 2_147_483_648.0,
                    _ => unreachable!(),
                }
            };
            acc += v;
        }
        out.push(acc / channels as f32);
    }

    Ok(DecodedPcm {
        samples: out,
        sample_rate: fmt.sample_rate,
        channels: fmt.channels,
        bits_per_sample: fmt.bits_per_sample,
    })
}

fn tag_str(tag: &[u8]) -> String {
    if tag.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        String::from_utf8_lossy(tag).into_owned()
    } else {
        format!("{:02X}{:02X}{:02X}{:02X}", tag[0], tag[1], tag[2], tag[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal WAV: 8 frames of 16-bit mono signed PCM.
    fn build_wav_16bit_mono(samples: &[i16], rate: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        let data_size = samples.len() * 2;
        bytes.extend_from_slice(&((36u32 + data_size as u32).to_le_bytes()));
        bytes.extend_from_slice(b"WAVE");
        // fmt chunk.
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
                                                       // data chunk.
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_size as u32).to_le_bytes());
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn decodes_16bit_mono_wav_to_f32() {
        // ramp -16384 → +16384 in i16.
        let in_samples: Vec<i16> = (0..8).map(|i| (i * 4096) as i16 - 16384).collect();
        let wav = build_wav_16bit_mono(&in_samples, 44_100);
        let pcm = decode_wav(&wav).expect("decode");
        assert_eq!(pcm.sample_rate, 44_100);
        assert_eq!(pcm.channels, 1);
        assert_eq!(pcm.bits_per_sample, 16);
        assert_eq!(pcm.samples.len(), 8);
        // Spot-check: first sample = -16384/32768 = -0.5.
        assert!(
            (pcm.samples[0] - -0.5).abs() < 1e-4,
            "got {}",
            pcm.samples[0]
        );
    }

    #[test]
    fn decodes_8bit_unsigned_pcm() {
        // 4 frames: 0, 64, 128, 192 → -1.0, -0.5, 0.0, 0.5.
        let body = [0u8, 64, 128, 192];
        let pcm = decode_pcm_bytes(&body, 22_050, 1, 8, 1).expect("decode");
        assert_eq!(pcm.samples.len(), 4);
        assert!((pcm.samples[0] - -1.0).abs() < 1e-4);
        assert!((pcm.samples[1] - -0.5).abs() < 1e-3);
        assert!((pcm.samples[2] - 0.0).abs() < 1e-4);
        assert!((pcm.samples[3] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn decodes_stereo_to_mono_average() {
        // 16-bit stereo: 4 frames of (L=-32768, R=+32766) = mean ≈ 0.
        let mut body: Vec<u8> = Vec::new();
        for _ in 0..4 {
            body.extend_from_slice(&(-32768i16).to_le_bytes());
            body.extend_from_slice(&32766i16.to_le_bytes());
        }
        let pcm = decode_pcm_bytes(&body, 44_100, 2, 16, 1).expect("decode");
        assert_eq!(pcm.samples.len(), 4);
        for s in &pcm.samples {
            assert!(s.abs() < 0.01, "expected near-zero from mono mix, got {s}");
        }
    }

    #[test]
    fn rejects_non_riff_input() {
        let err = decode_wav(b"not a wav").unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }
}
