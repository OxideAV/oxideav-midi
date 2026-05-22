//! Round-80 end-to-end DLS articulation test.
//!
//! Builds two minimal DLS banks that differ only in their `art1`
//! connection-block list, drives both through the [`MidiDecoder`], and
//! asserts the rendered PCM differs in a way that's only explainable by
//! the articulation having reached the SamplePlayer.
//!
//! Specifically:
//!  - **bank A**: no articulation (empty `lart` list) → SamplePlayer
//!    defaults (5 ms attack / 100 ms decay / 100 ms release).
//!  - **bank B**: `lart` carries a single `SRC_NONE → DST_PITCH`
//!    connection at +1200 cents (one-octave tuning shift). With that
//!    interpreted, the playback rate doubles, so the rendered audio
//!    advances through the sample buffer twice as fast, producing a
//!    measurably different RMS profile from bank A.
//!
//! This test fails as a tight functional contract: if a future refactor
//! breaks the articulation pipeline, both banks would render identical
//! output and the assertion below would trip.

use std::sync::Arc;

use oxideav_core::{Decoder, Error, Frame, Packet, TimeBase};
use oxideav_midi::instruments::dls::DlsInstrument;
use oxideav_midi::instruments::Instrument;
use oxideav_midi::{MidiDecoder, OUTPUT_SAMPLE_RATE};

/// Build a tiny SMF: one channel-0 key-60 note on at tick 0, note off
/// 0.5 s later, then EOT. Round-trips into ~0.5 s of audio.
fn one_note_smf() -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(b"MThd");
    blob.extend_from_slice(&6u32.to_be_bytes());
    blob.extend_from_slice(&0u16.to_be_bytes());
    blob.extend_from_slice(&1u16.to_be_bytes());
    blob.extend_from_slice(&480u16.to_be_bytes());
    let mut t: Vec<u8> = Vec::new();
    t.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // 120 BPM
    t.extend_from_slice(&[0x00, 0x90, 60, 0x64]);
    t.extend_from_slice(&[0x81, 0x70, 0x80, 60, 0x40]); // off
    t.extend_from_slice(&[0x81, 0x70, 0xFF, 0x2F, 0x00]); // EOT
    blob.extend_from_slice(b"MTrk");
    blob.extend_from_slice(&(t.len() as u32).to_be_bytes());
    blob.extend_from_slice(&t);
    blob
}

fn render_to_pcm(dec: &mut MidiDecoder, smf: Vec<u8>) -> Vec<i16> {
    let pkt = Packet::new(0, TimeBase::new(1, 44_100), smf);
    dec.send_packet(&pkt).unwrap();
    let mut all: Vec<i16> = Vec::new();
    for _ in 0..2048 {
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => {
                for chunk in af.data[0].chunks_exact(2) {
                    all.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            Err(Error::Eof) => break,
            Ok(_) => panic!("expected audio frame"),
            Err(other) => panic!("decoder error: {other:?}"),
        }
    }
    all
}

fn rms(samples: &[i16]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples
        .iter()
        .map(|&s| {
            let n = s as f64 / 32_768.0;
            n * n
        })
        .sum();
    (sum_sq / samples.len() as f64).sqrt()
}

#[test]
fn dls_articulation_tuning_shifts_pitch_through_voice() {
    // Bank A: no articulation. Bank B: +1200 cents (1 octave) tuning.
    let bank_a = build_dls_with_articulation(&[]);
    let bank_b = build_dls_with_articulation(&[ArtBlock {
        source: 0x0000,
        control: 0x0000,
        destination: 0x0003, // CONN_DST_PITCH
        transform: 0x0000,
        scale: 1200 * 65_536, // +1200 cents in 1/65536-cent units
    }]);

    let pcm_a = render_bank(&bank_a);
    let pcm_b = render_bank(&bank_b);

    assert!(!pcm_a.is_empty() && !pcm_b.is_empty());
    let rms_a = rms(&pcm_a);
    let rms_b = rms(&pcm_b);
    // Both should produce audible signal.
    assert!(rms_a > 0.0005, "bank A too quiet: rms={rms_a}");
    assert!(rms_b > 0.0005, "bank B too quiet: rms={rms_b}");

    // Sample-by-sample, the tuned bank must differ materially from the
    // untuned bank. We sum the absolute pointwise difference across the
    // first overlapping segment; an unaltered articulation pipeline
    // would produce identical bytes (diff sum ≈ 0).
    let n = pcm_a.len().min(pcm_b.len());
    let diff_total: u64 = pcm_a
        .iter()
        .zip(pcm_b.iter())
        .take(n)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
        .sum();
    let avg_diff = diff_total as f64 / n.max(1) as f64;
    assert!(
        avg_diff > 50.0,
        "tuning-shifted bank should differ from untuned bank, avg pointwise diff = {avg_diff}",
    );
}

/// Round-91 end-to-end check that the DLS `SRC_EG2 → DST_FILTER_CUTOFF`
/// routing actually reaches the SamplePlayer's biquad. We compare:
///   A) bank with `SRC_NONE → DST_FILTER_CUTOFF` set to a low cutoff
///      (~250 Hz) and no EG2 routing → the rendered signal should be
///      heavily attenuated throughout.
///   B) same bank with an additional `SRC_EG2 → DST_FILTER_CUTOFF`
///      routing of +6000 cents and an EG2 attack of 500 ms → the
///      filter sweeps open across the note; the late portion of the
///      rendered audio should be markedly louder than bank A's late
///      portion.
///
/// This is the same logical assertion as the unit test
/// `eg2_filter_sweep_changes_spectrum_over_note` but it travels the
/// full DLS articulation pipeline (`art2` parsing → `Articulation`
/// evaluator → `Articulation::mod_env()` + `Articulation::filter()` →
/// `SamplePlayerConfig` → biquad coefficients), so it fails as a tight
/// integration test if any layer drops the EG2 / filter values on the
/// floor.
#[test]
fn dls_articulation_eg2_sweeps_filter_cutoff_open() {
    // Time-cents = log2(0.5) * 1200 * 65536 ≈ -78_643_200 (500 ms).
    let tc_500ms: i32 = (0.5f64.log2() * 1200.0 * 65536.0) as i32;
    // EG2 → cutoff depth in DLS abs-pitch-cents units (1/65536 per cent):
    // +6000 cents at full EG2 scale = +6000 * 65536.
    let cutoff_sweep_cents: i32 = 6_000 * 65_536;
    // Initial cutoff at ~250 Hz: cents = 1200 * log2(250/8.176) ≈ 5938.
    let initial_cutoff_cents: i32 = 5_938 * 65_536;

    // Bank A: low static cutoff, no EG2 routing.
    let bank_a = build_dls_with_articulation(&[ArtBlock {
        source: 0x0000,
        control: 0x0000,
        destination: 0x0500, // CONN_DST_FILTER_CUTOFF
        transform: 0x0000,
        scale: initial_cutoff_cents,
    }]);
    // Bank B: low static cutoff + EG2 → cutoff routing + slow EG2.
    let bank_b = build_dls_with_articulation(&[
        ArtBlock {
            source: 0x0000,
            control: 0x0000,
            destination: 0x0500, // CONN_DST_FILTER_CUTOFF
            transform: 0x0000,
            scale: initial_cutoff_cents,
        },
        ArtBlock {
            source: 0x0000,
            control: 0x0000,
            destination: 0x030A, // CONN_DST_EG2_ATTACKTIME
            transform: 0x0000,
            scale: tc_500ms,
        },
        ArtBlock {
            source: 0x0000,
            control: 0x0000,
            destination: 0x030E, // CONN_DST_EG2_SUSTAINLEVEL
            transform: 0x0000,
            scale: 1_000, // 100 % sustain (kept at peak)
        },
        ArtBlock {
            source: 0x0005, // CONN_SRC_EG2
            control: 0x0000,
            destination: 0x0500, // CONN_DST_FILTER_CUTOFF
            transform: 0x0000,
            scale: cutoff_sweep_cents,
        },
    ]);

    let pcm_a = render_bank(&bank_a);
    let pcm_b = render_bank(&bank_b);

    // The SMF holds the note for 0.5 s @ 44.1 kHz output ≈ 22050
    // samples. We split the rendered audio into "early" (~ first
    // 100 ms) and "late" (≥ 300 ms in) windows.
    let n = pcm_a.len().min(pcm_b.len());
    assert!(n > 16_000, "expected >= 16 000 PCM samples, got {n}");
    let early_a = rms(&pcm_a[1_000..4_410]);
    let late_a = rms(&pcm_a[13_230..n.min(20_000)]);
    let early_b = rms(&pcm_b[1_000..4_410]);
    let late_b = rms(&pcm_b[13_230..n.min(20_000)]);

    // Bank A (static low-pass) should be roughly steady across the
    // two windows. We don't need bit-exactness — the volume envelope
    // can still nudge the level — but the late window shouldn't be
    // multiple-x louder than the early one.
    let a_ratio = late_a / early_a.max(1e-6);
    assert!(
        a_ratio < 2.0,
        "static-cutoff bank A should be steady; got early={early_a}, late={late_a} (ratio={a_ratio})"
    );

    // Bank B (EG2 sweep) must open the filter so the late window
    // grows substantially louder than the early window. The integer
    // 65536-unit DLS encoding plus the SamplePlayer's 50-cent biquad
    // re-coef gate introduces some slack; we require >= 1.6x
    // late/early gain.
    let b_ratio = late_b / early_b.max(1e-6);
    assert!(
        b_ratio > 1.6,
        "EG2 sweep bank B should open the filter; got early={early_b}, late={late_b} (ratio={b_ratio})"
    );

    // And bank B's late window must dominate bank A's late window —
    // a static-cutoff bank can't open up like a swept-cutoff one
    // does.
    assert!(
        late_b > late_a * 1.4,
        "swept-cutoff late RMS {late_b} should dominate static-cutoff late RMS {late_a}"
    );
}

#[test]
fn dls_articulation_long_release_extends_voice_tail() {
    // Bank A: default 100 ms release. Bank B: 2 s release (DLS
    // time-cents = log2(2) * 1200 * 65536 ≈ +78_643_200).
    let tc_2s: i32 = (2.0f64.log2() * 1200.0 * 65536.0) as i32;
    let bank_a = build_dls_with_articulation(&[]);
    let bank_b = build_dls_with_articulation(&[ArtBlock {
        source: 0x0000,
        control: 0x0000,
        destination: 0x0209, // CONN_DST_EG1_RELEASETIME
        transform: 0x0000,
        scale: tc_2s,
    }]);
    // Compute RMS on the trailing portion: the long-release bank should
    // have non-trivial energy past the point where the default-release
    // bank has gone silent.
    let pcm_a = render_bank(&bank_a);
    let pcm_b = render_bank(&bank_b);
    // Take the last 20 % of the rendered audio as the "tail".
    let tail_a = &pcm_a[(pcm_a.len() * 4 / 5)..];
    let tail_b = &pcm_b[(pcm_b.len() * 4 / 5)..];
    let rms_tail_a = rms(tail_a);
    let rms_tail_b = rms(tail_b);
    // The long-release bank should sustain a louder tail. We require
    // measurable inequality; a misrouted articulation pipeline would
    // produce equal tails (both at the default 100 ms release).
    assert!(
        rms_tail_b > rms_tail_a,
        "long-release bank tail RMS {rms_tail_b} should exceed default-release bank tail RMS {rms_tail_a}",
    );
}

fn render_bank(blob: &[u8]) -> Vec<i16> {
    let inst: Arc<dyn Instrument> =
        Arc::new(DlsInstrument::parse_bytes("articulation-test.dls", blob).expect("parse DLS"));
    let mut dec = MidiDecoder::new(inst, OUTPUT_SAMPLE_RATE);
    render_to_pcm(&mut dec, one_note_smf())
}

#[derive(Clone, Copy)]
struct ArtBlock {
    source: u16,
    control: u16,
    destination: u16,
    transform: u16,
    scale: i32,
}

/// Build a complete DLS-Level-2 bank with one instrument, one region,
/// one 16-bit PCM mono **looped** sample (256-frame sine), and the
/// requested articulation `lart` list (empty for "no articulation").
/// The looping keeps the sample sustaining for as long as the MIDI
/// note holds, so the rendered audio runs for the full ~0.5 s and
/// articulation-driven envelope / pitch changes have time to manifest.
fn build_dls_with_articulation(art1_blocks: &[ArtBlock]) -> Vec<u8> {
    // 440 Hz sine — 256 frames at 22 050 Hz repeats roughly every 5 ms,
    // so a looped voice can sustain for any note length without aliasing.
    let n: usize = 256;
    let rate: u32 = 22_050;
    let mut pcm = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        let v = (t * 440.0 * std::f32::consts::TAU).sin();
        let s = (v * 16_384.0) as i16;
        pcm.extend_from_slice(&s.to_le_bytes());
    }

    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&1u16.to_le_bytes()); // mono
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&(rate * 2).to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());

    // wsmp on the wave: unity_note=60, one forward loop over the whole
    // sample so the voice sustains throughout the held note.
    let mut wsmp = Vec::new();
    wsmp.extend_from_slice(&20u32.to_le_bytes()); // cbSize
    wsmp.extend_from_slice(&60u16.to_le_bytes()); // unity_note
    wsmp.extend_from_slice(&0i16.to_le_bytes()); // fine_tune
    wsmp.extend_from_slice(&0i32.to_le_bytes()); // gain (centibels)
    wsmp.extend_from_slice(&0u32.to_le_bytes()); // options
    wsmp.extend_from_slice(&1u32.to_le_bytes()); // 1 loop
                                                 // loop record: cbSize=16, type=0 (forward), start=0, length=n.
    wsmp.extend_from_slice(&16u32.to_le_bytes());
    wsmp.extend_from_slice(&0u32.to_le_bytes());
    wsmp.extend_from_slice(&0u32.to_le_bytes());
    wsmp.extend_from_slice(&(n as u32).to_le_bytes());

    let mut wave_body = Vec::from(b"wave" as &[u8]);
    push_riff(&mut wave_body, b"fmt ", &fmt);
    push_riff(&mut wave_body, b"data", &pcm);
    push_riff(&mut wave_body, b"wsmp", &wsmp);

    let mut wvpl_body = Vec::from(b"wvpl" as &[u8]);
    push_riff(&mut wvpl_body, b"LIST", &wave_body);

    // ptbl: 1 cue at offset 0.
    let mut ptbl = Vec::new();
    ptbl.extend_from_slice(&8u32.to_le_bytes()); // cbSize
    ptbl.extend_from_slice(&1u32.to_le_bytes()); // cCues
    ptbl.extend_from_slice(&0u32.to_le_bytes()); // pool offset 0

    // colh.
    let mut colh = Vec::new();
    colh.extend_from_slice(&1u32.to_le_bytes());

    // vers.
    let mut vers = Vec::new();
    vers.extend_from_slice(&((1u32 << 16) | 1u32).to_le_bytes());
    vers.extend_from_slice(&0u32.to_le_bytes());

    let mut info_body = Vec::from(b"INFO" as &[u8]);
    push_riff(&mut info_body, b"INAM", b"ArtTest\0");

    // insh: 1 region, bank 0, program 0.
    let mut insh = Vec::new();
    insh.extend_from_slice(&1u32.to_le_bytes());
    insh.extend_from_slice(&0u32.to_le_bytes());
    insh.extend_from_slice(&0u32.to_le_bytes());

    // rgnh: full keyboard, full velocity.
    let mut rgnh = Vec::new();
    rgnh.extend_from_slice(&0u16.to_le_bytes());
    rgnh.extend_from_slice(&127u16.to_le_bytes());
    rgnh.extend_from_slice(&0u16.to_le_bytes());
    rgnh.extend_from_slice(&127u16.to_le_bytes());
    rgnh.extend_from_slice(&0u16.to_le_bytes());
    rgnh.extend_from_slice(&0u16.to_le_bytes());

    // wlnk: table_index=0, mono left channel.
    let mut wlnk = Vec::new();
    wlnk.extend_from_slice(&0u16.to_le_bytes());
    wlnk.extend_from_slice(&0u16.to_le_bytes());
    wlnk.extend_from_slice(&1u32.to_le_bytes());
    wlnk.extend_from_slice(&0u32.to_le_bytes());

    let mut rgn_body = Vec::from(b"rgn " as &[u8]);
    push_riff(&mut rgn_body, b"rgnh", &rgnh);
    push_riff(&mut rgn_body, b"wlnk", &wlnk);

    // Attach a `lart` chunk if we have any articulation blocks to
    // express. Per the DLS spec, a region's lart list overrides the
    // instrument-level list; we put the test blocks at region level so
    // the test exercises the region-articulation path.
    if !art1_blocks.is_empty() {
        let mut art1_payload = Vec::new();
        art1_payload.extend_from_slice(&8u32.to_le_bytes()); // cbSize (header only)
        art1_payload.extend_from_slice(&(art1_blocks.len() as u32).to_le_bytes());
        for b in art1_blocks {
            art1_payload.extend_from_slice(&b.source.to_le_bytes());
            art1_payload.extend_from_slice(&b.control.to_le_bytes());
            art1_payload.extend_from_slice(&b.destination.to_le_bytes());
            art1_payload.extend_from_slice(&b.transform.to_le_bytes());
            art1_payload.extend_from_slice(&b.scale.to_le_bytes());
        }
        let mut lart_body = Vec::from(b"lart" as &[u8]);
        push_riff(&mut lart_body, b"art1", &art1_payload);
        push_riff(&mut rgn_body, b"LIST", &lart_body);
    }

    let mut lrgn_body = Vec::from(b"lrgn" as &[u8]);
    push_riff(&mut lrgn_body, b"LIST", &rgn_body);

    let mut ins_body = Vec::from(b"ins " as &[u8]);
    push_riff(&mut ins_body, b"insh", &insh);
    push_riff(&mut ins_body, b"LIST", &lrgn_body);

    let mut lins_body = Vec::from(b"lins" as &[u8]);
    push_riff(&mut lins_body, b"LIST", &ins_body);

    let mut body = Vec::from(b"DLS " as &[u8]);
    push_riff(&mut body, b"vers", &vers);
    push_riff(&mut body, b"colh", &colh);
    push_riff(&mut body, b"LIST", &lins_body);
    push_riff(&mut body, b"ptbl", &ptbl);
    push_riff(&mut body, b"LIST", &wvpl_body);
    push_riff(&mut body, b"LIST", &info_body);

    let mut out = Vec::from(b"RIFF" as &[u8]);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn push_riff(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(tag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}
