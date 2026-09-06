//! Native MIDI 2.0 resolution in the synthesis path (M2-104 §7.4),
//! measured on rendered PCM: the pure-tone voice is rendered through
//! the mixer once with a MIDI 1.0 (14-bit) controller value and once
//! with its MIDI 2.0 (32-bit) counterpart, and the two renders are
//! compared sample-for-sample.
//!
//! Identity where it must hold: a 32-bit value that sits exactly on a
//! 14-bit grid point renders **bit-identically** to the 14-bit value.
//! Difference where it must show: a 32-bit value *between* two 14-bit
//! steps — which the Appendix-D downscale folds onto the same 14-bit
//! step, and therefore onto the same PCM — renders differently.

use oxideav_midi::instruments::tone::ToneInstrument;
use oxideav_midi::instruments::Instrument;
use oxideav_midi::mixer::Mixer;
use oxideav_midi::ump::scaling::scale_32_to_14;

const RATE: u32 = 44_100;
const FRAMES: usize = 4096;

/// Render one held note through a fresh mixer with `setup` applied
/// before the note starts (so the first sample already reflects it).
fn render(setup: impl FnOnce(&mut Mixer)) -> (Vec<f32>, Vec<f32>) {
    let inst = ToneInstrument::new();
    let mut m = Mixer::new();
    setup(&mut m);
    let voice = inst.make_voice(0, 69, 100, RATE).expect("tone voice");
    m.note_on(0, 69, 100, voice);
    let mut l = vec![0.0f32; FRAMES];
    let mut r = vec![0.0f32; FRAMES];
    m.mix_stereo(&mut l, &mut r);
    (l, r)
}

fn peak_abs(a: &[f32]) -> f32 {
    a.iter().fold(0.0f32, |m, s| m.max(s.abs()))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn pitch_bend_32_on_grid_renders_bit_identically_to_14_bit() {
    // Centre: 0x8000_0000 ⇄ 0x2000. Bottom: 0 ⇄ 0 (both exactly −200 c).
    for (v32, v14) in [(0x8000_0000u32, 0x2000u16), (0, 0)] {
        assert_eq!(scale_32_to_14(v32), v14);
        let (l14, _) = render(|m| m.set_pitch_bend(0, v14));
        let (l32, _) = render(|m| m.set_pitch_bend_32(0, v32));
        assert!(peak_abs(&l14) > 0.01, "render must be audible");
        assert_eq!(
            l14, l32,
            "on-grid 32-bit bend {v32:#010x} must match 14-bit {v14:#06x}"
        );
    }
}

#[test]
fn pitch_bend_32_between_14_bit_steps_is_audible_natively_but_not_translated() {
    // Half a 14-bit step above centre (2^17 in 32-bit units).
    let v32 = 0x8000_0000u32 + (1 << 17);
    // The Appendix-D downscale folds it onto the centre step …
    assert_eq!(scale_32_to_14(v32), 0x2000);
    let (centre, _) = render(|m| m.set_pitch_bend(0, 0x2000));
    let (translated, _) = render(|m| m.set_pitch_bend(0, scale_32_to_14(v32)));
    assert_eq!(
        centre, translated,
        "translated path cannot tell it from centre"
    );
    // … while the native path renders a real (fractional-cent) bend.
    let (native, _) = render(|m| m.set_pitch_bend_32(0, v32));
    let d = max_abs_diff(&centre, &native);
    assert!(d > 0.0, "native 32-bit bend must move the PCM");
    // And the movement is tiny — a fraction of a cent (200 c / 2^14 ≈
    // 0.012 c), not a gross pitch change: the last sample of a 440 Hz
    // tone drifts by well under a full cycle over 4096 frames.
    assert!(
        d < 0.05 * peak_abs(&centre),
        "drift {d} is not a sub-cent bend"
    );
}

#[test]
fn pitch_bend_32_resolution_is_monotone_across_a_14_bit_step() {
    // Sweep 8 values inside one 14-bit step; each render must differ
    // from the previous one (strictly finer than 14-bit resolution).
    let mut prev: Option<Vec<f32>> = None;
    for i in 0..8u32 {
        let v32 = 0x8000_0000u32 + i * (1 << 15);
        assert_eq!(scale_32_to_14(v32), 0x2000);
        let (l, _) = render(|m| m.set_pitch_bend_32(0, v32));
        if let Some(p) = &prev {
            assert!(
                max_abs_diff(p, &l) > 0.0,
                "step {i} rendered identically to step {}",
                i - 1
            );
        }
        prev = Some(l);
    }
}

// ── 16-bit velocity + Pitch 7.9 attribute (M2-104 §7.4.2 / §7.4.15.3) ──

use oxideav_midi::mixer::{
    midi2_sample_key, midi2_velocity_to_7, ATTRIBUTE_TYPE_NONE, ATTRIBUTE_TYPE_PITCH_7_9,
};

/// Render a single MIDI 2.0 note (16-bit velocity + attribute) through
/// a fresh mixer.
fn render_midi2(velocity: u16, attribute_type: u8, attribute: u16) -> Vec<f32> {
    let inst = ToneInstrument::new();
    let mut m = Mixer::new();
    let key = midi2_sample_key(69, attribute_type, attribute);
    let voice = inst
        .make_voice(0, key, midi2_velocity_to_7(velocity), RATE)
        .expect("tone voice");
    m.note_on_midi2(0, 69, velocity, attribute_type, attribute, voice);
    let mut l = vec![0.0f32; FRAMES];
    let mut r = vec![0.0f32; FRAMES];
    m.mix_stereo(&mut l, &mut r);
    l
}

#[test]
fn velocity_16_on_the_7_bit_grid_renders_bit_identically_to_midi1() {
    for v7 in [1u8, 64, 100, 127] {
        let (l1, _) = render(|_| {});
        // `render` strikes at velocity 100; rebuild at v7 for both.
        let inst = ToneInstrument::new();
        let mut m = Mixer::new();
        m.note_on(0, 69, v7, inst.make_voice(0, 69, v7, RATE).unwrap());
        let mut l1b = vec![0.0f32; FRAMES];
        let mut r = vec![0.0f32; FRAMES];
        m.mix_stereo(&mut l1b, &mut r);
        let l2 = render_midi2(u16::from(v7) << 9, ATTRIBUTE_TYPE_NONE, 0);
        assert_eq!(l1b, l2, "velocity {v7}");
        if v7 == 100 {
            assert_eq!(l1, l2);
        }
    }
}

#[test]
fn velocity_16_low_bits_scale_the_render_where_translation_cannot() {
    let v_grid = 100u16 << 9;
    let v_half = v_grid | 256;
    // Appendix D folds both onto velocity 100 …
    assert_eq!(midi2_velocity_to_7(v_grid), midi2_velocity_to_7(v_half));
    let grid = render_midi2(v_grid, ATTRIBUTE_TYPE_NONE, 0);
    let half = render_midi2(v_half, ATTRIBUTE_TYPE_NONE, 0);
    // … natively the half-step note is exactly (v_half / v_grid) louder,
    // sample for sample (a static gain: same waveform, same envelope).
    let ratio = f32::from(v_half) / f32::from(v_grid);
    for (i, (g, h)) in grid.iter().zip(&half).enumerate() {
        let want = g * ratio;
        assert!(
            (want - h).abs() <= 1e-6 * want.abs().max(1e-3),
            "sample {i}: {h} vs {want}"
        );
    }
    assert!(peak_abs(&half) > peak_abs(&grid));
}

#[test]
fn pitch_7_9_half_semitone_equals_an_exact_50_cent_bend() {
    // Pitch 69 + 256/512 HCU is exactly +50 c above A4; a 32-bit bend
    // of 0xA000_0000 at the ±200 c default is exactly +50 c too. Both
    // feed the same fractional-cents voice hook → identical PCM.
    let attr = (69u16 << 9) | 256;
    let via_attr = render_midi2(100 << 9, ATTRIBUTE_TYPE_PITCH_7_9, attr);
    let (via_bend, _) = render(|m| m.set_pitch_bend_32(0, 0xA000_0000));
    assert_eq!(via_attr, via_bend);
    // And a Pitch 7.9 with a zero fraction is the plain note.
    let plain = render_midi2(100 << 9, ATTRIBUTE_TYPE_NONE, 0);
    let whole = render_midi2(100 << 9, ATTRIBUTE_TYPE_PITCH_7_9, 69 << 9);
    assert_eq!(plain, whole);
    // Note number is only an index for a Pitch 7.9 note: index 5 with
    // pitch 69.0 renders exactly like the plain A4.
    let inst = ToneInstrument::new();
    let mut m = Mixer::new();
    let key = midi2_sample_key(5, ATTRIBUTE_TYPE_PITCH_7_9, 69 << 9);
    assert_eq!(key, 69);
    m.note_on_midi2(
        0,
        5,
        100 << 9,
        ATTRIBUTE_TYPE_PITCH_7_9,
        69 << 9,
        inst.make_voice(0, key, 100, RATE).unwrap(),
    );
    let mut l = vec![0.0f32; FRAMES];
    let mut r = vec![0.0f32; FRAMES];
    m.mix_stereo(&mut l, &mut r);
    assert_eq!(l, plain);
}
