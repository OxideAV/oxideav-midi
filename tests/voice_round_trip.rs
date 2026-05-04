//! Round-9 end-to-end voice tests.
//!
//! Exercises [`InstrumentSource`] + [`MidiDecoder::with_instrument_source`]
//! by writing temporary instrument banks (SFZ + WAV / SF2 / DLS) to
//! disk, instantiating the decoder, feeding it a tiny SMF, and asserting
//! the rendered PCM has non-trivial RMS energy.

use std::path::PathBuf;
use std::sync::Arc;

use oxideav_core::{Decoder, Error, Frame, Packet, TimeBase};
use oxideav_midi::{instruments::Instrument, InstrumentSource, MidiDecoder, OUTPUT_SAMPLE_RATE};

/// Build a minimal one-track SMF: a single 60-key note at chan 0 vel
/// 100 lasting ~0.5 s plus an EOT.
fn one_note_smf(key: u8) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(b"MThd");
    blob.extend_from_slice(&6u32.to_be_bytes());
    blob.extend_from_slice(&0u16.to_be_bytes()); // format 0
    blob.extend_from_slice(&1u16.to_be_bytes()); // 1 track
    blob.extend_from_slice(&480u16.to_be_bytes());

    let mut track: Vec<u8> = Vec::new();
    // tick 0 set tempo 500 000 us/qn (= 120 BPM).
    track.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
    // tick 0 note on chan 0 key vel 100.
    track.extend_from_slice(&[0x00, 0x90, key, 0x64]);
    // tick 240 (≈ 0.5 s) note off.
    track.extend_from_slice(&[0x81, 0x70, 0x80, key, 0x40]);
    // tick + 240 EOT.
    track.extend_from_slice(&[0x81, 0x70, 0xFF, 0x2F, 0x00]);

    blob.extend_from_slice(b"MTrk");
    blob.extend_from_slice(&(track.len() as u32).to_be_bytes());
    blob.extend_from_slice(&track);
    blob
}

/// Drive the decoder until it returns Eof (or hits the chunk cap),
/// returning every i16 sample produced.
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

/// Compute the RMS of a sample buffer normalised to the i16 full-scale
/// range. Used as a "did this voice produce audible signal?" gate.
fn rms_normalised(samples: &[i16]) -> f64 {
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

// =========================================================================
// SFZ end-to-end.
// =========================================================================

/// Minimal WAV: 64 frames of a 1 kHz sine at 44 100 Hz, 16-bit mono.
/// Just enough sample to feed a sample-loop voice.
fn one_khz_sine_wav() -> Vec<u8> {
    let n: usize = 4096;
    let rate: u32 = 44_100;
    let mut samples: Vec<i16> = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        let v = (t * 1000.0 * std::f32::consts::TAU).sin();
        samples.push((v * 16_384.0) as i16);
    }
    let data_size = (samples.len() * 2) as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36u32 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&rate.to_le_bytes());
    bytes.extend_from_slice(&(rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for s in &samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

#[test]
fn sfz_end_to_end_renders_nonzero_rms() {
    // Write SFZ + WAV under tempdir.
    let tmp = std::env::temp_dir().join("oxideav-midi-roundtrip-sfz");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let wav_path = tmp.join("sine.wav");
    std::fs::write(&wav_path, one_khz_sine_wav()).unwrap();
    let sfz_path = tmp.join("sine.sfz");
    std::fs::write(
        &sfz_path,
        b"<region> sample=sine.wav lokey=0 hikey=127 pitch_keycenter=60 \
          loop_mode=loop_continuous loop_start=0 loop_end=4096\n",
    )
    .unwrap();

    let src = InstrumentSource::sfz(&sfz_path);
    let mut dec = MidiDecoder::with_instrument_source(src).unwrap();
    let pcm = render_to_pcm(&mut dec, one_note_smf(60));
    assert!(!pcm.is_empty(), "sfz render produced no samples");
    let rms = rms_normalised(&pcm);
    assert!(
        rms > 0.001,
        "SFZ render too quiet — rms {rms}, len {}",
        pcm.len(),
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

// =========================================================================
// SF2 end-to-end.
// =========================================================================

#[test]
fn sf2_end_to_end_renders_nonzero_rms() {
    let blob = build_sine_sf2();
    // The SF2 is in-memory; route through MidiDecoder::with_instrument
    // (no on-disk path needed). The InstrumentSource enum is exercised
    // by the SFZ + DLS tests.
    let inst: Arc<dyn Instrument> = Arc::new(
        oxideav_midi::instruments::sf2::Sf2Instrument::from_bytes("sine.sf2", &blob).unwrap(),
    );
    let mut dec = MidiDecoder::new(inst, OUTPUT_SAMPLE_RATE);
    let pcm = render_to_pcm(&mut dec, one_note_smf(60));
    assert!(!pcm.is_empty(), "sf2 render produced no samples");
    let rms = rms_normalised(&pcm);
    assert!(
        rms > 0.001,
        "SF2 render too quiet — rms {rms}, len {}",
        pcm.len(),
    );
}

/// Build an SF2 with a single looping 1 kHz-ish sine sample at 22 050 Hz.
fn build_sine_sf2() -> Vec<u8> {
    // 100-frame sine at 22 050 Hz (~ 220 Hz fundamental — roughly key A3).
    let n: usize = 100;
    let mut smpl_bytes = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = ((i as f32 / n as f32) * std::f32::consts::TAU).sin();
        let s = (v * 16_000.0) as i16;
        smpl_bytes.extend_from_slice(&s.to_le_bytes());
    }

    let mut info = Vec::new();
    push_riff(&mut info, b"ifil", &[0x02, 0x00, 0x04, 0x00]);
    push_riff(&mut info, b"INAM", b"SineBank\0");
    let mut info_list = Vec::from(b"INFO" as &[u8]);
    info_list.extend_from_slice(&info);

    let mut sdta = Vec::new();
    push_riff(&mut sdta, b"smpl", &smpl_bytes);
    let mut sdta_list = Vec::from(b"sdta" as &[u8]);
    sdta_list.extend_from_slice(&sdta);

    const GEN_SAMPLE_MODES: u16 = 54;
    const GEN_SAMPLE_ID: u16 = 53;
    const GEN_INSTRUMENT: u16 = 41;
    let phdr = concat(&[phdr_record("Sine", 0, 0, 0), phdr_record("EOP", 0, 0, 1)]);
    let pbag = concat(&[bag_record(0, 0), bag_record(1, 0)]);
    let pmod = vec![0u8; 10];
    let pgen = concat(&[gen_record(GEN_INSTRUMENT, 0), gen_record(0, 0)]);
    let inst = concat(&[inst_record("Sine", 0), inst_record("EOI", 2)]);
    let ibag = concat(&[bag_record(0, 0), bag_record(2, 0)]);
    let imod = vec![0u8; 10];
    let igen = concat(&[
        gen_record(GEN_SAMPLE_MODES, 1),
        gen_record(GEN_SAMPLE_ID, 0),
        gen_record(0, 0),
    ]);
    let shdr = concat(&[
        shdr_record("SineLoop", 0, n as u32, 0, n as u32, 22_050, 60, 0, 0, 1),
        shdr_record("EOS", 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]);

    let mut pdta = Vec::new();
    push_riff(&mut pdta, b"phdr", &phdr);
    push_riff(&mut pdta, b"pbag", &pbag);
    push_riff(&mut pdta, b"pmod", &pmod);
    push_riff(&mut pdta, b"pgen", &pgen);
    push_riff(&mut pdta, b"inst", &inst);
    push_riff(&mut pdta, b"ibag", &ibag);
    push_riff(&mut pdta, b"imod", &imod);
    push_riff(&mut pdta, b"igen", &igen);
    push_riff(&mut pdta, b"shdr", &shdr);
    let mut pdta_list = Vec::from(b"pdta" as &[u8]);
    pdta_list.extend_from_slice(&pdta);

    let mut body = Vec::from(b"sfbk" as &[u8]);
    push_riff(&mut body, b"LIST", &info_list);
    push_riff(&mut body, b"LIST", &sdta_list);
    push_riff(&mut body, b"LIST", &pdta_list);
    let mut out = Vec::from(b"RIFF" as &[u8]);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

// =========================================================================
// DLS end-to-end.
// =========================================================================

#[test]
fn dls_end_to_end_renders_nonzero_rms() {
    let blob = build_sine_dls();
    let tmp_path: PathBuf = std::env::temp_dir().join("oxideav-midi-roundtrip-dls.dls");
    std::fs::write(&tmp_path, &blob).unwrap();
    let src = InstrumentSource::dls(&tmp_path);
    let mut dec = MidiDecoder::with_instrument_source(src).unwrap();
    let pcm = render_to_pcm(&mut dec, one_note_smf(60));
    assert!(!pcm.is_empty(), "dls render produced no samples");
    let rms = rms_normalised(&pcm);
    assert!(
        rms > 0.001,
        "DLS render too quiet — rms {rms}, len {}",
        pcm.len(),
    );
    let _ = std::fs::remove_file(&tmp_path);
}

/// Build a one-instrument DLS bank with an 8-bit unsigned PCM 1 kHz
/// sine at 22 050 Hz. Single region covering the full keyboard.
fn build_sine_dls() -> Vec<u8> {
    // 100-frame 8-bit unsigned PCM sine.
    let n: usize = 100;
    let mut pcm = Vec::with_capacity(n);
    for i in 0..n {
        let v = ((i as f32 / n as f32) * std::f32::consts::TAU).sin();
        // 0x80 = silence; ±0x7F = peak.
        let s = (v * 100.0) as i32 + 128;
        pcm.push(s.clamp(0, 255) as u8);
    }

    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&22_050u32.to_le_bytes());
    fmt.extend_from_slice(&22_050u32.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&8u16.to_le_bytes());

    // wsmp on the wave: unity_note=60, no loops.
    let mut wsmp = Vec::new();
    wsmp.extend_from_slice(&20u32.to_le_bytes());
    wsmp.extend_from_slice(&60u16.to_le_bytes());
    wsmp.extend_from_slice(&0i16.to_le_bytes());
    wsmp.extend_from_slice(&0i32.to_le_bytes());
    wsmp.extend_from_slice(&0u32.to_le_bytes());
    wsmp.extend_from_slice(&0u32.to_le_bytes());

    let mut wave_body = Vec::from(b"wave" as &[u8]);
    push_riff(&mut wave_body, b"fmt ", &fmt);
    push_riff(&mut wave_body, b"data", &pcm);
    push_riff(&mut wave_body, b"wsmp", &wsmp);

    let mut wvpl_body = Vec::from(b"wvpl" as &[u8]);
    push_riff(&mut wvpl_body, b"LIST", &wave_body);

    let mut ptbl = Vec::new();
    ptbl.extend_from_slice(&8u32.to_le_bytes());
    ptbl.extend_from_slice(&1u32.to_le_bytes());
    ptbl.extend_from_slice(&0u32.to_le_bytes());

    let mut colh = Vec::new();
    colh.extend_from_slice(&1u32.to_le_bytes());

    let mut vers = Vec::new();
    vers.extend_from_slice(&((1u32 << 16) | 1u32).to_le_bytes());
    vers.extend_from_slice(&0u32.to_le_bytes());

    let mut info_body = Vec::from(b"INFO" as &[u8]);
    push_riff(&mut info_body, b"INAM", b"SineBank\0");

    let mut insh = Vec::new();
    insh.extend_from_slice(&1u32.to_le_bytes());
    insh.extend_from_slice(&0u32.to_le_bytes());
    insh.extend_from_slice(&0u32.to_le_bytes());

    let mut rgnh = Vec::new();
    rgnh.extend_from_slice(&0u16.to_le_bytes());
    rgnh.extend_from_slice(&127u16.to_le_bytes());
    rgnh.extend_from_slice(&0u16.to_le_bytes());
    rgnh.extend_from_slice(&127u16.to_le_bytes());
    rgnh.extend_from_slice(&0u16.to_le_bytes());
    rgnh.extend_from_slice(&0u16.to_le_bytes());

    let mut rgn_wsmp = Vec::new();
    rgn_wsmp.extend_from_slice(&20u32.to_le_bytes());
    rgn_wsmp.extend_from_slice(&60u16.to_le_bytes());
    rgn_wsmp.extend_from_slice(&0i16.to_le_bytes());
    rgn_wsmp.extend_from_slice(&0i32.to_le_bytes());
    rgn_wsmp.extend_from_slice(&0u32.to_le_bytes());
    rgn_wsmp.extend_from_slice(&0u32.to_le_bytes());

    let mut wlnk = Vec::new();
    wlnk.extend_from_slice(&0u16.to_le_bytes());
    wlnk.extend_from_slice(&0u16.to_le_bytes());
    wlnk.extend_from_slice(&1u32.to_le_bytes());
    wlnk.extend_from_slice(&0u32.to_le_bytes());

    let mut rgn_body = Vec::from(b"rgn " as &[u8]);
    push_riff(&mut rgn_body, b"rgnh", &rgnh);
    push_riff(&mut rgn_body, b"wsmp", &rgn_wsmp);
    push_riff(&mut rgn_body, b"wlnk", &wlnk);

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

// =========================================================================
// Shared RIFF helpers.
// =========================================================================

fn push_riff(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(tag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

fn concat(rs: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in rs {
        out.extend_from_slice(r);
    }
    out
}

fn name20(s: &str) -> [u8; 20] {
    let mut buf = [0u8; 20];
    let bytes = s.as_bytes();
    let n = bytes.len().min(19);
    buf[..n].copy_from_slice(&bytes[..n]);
    buf
}

fn phdr_record(name: &str, program: u16, bank: u16, pbag_start: u16) -> Vec<u8> {
    let mut r = vec![0u8; 38];
    r[0..20].copy_from_slice(&name20(name));
    r[20..22].copy_from_slice(&program.to_le_bytes());
    r[22..24].copy_from_slice(&bank.to_le_bytes());
    r[24..26].copy_from_slice(&pbag_start.to_le_bytes());
    r
}

fn inst_record(name: &str, ibag_start: u16) -> Vec<u8> {
    let mut r = vec![0u8; 22];
    r[0..20].copy_from_slice(&name20(name));
    r[20..22].copy_from_slice(&ibag_start.to_le_bytes());
    r
}

fn bag_record(gen_start: u16, mod_start: u16) -> Vec<u8> {
    let mut r = vec![0u8; 4];
    r[0..2].copy_from_slice(&gen_start.to_le_bytes());
    r[2..4].copy_from_slice(&mod_start.to_le_bytes());
    r
}

fn gen_record(oper: u16, amount: u16) -> Vec<u8> {
    let mut r = vec![0u8; 4];
    r[0..2].copy_from_slice(&oper.to_le_bytes());
    r[2..4].copy_from_slice(&amount.to_le_bytes());
    r
}

#[allow(clippy::too_many_arguments)]
fn shdr_record(
    name: &str,
    start: u32,
    end: u32,
    start_loop: u32,
    end_loop: u32,
    sample_rate: u32,
    original_key: u8,
    pitch_correction: i8,
    sample_link: u16,
    sample_type: u16,
) -> Vec<u8> {
    let mut r = vec![0u8; 46];
    r[0..20].copy_from_slice(&name20(name));
    r[20..24].copy_from_slice(&start.to_le_bytes());
    r[24..28].copy_from_slice(&end.to_le_bytes());
    r[28..32].copy_from_slice(&start_loop.to_le_bytes());
    r[32..36].copy_from_slice(&end_loop.to_le_bytes());
    r[36..40].copy_from_slice(&sample_rate.to_le_bytes());
    r[40] = original_key;
    r[41] = pitch_correction as u8;
    r[42..44].copy_from_slice(&sample_link.to_le_bytes());
    r[44..46].copy_from_slice(&sample_type.to_le_bytes());
    r
}
