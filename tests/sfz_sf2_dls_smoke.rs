//! Round-8 smoke: load an SFZ patch, dump regions; load an SF2 bank,
//! dump preset list; load a DLS bank, dump instrument + region table.
//! Exercises the public API surface that round-2 voice generation
//! will build on.

use std::path::Path;

use oxideav_midi::instruments::{
    dls::{DlsArtKind, DlsInstrument},
    sf2::Sf2Instrument,
    sfz::{LoopMode, SfzInstrument},
};

#[test]
fn sfz_dump_regions_basic_template() {
    // Tutorial-shaped patch — one global volume, one group with two
    // regions covering the bottom and top half of the keyboard.
    let text = "
        // basic SFZ template
        <control> default_path=
        <global> volume=-3
        <group>  cutoff=2200 trigger=attack loop_mode=no_loop
        <region> sample=lower.wav lokey=0 hikey=59 pitch_keycenter=60
        <region> sample=upper.wav lokey=60 hikey=127 pitch_keycenter=72 tune=15
    ";
    let inst = SfzInstrument::parse_str("template", text).expect("parse SFZ");
    let regions = inst.regions();
    assert_eq!(
        regions.len(),
        2,
        "expected two regions, got {}",
        regions.len()
    );

    // Region 1: lower half.
    assert_eq!(
        regions[0].sample_path.as_deref(),
        Some(Path::new("lower.wav"))
    );
    assert_eq!(regions[0].lokey, 0);
    assert_eq!(regions[0].hikey, 59);
    assert_eq!(regions[0].pitch_keycenter, 60);
    assert_eq!(regions[0].volume, -3.0); // inherited from global
    assert_eq!(regions[0].loop_mode, LoopMode::NoLoop); // from group
    assert_eq!(regions[0].trigger, "attack");
    assert_eq!(
        regions[0].opcodes.get("cutoff").map(String::as_str),
        Some("2200"),
        "group cutoff should propagate into region opcode map",
    );

    // Region 2: upper half + per-region tune.
    assert_eq!(
        regions[1].sample_path.as_deref(),
        Some(Path::new("upper.wav"))
    );
    assert_eq!(regions[1].lokey, 60);
    assert_eq!(regions[1].hikey, 127);
    assert_eq!(regions[1].pitch_keycenter, 72);
    assert_eq!(regions[1].tune, 15);
    assert_eq!(regions[1].volume, -3.0);

    // Pretty-print to make sure the public API is friendly enough to
    // diagnose a patch from a one-liner. (No assertion on the exact
    // text — just exercise Display via Debug for now.)
    let dump: Vec<String> = regions
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "region {}: sample={:?} key=[{}, {}] vel=[{}, {}] center={} vol={}dB",
                i, r.sample_path, r.lokey, r.hikey, r.lovel, r.hivel, r.pitch_keycenter, r.volume,
            )
        })
        .collect();
    assert_eq!(dump.len(), 2);
    assert!(dump[0].contains("lower.wav"));
    assert!(dump[1].contains("upper.wav"));
}

#[test]
fn sf2_dump_preset_list() {
    // Build a tiny in-memory SF2 with one preset (Test Preset, prog 0,
    // bank 0). Reuses the same fixture builder pattern as the lib-side
    // tests but inlined here so the round-7 dispatch leaves a public
    // smoke test that `cargo test` runs without any external file.
    let blob = build_one_preset_sf2();
    let inst = Sf2Instrument::from_bytes("smoke.sf2", &blob).expect("parse SF2");
    let presets = &inst.bank().presets;
    assert!(!presets.is_empty(), "expected at least one preset");
    let p = &presets[0];
    assert_eq!(p.program, 0);
    assert_eq!(p.bank, 0);
    assert!(p.name.starts_with("Test"), "got name {:?}", p.name);

    // Dump (the round-2 SFZ-to-MIDI bridge will iterate this list).
    let dump: Vec<String> = presets
        .iter()
        .map(|p| {
            format!(
                "preset bank={} prog={} name={:?}",
                p.bank, p.program, p.name
            )
        })
        .collect();
    assert!(dump.iter().any(|line| line.contains("prog=0")));
}

// ----- minimal SF2 fixture builder (mirrors the lib-side helper) -----

fn build_one_preset_sf2() -> Vec<u8> {
    // 20-frame ramp at 22 050 Hz, root key 60, looping.
    let mut smpl_bytes = Vec::with_capacity(40);
    for i in 0i32..20 {
        let v = (i * 800 - 8000) as i16;
        smpl_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let mut info = Vec::new();
    push_riff(&mut info, b"ifil", &[0x02, 0x00, 0x04, 0x00]);
    push_riff(&mut info, b"INAM", b"SmokeBank\0");
    let mut info_list = Vec::from(b"INFO" as &[u8]);
    info_list.extend_from_slice(&info);

    let mut sdta = Vec::new();
    push_riff(&mut sdta, b"smpl", &smpl_bytes);
    let mut sdta_list = Vec::from(b"sdta" as &[u8]);
    sdta_list.extend_from_slice(&sdta);

    const GEN_SAMPLE_MODES: u16 = 54;
    const GEN_SAMPLE_ID: u16 = 53;
    const GEN_INSTRUMENT: u16 = 41;
    let phdr = concat_records(&[
        phdr_record("Test Preset", 0, 0, 0),
        phdr_record("EOP", 0, 0, 1),
    ]);
    let pbag = concat_records(&[bag_record(0, 0), bag_record(1, 0)]);
    let pmod = vec![0u8; 10];
    let pgen = concat_records(&[gen_record(GEN_INSTRUMENT, 0), gen_record(0, 0)]);
    let inst = concat_records(&[inst_record("Test Inst", 0), inst_record("EOI", 2)]);
    let ibag = concat_records(&[bag_record(0, 0), bag_record(2, 0)]);
    let imod = vec![0u8; 10];
    let igen = concat_records(&[
        gen_record(GEN_SAMPLE_MODES, 1),
        gen_record(GEN_SAMPLE_ID, 0),
        gen_record(0, 0),
    ]);
    let shdr = concat_records(&[
        shdr_record("RampLoop", 0, 20, 5, 15, 22_050, 60, 0, 0, 1),
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

fn push_riff(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(tag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

fn concat_records(rs: &[Vec<u8>]) -> Vec<u8> {
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

// ----- DLS smoke: dump instrument + region table -----

#[test]
fn dls_dump_instrument_and_region_table() {
    // Hand-built minimal DLS Level 1 file: 1 instrument (bank 0,
    // program 5), 2 regions splitting the keyboard at C4, 1 wave-pool
    // sample shared across both regions, plus an instrument-level
    // articulation block. Exercises the parser end-to-end and
    // demonstrates the public API a round-2 voice generator will
    // consume.
    let blob = build_two_region_dls();
    let inst = DlsInstrument::parse_bytes("smoke.dls", &blob).expect("parse DLS");
    let bank = inst.bank();

    // Top-level metadata + pool table.
    assert_eq!(bank.declared_instrument_count, 1);
    assert_eq!(bank.version.map(|v| (v.0, v.1)), Some((1, 1)));
    assert_eq!(bank.info.name.as_deref(), Some("SmokeBank"));
    assert_eq!(bank.wave_pool_offsets.len(), 1);

    // Wave pool: one 8-bit mono sample @ 22050 Hz.
    assert_eq!(bank.waves.len(), 1);
    assert_eq!(bank.waves[0].channels, 1);
    assert_eq!(bank.waves[0].sample_rate, 22_050);
    assert_eq!(bank.waves[0].bits_per_sample, 8);

    // Instruments: one melodic instrument carrying two regions.
    assert_eq!(bank.instruments.len(), 1);
    let i0 = &bank.instruments[0];
    assert!(!i0.is_drum());
    assert_eq!(i0.program_number(), 5);
    assert_eq!(i0.regions.len(), 2);
    assert_eq!(i0.declared_region_count, 2);
    assert_eq!(i0.articulation.len(), 1);
    assert_eq!(i0.articulation[0].kind, DlsArtKind::Art1);

    // Region 0: lower half (0..=59), Region 1: upper half (60..=127).
    assert_eq!(i0.regions[0].key_lo, 0);
    assert_eq!(i0.regions[0].key_hi, 59);
    assert_eq!(i0.regions[1].key_lo, 60);
    assert_eq!(i0.regions[1].key_hi, 127);
    // Both regions point at wave-pool entry 0.
    assert_eq!(i0.regions[0].wlnk.unwrap().table_index, 0);
    assert_eq!(i0.regions[1].wlnk.unwrap().table_index, 0);

    // Dump (the round-2 voice generator will iterate this list).
    let dump: Vec<String> = i0
        .regions
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "region {}: key=[{}, {}] vel=[{}, {}] wlnk_idx={:?} arts={}",
                i,
                r.key_lo,
                r.key_hi,
                r.vel_lo,
                r.vel_hi,
                r.wlnk.map(|w| w.table_index),
                r.articulation.len(),
            )
        })
        .collect();
    assert_eq!(dump.len(), 2);
    assert!(dump[0].contains("[0, 59]"));
    assert!(dump[1].contains("[60, 127]"));
}

fn build_two_region_dls() -> Vec<u8> {
    // 8 frames of 8-bit unsigned PCM (silence + ramp).
    let pcm = vec![0x80u8, 0x90, 0xA0, 0xB0, 0xC0, 0xB0, 0xA0, 0x80];

    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&1u16.to_le_bytes()); // mono
    fmt.extend_from_slice(&22_050u32.to_le_bytes());
    fmt.extend_from_slice(&22_050u32.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes()); // block align
    fmt.extend_from_slice(&8u16.to_le_bytes()); // bits/sample

    let mut wave_body = Vec::from(b"wave" as &[u8]);
    push_riff_chunk(&mut wave_body, b"fmt ", &fmt);
    push_riff_chunk(&mut wave_body, b"data", &pcm);

    let mut wvpl_body = Vec::from(b"wvpl" as &[u8]);
    push_riff_chunk(&mut wvpl_body, b"LIST", &wave_body);

    // ptbl: one cue at offset 0.
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
    push_riff_chunk(&mut info_body, b"INAM", b"SmokeBank\0");

    // Instrument body: insh + lrgn (2 regions) + lart (1 connection).
    let mut insh = Vec::new();
    insh.extend_from_slice(&2u32.to_le_bytes()); // cRegions
    insh.extend_from_slice(&0u32.to_le_bytes()); // ulBank
    insh.extend_from_slice(&5u32.to_le_bytes()); // ulInstrument = 5

    // Region 0: keys 0..=59.
    let mut rgnh0 = Vec::new();
    rgnh0.extend_from_slice(&0u16.to_le_bytes());
    rgnh0.extend_from_slice(&59u16.to_le_bytes());
    rgnh0.extend_from_slice(&0u16.to_le_bytes());
    rgnh0.extend_from_slice(&127u16.to_le_bytes());
    rgnh0.extend_from_slice(&0u16.to_le_bytes());
    rgnh0.extend_from_slice(&0u16.to_le_bytes());

    let mut wlnk0 = Vec::new();
    wlnk0.extend_from_slice(&0u16.to_le_bytes());
    wlnk0.extend_from_slice(&0u16.to_le_bytes());
    wlnk0.extend_from_slice(&1u32.to_le_bytes());
    wlnk0.extend_from_slice(&0u32.to_le_bytes());

    let mut wsmp0 = Vec::new();
    wsmp0.extend_from_slice(&20u32.to_le_bytes());
    wsmp0.extend_from_slice(&60u16.to_le_bytes());
    wsmp0.extend_from_slice(&0i16.to_le_bytes());
    wsmp0.extend_from_slice(&0i32.to_le_bytes());
    wsmp0.extend_from_slice(&0u32.to_le_bytes());
    wsmp0.extend_from_slice(&0u32.to_le_bytes());

    let mut rgn0_body = Vec::from(b"rgn " as &[u8]);
    push_riff_chunk(&mut rgn0_body, b"rgnh", &rgnh0);
    push_riff_chunk(&mut rgn0_body, b"wsmp", &wsmp0);
    push_riff_chunk(&mut rgn0_body, b"wlnk", &wlnk0);

    // Region 1: keys 60..=127.
    let mut rgnh1 = Vec::new();
    rgnh1.extend_from_slice(&60u16.to_le_bytes());
    rgnh1.extend_from_slice(&127u16.to_le_bytes());
    rgnh1.extend_from_slice(&0u16.to_le_bytes());
    rgnh1.extend_from_slice(&127u16.to_le_bytes());
    rgnh1.extend_from_slice(&0u16.to_le_bytes());
    rgnh1.extend_from_slice(&0u16.to_le_bytes());

    let mut wlnk1 = Vec::new();
    wlnk1.extend_from_slice(&0u16.to_le_bytes());
    wlnk1.extend_from_slice(&0u16.to_le_bytes());
    wlnk1.extend_from_slice(&1u32.to_le_bytes());
    wlnk1.extend_from_slice(&0u32.to_le_bytes());

    let mut rgn1_body = Vec::from(b"rgn " as &[u8]);
    push_riff_chunk(&mut rgn1_body, b"rgnh", &rgnh1);
    push_riff_chunk(&mut rgn1_body, b"wlnk", &wlnk1);

    let mut lrgn_body = Vec::from(b"lrgn" as &[u8]);
    push_riff_chunk(&mut lrgn_body, b"LIST", &rgn0_body);
    push_riff_chunk(&mut lrgn_body, b"LIST", &rgn1_body);

    // art1 (instrument-level): EG1 → attenuation, scale 0.
    let mut art1 = Vec::new();
    art1.extend_from_slice(&8u32.to_le_bytes()); // cbSize
    art1.extend_from_slice(&1u32.to_le_bytes()); // 1 connection
    art1.extend_from_slice(&0x0004u16.to_le_bytes()); // CONN_SRC_EG1
    art1.extend_from_slice(&0x0000u16.to_le_bytes()); // CONN_SRC_NONE
    art1.extend_from_slice(&0x0001u16.to_le_bytes()); // CONN_DST_ATTENUATION (DLS1)
    art1.extend_from_slice(&0x0000u16.to_le_bytes());
    art1.extend_from_slice(&0i32.to_le_bytes());

    let mut lart_body = Vec::from(b"lart" as &[u8]);
    push_riff_chunk(&mut lart_body, b"art1", &art1);

    let mut ins_body = Vec::from(b"ins " as &[u8]);
    push_riff_chunk(&mut ins_body, b"insh", &insh);
    push_riff_chunk(&mut ins_body, b"LIST", &lrgn_body);
    push_riff_chunk(&mut ins_body, b"LIST", &lart_body);

    let mut lins_body = Vec::from(b"lins" as &[u8]);
    push_riff_chunk(&mut lins_body, b"LIST", &ins_body);

    let mut body = Vec::from(b"DLS " as &[u8]);
    push_riff_chunk(&mut body, b"vers", &vers);
    push_riff_chunk(&mut body, b"colh", &colh);
    push_riff_chunk(&mut body, b"LIST", &lins_body);
    push_riff_chunk(&mut body, b"ptbl", &ptbl);
    push_riff_chunk(&mut body, b"LIST", &wvpl_body);
    push_riff_chunk(&mut body, b"LIST", &info_body);

    let mut out = Vec::from(b"RIFF" as &[u8]);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

fn push_riff_chunk(out: &mut Vec<u8>, tag: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(tag);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}
