//! SMF→PCM synthesis wall-clock harness + output hasher.
//!
//! A `harness = false` bench so the same binary serves three jobs:
//!
//!   * **timing** (default): render a dense 32-voice SMF through
//!     [`MidiDecoder`] N times and print per-iteration wall time plus
//!     an FNV-1a-64 hash of every PCM byte produced — the hash is the
//!     bit-identity witness for optimization rounds.
//!   * **`--corpus`**: render every in-tree fixture SMF (the three
//!     `fuzz/corpus/smf/*.mid` files + the two synthetic scores below)
//!     through both the SF2 bank and the pure-tone fallback and print
//!     one `name instrument hash n_bytes` row each. Diff the table
//!     before/after a change to prove the output is untouched.
//!   * **`--spin SECS`**: render the dense score in a loop for at
//!     least `SECS` wall seconds — a stable target for a sampling
//!     profiler.
//!
//! Run with:
//!   cargo bench --bench synth_render -- [--iters N] [--corpus] [--spin SECS]

use std::sync::Arc;
use std::time::Instant;

use oxideav_core::{Decoder, Error, Frame, Packet, TimeBase};
use oxideav_midi::instruments::{sf2::Sf2Instrument, tone::ToneInstrument, Instrument};
use oxideav_midi::{MidiDecoder, OUTPUT_SAMPLE_RATE};

// =========================================================================
// FNV-1a 64 — tiny, dependency-free, stable across platforms.
// =========================================================================

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64(state: u64, bytes: &[u8]) -> u64 {
    let mut h = state;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// =========================================================================
// SMF builders.
// =========================================================================

/// Append a variable-length quantity.
fn push_vlq(out: &mut Vec<u8>, mut v: u32) {
    let mut stack = [0u8; 5];
    let mut n = 0;
    loop {
        stack[n] = (v & 0x7F) as u8;
        v >>= 7;
        n += 1;
        if v == 0 {
            break;
        }
    }
    for i in (0..n).rev() {
        let mut b = stack[i];
        if i != 0 {
            b |= 0x80;
        }
        out.push(b);
    }
}

/// One absolute-tick event destined for the single format-0 track.
struct Ev {
    tick: u32,
    /// Tie-break so simultaneous events keep a deterministic order.
    seq: u32,
    bytes: Vec<u8>,
}

/// Build a format-0 SMF (480 tpqn, 120 BPM) from absolute-tick events.
fn build_smf(mut events: Vec<Ev>) -> Vec<u8> {
    events.sort_by_key(|e| (e.tick, e.seq));
    let mut track: Vec<u8> = Vec::new();
    // tick 0: set tempo 500 000 us/qn (120 BPM).
    track.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);
    let mut last = 0u32;
    let mut end = 0u32;
    for ev in &events {
        push_vlq(&mut track, ev.tick - last);
        track.extend_from_slice(&ev.bytes);
        last = ev.tick;
        end = end.max(ev.tick);
    }
    // EOT one beat after the last event.
    push_vlq(&mut track, 480);
    let _ = end;
    track.extend_from_slice(&[0xFF, 0x2F, 0x00]);

    let mut blob = Vec::new();
    blob.extend_from_slice(b"MThd");
    blob.extend_from_slice(&6u32.to_be_bytes());
    blob.extend_from_slice(&0u16.to_be_bytes()); // format 0
    blob.extend_from_slice(&1u16.to_be_bytes()); // 1 track
    blob.extend_from_slice(&480u16.to_be_bytes());
    blob.extend_from_slice(b"MTrk");
    blob.extend_from_slice(&(track.len() as u32).to_be_bytes());
    blob.extend_from_slice(&track);
    blob
}

/// Dense score: 8 melodic channels, each holding 4 overlapping notes at
/// any instant (note-on every 60 ticks, note-off 240 ticks later) — a
/// steady 32 sounding voices, which saturates the mixer pool. Per-channel
/// volume/pan CCs up front plus a pitch-bend sweep on half the channels
/// so the bend → phase-increment path is exercised too. ~24 s of music.
fn dense_smf() -> Vec<u8> {
    let mut events = Vec::new();
    let mut seq = 0u32;
    let mut push = |events: &mut Vec<Ev>, tick: u32, bytes: Vec<u8>| {
        events.push(Ev { tick, seq, bytes });
        seq += 1;
    };

    for ch in 0u8..8 {
        // CC 7 volume (90..104), CC 10 pan spread across the field.
        push(&mut events, 0, vec![0xB0 | ch, 7, 90 + ch * 2]);
        push(&mut events, 0, vec![0xB0 | ch, 10, 16 + ch * 13]);
    }

    // 480 tpqn at 120 BPM = 960 ticks/s; 23 040 ticks = 24 s.
    let total_ticks = 23_040u32;
    let step = 60u32; // new note every 62.5 ms per channel
    let dur = 240u32; // each note rings 250 ms → 4-deep overlap
    let scale = [60u8, 62, 64, 65, 67, 69, 71, 72];
    for ch in 0u8..8 {
        let mut i = 0u32;
        let mut t = 0u32;
        while t + dur <= total_ticks {
            let key = scale[((i + ch as u32) % scale.len() as u32) as usize] - 12 + (ch % 3) * 12;
            push(&mut events, t, vec![0x90 | ch, key, 80 + (i % 40) as u8]);
            push(&mut events, t + dur, vec![0x80 | ch, key, 64]);
            i += 1;
            t += step;
        }
    }

    // Pitch-bend sweeps on channels 0..4: a new 14-bit value every 120
    // ticks walking a triangle around centre.
    for ch in 0u8..4 {
        let mut t = 0u32;
        let mut k = 0i32;
        while t < total_ticks {
            let tri = (k % 64 - 32).abs() - 16; // -16..=16
            let v = (0x2000 + tri * 200).clamp(0, 0x3FFF) as u16;
            push(
                &mut events,
                t,
                vec![0xE0 | ch, (v & 0x7F) as u8, (v >> 7) as u8],
            );
            k += 1;
            t += 120;
        }
    }

    build_smf(events)
}

/// Sparse score: one channel, short melody — small fixture so the
/// corpus table also covers the mostly-idle mixer path.
fn sparse_smf() -> Vec<u8> {
    let mut events = Vec::new();
    let mut seq = 0u32;
    for (i, key) in [60u8, 64, 67, 72, 67, 64, 60].iter().enumerate() {
        let t = i as u32 * 480;
        events.push(Ev {
            tick: t,
            seq,
            bytes: vec![0x90, *key, 100],
        });
        seq += 1;
        events.push(Ev {
            tick: t + 420,
            seq,
            bytes: vec![0x80, *key, 64],
        });
        seq += 1;
    }
    build_smf(events)
}

// =========================================================================
// SF2 fixture bank (RIFF builders shared shape with tests/).
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

/// Build an SF2 with a single looping harmonic-rich sample (2048
/// frames at 22 050 Hz) on preset 0. Long enough that the phase walk
/// + loop wrap both matter; deterministic content (sum of 3 sines).
fn build_bench_sf2() -> Vec<u8> {
    let n: usize = 2048;
    let mut smpl_bytes = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = i as f32 / n as f32 * std::f32::consts::TAU;
        // 8 / 17 / 31 cycles over the loop — all integral so the loop
        // is click-free; mixed amplitudes keep it below full scale.
        let v = (x * 8.0).sin() * 0.6 + (x * 17.0).sin() * 0.25 + (x * 31.0).sin() * 0.1;
        let s = (v * 16_000.0) as i16;
        smpl_bytes.extend_from_slice(&s.to_le_bytes());
    }

    let mut info = Vec::new();
    push_riff(&mut info, b"ifil", &[0x02, 0x00, 0x04, 0x00]);
    push_riff(&mut info, b"INAM", b"BenchBank\0");
    let mut info_list = Vec::from(b"INFO" as &[u8]);
    info_list.extend_from_slice(&info);

    let mut sdta = Vec::new();
    push_riff(&mut sdta, b"smpl", &smpl_bytes);
    let mut sdta_list = Vec::from(b"sdta" as &[u8]);
    sdta_list.extend_from_slice(&sdta);

    const GEN_SAMPLE_MODES: u16 = 54;
    const GEN_SAMPLE_ID: u16 = 53;
    const GEN_INSTRUMENT: u16 = 41;
    let phdr = concat(&[phdr_record("Bench", 0, 0, 0), phdr_record("EOP", 0, 0, 1)]);
    let pbag = concat(&[bag_record(0, 0), bag_record(1, 0)]);
    let pmod = vec![0u8; 10];
    let pgen = concat(&[gen_record(GEN_INSTRUMENT, 0), gen_record(0, 0)]);
    let inst = concat(&[inst_record("Bench", 0), inst_record("EOI", 2)]);
    let ibag = concat(&[bag_record(0, 0), bag_record(2, 0)]);
    let imod = vec![0u8; 10];
    let igen = concat(&[
        gen_record(GEN_SAMPLE_MODES, 1),
        gen_record(GEN_SAMPLE_ID, 0),
        gen_record(0, 0),
    ]);
    let shdr = concat(&[
        shdr_record("BenchLoop", 0, n as u32, 0, n as u32, 22_050, 60, 0, 0, 1),
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
// Render driver.
// =========================================================================

/// Render one SMF through one instrument; return (pcm_bytes, fnv hash).
fn render(instrument: Arc<dyn Instrument>, smf: &[u8]) -> (usize, u64) {
    let mut dec = MidiDecoder::new(instrument, OUTPUT_SAMPLE_RATE);
    let pkt = Packet::new(0, TimeBase::new(1, OUTPUT_SAMPLE_RATE as i64), smf.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    let mut hash = FNV_OFFSET;
    let mut total = 0usize;
    // 24 s of music + tail at 1024-sample chunks ≈ 1070 frames; the
    // 8192 cap is purely a runaway guard.
    for _ in 0..8192 {
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => {
                for plane in &af.data {
                    hash = fnv1a64(hash, plane);
                    total += plane.len();
                }
            }
            Ok(_) => panic!("expected audio frame"),
            Err(Error::Eof) => break,
            Err(other) => panic!("decoder error: {other:?}"),
        }
    }
    (total, hash)
}

fn corpus_table(sf2: &Arc<dyn Instrument>, tone: &Arc<dyn Instrument>) {
    let dir = concat_path(env!("CARGO_MANIFEST_DIR"), "fuzz/corpus/smf");
    let mut entries: Vec<(String, Vec<u8>)> = vec![
        ("dense".into(), dense_smf()),
        ("sparse".into(), sparse_smf()),
    ];
    let mut disk: Vec<_> = std::fs::read_dir(&dir)
        .expect("fuzz/corpus/smf readable")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "mid").unwrap_or(false))
        .collect();
    disk.sort_by_key(|e| e.file_name());
    for e in disk {
        let name = e.file_name().to_string_lossy().into_owned();
        let blob = std::fs::read(e.path()).expect("fixture readable");
        entries.push((name, blob));
    }
    for (name, smf) in &entries {
        for (label, inst) in [("sf2", sf2), ("tone", tone)] {
            let (n, h) = render(Arc::clone(inst), smf);
            println!("{name:<24} {label:<5} {h:016x} {n}");
        }
    }
}

fn concat_path(a: &str, b: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(a);
    p.push(b);
    p
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // `cargo bench` passes `--bench`; ignore it.
    let args: Vec<&str> = args
        .iter()
        .skip(1)
        .map(|s| s.as_str())
        .filter(|s| *s != "--bench")
        .collect();

    let sf2_blob = build_bench_sf2();
    let sf2: Arc<dyn Instrument> =
        Arc::new(Sf2Instrument::from_bytes("bench.sf2", &sf2_blob).expect("parse bench SF2"));
    let tone: Arc<dyn Instrument> = Arc::new(ToneInstrument::new());

    if args.contains(&"--corpus") {
        corpus_table(&sf2, &tone);
        return;
    }

    if let Some(i) = args.iter().position(|a| *a == "--spin") {
        let secs: u64 = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(20);
        let smf = dense_smf();
        let t0 = Instant::now();
        let mut iters = 0u32;
        let mut hash = 0u64;
        while t0.elapsed().as_secs() < secs {
            let (_, h) = render(Arc::clone(&sf2), &smf);
            hash = h;
            iters += 1;
        }
        println!(
            "spin: {iters} iters in {:?}, hash {hash:016x}",
            t0.elapsed()
        );
        return;
    }

    let iters: u32 = args
        .iter()
        .position(|a| *a == "--iters")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);

    let smf = dense_smf();
    // Warm-up (page in the bank, JIT the allocator pools).
    let (n0, h0) = render(Arc::clone(&sf2), &smf);
    println!("warmup: {n0} bytes, hash {h0:016x}");
    let mut best = f64::MAX;
    for i in 0..iters {
        let t0 = Instant::now();
        let (n, h) = render(Arc::clone(&sf2), &smf);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(h, h0, "non-deterministic render");
        assert_eq!(n, n0);
        best = best.min(dt);
        println!("iter {i}: {dt:.2} ms");
    }
    println!("best: {best:.2} ms, hash {h0:016x}, bytes {n0}");
}
