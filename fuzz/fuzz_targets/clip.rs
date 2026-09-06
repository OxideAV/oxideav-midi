#![no_main]

//! Fuzz the MIDI Clip File (`.midi2`, M2-116) parser against
//! arbitrary bytes.
//!
//! The contract matches the other targets: every input — well-formed
//! clips, random bytes, truncated UMP word streams, hostile Delta
//! Clockstamp runs, missing DCTPQ / Start / End of Clip framing —
//! returns a `Result`; panics, debug overflows, out-of-bounds reads,
//! or attacker-controlled allocations are bugs.
//!
//! When the parse succeeds we additionally require the writer to be a
//! fixed point of the reader (`parse(write(clip)) == clip`), run the
//! Appendix-D translation to SMF, and render a bounded slice of the
//! clip through the **native** MIDI 2.0 scheduler + mixer path (16-bit
//! velocity, 32-bit controllers, per-note messages, Registered /
//! Assignable Controllers, Profiles) with the pure-tone instrument —
//! so fuzz-discovered message shapes reach the synthesis code too.

use libfuzzer_sys::fuzz_target;
use oxideav_midi::instruments::tone::ToneInstrument;
use oxideav_midi::mixer::Mixer;
use oxideav_midi::scheduler::{dispatch_universal_sysex, Scheduler};

fuzz_target!(|data: &[u8]| {
    if let Ok(clip) = oxideav_midi::clip::parse(data) {
        let _ = clip.to_smf();
        let bytes = oxideav_midi::clip::write(&clip).expect("a parsed clip always writes");
        let again = oxideav_midi::clip::parse(&bytes).expect("a written clip parses");
        assert_eq!(again, clip, "clip writer must be a fixed point");
        // Bounded native render: 8 blocks of 1024 frames.
        let inst = ToneInstrument::new();
        let mut mixer = Mixer::new();
        for p in &clip.profiles {
            dispatch_universal_sysex(p, &mut mixer);
        }
        let mut sched = Scheduler::from_clip(&clip, 44_100);
        let (mut l, mut r) = (vec![0.0f32; 1024], vec![0.0f32; 1024]);
        for _ in 0..8 {
            sched.step(1024, &mut mixer, &inst);
            mixer.mix_stereo(&mut l, &mut r);
        }
    }
});
