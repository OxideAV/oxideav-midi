//! GM2 Percussion Sound Set — drum-set selection, per-key preset data,
//! and the §2.8.1 mutually-exclusive Note groups.
//!
//! Pure data + selection logic transcribed from the **General MIDI 2**
//! specification (RP-024 v1.2a), Appendix B "GM 2 Percussion Sound Set"
//! and §2.8.1 "Rhythm Channels" — see
//! `docs/audio/midi/gm2-percussion-sets.md`. Facts / tables only.
//!
//! A GM2 Rhythm Channel selects one of nine drum sets by Program Change
//! while Bank Select MSB/LSB = 78H/00H (§3.2, §3.3.1). Each drum set
//! maps MIDI note numbers to individual percussion sounds; some sounds
//! belong to a **mutually-exclusive (EXC) group** so that striking one
//! member promptly mutes any sounding member of the same group (the
//! classic open/closed hi-hat "choke"). Several sets reuse the STANDARD
//! Set for keys they do not redefine (`@` inheritance in Appendix B).
//!
//! The [`Mixer`](crate::mixer::Mixer) consults this module at note-on /
//! note-off time on Rhythm Channels: [`DrumSet::from_program`] resolves
//! the active set, [`DrumSet::exc_group`] drives the choke, and
//! [`DrumSet::honors_note_off`] decides whether a Note Off releases the
//! voice or is ignored (drum one-shots ring out).

/// One of the nine GM2 drum sets selectable in Bank 78H/00H by a
/// Program Change (Appendix B).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrumSet {
    /// PC #1 (00H) — also the GM1 Drum Set.
    Standard,
    /// PC #9 (08H).
    Room,
    /// PC #17 (10H).
    Power,
    /// PC #25 (18H).
    Electronic,
    /// PC #26 (19H).
    Analog,
    /// PC #33 (20H).
    Jazz,
    /// PC #41 (28H).
    Brush,
    /// PC #49 (30H).
    Orchestra,
    /// PC #57 (38H).
    Sfx,
}

/// One resolved percussion sound: its display name, recommended preset
/// pan (0..=127, 64 = centre), and EXC group membership (if any).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrumKey {
    /// Instrument name as printed in Appendix B.
    pub name: &'static str,
    /// Recommended preset pan for this sound (§3.3.5): the channel's
    /// CC 10 Pan *offsets* this per-instrument value rather than
    /// replacing it.
    pub pan: u8,
    /// EXC group number within this set (`Some(n)` = member of group
    /// `n`), or `None` for sounds with no mutual exclusion. Group
    /// numbers are scoped per set; only the note membership is
    /// meaningful across sets.
    pub exc: Option<u8>,
}

/// Internal Appendix-B cell for an inheriting override set: an explicit
/// sound, or an `@` inheritance of the STANDARD Set instrument. (The
/// inheriting sets — ROOM/POWER/ELECTRONIC/ANALOG/JAZZ/BRUSH/ORCHESTRA
/// — have no `---` keys; the only self-contained set, SFX, is resolved
/// directly by [`sfx_key`].)
enum Cell {
    /// Explicit per-set instrument (`[EXCn]` folded into `.exc`).
    Over(DrumKey),
    /// `@` — use the STANDARD Set instrument for this key.
    Inherit,
}

/// Const helper — build a [`DrumKey`].
const fn dk(name: &'static str, pan: u8, exc: Option<u8>) -> DrumKey {
    DrumKey { name, pan, exc }
}

impl DrumSet {
    /// Map a Program Change wire byte (0-based, as it appears on the
    /// MIDI wire) to a drum set while Bank Select is 78H/00H. An
    /// undefined program falls back to the STANDARD Set (the GM1 Drum
    /// Set), per §2.5.
    pub fn from_program(program: u8) -> DrumSet {
        match program {
            0x00 => DrumSet::Standard,
            0x08 => DrumSet::Room,
            0x10 => DrumSet::Power,
            0x18 => DrumSet::Electronic,
            0x19 => DrumSet::Analog,
            0x20 => DrumSet::Jazz,
            0x28 => DrumSet::Brush,
            0x30 => DrumSet::Orchestra,
            0x38 => DrumSet::Sfx,
            // §2.5: an undefined program in Bank 78H/00H uses the
            // STANDARD Set (Program 1 / the GM1 Drum Set).
            _ => DrumSet::Standard,
        }
    }

    /// The 1-based Program number as printed in Appendix B.
    pub fn program_number(self) -> u8 {
        match self {
            DrumSet::Standard => 1,
            DrumSet::Room => 9,
            DrumSet::Power => 17,
            DrumSet::Electronic => 25,
            DrumSet::Analog => 26,
            DrumSet::Jazz => 33,
            DrumSet::Brush => 41,
            DrumSet::Orchestra => 49,
            DrumSet::Sfx => 57,
        }
    }

    /// Human-readable set name.
    pub fn name(self) -> &'static str {
        match self {
            DrumSet::Standard => "STANDARD",
            DrumSet::Room => "ROOM",
            DrumSet::Power => "POWER",
            DrumSet::Electronic => "ELECTRONIC",
            DrumSet::Analog => "ANALOG",
            DrumSet::Jazz => "JAZZ",
            DrumSet::Brush => "BRUSH",
            DrumSet::Orchestra => "ORCHESTRA",
            DrumSet::Sfx => "SFX",
        }
    }

    /// Resolve one MIDI note to its percussion sound in this set,
    /// following `@` inheritance to the STANDARD Set. Returns `None`
    /// for keys that do not sound (`---`, or STANDARD's silent keys).
    pub fn key(self, note: u8) -> Option<DrumKey> {
        match self {
            DrumSet::Standard => standard_key(note),
            // SFX is self-contained: undefined keys are silent, no `@`.
            DrumSet::Sfx => sfx_key(note),
            // Every other set overrides some keys and inherits the rest
            // (`@`) from the STANDARD Set.
            _ => match self.override_cell(note) {
                Cell::Over(k) => Some(k),
                Cell::Inherit => standard_key(note),
            },
        }
    }

    /// `true` when `note` produces a sound in this set.
    pub fn sounds(self, note: u8) -> bool {
        self.key(note).is_some()
    }

    /// EXC group number for `note` in this set, or `None` when the note
    /// belongs to no mutually-exclusive group (or does not sound). Two
    /// notes in the same set choke each other iff their groups compare
    /// equal (both `Some(n)`), per §2.8.1 / the Appendix-B `[EXCn]` tags.
    pub fn exc_group(self, note: u8) -> Option<u8> {
        self.key(note).and_then(|k| k.exc)
    }

    /// `true` when a Note Off on `note` should release the voice. On
    /// Rhythm Channels Note Off is **ignored** (§2.8.1) so percussion
    /// one-shots ring out — except the ORCHESTRA Set's Note 88
    /// (Applause) and the SFX Set's Notes 47–84, which honour Note Off.
    pub fn honors_note_off(self, note: u8) -> bool {
        match self {
            DrumSet::Orchestra => note == 88,
            DrumSet::Sfx => (47..=84).contains(&note),
            _ => false,
        }
    }

    /// Per-set Appendix-B override cell (only for the inheriting sets;
    /// STANDARD / SFX are resolved directly by [`Self::key`]).
    fn override_cell(self, note: u8) -> Cell {
        match self {
            DrumSet::Room => match note {
                41 => Cell::Over(dk("Room Low Tom 2", 34, None)),
                43 => Cell::Over(dk("Room Low Tom 1", 46, None)),
                45 => Cell::Over(dk("Room Mid Tom 2", 58, None)),
                47 => Cell::Over(dk("Room Mid Tom 1", 70, None)),
                48 => Cell::Over(dk("Room Hi Tom 2", 82, None)),
                50 => Cell::Over(dk("Room Hi Tom 1", 94, None)),
                _ => Cell::Inherit,
            },
            DrumSet::Power => match note {
                36 => Cell::Over(dk("Power Kick Drum", 64, None)),
                38 => Cell::Over(dk("Power Snare Drum", 64, None)),
                41 => Cell::Over(dk("Power Low Tom 2", 34, None)),
                43 => Cell::Over(dk("Power Low Tom 1", 46, None)),
                45 => Cell::Over(dk("Power Mid Tom 2", 58, None)),
                47 => Cell::Over(dk("Power Mid Tom 1", 70, None)),
                48 => Cell::Over(dk("Power Hi Tom 2", 82, None)),
                50 => Cell::Over(dk("Power Hi Tom 1", 94, None)),
                _ => Cell::Inherit,
            },
            DrumSet::Electronic => match note {
                36 => Cell::Over(dk("Electric Bass Drum", 64, None)),
                38 => Cell::Over(dk("Electric Snare 1", 64, None)),
                40 => Cell::Over(dk("Electric Snare 2", 64, None)),
                41 => Cell::Over(dk("Electric Low Tom 2", 34, None)),
                43 => Cell::Over(dk("Electric Low Tom 1", 46, None)),
                45 => Cell::Over(dk("Electric Mid Tom 2", 58, None)),
                47 => Cell::Over(dk("Electric Mid Tom 1", 70, None)),
                48 => Cell::Over(dk("Electric Hi Tom 2", 82, None)),
                50 => Cell::Over(dk("Electric Hi Tom 1", 94, None)),
                52 => Cell::Over(dk("Reverse Cymbal", 44, None)),
                _ => Cell::Inherit,
            },
            DrumSet::Analog => match note {
                36 => Cell::Over(dk("Analog Bass Drum", 64, None)),
                37 => Cell::Over(dk("Analog Rim Shot", 64, None)),
                38 => Cell::Over(dk("Analog Snare 1", 64, None)),
                41 => Cell::Over(dk("Analog Low Tom 2", 34, None)),
                42 => Cell::Over(dk("Analog CHH 1", 84, Some(1))),
                43 => Cell::Over(dk("Analog Low Tom 1", 46, None)),
                44 => Cell::Over(dk("Analog CHH 2", 84, Some(1))),
                45 => Cell::Over(dk("Analog Mid Tom 2", 58, None)),
                46 => Cell::Over(dk("Analog OHH", 84, Some(1))),
                47 => Cell::Over(dk("Analog Mid Tom 1", 70, None)),
                48 => Cell::Over(dk("Analog Hi Tom 2", 82, None)),
                49 => Cell::Over(dk("Analog Cymbal", 84, None)),
                50 => Cell::Over(dk("Analog Hi Tom 1", 94, None)),
                56 => Cell::Over(dk("Analog Cowbell", 84, None)),
                62 => Cell::Over(dk("Analog High Conga", 39, None)),
                63 => Cell::Over(dk("Analog Mid Conga", 44, None)),
                64 => Cell::Over(dk("Analog Low Conga", 49, None)),
                70 => Cell::Over(dk("Analog Maracas", 24, None)),
                75 => Cell::Over(dk("Analog Claves", 84, None)),
                _ => Cell::Inherit,
            },
            DrumSet::Jazz => match note {
                35 => Cell::Over(dk("Jazz Kick 2", 64, None)),
                36 => Cell::Over(dk("Jazz Kick 1", 64, None)),
                _ => Cell::Inherit,
            },
            DrumSet::Brush => match note {
                35 => Cell::Over(dk("Jazz Kick 2", 64, None)),
                36 => Cell::Over(dk("Jazz Kick 1", 64, None)),
                38 => Cell::Over(dk("Brush Tap", 64, None)),
                39 => Cell::Over(dk("Brush Slap", 64, None)),
                40 => Cell::Over(dk("Brush Swirl", 64, None)),
                _ => Cell::Inherit,
            },
            DrumSet::Orchestra => match note {
                27 => Cell::Over(dk("Closed Hi-hat 2", 84, Some(1))),
                28 => Cell::Over(dk("Pedal Hi-hat", 84, Some(1))),
                29 => Cell::Over(dk("Open Hi-hat 2", 84, Some(1))),
                30 => Cell::Over(dk("Ride Cymbal 1", 44, None)),
                35 => Cell::Over(dk("Concert BD 2", 24, None)),
                36 => Cell::Over(dk("Concert BD 1", 24, None)),
                38 => Cell::Over(dk("Concert SD", 44, None)),
                39 => Cell::Over(dk("Castanets", 34, None)),
                41 => Cell::Over(dk("Timpani F", 34, None)),
                42 => Cell::Over(dk("Timpani F#", 34, None)),
                43 => Cell::Over(dk("Timpani G", 34, None)),
                44 => Cell::Over(dk("Timpani G#", 34, None)),
                45 => Cell::Over(dk("Timpani A", 34, None)),
                46 => Cell::Over(dk("Timpani A#", 34, None)),
                47 => Cell::Over(dk("Timpani B", 34, None)),
                48 => Cell::Over(dk("Timpani c", 34, None)),
                49 => Cell::Over(dk("Timpani c#", 34, None)),
                50 => Cell::Over(dk("Timpani d", 34, None)),
                51 => Cell::Over(dk("Timpani d#", 34, None)),
                52 => Cell::Over(dk("Timpani e", 34, None)),
                53 => Cell::Over(dk("Timpani f", 34, None)),
                57 => Cell::Over(dk("Concert Cymbal 2", 34, None)),
                59 => Cell::Over(dk("Concert Cymbal 1", 34, None)),
                88 => Cell::Over(dk("Applause", 64, None)),
                _ => Cell::Inherit,
            },
            // STANDARD / SFX never reach here.
            DrumSet::Standard | DrumSet::Sfx => Cell::Inherit,
        }
    }
}

/// STANDARD Set (PC #1) full key map — the base every inheriting set
/// falls back to for `@` keys. Notes outside 27..=87 and Note 88 do not
/// sound.
fn standard_key(note: u8) -> Option<DrumKey> {
    let k = match note {
        27 => dk("High Q", 49, None),
        28 => dk("Slap", 49, None),
        29 => dk("Scratch Push", 54, Some(7)),
        30 => dk("Scratch Pull", 54, Some(7)),
        31 => dk("Sticks", 64, None),
        32 => dk("Square Click", 54, None),
        33 => dk("Metronome Click", 64, None),
        34 => dk("Metronome Bell", 64, None),
        35 => dk("Acoustic Bass Drum", 64, None),
        36 => dk("Bass Drum 1", 64, None),
        37 => dk("Side Stick", 64, None),
        38 => dk("Acoustic Snare", 64, None),
        39 => dk("Hand Clap", 54, None),
        40 => dk("Electric Snare", 64, None),
        41 => dk("Low Floor Tom", 34, None),
        42 => dk("Closed Hi-hat", 84, Some(1)),
        43 => dk("High Floor Tom", 46, None),
        44 => dk("Pedal Hi-hat", 84, Some(1)),
        45 => dk("Low Tom", 58, None),
        46 => dk("Open Hi-hat", 84, Some(1)),
        47 => dk("Low-Mid Tom", 70, None),
        48 => dk("High Mid Tom", 82, None),
        49 => dk("Crash Cymbal 1", 84, None),
        50 => dk("High Tom", 94, None),
        51 => dk("Ride Cymbal 1", 44, None),
        52 => dk("Chinese Cymbal", 44, None),
        53 => dk("Ride Bell", 44, None),
        54 => dk("Tambourine", 74, None),
        55 => dk("Splash Cymbal", 54, None),
        56 => dk("Cowbell", 84, None),
        57 => dk("Crash Cymbal 2", 44, None),
        58 => dk("Vibra-slap", 29, None),
        59 => dk("Ride Cymbal 2", 44, None),
        60 => dk("High Bongo", 99, None),
        61 => dk("Low Bongo", 99, None),
        62 => dk("Mute Hi Conga", 39, None),
        63 => dk("Open Hi Conga", 39, None),
        64 => dk("Low Conga", 44, None),
        65 => dk("High Timbale", 84, None),
        66 => dk("Low Timbale", 84, None),
        67 => dk("High Agogo", 29, None),
        68 => dk("Low Agogo", 29, None),
        69 => dk("Cabasa", 29, None),
        70 => dk("Maracas", 24, None),
        71 => dk("Short Whistle", 99, Some(2)),
        72 => dk("Long Whistle", 99, Some(2)),
        73 => dk("Short Guiro", 94, Some(3)),
        74 => dk("Long Guiro", 94, Some(3)),
        75 => dk("Claves", 84, None),
        76 => dk("Hi Wood Block", 99, None),
        77 => dk("Low Wood Block", 99, None),
        78 => dk("Mute Cuica", 44, Some(4)),
        79 => dk("Open Cuica", 44, Some(4)),
        80 => dk("Mute Triangle", 24, Some(5)),
        81 => dk("Open Triangle", 24, Some(5)),
        82 => dk("Shaker", 94, None),
        83 => dk("Jingle Bell", 99, None),
        84 => dk("Bell Tree", 104, None),
        85 => dk("Castanets", 34, None),
        86 => dk("Mute Surdo", 44, Some(6)),
        87 => dk("Open Surdo", 44, Some(6)),
        // Note 88 "(does not sound)" and everything outside 27..=87.
        _ => return None,
    };
    Some(k)
}

/// SFX Set (PC #57) — self-contained (no `@` inheritance). Only Notes
/// 39–84 sound; 27–38 and 85–88 are `---`.
fn sfx_key(note: u8) -> Option<DrumKey> {
    let k = match note {
        39 => dk("High Q", 49, None),
        40 => dk("Slap", 49, None),
        41 => dk("Scratch Push", 54, Some(7)),
        42 => dk("Scratch Pull", 54, Some(7)),
        43 => dk("Sticks", 64, None),
        44 => dk("Square Click", 54, None),
        45 => dk("Metronome Click", 64, None),
        46 => dk("Metronome Bell", 64, None),
        47 => dk("Guitar Fret Noise", 64, None),
        48 => dk("Guitar Cutting Noise Up", 64, None),
        49 => dk("Guitar Cutting Noise Down", 64, None),
        50 => dk("String Slap of Double Bass", 64, None),
        51 => dk("Fl.Key Click", 64, None),
        52 => dk("Laughing", 64, None),
        53 => dk("Scream", 64, None),
        54 => dk("Punch", 64, None),
        55 => dk("Heart Beat", 64, None),
        56 => dk("Footsteps 1", 64, None),
        57 => dk("Footsteps 2", 64, None),
        58 => dk("Applause", 64, None),
        59 => dk("Door Creaking", 64, None),
        60 => dk("Door", 64, None),
        61 => dk("Scratch", 64, None),
        62 => dk("Wind Chimes", 64, None),
        63 => dk("Car-Engine", 64, None),
        64 => dk("Car-Stop", 64, None),
        65 => dk("Car-Pass", 64, None),
        66 => dk("Car-Crash", 64, None),
        67 => dk("Siren", 64, None),
        68 => dk("Train", 64, None),
        69 => dk("Jetplane", 64, None),
        70 => dk("Helicopter", 64, None),
        71 => dk("Starship", 64, None),
        72 => dk("Gun Shot", 64, None),
        73 => dk("Machine Gun", 64, None),
        74 => dk("Lasergun", 64, None),
        75 => dk("Explosion", 64, None),
        76 => dk("Dog", 64, None),
        77 => dk("Horse-Gallop", 64, None),
        78 => dk("Birds", 64, None),
        79 => dk("Rain", 64, None),
        80 => dk("Thunder", 64, None),
        81 => dk("Wind", 64, None),
        82 => dk("Seashore", 64, None),
        83 => dk("Stream", 64, None),
        84 => dk("Bubble", 64, None),
        _ => return None,
    };
    Some(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_maps_to_defined_sets_with_standard_fallback() {
        // The nine defined Bank 78H/00H programs (wire bytes).
        assert_eq!(DrumSet::from_program(0x00), DrumSet::Standard);
        assert_eq!(DrumSet::from_program(0x08), DrumSet::Room);
        assert_eq!(DrumSet::from_program(0x10), DrumSet::Power);
        assert_eq!(DrumSet::from_program(0x18), DrumSet::Electronic);
        assert_eq!(DrumSet::from_program(0x19), DrumSet::Analog);
        assert_eq!(DrumSet::from_program(0x20), DrumSet::Jazz);
        assert_eq!(DrumSet::from_program(0x28), DrumSet::Brush);
        assert_eq!(DrumSet::from_program(0x30), DrumSet::Orchestra);
        assert_eq!(DrumSet::from_program(0x38), DrumSet::Sfx);
        // §2.5: every undefined program uses the STANDARD Set.
        for p in [0x01u8, 0x07, 0x09, 0x11, 0x1A, 0x21, 0x31, 0x39, 0x7F] {
            assert_eq!(
                DrumSet::from_program(p),
                DrumSet::Standard,
                "program {p:#x}"
            );
        }
        // Program numbers round-trip to their wire byte.
        for s in [
            DrumSet::Standard,
            DrumSet::Room,
            DrumSet::Power,
            DrumSet::Electronic,
            DrumSet::Analog,
            DrumSet::Jazz,
            DrumSet::Brush,
            DrumSet::Orchestra,
            DrumSet::Sfx,
        ] {
            assert_eq!(DrumSet::from_program(s.program_number() - 1), s);
        }
    }

    #[test]
    fn standard_hi_hat_choke_group() {
        // §2.8.1: Closed / Pedal / Open Hi-hat share EXC1 and choke.
        let s = DrumSet::Standard;
        assert_eq!(s.exc_group(42), Some(1)); // Closed Hi-hat
        assert_eq!(s.exc_group(44), Some(1)); // Pedal Hi-hat
        assert_eq!(s.exc_group(46), Some(1)); // Open Hi-hat
        assert_eq!(s.exc_group(42), s.exc_group(46));
        // A non-EXC key is in no group.
        assert_eq!(s.exc_group(36), None); // Bass Drum 1
                                           // The §2.8.1 worked example: Note 42 (Closed) mutes Note 46
                                           // (Open) — same group.
        assert_eq!(s.exc_group(42), s.exc_group(46));
        assert_ne!(s.exc_group(42), s.exc_group(51)); // vs Ride Cymbal 1
    }

    #[test]
    fn all_standard_exc_groups_present() {
        let s = DrumSet::Standard;
        // EXC1..EXC7 exactly as tabulated in the doc.
        let groups: &[(&[u8], u8)] = &[
            (&[42, 44, 46], 1),
            (&[71, 72], 2),
            (&[73, 74], 3),
            (&[78, 79], 4),
            (&[80, 81], 5),
            (&[86, 87], 6),
            (&[29, 30], 7),
        ];
        for (notes, g) in groups {
            for &n in *notes {
                assert_eq!(s.exc_group(n), Some(*g), "note {n} → EXC{g}");
            }
        }
    }

    #[test]
    fn inheriting_sets_keep_standard_exc_but_override_instruments() {
        // ROOM redefines toms but inherits hi-hats and their choke.
        let r = DrumSet::Room;
        assert_eq!(r.key(41).unwrap().name, "Room Low Tom 2");
        assert_eq!(r.key(42).unwrap().name, "Closed Hi-hat"); // @ inherit
        assert_eq!(r.exc_group(42), Some(1));
        assert_eq!(r.exc_group(46), Some(1));
        // POWER redefines kick + snare.
        assert_eq!(DrumSet::Power.key(36).unwrap().name, "Power Kick Drum");
        assert_eq!(DrumSet::Power.key(38).unwrap().name, "Power Snare Drum");
        // ELECTRONIC adds a Reverse Cymbal on 52 and a second snare on 40.
        assert_eq!(DrumSet::Electronic.key(52).unwrap().name, "Reverse Cymbal");
        assert_eq!(
            DrumSet::Electronic.key(40).unwrap().name,
            "Electric Snare 2"
        );
        // JAZZ / BRUSH replace the kicks.
        assert_eq!(DrumSet::Jazz.key(35).unwrap().name, "Jazz Kick 2");
        assert_eq!(DrumSet::Brush.key(38).unwrap().name, "Brush Tap");
        // BRUSH inherits the hi-hats.
        assert_eq!(DrumSet::Brush.exc_group(44), Some(1));
    }

    #[test]
    fn analog_overrides_hi_hats_but_keeps_the_group() {
        let a = DrumSet::Analog;
        assert_eq!(a.key(42).unwrap().name, "Analog CHH 1");
        assert_eq!(a.key(44).unwrap().name, "Analog CHH 2");
        assert_eq!(a.key(46).unwrap().name, "Analog OHH");
        // Still a mutually-exclusive group.
        assert_eq!(a.exc_group(42), Some(1));
        assert_eq!(a.exc_group(46), Some(1));
        // EXC7 (Scratch) inherited from STANDARD.
        assert_eq!(a.exc_group(29), Some(7));
        assert_eq!(a.exc_group(30), Some(7));
    }

    #[test]
    fn orchestra_hi_hats_move_an_octave_low() {
        let o = DrumSet::Orchestra;
        assert_eq!(o.key(27).unwrap().name, "Closed Hi-hat 2");
        assert_eq!(o.key(28).unwrap().name, "Pedal Hi-hat");
        assert_eq!(o.key(29).unwrap().name, "Open Hi-hat 2");
        // The lower hi-hats form EXC1.
        assert_eq!(o.exc_group(27), Some(1));
        assert_eq!(o.exc_group(28), Some(1));
        assert_eq!(o.exc_group(29), Some(1));
        // Notes 42/44/46 are now Timpani (not hi-hats), so no EXC1 there.
        assert_eq!(o.key(42).unwrap().name, "Timpani F#");
        assert_eq!(o.exc_group(42), None);
        // Note 30 is Ride Cymbal 1 now — the STANDARD EXC7 is gone.
        assert_eq!(o.key(30).unwrap().name, "Ride Cymbal 1");
        assert_eq!(o.exc_group(30), None);
        assert_eq!(o.exc_group(29), Some(1)); // was EXC7 in STANDARD
                                              // §2.8.1: ORCHESTRA Note 88 honours Note Off (Applause).
        assert_eq!(o.key(88).unwrap().name, "Applause");
        assert!(o.honors_note_off(88));
        assert!(!o.honors_note_off(36));
    }

    #[test]
    fn sfx_is_self_contained_and_honors_note_off_47_to_84() {
        let x = DrumSet::Sfx;
        // Only 39..=84 sound; no `@` inheritance.
        assert!(!x.sounds(27));
        assert!(!x.sounds(38));
        assert!(x.sounds(39));
        assert!(x.sounds(84));
        assert!(!x.sounds(85));
        assert!(!x.sounds(88));
        assert_eq!(x.key(39).unwrap().name, "High Q");
        assert_eq!(x.key(72).unwrap().name, "Gun Shot");
        // Its only EXC group is EXC7 = 41/42.
        assert_eq!(x.exc_group(41), Some(7));
        assert_eq!(x.exc_group(42), Some(7));
        assert_eq!(x.exc_group(43), None);
        // Notes 47–84 honour Note Off; 39–46 do not.
        assert!(!x.honors_note_off(41));
        assert!(!x.honors_note_off(46));
        assert!(x.honors_note_off(47));
        assert!(x.honors_note_off(84));
        assert!(!x.honors_note_off(85));
    }

    #[test]
    fn standard_note_88_is_silent_and_ignores_note_off() {
        let s = DrumSet::Standard;
        assert!(!s.sounds(88));
        assert!(s.sounds(87));
        assert!(!s.sounds(26));
        // Rhythm Channels ignore Note Off outside the ORCHESTRA/SFX
        // exceptions.
        for n in 0..=127u8 {
            assert!(!s.honors_note_off(n), "STANDARD note {n}");
        }
    }

    #[test]
    fn preset_pans_match_appendix_b() {
        let s = DrumSet::Standard;
        assert_eq!(s.key(41).unwrap().pan, 34); // Low Floor Tom
        assert_eq!(s.key(42).unwrap().pan, 84); // Closed Hi-hat
        assert_eq!(s.key(84).unwrap().pan, 104); // Bell Tree
        assert_eq!(s.key(70).unwrap().pan, 24); // Maracas
        assert_eq!(s.key(31).unwrap().pan, 64); // Sticks (centre)
    }

    #[test]
    fn every_note_resolves_without_panic() {
        // Robustness: no note number panics on any set (full 0..=127).
        for set in [
            DrumSet::Standard,
            DrumSet::Room,
            DrumSet::Power,
            DrumSet::Electronic,
            DrumSet::Analog,
            DrumSet::Jazz,
            DrumSet::Brush,
            DrumSet::Orchestra,
            DrumSet::Sfx,
        ] {
            for n in 0..=127u8 {
                let _ = set.key(n);
                let _ = set.exc_group(n);
                let _ = set.honors_note_off(n);
                let _ = set.sounds(n);
            }
        }
    }
}
