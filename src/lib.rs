//! MIDI — Standard MIDI File (SMF) parser + transport metadata + soft-synth scaffold.
//!
//! Round-1 ships:
//!
//! * **[`smf`]** — pure-Rust parser for the Standard MIDI File format
//!   (Type 0 / 1 / 2). Header (`MThd`) + tracks (`MTrk`) + every common
//!   channel-voice message, sysex (`F0` / `F7`), and meta event
//!   (tempo, time signature, key signature, text, marker, end-of-track,
//!   SMPTE offset, sequencer-specific). Running status is honoured;
//!   VLQs are bounded to 4 bytes per spec; chunk lengths are validated
//!   against remaining bytes; total events per file are capped at
//!   [`smf::MAX_EVENTS_PER_FILE`].
//! * **[`paths`]** — per-OS SoundFont/SFZ/DLS search paths plus the
//!   `OXIDEAV_SOUNDFONT_PATH` environment override. `find_soundfonts`
//!   walks them and returns every instrument-bank file present.
//! * **[`instruments`]** — [`instruments::Instrument`] trait, magic-byte
//!   detector stubs for SoundFont 2 / SFZ / DLS, and a working
//!   pure-tone fallback ([`instruments::tone::ToneInstrument`]) so the
//!   synth produces *something* even with no on-disk bank.
//! * **[`downloader`]** — stub that names a planned default bank
//!   (TimGM6mb) but currently returns `Error::Unsupported`.
//!
//! The decoder factory ([`make_decoder`]) is registered under codec id
//! [`CODEC_ID_STR`] = `"midi"`. Round-1 returns
//! `Error::Unsupported("MIDI synthesis not yet implemented; round 2")`
//! when invoked — the crate is a parser + path finder + scaffold for now.
//! Synthesis (mixing voices into PCM frames, mapping SMF events to
//! voice on/off, tempo + division → real-time scheduling) is round-2.

pub mod downloader;
pub mod instruments;
pub mod paths;
pub mod smf;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, Decoder, Error, Frame,
    Packet, Result,
};

/// Public codec id string. Matches the aggregator feature name `midi`.
pub const CODEC_ID_STR: &str = "midi";

/// Register the MIDI codec stub. Round-1 only emits a clear
/// `Error::Unsupported` from `send_packet` — the registration exists
/// so the codec id resolves cleanly through the `CodecRegistry` and
/// downstream tooling (`oxideav list`, pipeline JSON validation) can
/// see the placeholder.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::audio("midi_synth")
        .with_lossy(false)
        .with_lossless(true)
        .with_intra_only(false)
        .with_max_channels(16);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder),
    );
}

fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(MidiStubDecoder {
        codec_id: CodecId::new(CODEC_ID_STR),
    }))
}

/// Round-1 stub decoder. Validates incoming packets as SMF (so a caller
/// gets a clear "wrong bytes" rather than a silent miss), then surfaces
/// `Error::Unsupported` from `send_packet`. Synthesis lands in round-2.
struct MidiStubDecoder {
    codec_id: CodecId,
}

impl Decoder for MidiStubDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Confirm the packet at least *looks* like an SMF — saves the
        // user from "synthesis pending" when the real issue is a
        // mis-routed packet.
        if packet.data.len() < 4 || &packet.data[0..4] != b"MThd" {
            return Err(Error::invalid(
                "MIDI: packet does not start with the 'MThd' header chunk",
            ));
        }
        // Light structural validation so malformed files surface as
        // `invalid`, not `unsupported`.
        crate::smf::parse(&packet.data)?;
        Err(Error::unsupported(
            "MIDI synthesis not yet implemented; round 2 will wire the SMF \
             event stream through a soft-synth driven by SoundFont 2 / SFZ / \
             DLS or the pure-tone fallback (see oxideav_midi::instruments)",
        ))
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(Error::Eof)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    fn minimal_smf() -> Vec<u8> {
        // MThd format-0, ntrks=1, division=96; one MTrk with EOT.
        let mut b = vec![];
        b.extend_from_slice(b"MThd");
        b.extend_from_slice(&6u32.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&96u16.to_be_bytes());
        b.extend_from_slice(b"MTrk");
        b.extend_from_slice(&4u32.to_be_bytes());
        b.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);
        b
    }

    #[test]
    fn registers_codec_under_midi_id() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        assert!(reg.has_decoder(&CodecId::new(CODEC_ID_STR)));
    }

    #[test]
    fn stub_decoder_rejects_non_smf_packets() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        let mut dec = reg.make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), b"not midi".to_vec());
        let err = dec.send_packet(&pkt).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn stub_decoder_returns_unsupported_for_valid_smf() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        let mut dec = reg.make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 44_100), minimal_smf());
        let err = dec.send_packet(&pkt).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }
}
