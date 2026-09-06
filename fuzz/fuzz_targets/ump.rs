#![no_main]

//! Fuzz the Universal MIDI Packet layer (M2-104): the word-stream
//! walker, every Message Type's decoder, and the encoders.
//!
//! Arbitrary bytes are read as big-endian 32-bit words (the on-disk
//! form of a MIDI Clip File) and walked with `UmpStream`, which sizes
//! every packet from its Message Type nibble — Reserved types
//! included. Each packet must decode to a `Result`; a decoded message
//! must re-encode to a packet that decodes to the *same* message (the
//! encoder is a fixed point of the decoder after one pass). Panics,
//! debug overflows and out-of-bounds reads are bugs.

use libfuzzer_sys::fuzz_target;
use oxideav_midi::ump::{Ump, UmpMessage, UmpStream};

fuzz_target!(|data: &[u8]| {
    let words: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for packet in UmpStream::new(&words) {
        let Ok(packet) = packet else { break };
        let Ok(message) = UmpMessage::decode(&packet) else {
            continue;
        };
        let encoded: Ump = message.encode();
        let again = UmpMessage::decode(&encoded)
            .expect("re-encoded packet must decode");
        assert_eq!(again, message, "encode/decode must be a fixed point");
        assert_eq!(
            encoded.words().len(),
            packet.words().len(),
            "re-encoded packet keeps its Message Type size"
        );
    }
});
