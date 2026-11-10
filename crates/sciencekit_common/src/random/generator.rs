//! The shared SplitMix64 generator and its index/sampling operations (spec
//! `deterministic-randomness`).
//!
//! SplitMix64 is a 64-bit counter advanced by the golden-ratio increment and
//! finalized with two xor-shift-multiply rounds. It needs no external random
//! source, no allocation and no platform state, so the same seed always yields
//! the same stream.
//!
//! Every operation is driven by the same underlying [`SKRandomGenerator::next_u64`]
//! stream. A generator carries mutable state: it is `Send`, so a task can own
//! one and move it across threads, but sharing one between threads requires
//! external synchronization. Stochastic estimators therefore give each
//! independent component its own generator seeded through
//! [`crate::sk_derive_seed`].

/// A deterministic seeded pseudo-random generator.
///
/// The stream is a pure function of the seed: no ambient entropy is ever read.
/// See the module documentation for the reproducibility scope.
#[derive(Debug, Clone)]
pub struct SKRandomGenerator {
    state: u64,
}

impl SKRandomGenerator {
    /// Seeds the generator.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64-bit value of the stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Returns a uniform index in `0..bound`, or zero for an empty range.
    ///
    /// Rejection sampling removes the modulo bias, so the order stays uniform
    /// even when the range does not divide the generator's period.
    pub fn next_index(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        let bound = bound as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }

    /// Returns a Fisher–Yates permutation of `0..count`.
    ///
    /// The result is a permutation: every index appears exactly once, and the
    /// same seed and count always yield the same order.
    pub fn shuffled_indices(&mut self, count: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..count).collect();
        for position in (1..count).rev() {
            let drawn = self.next_index(position + 1);
            indices.swap(position, drawn);
        }
        indices
    }
}
