//! The self-contained seeded shuffler (spec `stochastic-gradient-solver`,
//! decision 6).
//!
//! SplitMix64 plus a Fisher–Yates shuffle over row indices: the same seed and
//! row count yield the same order for a dense matrix, a CSR matrix and a
//! streaming batch, with no external random-number dependency.

/// A deterministic SplitMix64 generator.
#[derive(Debug, Clone)]
pub(crate) struct SKRandomGenerator {
    state: u64,
}

impl SKRandomGenerator {
    /// Seed the generator.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64-bit value of the stream.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// A uniform index in `0..bound`, or zero for an empty range.
    ///
    /// Rejection sampling removes the modulo bias, so the ordering stays
    /// uniform even when the range does not divide the generator's period.
    pub(crate) fn next_index(&mut self, bound: usize) -> usize {
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

    /// A Fisher–Yates permutation of `0..count`.
    pub(crate) fn shuffled_indices(&mut self, count: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..count).collect();
        for position in (1..count).rev() {
            let drawn = self.next_index(position + 1);
            indices.swap(position, drawn);
        }
        indices
    }
}
