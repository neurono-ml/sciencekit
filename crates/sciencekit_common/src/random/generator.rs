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
    cached_normal: Option<f64>,
}

impl SKRandomGenerator {
    /// Seeds the generator.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
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

    /// Returns a uniform draw in `[0, 1)` built from the top 53 bits.
    ///
    /// Fifty-three bits is exactly the mantissa width of an `f64`, so every
    /// representable value in the interval is reachable with equal spacing.
    pub fn next_f64(&mut self) -> f64 {
        // 2^-53 as an exact power of two.
        const SCALE: f64 = 1.0 / (1_u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Returns a uniform draw in `[low, high)`.
    ///
    /// The endpoints follow [`SKRandomGenerator::next_f64`], so the draw is
    /// never equal to `high`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// Returns a standard normal draw (mean `0`, variance `1`).
    ///
    /// Uses the Box–Muller transform and caches the second of the pair, so two
    /// consecutive calls consume a single uniform pair. The values depend only
    /// on the stream and the platform's `sqrt`/`ln`/`cos`, never on ambient
    /// entropy.
    pub fn normal(&mut self) -> f64 {
        if let Some(spare) = self.cached_normal.take() {
            return spare;
        }
        // Clamp the radial uniform away from zero: `ln(0)` is not finite and
        // the generator can legitimately emit exactly zero.
        let radial_uniform = self.next_f64().max(f64::MIN_POSITIVE);
        let angular_uniform = self.next_f64();
        let radius = (-2.0 * radial_uniform.ln()).sqrt();
        let angle = std::f64::consts::TAU * angular_uniform;
        self.cached_normal = Some(radius * angle.sin());
        radius * angle.cos()
    }

    /// Shuffles a slice in place with a Fisher–Yates permutation.
    ///
    /// The resulting order is the same one [`SKRandomGenerator::shuffled_indices`]
    /// returns for the slice length and seed.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for position in (1..values.len()).rev() {
            let drawn = self.next_index(position + 1);
            values.swap(position, drawn);
        }
    }

    /// Returns `sample_count` distinct indices drawn uniformly from `0..count`.
    ///
    /// Implements a partial Fisher–Yates shuffle: only the first
    /// `sample_count` positions are settled, so sampling `k` of `n` costs `k`
    /// draws instead of a full permutation. `sample_count` is clamped to
    /// `count`; the order of the returned indices is part of the seeded stream.
    pub fn sample_indices_without_replacement(
        &mut self,
        count: usize,
        sample_count: usize,
    ) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..count).collect();
        let sample_count = sample_count.min(count);
        for position in 0..sample_count {
            let drawn = position + self.next_index(count - position);
            indices.swap(position, drawn);
        }
        indices.truncate(sample_count);
        indices
    }
}
