//! The observed set of a single column and its strategy statistic
//! (spec `simple-imputer`).
//!
//! The observed set is the explicit non-missing values plus every structural
//! zero that counts as observed. Keeping the zeros as a multiplicity instead
//! of materializing them makes the sparse path exact without densifying, and
//! the dense path is the same code with a zero multiplicity of `0`.

use sciencekit_common::SKFloat;

use crate::simple_imputer::strategy::SKImputationStrategy;

/// The observed cells of one column: the explicit values and how many implicit
/// zeros join them.
pub(super) struct SKObservedColumn<F> {
    values: Vec<F>,
    implicit_zeros: usize,
}

impl<F: SKFloat> SKObservedColumn<F> {
    /// An observed set of `values` plus `implicit_zeros` structural zeros.
    pub(super) fn new(values: Vec<F>, implicit_zeros: usize) -> Self {
        Self {
            values,
            implicit_zeros,
        }
    }

    /// The number of observed cells.
    pub(super) fn count(&self) -> usize {
        self.values.len() + self.implicit_zeros
    }

    /// The statistic of this observed set under `strategy`.
    pub(super) fn statistic(&self, strategy: SKImputationStrategy, fill_value: Option<F>) -> F {
        match strategy {
            SKImputationStrategy::Mean => self.mean(),
            SKImputationStrategy::Median => self.median(),
            SKImputationStrategy::MostFrequent => self.most_frequent(),
            SKImputationStrategy::Constant => fill_value.unwrap_or_else(F::zero),
        }
    }

    /// The mean, computed with Neumaier compensation so a long column of
    /// similar magnitudes does not lose its low-order digits. The implicit
    /// zeros add nothing to the sum but do count in the denominator.
    fn mean(&self) -> F {
        let count = self.count();
        if count == 0 {
            return F::nan();
        }
        let mut sum = F::zero();
        let mut compensation = F::zero();
        for &value in &self.values {
            let next = sum + value;
            if sum.abs() >= value.abs() {
                compensation = compensation + (sum - next) + value;
            } else {
                compensation = compensation + (value - next) + sum;
            }
            sum = next;
        }
        (sum + compensation) / F::from(count).expect("column count fits the scalar")
    }

    /// The median: the middle order statistic of the multiset, averaging the
    /// central pair on an even count.
    fn median(&self) -> F {
        let count = self.count();
        if count == 0 {
            return F::nan();
        }
        let sorted = self.sorted_values();
        let middle = count / 2;
        if count % 2 == 1 {
            order_statistic(&sorted, self.implicit_zeros, middle)
        } else {
            (order_statistic(&sorted, self.implicit_zeros, middle - 1)
                + order_statistic(&sorted, self.implicit_zeros, middle))
                / F::from(2u64).expect("two fits the scalar")
        }
    }

    /// The most frequent value, with ties broken toward the smallest.
    ///
    /// The values are sorted once and scanned as runs; because the scan is
    /// ascending and the replacement test is strict, the smallest of several
    /// tied modes wins deterministically, independent of arrival order.
    fn most_frequent(&self) -> F {
        if self.count() == 0 {
            return F::nan();
        }
        let sorted = self.sorted_values();
        let mut runs: Vec<(F, usize)> = Vec::new();
        for &value in &sorted {
            match runs.last_mut() {
                Some((last_value, count)) if *last_value == value => *count += 1,
                _ => runs.push((value, 1)),
            }
        }
        if self.implicit_zeros > 0 {
            match runs.binary_search_by(|(value, _)| {
                value
                    .partial_cmp(&F::zero())
                    .unwrap_or(core::cmp::Ordering::Equal)
            }) {
                Ok(index) => runs[index].1 += self.implicit_zeros,
                Err(index) => runs.insert(index, (F::zero(), self.implicit_zeros)),
            }
        }
        let mut best = runs[0];
        for run in &runs[1..] {
            if run.1 > best.1 {
                best = *run;
            }
        }
        best.0
    }

    /// The explicit values in ascending order.
    fn sorted_values(&self) -> Vec<F> {
        let mut sorted = self.values.clone();
        sorted.sort_by(|left, right| {
            left.partial_cmp(right)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        sorted
    }
}

/// The `rank`-th smallest element of the sorted explicit values together with
/// `implicit_zeros` zeros, without materializing the zeros.
///
/// The zeros occupy the contiguous rank block right after the values below
/// zero, so a rank inside that block answers `0` and a rank past it indexes
/// the explicit array shifted by the zero multiplicity.
fn order_statistic<F: SKFloat>(sorted: &[F], implicit_zeros: usize, rank: usize) -> F {
    let below_zero = sorted.partition_point(|value| *value < F::zero());
    if rank < below_zero {
        sorted[rank]
    } else if rank < below_zero + implicit_zeros {
        F::zero()
    } else {
        sorted[rank - implicit_zeros]
    }
}
