//! The accumulating state that `fit` seeds and `partial_fit` extends
//! (spec `simple-imputer`).
//!
//! One accumulator per strategy, allocated only for the active strategy:
//! count and compensated sum for the mean, value counts for the mode, and a
//! `tdigest` sketch plus a structural-zero multiplicity for the median. Every
//! accumulator is exact except the median sketch, whose approximation is
//! confined to the incremental path — the dense fit still reads the exact
//! order statistic.

use std::collections::HashMap;

use ndarray::Array1;
use sciencekit_common::SKFloat;
use tdigest::TDigest;

use crate::simple_imputer::strategy::SKImputationStrategy;

/// Compression of the per-column median sketch, matching the robust scaler:
/// a memory/accuracy trade-off deliberately not exposed as a builder option.
const SKETCH_COMPRESSION: usize = 200;

/// The state that turns one batch at a time into the same statistics the whole
/// dataset would produce.
pub(in crate::simple_imputer) struct SKImputationState<F> {
    pub(super) counts: Vec<u64>,
    pub(super) sums: Vec<F>,
    pub(super) compensations: Vec<F>,
    pub(super) value_counts: Vec<HashMap<u64, u64>>,
    pub(super) sketches: Vec<TDigest>,
    pub(super) implicit_zeros: Vec<u64>,
    pub(super) missing_columns: Vec<bool>,
}

impl<F: SKFloat> SKImputationState<F> {
    /// An empty state for `columns_count` columns, allocating only the
    /// accumulators the active strategy reads.
    pub(in crate::simple_imputer) fn new(
        columns_count: usize,
        strategy: SKImputationStrategy,
    ) -> Self {
        Self {
            counts: vec![0; columns_count],
            sums: if strategy == SKImputationStrategy::Mean {
                vec![F::zero(); columns_count]
            } else {
                Vec::new()
            },
            compensations: if strategy == SKImputationStrategy::Mean {
                vec![F::zero(); columns_count]
            } else {
                Vec::new()
            },
            value_counts: if strategy == SKImputationStrategy::MostFrequent {
                (0..columns_count).map(|_| HashMap::new()).collect()
            } else {
                Vec::new()
            },
            sketches: if strategy == SKImputationStrategy::Median {
                (0..columns_count)
                    .map(|_| TDigest::new_with_size(SKETCH_COMPRESSION))
                    .collect()
            } else {
                Vec::new()
            },
            implicit_zeros: if strategy == SKImputationStrategy::Median {
                vec![0; columns_count]
            } else {
                Vec::new()
            },
            missing_columns: vec![false; columns_count],
        }
    }

    /// The observed count of every column: zero means the column carries no
    /// statistic and, unless retained, is dropped from the output.
    pub(in crate::simple_imputer) fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// The columns with at least one missing cell, accumulated batch by batch;
    /// the composed indicator rebuilds its scope from it.
    pub(in crate::simple_imputer) fn missing_columns(&self) -> &[bool] {
        &self.missing_columns
    }

    /// The current statistic of every column under `strategy`.
    ///
    /// A column with no observed value yields `NaN`; under `Constant` the
    /// statistic is the configured fill and no column can be empty.
    pub(in crate::simple_imputer) fn statistics(
        &self,
        strategy: SKImputationStrategy,
        fill_value: Option<F>,
    ) -> Array1<F> {
        let statistics: Vec<F> = (0..self.counts.len())
            .map(|column| match strategy {
                SKImputationStrategy::Mean => self.mean(column),
                SKImputationStrategy::Median => self.median(column),
                SKImputationStrategy::MostFrequent => self.most_frequent(column),
                SKImputationStrategy::Constant => fill_value.unwrap_or_else(F::zero),
            })
            .collect();
        Array1::from_vec(statistics)
    }

    /// The running mean: the compensated sum over the observed count.
    fn mean(&self, column: usize) -> F {
        let count = self.counts[column];
        if count == 0 {
            return F::nan();
        }
        (self.sums[column] + self.compensations[column])
            / F::from(count).expect("column count fits the scalar")
    }

    /// The running median.
    ///
    /// The sketch holds only the explicit observed values; the structural
    /// zeros are a multiplicity. Inverting the combined CDF
    /// `(explicit·F_explicit + zeros·[v ≥ 0]) / total` places the zero block
    /// exactly, so the median of a mostly-structural-zero column is `0.0`
    /// instead of an interpolation artifact between the neighbouring
    /// centroids.
    fn median(&self, column: usize) -> F {
        let sketch = &self.sketches[column];
        let explicit_count = sketch.count();
        let zeros = self.implicit_zeros[column] as f64;
        let total = explicit_count + zeros;
        if total == 0.0 {
            return F::nan();
        }
        let value = if explicit_count == 0.0 {
            0.0
        } else {
            let at_or_below_zero = sketch.estimate_rank(0.0).unwrap_or(0.0) * explicit_count;
            let target = 0.5 * total;
            let quantile = if at_or_below_zero + zeros < target {
                (target - zeros) / explicit_count
            } else if at_or_below_zero < target {
                return F::zero();
            } else {
                target / explicit_count
            };
            match sketch.estimate_quantile(quantile.clamp(0.0, 1.0)) {
                Some(value) => value,
                None => return F::nan(),
            }
        };
        F::from(value).expect("the sketch stores finite values")
    }

    /// The running mode: the most frequent value, ties broken toward the
    /// smallest, matching the exact fit's deterministic rule.
    fn most_frequent(&self, column: usize) -> F {
        let mut best: Option<(f64, u64)> = None;
        for (&key, &count) in &self.value_counts[column] {
            let value = f64::from_bits(key);
            best = Some(match best {
                Some((best_value, best_count))
                    if count < best_count || (count == best_count && value >= best_value) =>
                {
                    (best_value, best_count)
                }
                _ => (value, count),
            });
        }
        best.map_or_else(F::nan, |(value, _)| {
            F::from(value).expect("the counts store finite values")
        })
    }
}

/// The map key of a value: its IEEE-754 bit pattern, with both zeros folded
/// onto `0` so `-0.0` and `0.0` count as the same value.
pub(super) fn float_key<F: SKFloat>(value: F) -> u64 {
    let value = value.to_f64().unwrap_or(0.0);
    if value == 0.0 { 0 } else { value.to_bits() }
}
