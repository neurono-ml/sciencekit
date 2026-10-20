//! Feeding one batch into the incremental state (spec `simple-imputer`).
//!
//! The dense and sparse iterations feed the same accumulators; the sparse path
//! additionally hands the structural zeros to the accumulator in bulk — one
//! count for the mean, one count for the mode, and a multiplicity for the
//! median sketch, so a matrix with millions of structural zeros keeps its
//! bounded memory and never densifies. Both paths record the missing columns
//! in the same pass, so the composed indicator learns its scope batch by batch.

use ndarray::ArrayView2;
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use crate::missingness::{sk_is_missing, sk_sparse_marks_implicit_zeros};
use crate::simple_imputer::state::SKImputationState;
use crate::simple_imputer::strategy::SKImputationStrategy;

impl<F: SKFloat> SKImputationState<F> {
    /// Feed every observed cell of a dense batch and record which columns hold
    /// at least one missing cell.
    pub(in crate::simple_imputer) fn extend_dense(
        &mut self,
        features: &ArrayView2<'_, F>,
        strategy: SKImputationStrategy,
        missing_values: F,
    ) {
        match strategy {
            SKImputationStrategy::Mean => {
                for (column, values) in features.columns().into_iter().enumerate() {
                    let mut observed_count = 0u64;
                    {
                        let (sum, compensation) =
                            (&mut self.sums[column], &mut self.compensations[column]);
                        for &value in values.iter() {
                            if sk_is_missing(value, missing_values) {
                                self.missing_columns[column] = true;
                                continue;
                            }
                            observed_count += 1;
                            let next = *sum + value;
                            if sum.abs() >= value.abs() {
                                *compensation = *compensation + (*sum - next) + value;
                            } else {
                                *compensation = *compensation + (value - next) + *sum;
                            }
                            *sum = next;
                        }
                    }
                    self.counts[column] += observed_count;
                }
            }
            SKImputationStrategy::Median => {
                for (column, values) in features.columns().into_iter().enumerate() {
                    let mut observed: Vec<f64> = Vec::with_capacity(values.len());
                    let mut has_missing = false;
                    for &value in values.iter() {
                        if sk_is_missing(value, missing_values) {
                            has_missing = true;
                            continue;
                        }
                        observed.push(value.to_f64().unwrap_or(0.0));
                    }
                    if has_missing {
                        self.missing_columns[column] = true;
                    }
                    self.counts[column] += observed.len() as u64;
                    if !observed.is_empty() {
                        let sketch = &mut self.sketches[column];
                        sketch.extend_values(observed);
                        sketch.flush();
                    }
                }
            }
            SKImputationStrategy::MostFrequent => {
                for (column, values) in features.columns().into_iter().enumerate() {
                    let mut observed_count = 0u64;
                    {
                        let counts = &mut self.value_counts[column];
                        for &value in values.iter() {
                            if sk_is_missing(value, missing_values) {
                                self.missing_columns[column] = true;
                                continue;
                            }
                            observed_count += 1;
                            *counts
                                .entry(super::core_implementation::float_key(value))
                                .or_insert(0) += 1;
                        }
                    }
                    self.counts[column] += observed_count;
                }
            }
            SKImputationStrategy::Constant => {
                for (column, values) in features.columns().into_iter().enumerate() {
                    if values
                        .iter()
                        .any(|&value| sk_is_missing(value, missing_values))
                    {
                        self.missing_columns[column] = true;
                    }
                }
            }
        }
    }

    /// Feed every observed cell of a sparse batch, counting the structural
    /// zeros the marker observes, and record which columns hold a missing
    /// explicit entry or — under the `0.0` marker — an implicit zero.
    pub(in crate::simple_imputer) fn extend_sparse(
        &mut self,
        features: &CsMatView<'_, F>,
        strategy: SKImputationStrategy,
        missing_values: F,
    ) {
        let columns_count = self.counts.len();
        let accumulates = strategy != SKImputationStrategy::Constant;
        let mut explicit_counts = vec![0usize; columns_count];
        for row in features.outer_iterator() {
            for (column, &value) in row.iter() {
                explicit_counts[column] += 1;
                if sk_is_missing(value, missing_values) {
                    self.missing_columns[column] = true;
                    continue;
                }
                if !accumulates {
                    continue;
                }
                self.counts[column] += 1;
                match strategy {
                    SKImputationStrategy::Mean => {
                        let (sum, compensation) =
                            (&mut self.sums[column], &mut self.compensations[column]);
                        let next = *sum + value;
                        if sum.abs() >= value.abs() {
                            *compensation = *compensation + (*sum - next) + value;
                        } else {
                            *compensation = *compensation + (value - next) + *sum;
                        }
                        *sum = next;
                    }
                    SKImputationStrategy::Median => {
                        self.sketches[column].push(value.to_f64().unwrap_or(0.0));
                    }
                    SKImputationStrategy::MostFrequent => {
                        *self.value_counts[column]
                            .entry(super::core_implementation::float_key(value))
                            .or_insert(0) += 1;
                    }
                    SKImputationStrategy::Constant => {}
                }
            }
        }
        if accumulates && strategy == SKImputationStrategy::Median {
            for sketch in &mut self.sketches {
                sketch.flush();
            }
        }
        let number_of_rows = features.rows();
        if sk_sparse_marks_implicit_zeros(missing_values) {
            for (column, &explicit_count) in explicit_counts.iter().enumerate() {
                if explicit_count < number_of_rows {
                    self.missing_columns[column] = true;
                }
            }
            return;
        }
        if !accumulates {
            return;
        }
        for (column, explicit_count) in explicit_counts.into_iter().enumerate() {
            let implicit_zeros = (number_of_rows - explicit_count) as u64;
            if implicit_zeros == 0 {
                continue;
            }
            self.counts[column] += implicit_zeros;
            match strategy {
                SKImputationStrategy::Mean => {}
                SKImputationStrategy::Median => {
                    self.implicit_zeros[column] += implicit_zeros;
                }
                SKImputationStrategy::MostFrequent => {
                    *self.value_counts[column].entry(0).or_insert(0) += implicit_zeros;
                }
                SKImputationStrategy::Constant => {}
            }
        }
    }
}
