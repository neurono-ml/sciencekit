//! Sparse per-column statistics (spec `simple-imputer`).
//!
//! The structural zeros of a CSR matrix are values too: under any marker other
//! than `0.0` each implicit zero is one observed `0.0`, so the statistics come
//! from the explicit non-missing entries plus a per-column multiplicity of
//! zeros — never from densifying.

use ndarray::Array1;
use rayon::prelude::*;
use sciencekit_common::SKFloat;
use sprs::CsMatView;

use crate::missingness::{sk_is_missing, sk_sparse_marks_implicit_zeros};
use crate::simple_imputer::statistics::observed_column::SKObservedColumn;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// The strategy statistic of every sparse column, one entry per input column.
pub(in crate::simple_imputer) fn sparse_column_statistics<F: SKFloat>(
    features: &CsMatView<'_, F>,
    strategy: SKImputationStrategy,
    missing_values: F,
    fill_value: Option<F>,
) -> Array1<F> {
    let columns_count = features.cols();
    if strategy == SKImputationStrategy::Constant {
        return Array1::from_elem(columns_count, fill_value.unwrap_or_else(F::zero));
    }
    let mut values: Vec<Vec<F>> = (0..columns_count).map(|_| Vec::new()).collect();
    let mut explicit_counts = vec![0usize; columns_count];
    for row in features.outer_iterator() {
        for (column, &value) in row.iter() {
            explicit_counts[column] += 1;
            if !sk_is_missing(value, missing_values) {
                values[column].push(value);
            }
        }
    }
    let implicit_zeros_are_observed = !sk_sparse_marks_implicit_zeros(missing_values);
    let number_of_rows = features.rows();
    let statistics: Vec<F> = values
        .into_par_iter()
        .zip(explicit_counts.into_par_iter())
        .map(|(values, explicit_count)| {
            let implicit_zeros = if implicit_zeros_are_observed {
                number_of_rows - explicit_count
            } else {
                0
            };
            SKObservedColumn::new(values, implicit_zeros).statistic(strategy, fill_value)
        })
        .collect();
    Array1::from_vec(statistics)
}
