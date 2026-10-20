//! Dense per-column statistics (spec `simple-imputer`).
//!
//! Columns are independent, so the computation iterates them with a rayon
//! `par_iter` and stays sequential inside each column.

use ndarray::{Array1, ArrayView2, Axis};
use rayon::prelude::*;
use sciencekit_common::SKFloat;

use crate::missingness::sk_is_missing;
use crate::simple_imputer::statistics::observed_column::SKObservedColumn;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// The strategy statistic of every dense column, one entry per input column.
///
/// A `NaN` entry marks a column with no observed value under a non-constant
/// strategy. Under `Constant` the statistic is the configured fill and no
/// column can be empty.
pub(in crate::simple_imputer) fn dense_column_statistics<F: SKFloat>(
    features: &ArrayView2<'_, F>,
    strategy: SKImputationStrategy,
    missing_values: F,
    fill_value: Option<F>,
) -> Array1<F> {
    let statistics: Vec<F> = features
        .axis_iter(Axis(1))
        .into_par_iter()
        .map(|column| {
            let values: Vec<F> = column
                .iter()
                .copied()
                .filter(|&value| !sk_is_missing(value, missing_values))
                .collect();
            SKObservedColumn::new(values, 0).statistic(strategy, fill_value)
        })
        .collect();
    Array1::from_vec(statistics)
}
