//! Per-column fit statistics (spec `simple-imputer`).
//!
//! Columns are independent, so the computation iterates them with a rayon
//! `par_iter` and stays sequential inside each column. Every strategy skips
//! the missing cells under the configured marker and returns a `NaN`
//! statistic for a column with no observed value, which the fit records as
//! an empty column.

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use sciencekit_common::SKFloat;

use crate::missingness::sk_is_missing;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// The strategy statistic of every column, one entry per input column.
///
/// A `NaN` entry marks a column with no observed value under a non-constant
/// strategy; the `Constant` strategy is stateless and always fills.
pub(super) fn column_statistics<F: SKFloat>(
    features: &ArrayView2<F>,
    strategy: SKImputationStrategy,
    missing_values: F,
    fill_value: Option<F>,
) -> Array1<F> {
    let statistics: Vec<F> = features
        .axis_iter(Axis(1))
        .into_par_iter()
        .map(|column| column_statistic(&column, strategy, missing_values, fill_value))
        .collect();
    Array1::from_vec(statistics)
}

/// The statistic of a single column under `strategy`.
fn column_statistic<F: SKFloat>(
    column: &ArrayView1<'_, F>,
    strategy: SKImputationStrategy,
    missing_values: F,
    fill_value: Option<F>,
) -> F {
    match strategy {
        SKImputationStrategy::Mean => mean_over_observed(column, missing_values),
        SKImputationStrategy::Median => median_over_observed(column, missing_values),
        SKImputationStrategy::MostFrequent => most_frequent_over_observed(column, missing_values),
        SKImputationStrategy::Constant => fill_value.unwrap_or_else(F::zero),
    }
}

/// The mean of the observed cells, computed with Neumaier compensation so a
/// long column of similar magnitudes does not lose its low-order digits.
fn mean_over_observed<F: SKFloat>(column: &ArrayView1<'_, F>, missing_values: F) -> F {
    let mut sum = F::zero();
    let mut compensation = F::zero();
    let mut count = 0u64;
    for &value in column.iter() {
        if sk_is_missing(value, missing_values) {
            continue;
        }
        count += 1;
        let next = sum + value;
        if sum.abs() >= value.abs() {
            compensation = compensation + (sum - next) + value;
        } else {
            compensation = compensation + (value - next) + sum;
        }
        sum = next;
    }
    if count == 0 {
        F::nan()
    } else {
        (sum + compensation) / F::from(count).expect("column count fits the scalar")
    }
}

/// The median of the observed cells: the middle order statistic of the
/// sorted values, averaging the central pair on an even count.
fn median_over_observed<F: SKFloat>(column: &ArrayView1<'_, F>, missing_values: F) -> F {
    let Some(mut observed) = collect_observed(column, missing_values) else {
        return F::nan();
    };
    observed.sort_by(|left, right| {
        left.partial_cmp(right)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let middle = observed.len() / 2;
    if observed.len() % 2 == 1 {
        observed[middle]
    } else {
        (observed[middle - 1] + observed[middle]) / F::from(2u64).expect("two fits the scalar")
    }
}

/// The most frequent observed value, with ties broken toward the smallest.
///
/// The values are sorted once and scanned as runs; because the scan is
/// ascending and the replacement test is strict, the smallest of several
/// tied modes wins deterministically, independent of arrival order.
fn most_frequent_over_observed<F: SKFloat>(column: &ArrayView1<'_, F>, missing_values: F) -> F {
    let Some(mut observed) = collect_observed(column, missing_values) else {
        return F::nan();
    };
    observed.sort_by(|left, right| {
        left.partial_cmp(right)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut best_value = observed[0];
    let mut best_count = 0usize;
    let mut run_value = observed[0];
    let mut run_count = 0usize;
    for &value in &observed {
        if value == run_value {
            run_count += 1;
        } else {
            if run_count > best_count {
                best_count = run_count;
                best_value = run_value;
            }
            run_value = value;
            run_count = 1;
        }
    }
    if run_count > best_count {
        best_value = run_value;
    }
    best_value
}

/// The observed cells of a column, or `None` when there are none.
fn collect_observed<F: SKFloat>(column: &ArrayView1<'_, F>, missing_values: F) -> Option<Vec<F>> {
    let observed: Vec<F> = column
        .iter()
        .copied()
        .filter(|&value| !sk_is_missing(value, missing_values))
        .collect();
    if observed.is_empty() {
        None
    } else {
        Some(observed)
    }
}
