//! Cross-product reduction kernels: the Gram matrix `XᵀX` and the Gram vector
//! `Xᵀy` (spec `higher-order-kernels`).
//!
//! Given a 2-D array `X` (rows × features), [`sk_gram_matrix`] returns the
//! symmetric features × features matrix `XᵀX` and [`sk_gram_vector`] returns
//! the features vector `Xᵀy`. Both are the matrix/vector companions of the axis
//! reductions and share the [`SK_AXIS_SUM_GRAIN`](super::SK_AXIS_SUM_GRAIN)
//! reduction grain.
//!
//! Both kernels are pure functions over [`SKFloat`], free of manual index loops
//! (the dot products run through ndarray's iterator API) and layout-agnostic
//! (they read columns, whatever the storage order). They honour the resolved
//! `parallelism`: `1` runs the sequential form; `> 1` dispatches across the
//! compute pool with each output entry computed independently — no shared
//! mutable accumulator.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use sciencekit_common::SKFloat;

/// Compute the Gram matrix `XᵀX` of a 2-D array.
///
/// The result is a symmetric `features × features` matrix whose entry `(i, j)`
/// is the dot product of columns `i` and `j` of `input`.
///
/// Honours the resolved `parallelism`: `1` runs the sequential form; `> 1`
/// dispatches the independent output rows across the compute pool.
pub fn sk_gram_matrix<F: SKFloat>(input: &ArrayView2<F>, parallelism: usize) -> Array2<F> {
    let (_, columns_count) = input.dim();
    let compute_row = |row_index: usize| -> Array1<F> {
        let reference = input.column(row_index);
        input
            .columns()
            .into_iter()
            .map(|column| column.dot(&reference))
            .collect()
    };
    let rows: Vec<Array1<F>> = if parallelism > 1 {
        (0..columns_count)
            .into_par_iter()
            .map(compute_row)
            .collect()
    } else {
        (0..columns_count).map(compute_row).collect()
    };
    let flattened: Vec<F> = rows.into_iter().flat_map(Array1::into_iter).collect();
    Array2::from_shape_vec((columns_count, columns_count), flattened)
        .expect("every Gram row carries one entry per column")
}

/// Compute the Gram vector `Xᵀy` of a 2-D array and a 1-D target.
///
/// The result has one entry per feature: the dot product of that column of
/// `input` with `target`.
///
/// Honours the resolved `parallelism`: `1` runs the sequential form; `> 1`
/// dispatches the independent output entries across the compute pool.
pub fn sk_gram_vector<F: SKFloat>(
    input: &ArrayView2<F>,
    target: &ArrayView1<F>,
    parallelism: usize,
) -> Array1<F> {
    let columns: Vec<ArrayView1<F>> = input.columns().into_iter().collect();
    let values: Vec<F> = if parallelism > 1 {
        columns
            .into_par_iter()
            .map(|column| column.dot(target))
            .collect()
    } else {
        columns
            .into_iter()
            .map(|column| column.dot(target))
            .collect()
    };
    Array1::from_vec(values)
}
