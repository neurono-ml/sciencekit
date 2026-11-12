//! Radial-basis-function kernel `K(X, Y) = exp(-gamma · ‖X − Y‖²)`
//! (spec `kernel-functions`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::config::{SKKernel, SKKernelGamma, sk_validate_kernel_hyperparameters};
use super::linear::sk_dense_row_dot;
use super::parallel::{sk_fill_kernel_rows, sk_transform_kernel_in_place};
use super::validation::sk_validate_kernel_pair;
use crate::pairwise::{SKPairwiseMetric, sk_pairwise_distance_matrix};

fn sk_dense_rbf_kernel<F: SKFloat>(
    left: &ArrayView2<'_, F>,
    right: &ArrayView2<'_, F>,
    gamma: F,
    allow_parallel: bool,
) -> Array2<F> {
    let left_norms: Vec<F> = left
        .rows()
        .into_iter()
        .map(|row| sk_dense_row_dot(row, row))
        .collect();
    let right_rows: Vec<_> = right.rows().into_iter().collect();
    let right_norms: Vec<F> = right_rows
        .iter()
        .map(|row| sk_dense_row_dot(*row, *row))
        .collect();

    sk_fill_kernel_rows(
        left.nrows(),
        right.nrows(),
        allow_parallel,
        |row_index, mut output_row| {
            let left_row = left.row(row_index);
            let left_norm = left_norms[row_index];
            output_row
                .iter_mut()
                .zip(right_rows.iter().zip(right_norms.iter()))
                .for_each(|(cell, (right_row, &right_norm))| {
                    let dot = sk_dense_row_dot(left_row, *right_row);
                    let squared = (left_norm + right_norm - dot - dot).max(F::zero());
                    *cell = (-gamma * squared).exp();
                });
        },
    )
}

/// Internal RBF entry point honoring `allow_parallel`.
pub(crate) fn sk_radial_basis_kernel_with_mode<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
    gamma: F,
    allow_parallel: bool,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_hyperparameters(
        SKKernel::RadialBasisFunction,
        SKKernelGamma::Value(gamma),
        1,
        F::zero(),
    )?;
    sk_validate_kernel_pair(&left, &right)?;
    if let (SKDataView::Dense(left_dense), SKDataView::Dense(right_dense)) = (left, right) {
        return Ok(sk_dense_rbf_kernel(
            &left_dense,
            &right_dense,
            gamma,
            allow_parallel,
        ));
    }
    let mut matrix = sk_pairwise_distance_matrix(SKPairwiseMetric::SquaredEuclidean, left, right)?;
    sk_transform_kernel_in_place(&mut matrix, allow_parallel, |squared_distance| {
        (-gamma * squared_distance.max(F::zero())).exp()
    });
    Ok(matrix)
}

/// Compute the radial-basis-function kernel matrix
/// `K[i, j] = exp(-gamma · max(‖left[i] − right[j]‖², 0))`.
pub fn sk_radial_basis_function_kernel_matrix<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
    gamma: F,
) -> Result<Array2<F>, SKError> {
    sk_radial_basis_kernel_with_mode(left, right, gamma, true)
}
