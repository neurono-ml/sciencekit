//! Stored-value row statistics shared by the sparse kernels.

use sciencekit_common::SKFloat;

/// Sum of the squares of a sparse row's stored values (`‖x‖²`).
///
/// Implicit coordinates are observed zeros and contribute nothing, so iterating
/// the stored values is exact.
pub(crate) fn sk_sparse_row_norm_squared<F: SKFloat>(row_values: &[F]) -> F {
    row_values
        .iter()
        .fold(F::zero(), |accumulator, &value| {
            accumulator + value * value
        })
}

/// Euclidean norm of a sparse row (`‖x‖`).
pub(crate) fn sk_sparse_row_norm<F: SKFloat>(row_values: &[F]) -> F {
    sk_sparse_row_norm_squared(row_values).sqrt()
}
