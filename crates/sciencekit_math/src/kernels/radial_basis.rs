//! Radial-basis-function kernel `K(X, Y) = exp(-gamma · ‖X − Y‖²)`
//! (spec `kernel-functions`).

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::config::{SKKernel, SKKernelGamma, sk_validate_kernel_hyperparameters};
use super::validation::sk_validate_kernel_pair;
use crate::pairwise::{SKPairwiseMetric, sk_pairwise_distance_matrix};

/// Compute the radial-basis-function kernel matrix
/// `K[i, j] = exp(-gamma · max(‖left[i] − right[j]‖², 0))`.
///
/// Reuses the dense/sparse/mixed squared-Euclidean distance dispatcher and
/// clamps tiny negative cancellation errors to zero before exponentiation.
pub fn sk_radial_basis_function_kernel_matrix<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
    gamma: F,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_hyperparameters(
        SKKernel::RadialBasisFunction,
        SKKernelGamma::Value(gamma),
        1,
        F::zero(),
    )?;
    sk_validate_kernel_pair(&left, &right)?;
    let mut matrix =
        sk_pairwise_distance_matrix(SKPairwiseMetric::SquaredEuclidean, left, right)?;
    matrix.mapv_inplace(|squared_distance| {
        let clamped = squared_distance.max(F::zero());
        (-gamma * clamped).exp()
    });
    Ok(matrix)
}
