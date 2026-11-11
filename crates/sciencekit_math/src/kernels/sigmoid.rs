//! Sigmoid kernel `K(X, Y) = tanh(gamma · X · Yᵀ + coef0)`
//! (spec `kernel-functions`).

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::config::{SKKernel, SKKernelGamma, sk_validate_kernel_hyperparameters};
use super::linear::sk_linear_kernel_matrix;

/// Compute the sigmoid kernel matrix
/// `K[i, j] = tanh(gamma · ⟨left[i], right[j]⟩ + coef0)`.
pub fn sk_sigmoid_kernel_matrix<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
    gamma: F,
    coef0: F,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_hyperparameters(
        SKKernel::Sigmoid,
        SKKernelGamma::Value(gamma),
        1,
        coef0,
    )?;
    let mut matrix = sk_linear_kernel_matrix(left, right)?;
    matrix.mapv_inplace(|dot_product| (gamma * dot_product + coef0).tanh());
    Ok(matrix)
}
