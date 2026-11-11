//! Polynomial kernel `K(X, Y) = (gamma · X · Yᵀ + coef0)^degree`
//! (spec `kernel-functions`).

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::config::{SKKernel, SKKernelGamma, sk_validate_kernel_hyperparameters};
use super::linear::sk_linear_kernel_matrix;
use super::parallel::sk_transform_kernel_in_place;

/// Fast exponentiation by squaring for a strictly positive integer degree.
pub(crate) fn sk_integer_power<F: SKFloat>(mut base: F, mut exponent: u32) -> F {
    let mut result = F::one();
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base;
        }
        exponent >>= 1;
        if exponent > 0 {
            base = base * base;
        }
    }
    result
}

/// Compute the polynomial kernel matrix
/// `K[i, j] = (gamma · ⟨left[i], right[j]⟩ + coef0)^degree`.
pub fn sk_polynomial_kernel_matrix<F: SKFloat>(
    left: SKDataView<'_, F>,
    right: SKDataView<'_, F>,
    gamma: F,
    coef0: F,
    degree: u32,
) -> Result<Array2<F>, SKError> {
    sk_validate_kernel_hyperparameters(
        SKKernel::Polynomial,
        SKKernelGamma::Value(gamma),
        degree,
        coef0,
    )?;
    let mut matrix = sk_linear_kernel_matrix(left, right)?;
    sk_transform_kernel_in_place(&mut matrix, true, |dot_product| {
        sk_integer_power(gamma * dot_product + coef0, degree)
    });
    Ok(matrix)
}
