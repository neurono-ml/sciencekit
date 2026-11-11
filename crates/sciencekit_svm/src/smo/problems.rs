//! Binary C-SVC and epsilon-SVR SMO problem specializations
//! (spec `support-vector-machines`).

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::SKKernelParameters;

use super::solver::{SKSmoOptions, sk_solve_smo_quadratic_program};

/// Result of a binary C-SVC SMO solve over `n` training rows.
#[derive(Debug, Clone, PartialEq)]
pub struct SKBinarySvcSmoResult<F: SKFloat> {
    /// Indices of retained support vectors (`alpha_i > 0`).
    pub support_indices: Vec<usize>,
    /// Signed dual coefficients `y_i * alpha_i` for each support vector.
    pub dual_coefficients: Vec<F>,
    /// Decision bias `b`.
    pub bias: F,
    /// Number of SMO iterations executed.
    pub iterations: usize,
}

/// Solve binary C-SVC with per-sample box bounds `upper_bounds[i] = C * weight[i]`
/// and labels `signs[i] ∈ {-1, +1}`.
pub fn sk_solve_binary_c_svc<F: SKFloat>(
    signs: &[i8],
    upper_bounds: &[F],
    features: SKDataView<'_, F>,
    parameters: SKKernelParameters<F>,
    options: SKSmoOptions<F>,
) -> Result<SKBinarySvcSmoResult<F>, SKError> {
    let linear_terms = vec![-F::one(); signs.len()];
    let solution = sk_solve_smo_quadratic_program(
        signs.to_vec(),
        upper_bounds.to_vec(),
        linear_terms,
        features,
        parameters,
        options,
    )?;

    let mut support_indices = Vec::new();
    let mut dual_coefficients = Vec::new();
    for (index, &alpha) in solution.alphas.iter().enumerate() {
        if alpha > F::zero() {
            support_indices.push(index);
            let signed = if signs[index] > 0 { alpha } else { -alpha };
            dual_coefficients.push(signed);
        }
    }
    Ok(SKBinarySvcSmoResult {
        support_indices,
        dual_coefficients,
        bias: solution.bias,
        iterations: solution.iterations,
    })
}
