//! The sparse conjugate-gradient ridge solver (spec `ridge-regression`,
//! Decision 6).
//!
//! One [`sk_conjugate_gradient`] solve per `alpha` over the zero-copy augmented
//! normal-equation operator; when the intercept is included it is the free first
//! coordinate of the solution. The design is never densified and `XᵀX` is never
//! formed.

use ndarray::{Array1, Array2, ArrayView1, s};
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::{SKBackendSelection, sk_conjugate_gradient};
use sprs::CsMatView;

use super::normal_equation_operator::{
    SKRidgeSparseNormalEquationOperator, sk_csr_column_sums, sk_csr_transpose_vector_product,
};
use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::solving::SKRidgeSolution;

/// Solve `(DᵀD + αP)β = Dᵀy` by conjugate gradient, once per `alpha`.
///
/// `D = [1 ‖ X]` with `P = diag(0, 1, …, 1)` when the intercept is included, and
/// `D = X` with `P` all-ones otherwise. The backend kind is carried for span
/// uniformity but the iteration is host-side over `sciencekit_math` kernels, so
/// no concrete backend is dispatched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_with_sparse_cg<F: SKBackendSelection>(
    features: CsMatView<F>,
    target: ArrayView1<F>,
    include_intercept: bool,
    alphas: &[F],
    tolerance: F,
    maximum_number_of_iterations: usize,
    _backend: SKBackendKind,
) -> Result<SKRidgeSolution<F>, SKError> {
    let features_count = features.cols();
    let column_sums = sk_csr_column_sums(&features);
    let cross = sk_csr_transpose_vector_product(&features, target);
    let right_hand_side = if include_intercept {
        let mut rhs = Array1::zeros(features_count + 1);
        rhs[0] = target.sum();
        rhs.slice_mut(s![1..]).assign(&cross);
        rhs
    } else {
        cross
    };
    let row_count = F::from(features.rows()).expect("row counts fit in the scalar");
    let mut coefficients = Array2::zeros((alphas.len(), features_count));
    let mut intercepts = Array1::zeros(alphas.len());
    for (row_index, &alpha) in alphas.iter().enumerate() {
        let operator = SKRidgeSparseNormalEquationOperator::new(
            features,
            include_intercept,
            alpha,
            row_count,
            column_sums.clone(),
        );
        let solution = sk_conjugate_gradient(
            &operator,
            right_hand_side.view(),
            tolerance,
            maximum_number_of_iterations,
        )?;
        if include_intercept {
            intercepts[row_index] = solution[0];
            coefficients
                .row_mut(row_index)
                .assign(&solution.slice(s![1..]));
        } else {
            coefficients.row_mut(row_index).assign(&solution);
        }
    }
    Ok(SKRidgeSolution {
        resolved: SKRidgeSolver::SparseCg,
        coefficients,
        intercepts,
        rank: features_count,
        singular_values: Vec::new(),
    })
}
