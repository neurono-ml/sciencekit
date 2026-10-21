//! The SVD path: one centered decomposition applied per `alpha`
//! (spec `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::SKBackendSelection;

/// Solve every `alpha` from a single singular-value decomposition of the
/// (centered) design.
///
/// With `A = U Σ Vᵀ`, the ridge solution is
/// `β = V diag(σ/(σ² + α)) Uᵀ ỹ`, so the expensive factorization is amortized
/// across the `alpha` sequence. Singular values at or below the truncation
/// cutoff contribute nothing, so a rank-deficient design with `alpha = 0`
/// yields the minimum-norm solution rather than an amplified null-space
/// direction. The factors are sliced to the thin `min(rows, columns)` width, so
/// tall and wide designs both work. Returns the `n_alphas × n_features`
/// coefficients plus the reported rank and the singular values.
pub(super) fn solve_with_svd<F: SKBackendSelection>(
    backend: SKBackendKind,
    design: ArrayView2<F>,
    target: ArrayView1<F>,
    alphas: &[F],
) -> Result<(Array2<F>, usize, Vec<F>), SKError> {
    let machine = F::select_math_backend(backend);
    let decomposition = machine.svd(design)?;
    let (rows_count, columns_count) = design.dim();
    let singular_values = decomposition.singular_values;
    let factor_count = singular_values.len();
    let cutoff = singular_values.first().copied().unwrap_or(F::zero())
        * F::epsilon()
        * F::from(rows_count.max(columns_count)).expect("dimensions fit in the scalar");
    let rank = singular_values
        .iter()
        .filter(|&&value| value > cutoff)
        .count();
    // Thin factors: only the first `min(rows, columns)` singular directions are
    // used, whatever the backend's full-vs-thin convention.
    let left = decomposition.u.slice(s![.., ..factor_count]);
    let right = decomposition.v.slice(s![.., ..factor_count]);
    let projected = left.t().dot(&target);
    let mut coefficients = Array2::zeros((alphas.len(), columns_count));
    for (row_index, &alpha) in alphas.iter().enumerate() {
        let factors: Vec<F> = singular_values
            .iter()
            .zip(projected.iter())
            .map(|(&sigma, &value)| {
                if sigma > cutoff {
                    sigma / (sigma * sigma + alpha) * value
                } else {
                    F::zero()
                }
            })
            .collect();
        let row = right.dot(&Array1::from_vec(factors));
        coefficients.row_mut(row_index).assign(&row);
    }
    Ok((coefficients, rank, singular_values))
}
