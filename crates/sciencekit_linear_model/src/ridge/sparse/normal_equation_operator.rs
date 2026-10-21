//! The zero-copy augmented ridge normal-equation operator and its sparse
//! helpers (spec `ridge-regression`, Decision 6).
//!
//! The operator applies `DᵀD·v + α·(P·v)` over a **borrowed** sparse design `X`
//! without ever densifying it or forming `XᵀX`: `D = [1 ‖ X]` when the intercept
//! is included (with `P = diag(0, 1, …, 1)`) and `D = X` otherwise (with `P`
//! all-ones). The `XᵀX` part reuses [`SKCsrNormalEquationOperator`]; only the
//! intercept column's constant and `Σx` are handled explicitly.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, azip, s};
use sciencekit_common::SKFloat;
use sciencekit_math::{SKCsrNormalEquationOperator, SKLinearOperator};
use sprs::CsMatView;

/// The column sums `Σx = Xᵀ·1` of a borrowed sparse design, never densifying it.
pub(crate) fn sk_csr_column_sums<F: SKFloat>(features: &CsMatView<F>) -> Array1<F> {
    let mut sums = Array1::zeros(features.cols());
    for sparse_row in features.outer_iterator() {
        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
            sums[column] = sums[column] + value;
        }
    }
    sums
}

/// The transpose product `Xᵀ·vector` of a borrowed sparse design.
pub(crate) fn sk_csr_transpose_vector_product<F: SKFloat>(
    features: &CsMatView<F>,
    vector: ArrayView1<F>,
) -> Array1<F> {
    let mut product = Array1::zeros(features.cols());
    for (row_index, sparse_row) in features.outer_iterator().enumerate() {
        let weight = vector[row_index];
        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
            product[column] = product[column] + value * weight;
        }
    }
    product
}

/// The sparse `XᵀX` Gram of a borrowed design, one operator column at a time.
///
/// Each unit column routed through [`SKCsrNormalEquationOperator`] yields one
/// Gram column; only the `p × p` Gram is dense, never the design.
pub(crate) fn sk_sparse_gram<F: SKFloat>(features: &CsMatView<F>) -> Array2<F> {
    let operator = SKCsrNormalEquationOperator::new(*features);
    let columns_count = features.cols();
    let mut gram = Array2::zeros((columns_count, columns_count));
    (0..columns_count).for_each(|column_index| {
        let mut unit = Array1::zeros(columns_count);
        unit[column_index] = F::one();
        let product = operator.apply(unit.view());
        gram.column_mut(column_index).assign(&product);
    });
    gram
}

/// The zero-copy augmented ridge normal-equation operator over a sparse design.
///
/// Holds the borrowed design and the small `O(p)` intercept data (`Σx` and the
/// row count); the design itself is neither copied nor densified.
pub(super) struct SKRidgeSparseNormalEquationOperator<'a, F: SKFloat> {
    normal_equation: SKCsrNormalEquationOperator<'a, F>,
    include_intercept: bool,
    alpha: F,
    row_count: F,
    column_sums: Array1<F>,
}

impl<'a, F: SKFloat> SKRidgeSparseNormalEquationOperator<'a, F> {
    /// Build the operator over a borrowed CSR view, the intercept flag, the
    /// penalty `alpha`, the row count and the precomputed column sums.
    pub(super) fn new(
        features: CsMatView<'a, F>,
        include_intercept: bool,
        alpha: F,
        row_count: F,
        column_sums: Array1<F>,
    ) -> Self {
        Self {
            normal_equation: SKCsrNormalEquationOperator::new(features),
            include_intercept,
            alpha,
            row_count,
            column_sums,
        }
    }
}

impl<F: SKFloat> SKLinearOperator<F> for SKRidgeSparseNormalEquationOperator<'_, F> {
    fn dimension(&self) -> usize {
        self.normal_equation.dimension() + self.include_intercept as usize
    }

    fn apply_into(&self, input: ArrayView1<F>, mut output: ArrayViewMut1<F>) {
        if self.include_intercept {
            let intercept_input = input[0];
            let features_input = input.slice(s![1..]);
            output[0] = self.row_count * intercept_input + self.column_sums.dot(&features_input);
            let mut features_output = output.slice_mut(s![1..]);
            self.normal_equation
                .apply_into(features_input, features_output.view_mut());
            azip!((entry in &mut features_output,
                   &sum in &self.column_sums,
                   &value in &features_input) {
                *entry = *entry + sum * intercept_input + self.alpha * value;
            });
        } else {
            self.normal_equation.apply_into(input, output.view_mut());
            azip!((entry in &mut output, &value in &input) {
                *entry = *entry + self.alpha * value;
            });
        }
    }
}
