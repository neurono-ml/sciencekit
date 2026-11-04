//! The running cross-product state shared by the incremental and streaming
//! regimes (spec `linear-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, azip, s};
use sciencekit_common::SKFloat;
use sciencekit_math::{sk_gram_matrix, sk_gram_vector};
use sprs::CsMatView;

use crate::ridge::sparse::{sk_csr_column_sums, sk_csr_transpose_vector_product, sk_sparse_gram};

/// Running cross products of a (possibly augmented) design matrix and its
/// targets: `XᵀX`, `Xᵀy` and the accumulated row count.
///
/// The single source of truth for the incremental and streaming regimes — the
/// coefficients are a pure function of these quantities, so batch order and
/// sizes do not change the result. The dense `fit` records the same state so a
/// fitted model can keep learning through `partial_fit`.
#[derive(Debug, Clone)]
pub(crate) struct SKGramState<F> {
    /// The design width (`features + intercept`).
    pub(super) design_width: usize,
    /// The accumulated `XᵀX`, `design_width × design_width`.
    pub(super) gram: Array2<F>,
    /// The accumulated `Xᵀy`, length `design_width`.
    pub(super) right_hand_side: Array1<F>,
    /// The number of rows accumulated across every batch.
    pub(super) row_count: u64,
}

impl<F: SKFloat> SKGramState<F> {
    /// An all-zero state for a design of the given width.
    pub(super) fn empty(design_width: usize) -> Self {
        Self {
            design_width,
            gram: Array2::zeros((design_width, design_width)),
            right_hand_side: Array1::zeros(design_width),
            row_count: 0,
        }
    }

    /// Accumulate one dense batch's cross products into the running state.
    pub(super) fn accumulate(
        &mut self,
        design: &ArrayView2<F>,
        target: &ArrayView1<F>,
        parallelism: usize,
    ) {
        self.gram = &self.gram + &sk_gram_matrix(design, parallelism);
        self.right_hand_side = &self.right_hand_side + &sk_gram_vector(design, target, parallelism);
        self.row_count += design.nrows() as u64;
    }

    /// Accumulate one sparse batch's cross products into the running state.
    ///
    /// `XᵀX` comes from the zero-copy normal-equation operator, `Σx` from
    /// `Xᵀ·1` and `Xᵀy` from `Xᵀ·y`; the implicit intercept column is assembled
    /// from the row count and `Σx`. The design is never densified — only the
    /// `p × p` Gram and the `O(p)` vectors are materialized.
    pub(super) fn accumulate_sparse(
        &mut self,
        features: &CsMatView<'_, F>,
        target: &ArrayView1<F>,
        include_intercept: bool,
    ) {
        let sparse_gram = sk_sparse_gram(features);
        let cross = sk_csr_transpose_vector_product(features, target.view());
        let column_sums = sk_csr_column_sums(features);
        if include_intercept {
            let rows = F::from(features.rows()).expect("row counts fit in the scalar");
            self.gram[[0, 0]] = self.gram[[0, 0]] + rows;
            for (index, &sum) in column_sums.iter().enumerate() {
                self.gram[[0, index + 1]] = self.gram[[0, index + 1]] + sum;
                self.gram[[index + 1, 0]] = self.gram[[index + 1, 0]] + sum;
            }
            azip!((gram in self.gram.slice_mut(s![1.., 1..]), &value in &sparse_gram) {
                *gram = *gram + value;
            });
            self.right_hand_side[0] = self.right_hand_side[0] + target.sum();
            azip!((rhs in self.right_hand_side.slice_mut(s![1..]), &value in &cross) {
                *rhs = *rhs + value;
            });
        } else {
            self.gram = &self.gram + &sparse_gram;
            self.right_hand_side = &self.right_hand_side + &cross;
        }
        self.row_count += features.rows() as u64;
    }
}
