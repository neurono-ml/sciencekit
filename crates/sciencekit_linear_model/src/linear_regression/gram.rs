//! The running cross-product state shared by the incremental and streaming
//! regimes (spec `linear-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;
use sciencekit_math::{sk_gram_matrix, sk_gram_vector};

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

    /// Accumulate one batch's cross products into the running state.
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
}
