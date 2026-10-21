//! The centered-Gram accumulator shared by the dense ridge fit and its later
//! incremental/streaming regimes (spec `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, azip};
use sciencekit_common::SKFloat;
use sciencekit_math::{sk_axis_sum, sk_gram_matrix, sk_gram_vector};
use sprs::CsMatView;

use crate::ridge::sparse::{sk_csr_column_sums, sk_csr_transpose_vector_product, sk_sparse_gram};

/// Running centered cross products of a design matrix and its targets.
///
/// Holds the raw sums `XᵀX`, `Xᵀy`, `Σx` and `Σy` plus the accumulated row
/// count; [`SKCenteredGramState::centered`] derives the centered Gram
/// `Ĉ = XᵀX − n·x̄x̄ᵀ` and the centered cross `ĉ = Xᵀy − n·x̄·ȳ` the Cholesky and
/// SVD solvers consume. The coefficients are a pure function of these sums, so
/// batch order and sizes do not change a later result.
#[derive(Debug, Clone)]
pub(crate) struct SKCenteredGramState<F> {
    /// The accumulated `XᵀX`, `features_count × features_count` (all-zero when
    /// [`SKCenteredGramState::has_full_gram`] is false).
    pub(super) gram: Array2<F>,
    /// The accumulated `Xᵀy`, length `features_count`.
    pub(super) cross: Array1<F>,
    /// The accumulated per-feature sums `Σx`, length `features_count`.
    pub(super) column_sums: Array1<F>,
    /// The accumulated target sum `Σy`.
    pub(super) target_sum: F,
    /// The number of rows accumulated across every batch.
    pub(super) row_count: u64,
    /// Whether the `p × p` Gram was materialized.
    ///
    /// The `SparseCg` path never forms `XᵀX` (Decision 6), so its state keeps
    /// the `O(p)` statistics but an all-zero Gram; the incremental regime
    /// refuses to continue from such a state rather than solve a partial system.
    pub(super) has_full_gram: bool,
}

impl<F: SKFloat> SKCenteredGramState<F> {
    /// An all-zero state with an empty (but present) Gram for a design of the
    /// given width.
    pub(super) fn empty(features_count: usize) -> Self {
        Self {
            gram: Array2::zeros((features_count, features_count)),
            cross: Array1::zeros(features_count),
            column_sums: Array1::zeros(features_count),
            target_sum: F::zero(),
            row_count: 0,
            has_full_gram: true,
        }
    }

    /// The `O(p)` statistics of one sparse batch, **without** materializing the
    /// `p × p` Gram.
    ///
    /// This is the state a `SparseCg` fit keeps: `Xᵀy`, `Σx`, `Σy` and the row
    /// count, so the model can report its row count, while `XᵀX` is never formed
    /// (the conjugate-gradient operator applies `DᵀD` on the fly).
    pub(super) fn statistics_only(features: &CsMatView<F>, target: &ArrayView1<F>) -> Self {
        Self {
            gram: Array2::zeros((features.cols(), features.cols())),
            cross: sk_csr_transpose_vector_product(features, target.view()),
            column_sums: sk_csr_column_sums(features),
            target_sum: target.sum(),
            row_count: features.rows() as u64,
            has_full_gram: false,
        }
    }

    /// Accumulate one dense batch's cross products into the running state.
    pub(super) fn accumulate_dense(
        &mut self,
        features: &ArrayView2<F>,
        target: &ArrayView1<F>,
        parallelism: usize,
    ) {
        self.gram = &self.gram + &sk_gram_matrix(features, parallelism);
        self.cross = &self.cross + &sk_gram_vector(features, target, parallelism);
        self.column_sums = &self.column_sums + &sk_axis_sum(features, 0, parallelism);
        self.target_sum = self.target_sum + target.sum();
        self.row_count += features.nrows() as u64;
    }

    /// Accumulate one sparse batch's cross products into the running state.
    ///
    /// `XᵀX` comes from the zero-copy normal-equation operator applied to unit
    /// columns, `Σx` from `Xᵀ·1` and `Xᵀy` from `Xᵀ·y`; the design is never
    /// densified, only the `p × p` Gram and the `O(p)` vectors are materialized.
    pub(super) fn accumulate_sparse(&mut self, features: &CsMatView<F>, target: &ArrayView1<F>) {
        self.gram = &self.gram + &sk_sparse_gram(features);
        self.cross = &self.cross + &sk_csr_transpose_vector_product(features, target.view());
        self.column_sums = &self.column_sums + &sk_csr_column_sums(features);
        self.target_sum = self.target_sum + target.sum();
        self.row_count += features.rows() as u64;
    }

    /// The centered cross products `(Ĉ, ĉ)` plus the feature means `x̄` and the
    /// target mean `ȳ`.
    ///
    /// `Ĉ = XᵀX − n·x̄x̄ᵀ` and `ĉ = Xᵀy − n·x̄·ȳ`, computed from the accumulated
    /// sums without index loops.
    pub(super) fn centered(&self) -> (Array2<F>, Array1<F>, Array1<F>, F) {
        let row_count = F::from(self.row_count).expect("row counts fit in the scalar");
        let feature_means = self.column_sums.mapv(|sum| sum / row_count);
        let target_mean = self.target_sum / row_count;
        let mut centered_gram = self.gram.clone();
        centered_gram
            .rows_mut()
            .into_iter()
            .zip(feature_means.iter())
            .for_each(|(mut gram_row, &row_mean)| {
                azip!((entry in &mut gram_row, &column_mean in &feature_means) {
                    *entry = *entry - row_count * row_mean * column_mean;
                });
            });
        let mut centered_cross = self.cross.clone();
        azip!((entry in &mut centered_cross, &sum in &self.column_sums) {
            *entry = *entry - sum * target_mean;
        });
        (centered_gram, centered_cross, feature_means, target_mean)
    }
}
