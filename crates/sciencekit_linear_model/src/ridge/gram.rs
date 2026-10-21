//! The centered-Gram accumulator shared by the dense ridge fit and its later
//! incremental/streaming regimes (spec `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, azip};
use sciencekit_common::SKFloat;
use sciencekit_math::{sk_axis_sum, sk_gram_matrix, sk_gram_vector};

/// Running centered cross products of a design matrix and its targets.
///
/// Holds the raw sums `XᵀX`, `Xᵀy`, `Σx` and `Σy` plus the accumulated row
/// count; [`SKCenteredGramState::centered`] derives the centered Gram
/// `Ĉ = XᵀX − n·x̄x̄ᵀ` and the centered cross `ĉ = Xᵀy − n·x̄·ȳ` the Cholesky and
/// SVD solvers consume. The coefficients are a pure function of these sums, so
/// batch order and sizes do not change a later result.
#[derive(Debug, Clone)]
pub(crate) struct SKCenteredGramState<F> {
    /// The design width (number of features, excluding the intercept).
    ///
    /// Carried for the streaming/incremental regimes that reuse this state;
    /// the dense fit derives the width from the Gram's own shape.
    #[allow(dead_code)]
    pub(super) features_count: usize,
    /// The accumulated `XᵀX`, `features_count × features_count`.
    pub(super) gram: Array2<F>,
    /// The accumulated `Xᵀy`, length `features_count`.
    pub(super) cross: Array1<F>,
    /// The accumulated per-feature sums `Σx`, length `features_count`.
    pub(super) column_sums: Array1<F>,
    /// The accumulated target sum `Σy`.
    pub(super) target_sum: F,
    /// The number of rows accumulated across every batch.
    pub(super) row_count: u64,
}

impl<F: SKFloat> SKCenteredGramState<F> {
    /// An all-zero state for a design of the given width.
    pub(super) fn empty(features_count: usize) -> Self {
        Self {
            features_count,
            gram: Array2::zeros((features_count, features_count)),
            cross: Array1::zeros(features_count),
            column_sums: Array1::zeros(features_count),
            target_sum: F::zero(),
            row_count: 0,
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
