//! Representation dispatch for the classifier's per-head loop (spec decision
//! 9): one driver over dense and zero-copy CSR designs, plus validation scoring.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsMatView;

use crate::sgd::config::SKSGDConfig;
use crate::sgd::{
    SKLearningState, SKSGDClassifierLoss, SKSGDLoss, sk_run_dense_epoch, sk_run_sparse_epoch,
};

/// A borrowed feature design, dense or CSR.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SKFeatureDesign<'a, F> {
    /// A dense matrix.
    Dense(ndarray::ArrayView2<'a, F>),
    /// A CSR matrix, never densified.
    Sparse(CsMatView<'a, F>),
}

impl<'a, F: SKFloat> SKFeatureDesign<'a, F> {
    /// Dispatch the feature view the estimator received.
    pub(crate) fn from_view(data: SKDataView<'a, F>) -> Result<Self, SKError> {
        match data {
            SKDataView::Dense(dense) => Ok(Self::Dense(dense)),
            SKDataView::Sparse(sparse) => Ok(Self::Sparse(sparse)),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            }),
        }
    }

    /// The number of rows.
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.nrows(),
            Self::Sparse(sparse) => sparse.rows(),
        }
    }

    /// The number of columns.
    pub(crate) fn columns(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ncols(),
            Self::Sparse(sparse) => sparse.cols(),
        }
    }

    /// The linear predictor of one row for the given weights and intercept.
    pub(crate) fn prediction(&self, row: usize, weights: ArrayView1<F>, intercept: F) -> F {
        match self {
            Self::Dense(dense) => dense.row(row).dot(&weights) + intercept,
            Self::Sparse(sparse) => {
                let sparse_row = sparse.outer_view(row).expect("row within bounds");
                let score = sparse_row
                    .indices()
                    .iter()
                    .zip(sparse_row.data())
                    .fold(F::zero(), |accumulator, (&column, &value)| {
                        accumulator + value * weights[column]
                    });
                score + intercept
            }
        }
    }
}

/// Run one epoch of one head, returning its mean weighted loss.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_head_epoch<F: SKFloat>(
    state: &mut SKLearningState<F>,
    design: SKFeatureDesign<'_, F>,
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: SKSGDClassifierLoss,
    config: &SKSGDConfig<F>,
    order: &[usize],
) -> Result<F, SKError> {
    let objective = match design {
        SKFeatureDesign::Dense(dense) => {
            sk_run_dense_epoch(state, dense, targets, sample_weights, &loss, config, order)?
        }
        SKFeatureDesign::Sparse(sparse) => {
            sk_run_sparse_epoch(state, sparse, targets, sample_weights, &loss, config, order)?
        }
    };
    Ok(objective.mean())
}

/// Score one head's reported iterate on the selected rows, without updating.
pub(crate) fn sk_score_rows<F: SKFloat>(
    state: &SKLearningState<F>,
    design: SKFeatureDesign<'_, F>,
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: SKSGDClassifierLoss,
    rows: &[usize],
) -> F {
    if rows.is_empty() {
        return F::zero();
    }
    let weights = state.reported_weights();
    let intercept = state.reported_intercept();
    let total = rows.iter().fold(F::zero(), |accumulator, &row| {
        let prediction = design.prediction(row, weights, intercept);
        accumulator + sample_weights[row] * SKSGDLoss::<F>::value(&loss, prediction, targets[row])
    });
    total / F::from(rows.len()).expect("the row count fits in the scalar")
}
