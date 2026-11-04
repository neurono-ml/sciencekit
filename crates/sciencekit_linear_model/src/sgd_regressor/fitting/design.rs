//! Representation dispatch for the regressor's epoch loop (spec decision 9):
//! one driver over dense and zero-copy CSR designs, plus validation scoring.

use ndarray::ArrayView1;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sprs::CsMatView;

use crate::sgd::config::SKSGDConfig;
use crate::sgd::{SKLearningState, SKSGDLoss, sk_run_dense_epoch, sk_run_sparse_epoch};

/// A borrowed feature design, dense or CSR.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SKRegressorDesign<'a, F> {
    /// A dense matrix.
    Dense(ndarray::ArrayView2<'a, F>),
    /// A CSR matrix, never densified.
    Sparse(CsMatView<'a, F>),
}

impl<'a, F: SKFloat> SKRegressorDesign<'a, F> {
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

/// Run one epoch over the design, returning the mean weighted loss.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_run_design_epoch<F, L>(
    state: &mut SKLearningState<F>,
    design: SKRegressorDesign<'_, F>,
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: &L,
    config: &SKSGDConfig<F>,
    order: &[usize],
) -> Result<F, SKError>
where
    F: SKFloat,
    L: SKSGDLoss<F>,
{
    let objective = match design {
        SKRegressorDesign::Dense(dense) => {
            sk_run_dense_epoch(state, dense, targets, sample_weights, loss, config, order)?
        }
        SKRegressorDesign::Sparse(sparse) => {
            sk_run_sparse_epoch(state, sparse, targets, sample_weights, loss, config, order)?
        }
    };
    Ok(objective.mean())
}

/// Score the reported iterate on the selected rows, without updating.
pub(crate) fn sk_score_rows<F, L>(
    state: &SKLearningState<F>,
    design: SKRegressorDesign<'_, F>,
    targets: ArrayView1<F>,
    sample_weights: &[F],
    loss: &L,
    rows: &[usize],
) -> F
where
    F: SKFloat,
    L: SKSGDLoss<F>,
{
    if rows.is_empty() {
        return F::zero();
    }
    let weights = state.reported_weights();
    let intercept = state.reported_intercept();
    let total = rows.iter().fold(F::zero(), |accumulator, &row| {
        let prediction = design.prediction(row, weights, intercept);
        accumulator + sample_weights[row] * loss.value(prediction, targets[row])
    });
    total / F::from(rows.len()).expect("the row count fits in the scalar")
}
