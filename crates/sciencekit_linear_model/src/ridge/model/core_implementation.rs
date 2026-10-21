//! The fitted ridge model, its accessors and its per-alpha prediction
//! (spec `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};

use crate::ridge::builder::SKRidgeSolver;
use crate::ridge::gram::SKCenteredGramState;

/// Fitted ridge model: one coefficient vector and one intercept per `alpha`,
/// the resolved solver, the centered Gram state and the solve's rank and
/// singular values.
///
/// Distinct type from the estimator (`SKRidge`): the estimator is immutable
/// configuration, the model is the learned state and the export boundary. The
/// model is `Send + Sync` by construction, so one model serves concurrent
/// predictions.
#[derive(Debug, Clone)]
pub struct SKRidgeModel<F> {
    pub(super) alphas: Vec<F>,
    pub(super) coefficients: Array2<F>,
    pub(super) intercepts: Array1<F>,
    pub(super) include_intercept: bool,
    pub(super) solver: SKRidgeSolver,
    pub(super) rank: usize,
    pub(super) singular_values: Vec<F>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
    pub(super) gram: SKCenteredGramState<F>,
}

impl<F: SKFloat> SKRidgeModel<F> {
    /// Build a fitted model from a dense ridge solve and the design's centered
    /// Gram state. Crate-internal: models are produced by `fit`, never
    /// constructed directly by consumers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_solve(
        alphas: Vec<F>,
        coefficients: Array2<F>,
        intercepts: Array1<F>,
        rank: usize,
        singular_values: Vec<F>,
        solver: SKRidgeSolver,
        include_intercept: bool,
        backend: SKBackendKind,
        execution_intent: SKExecutionMode,
        gram: SKCenteredGramState<F>,
    ) -> Self {
        Self {
            alphas,
            coefficients,
            intercepts,
            include_intercept,
            solver,
            rank,
            singular_values,
            backend,
            execution_intent,
            gram,
        }
    }

    /// The configured penalty values, in configuration order.
    pub fn alphas(&self) -> &[F] {
        &self.alphas
    }

    /// The learned coefficients, shaped `n_alphas × n_features`.
    pub fn coefficients(&self) -> ArrayView2<F> {
        self.coefficients.view()
    }

    /// The learned intercepts, one per `alpha` (zeros when the intercept is
    /// disabled).
    pub fn intercepts(&self) -> ArrayView1<F> {
        self.intercepts.view()
    }

    /// The solver actually used, which may differ from the configured one after
    /// `Automatic` resolution or the Cholesky-to-SVD fallback.
    pub fn solver(&self) -> SKRidgeSolver {
        self.solver
    }

    /// Whether the fit solved for an unpenalized intercept.
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The reported rank of the matrix the last solve factored.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The singular values of the matrix the last solve factored, non-increasing
    /// (empty when the Cholesky path ran).
    pub fn singular_values(&self) -> &[F] {
        &self.singular_values
    }

    /// The number of features the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.coefficients.ncols()
    }

    /// The number of rows the model has learned from.
    pub fn row_count(&self) -> u64 {
        self.gram.row_count
    }

    /// The coefficient row for one `alpha` index.
    ///
    /// Panics when `index` is out of range; use [`SKRidgeModel::alphas`] to bound
    /// the valid range.
    pub fn coefficient_for(&self, index: usize) -> ArrayView1<F> {
        self.coefficients.row(index)
    }

    /// The intercept for one `alpha` index.
    ///
    /// Panics when `index` is out of range.
    pub fn intercept_for(&self, index: usize) -> F {
        self.intercepts[index]
    }

    /// Predict responses for one explicit `alpha` index.
    pub fn predict_with_alpha_index<'a, X>(
        &self,
        index: usize,
        features: X,
    ) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
        if dense.ncols() != self.feature_count() {
            return Err(SKError::shape_mismatch_2d(
                dense.nrows(),
                self.feature_count(),
                dense.nrows(),
                dense.ncols(),
            ));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count: dense.nrows(),
            columns_count: dense.ncols(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let intercept = self.intercepts[index];
            Ok(dense
                .dot(&self.coefficients.row(index))
                .mapv(|value| value + intercept))
        })
    }
}
