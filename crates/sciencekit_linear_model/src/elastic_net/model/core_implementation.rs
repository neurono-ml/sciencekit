//! The fitted elastic-net model: accessors, prediction and incremental learning
//! (spec `elastic-net`).

use ndarray::{Array1, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKDataView, SKError, SKFloat, SKTargetView};

use crate::coordinate_descent::SKCoordinateDescentModelCore;

/// Fitted elastic-net model: one coefficient vector and one intercept per
/// `alpha`, the `l1_ratio`, the backend kind and the retained design that backs
/// incremental updates.
///
/// Distinct type from the estimator ([`crate::elastic_net::SKElasticNet`]): the
/// estimator is immutable configuration, the model is the learned state and the
/// export boundary. The model is `Send + Sync` by construction.
#[derive(Debug, Clone)]
pub struct SKElasticNetModel<F> {
    pub(crate) core: SKCoordinateDescentModelCore<F>,
}

impl<F: SKFloat> SKElasticNetModel<F> {
    /// Wrap the shared learned core (crate-internal: models come from `fit`).
    pub(crate) fn from_core(core: SKCoordinateDescentModelCore<F>) -> Self {
        Self { core }
    }

    /// The configured penalty values, in configuration order.
    pub fn alphas(&self) -> &[F] {
        self.core.alphas()
    }

    /// The learned coefficients, shaped `n_alphas × n_features`.
    pub fn coefficients(&self) -> ArrayView2<F> {
        self.core.coefficients()
    }

    /// The learned intercepts, one per `alpha`.
    pub fn intercepts(&self) -> ArrayView1<F> {
        self.core.intercepts()
    }

    /// The configured L1/L2 mixing ratio.
    pub fn l1_ratio(&self) -> F {
        self.core.l1_ratio()
    }

    /// Whether the model reports an intercept.
    pub fn include_intercept(&self) -> bool {
        self.core.include_intercept()
    }

    /// Whether every coefficient is constrained to be non-negative.
    pub fn positive(&self) -> bool {
        self.core.positive()
    }

    /// The backend kind carried from the estimator that produced this model.
    pub fn backend(&self) -> SKBackendKind {
        self.core.backend()
    }

    /// The number of coordinates screening excluded across the alphas.
    pub fn screened_count(&self) -> usize {
        self.core.screened_count()
    }

    /// The number of features the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.core.feature_count()
    }

    /// The number of rows the model has learned from.
    pub fn row_count(&self) -> u64 {
        self.core.row_count()
    }

    /// The coefficient row for one `alpha` index.
    pub fn coefficient_for(&self, index: usize) -> ArrayView1<F> {
        self.core.coefficient_for(index)
    }

    /// The intercept for one `alpha` index.
    pub fn intercept_for(&self, index: usize) -> F {
        self.core.intercept_for(index)
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
        self.core.predict_with_alpha_index(index, features)
    }

    /// Continue learning from one more batch.
    ///
    /// The batch is retained and the whole accumulated design is re-solved;
    /// dense and sparse batches are both accepted.
    pub fn partial_fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        self.core.partial_fit(features, targets)?;
        Ok(self)
    }
}
