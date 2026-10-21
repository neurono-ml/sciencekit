//! Accessors shared by the L1-regularized fitted models (spec
//! `coordinate-descent-solver`).

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKError, SKFloat};

use super::definition::SKCoordinateDescentModelCore;

impl<F: SKFloat> SKCoordinateDescentModelCore<F> {
    /// The configured penalty values, in configuration order.
    pub(crate) fn alphas(&self) -> &[F] {
        &self.alphas
    }

    /// The configured `l1_ratio` (one for lasso).
    pub(crate) fn l1_ratio(&self) -> F {
        self.l1_ratio
    }

    /// The learned coefficients, shaped `n_alphas × n_features`.
    pub(crate) fn coefficients(&self) -> ArrayView2<F> {
        self.coefficients.view()
    }

    /// The learned intercepts, one per `alpha`.
    pub(crate) fn intercepts(&self) -> ArrayView1<F> {
        self.intercepts.view()
    }

    /// Whether the fit solved for an unpenalized intercept.
    pub(crate) fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Whether every coordinate is clamped at zero.
    pub(crate) fn positive(&self) -> bool {
        self.config.positive
    }

    /// The backend kind carried from the estimator that produced this model.
    pub(crate) fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The number of coordinates screening excluded across the alphas.
    pub(crate) fn screened_count(&self) -> usize {
        self.screened_count
    }

    /// The number of features the model was fitted on.
    pub(crate) fn feature_count(&self) -> usize {
        self.coefficients.ncols()
    }

    /// The number of rows the model has learned from.
    pub(crate) fn row_count(&self) -> u64 {
        self.state.row_count()
    }

    /// The coefficient row for one `alpha` index.
    pub(crate) fn coefficient_for(&self, index: usize) -> ArrayView1<F> {
        self.coefficients.row(index)
    }

    /// The intercept for one `alpha` index.
    pub(crate) fn intercept_for(&self, index: usize) -> F {
        self.intercepts[index]
    }

    /// The error for a plain prediction when several alphas are configured.
    pub(crate) fn ambiguous_prediction_error(&self) -> SKError {
        SKError::InvalidHyperparameter {
            name: "alpha",
            reason: "several alphas were configured; use predict_with_alpha_index to select one"
                .to_string(),
        }
    }
}
