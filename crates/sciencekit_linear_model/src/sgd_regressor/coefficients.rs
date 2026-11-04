//! The explicit warm-start source of the SGD regressor (spec decision 13).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::SKFloat;

/// Coefficients and intercept used to seed a fit.
///
/// The estimator stays immutable: a warm start is an explicit value supplied to
/// the builder, never mutable instance state. A source whose feature count
/// disagrees with the fit is rejected by `fit`.
#[derive(Debug, Clone)]
pub struct SKSGDRegressorCoefficients<F> {
    pub(crate) coefficients: Array1<F>,
    pub(crate) intercept: F,
}

impl<F: SKFloat> SKSGDRegressorCoefficients<F> {
    /// Assemble a warm-start source from its coefficients and intercept.
    pub fn new(coefficients: ArrayView1<'_, F>, intercept: F) -> Self {
        Self {
            coefficients: coefficients.to_owned(),
            intercept,
        }
    }

    /// The coefficients, one per feature.
    pub fn coefficients(&self) -> ArrayView1<'_, F> {
        self.coefficients.view()
    }

    /// The intercept.
    pub fn intercept(&self) -> F {
        self.intercept
    }

    /// The number of features the coefficients were fitted on.
    pub fn feature_count(&self) -> usize {
        self.coefficients.len()
    }
}
