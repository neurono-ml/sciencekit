//! The builder for [`SKLinearRegression`](super::core_implementation::SKLinearRegression).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use crate::linear_regression::core_implementation::SKLinearRegression;

/// Builder for [`SKLinearRegression`]: the only construction surface.
///
/// Defaults: intercept included, backend `Faer`, execution intent `Automatic`.
pub struct SKLinearRegressionBuilder {
    pub(super) state: SKBuilderState,
    pub(super) include_intercept: bool,
    pub(super) backend: SKBackendKind,
}

impl SKLinearRegressionBuilder {
    /// A fresh builder with the documented defaults.
    pub fn new() -> SKLinearRegressionBuilder {
        Self {
            state: SKBuilderState::new(),
            include_intercept: true,
            backend: SKBackendKind::Faer,
        }
    }

    /// Whether the fit solves for an intercept (default `true`).
    pub fn include_intercept(mut self, value: bool) -> Self {
        self.include_intercept = value;
        self
    }

    /// The math backend kind (default `SKBackendKind::Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The execution intent (default `SKExecutionMode::Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl Default for SKLinearRegressionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKLinearRegression> for SKLinearRegressionBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKLinearRegression, SKError> {
        Ok(SKLinearRegression {
            plan: self.state,
            include_intercept: self.include_intercept,
            backend: self.backend,
        })
    }
}
