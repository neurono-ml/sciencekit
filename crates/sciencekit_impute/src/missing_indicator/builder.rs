//! The `SKMissingIndicator` builder (spec `missing-indicator`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use crate::missing_indicator::core_implementation::SKMissingIndicator;
use crate::missing_indicator::scope::SKMissingFeatureScope;

/// Builder for [`SKMissingIndicator`]; the only construction path.
pub struct SKMissingIndicatorBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) features: SKMissingFeatureScope,
    pub(super) error_on_new: bool,
    pub(super) sparse_output: bool,
    pub(super) missing_values: F,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKMissingIndicatorBuilder<F> {
    /// A builder with the documented defaults.
    pub fn new() -> SKMissingIndicatorBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            features: SKMissingFeatureScope::MissingOnly,
            error_on_new: true,
            sparse_output: false,
            missing_values: F::nan(),
            backend: SKBackendKind::Faer,
        }
    }

    /// Which feature columns the mask covers; defaults to `MissingOnly`.
    pub fn features(mut self, value: SKMissingFeatureScope) -> Self {
        self.features = value;
        self
    }

    /// Reject missingness at transform in a feature complete at fit time;
    /// defaults to `true`.
    pub fn error_on_new(mut self, value: bool) -> Self {
        self.error_on_new = value;
        self
    }

    /// Emit a CSR mask even for dense input; defaults to `false`.
    pub fn sparse_output(mut self, value: bool) -> Self {
        self.sparse_output = value;
        self
    }

    /// The value that marks a cell as missing; defaults to `NaN`.
    pub fn missing_values(mut self, value: F) -> Self {
        self.missing_values = value;
        self
    }

    /// The backend kind recorded in operation spans; defaults to `Faer`.
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.backend = value;
        self
    }

    /// Set the execution intent; the default is `Automatic`.
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat> Default for SKMissingIndicatorBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKMissingIndicator<F>> for SKMissingIndicatorBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKMissingIndicator<F>, SKError> {
        Ok(SKMissingIndicator {
            plan: self.state,
            features: self.features,
            error_on_new: self.error_on_new,
            sparse_output: self.sparse_output,
            missing_values: self.missing_values,
            backend: self.backend,
        })
    }
}
