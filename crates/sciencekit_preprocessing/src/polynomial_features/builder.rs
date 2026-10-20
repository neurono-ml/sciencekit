//! The `SKPolynomialFeatures` builder (spec `polynomial-features`).

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, sk_validate_hyperparameter,
};

use crate::polynomial_features::core_implementation::SKPolynomialFeatures;

/// Builder for [`SKPolynomialFeatures`]; the only construction path.
pub struct SKPolynomialFeaturesBuilder {
    pub(super) state: SKBuilderState,
    pub(super) degree: usize,
    pub(super) interaction_only: bool,
    pub(super) include_bias: bool,
    pub(super) sparse_output: bool,
    pub(super) backend: SKBackendKind,
}

impl SKPolynomialFeaturesBuilder {
    /// A builder with the documented defaults.
    pub fn new() -> SKPolynomialFeaturesBuilder {
        Self {
            state: SKBuilderState::new(),
            degree: 2,
            interaction_only: false,
            include_bias: true,
            sparse_output: false,
            backend: SKBackendKind::Faer,
        }
    }

    /// The maximum total degree of the expansion.
    pub fn degree(mut self, value: usize) -> Self {
        self.degree = value;
        self
    }

    /// Restrict terms to distinct features (no power terms).
    pub fn interaction_only(mut self, value: bool) -> Self {
        self.interaction_only = value;
        self
    }

    /// Include the constant (bias) term.
    pub fn include_bias(mut self, value: bool) -> Self {
        self.include_bias = value;
        self
    }

    /// Whether the default transform returns a sparse CSR matrix.
    pub fn sparse_output(mut self, value: bool) -> Self {
        self.sparse_output = value;
        self
    }

    /// The math backend kind recorded in operation spans and carried into the
    /// fitted model.
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The execution intent.
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl Default for SKPolynomialFeaturesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKPolynomialFeatures> for SKPolynomialFeaturesBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKPolynomialFeatures, SKError> {
        sk_validate_hyperparameter("degree", self.degree >= 1, "degree must be at least one")?;
        Ok(SKPolynomialFeatures {
            state: self.state,
            degree: self.degree,
            interaction_only: self.interaction_only,
            include_bias: self.include_bias,
            sparse_output: self.sparse_output,
            backend: self.backend,
        })
    }
}
