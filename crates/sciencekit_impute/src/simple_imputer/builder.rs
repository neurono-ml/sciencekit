//! The `SKSimpleImputer` builder (spec `simple-imputer`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use crate::simple_imputer::core_implementation::SKSimpleImputer;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// Builder for [`SKSimpleImputer`]; the only construction path.
pub struct SKSimpleImputerBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) strategy: SKImputationStrategy,
    pub(super) missing_values: F,
    pub(super) fill_value: Option<F>,
    pub(super) keep_empty_features: bool,
    pub(super) add_indicator: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKSimpleImputerBuilder<F> {
    /// A builder with the documented defaults.
    pub fn new() -> SKSimpleImputerBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            strategy: SKImputationStrategy::Median,
            missing_values: F::zero(),
            fill_value: None,
            keep_empty_features: true,
            add_indicator: true,
            backend: SKBackendKind::NdArrayLinalg,
        }
    }

    /// The per-column statistic that fills the missing cells.
    pub fn strategy(mut self, value: SKImputationStrategy) -> Self {
        self.strategy = value;
        self
    }

    /// The value that marks a cell as missing; defaults to `NaN`.
    pub fn missing_values(mut self, value: F) -> Self {
        self.missing_values = value;
        self
    }

    /// The constant to fill with under [`SKImputationStrategy::Constant`].
    pub fn fill_value(mut self, value: Option<F>) -> Self {
        self.fill_value = value;
        self
    }

    /// Keep all-missing columns as `0.0` instead of dropping them.
    pub fn keep_empty_features(mut self, value: bool) -> Self {
        self.keep_empty_features = value;
        self
    }

    /// Stack a `0.0`/`1.0` column per fit-time missing feature.
    pub fn add_indicator(mut self, value: bool) -> Self {
        self.add_indicator = value;
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

impl<F: SKFloat> Default for SKSimpleImputerBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKSimpleImputer<F>> for SKSimpleImputerBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKSimpleImputer<F>, SKError> {
        Ok(SKSimpleImputer {
            plan: self.state,
            strategy: self.strategy,
            missing_values: self.missing_values,
            fill_value: self.fill_value,
            keep_empty_features: self.keep_empty_features,
            add_indicator: self.add_indicator,
            backend: self.backend,
        })
    }
}
