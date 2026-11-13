//! The nearest-neighbor imputer builder
//! (spec `nearest-neighbor-imputation`).

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
};

use crate::knn::core_implementation::SKKNNImputer;
use crate::knn::weights::SKImputationWeights;

/// Builder for [`SKKNNImputer`]; the only construction path.
pub struct SKKNNImputerBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) missing_values: F,
    pub(super) nearest_neighbors_count: usize,
    pub(super) weights: SKImputationWeights,
    pub(super) add_indicator: bool,
    pub(super) keep_empty_features: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKKNNImputerBuilder<F> {
    /// A builder with the documented defaults.
    pub fn new() -> SKKNNImputerBuilder<F> {
        Self {
            state: SKBuilderState::new(),
            missing_values: F::nan(),
            nearest_neighbors_count: 5,
            weights: SKImputationWeights::Uniform,
            add_indicator: false,
            keep_empty_features: false,
            backend: SKBackendKind::Faer,
        }
    }

    /// The value that marks a cell as missing; defaults to `NaN`.
    pub fn missing_values(mut self, value: F) -> Self {
        self.missing_values = value;
        self
    }

    /// The number of donors consulted per missing cell; defaults to five.
    pub fn nearest_neighbors_count(mut self, value: usize) -> Self {
        self.nearest_neighbors_count = value;
        self
    }

    /// The donor weighting scheme; defaults to `Uniform`.
    pub fn weights(mut self, value: SKImputationWeights) -> Self {
        self.weights = value;
        self
    }

    /// Stack a `0.0`/`1.0` column per fit-time missing feature; defaults to
    /// `false`.
    pub fn add_indicator(mut self, value: bool) -> Self {
        self.add_indicator = value;
        self
    }

    /// Keep all-missing columns as `0.0` instead of dropping them; defaults to
    /// `false`.
    pub fn keep_empty_features(mut self, value: bool) -> Self {
        self.keep_empty_features = value;
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

impl<F: SKFloat> Default for SKKNNImputerBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKKNNImputer<F>> for SKKNNImputerBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKKNNImputer<F>, SKError> {
        if self.nearest_neighbors_count == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "nearest_neighbors_count",
                reason: "at least one donor is required to impute a cell".to_owned(),
            });
        }
        if self.missing_values.is_infinite() {
            return Err(SKError::InvalidHyperparameter {
                name: "missing_values",
                reason: "the missing marker must be finite or NaN".to_owned(),
            });
        }
        Ok(SKKNNImputer {
            plan: self.state,
            missing_values: self.missing_values,
            nearest_neighbors_count: self.nearest_neighbors_count,
            weights: self.weights,
            add_indicator: self.add_indicator,
            keep_empty_features: self.keep_empty_features,
            backend: self.backend,
        })
    }
}
