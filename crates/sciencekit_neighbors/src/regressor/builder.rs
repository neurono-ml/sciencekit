//! Builder-only construction for `SKKNeighborsRegressor`
//! (spec `nearest-neighbor-estimators`).

use std::marker::PhantomData;

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
};

use super::SKKNeighborsRegressor;
use crate::neighbors::{
    SKNeighborsAlgorithm, SKNeighborsConfiguration, SKNeighborsMetric, SKNeighborsWeights,
};

/// Builder carrying the documented regressor defaults.
pub struct SKKNeighborsRegressorBuilder<F: SKFloat> {
    state: SKBuilderState,
    configuration: SKNeighborsConfiguration,
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKKNeighborsRegressorBuilder<F> {
    /// A builder with the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            configuration: SKNeighborsConfiguration {
                nearest_neighbors_count: 5,
                weights: SKNeighborsWeights::Uniform,
                algorithm: SKNeighborsAlgorithm::Auto,
                metric: SKNeighborsMetric::Euclidean,
                leaf_size: 40,
                backend: SKBackendKind::Faer,
                execution_mode: SKExecutionMode::Automatic,
            },
            marker: PhantomData,
        }
    }

    /// Sets the number of neighbours consulted per query row.
    pub fn nearest_neighbors_count(mut self, value: usize) -> Self {
        self.configuration.nearest_neighbors_count = value;
        self
    }

    /// Sets the neighbour weighting scheme.
    pub fn weights(mut self, value: SKNeighborsWeights) -> Self {
        self.configuration.weights = value;
        self
    }

    /// Sets the search algorithm (`Auto` by default).
    pub fn algorithm(mut self, value: SKNeighborsAlgorithm) -> Self {
        self.configuration.algorithm = value;
        self
    }

    /// Sets the distance metric.
    pub fn metric(mut self, value: SKNeighborsMetric) -> Self {
        self.configuration.metric = value;
        self
    }

    /// Sets the maximum number of rows in a tree leaf.
    pub fn leaf_size(mut self, value: usize) -> Self {
        self.configuration.leaf_size = value;
        self
    }

    /// Sets the backend kind recorded in operation spans.
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.configuration.backend = value;
        self
    }

    /// Sets the execution intent.
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self.configuration.execution_mode = self.state.execution_intent();
        self
    }
}

impl<F: SKFloat> Default for SKKNeighborsRegressorBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKKNeighborsRegressor<F>> for SKKNeighborsRegressorBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self.configuration.execution_mode = self.state.execution_intent();
        self
    }

    fn build(self) -> Result<SKKNeighborsRegressor<F>, SKError> {
        self.configuration.validate()?;
        Ok(SKKNeighborsRegressor::from_configuration(
            self.configuration,
        ))
    }
}
