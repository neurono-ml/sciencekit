//! The mandatory DBSCAN builder (spec `dbscan`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionMode, SKFloat, builders::SKBuilder,
    sk_validate_hyperparameter,
};

use super::config::{SKDBSCANAlgorithm, SKDBSCANConfiguration, SKDBSCANMetric};
use super::core_implementation::SKDBSCAN;

/// Builder-only construction surface for [`SKDBSCAN`].
pub struct SKDBSCANBuilder<F: SKFloat> {
    pub(super) state: SKBuilderState,
    pub(super) eps: F,
    pub(super) minimum_samples: usize,
    pub(super) metric: SKDBSCANMetric,
    pub(super) algorithm: SKDBSCANAlgorithm,
    pub(super) leaf_size: usize,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKDBSCANBuilder<F> {
    /// A builder carrying the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            eps: F::from(0.5).expect("the default eps fits the scalar"),
            minimum_samples: 5,
            metric: SKDBSCANMetric::Euclidean,
            algorithm: SKDBSCANAlgorithm::Auto,
            leaf_size: 40,
            backend: SKBackendKind::Faer,
        }
    }

    /// Sets the inclusive neighbourhood radius (must be finite and strictly
    /// positive).
    pub fn eps(mut self, value: F) -> Self {
        self.eps = value;
        self
    }

    /// Sets the minimum number of neighbours (the point itself included) that
    /// make a row a core sample (must be strictly positive).
    pub fn minimum_samples(mut self, value: usize) -> Self {
        self.minimum_samples = value;
        self
    }

    /// Sets the distance metric (`Euclidean` by default).
    pub fn metric(mut self, value: SKDBSCANMetric) -> Self {
        self.metric = value;
        self
    }

    /// Sets the neighbourhood-search algorithm (`Auto` by default).
    pub fn algorithm(mut self, value: SKDBSCANAlgorithm) -> Self {
        self.algorithm = value;
        self
    }

    /// Sets the maximum number of rows in a spatial-index leaf (must be
    /// strictly positive).
    pub fn leaf_size(mut self, value: usize) -> Self {
        self.leaf_size = value;
        self
    }

    /// Sets the backend kind recorded in operation spans.
    pub fn backend(mut self, value: SKBackendKind) -> Self {
        self.backend = value;
        self
    }

    /// Sets the execution intent.
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    /// Validates the accumulated configuration.
    fn validated_configuration(&self) -> Result<SKDBSCANConfiguration<F>, SKError> {
        sk_validate_hyperparameter(
            "eps",
            self.eps.is_finite() && self.eps > F::zero(),
            "must be finite and strictly positive",
        )?;
        sk_validate_hyperparameter(
            "minimum_samples",
            self.minimum_samples > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter("leaf_size", self.leaf_size > 0, "must be strictly positive")?;
        if matches!(
            self.algorithm,
            SKDBSCANAlgorithm::KdTree | SKDBSCANAlgorithm::BallTree
        ) && !self.metric.is_tree_compatible()
        {
            return Err(SKError::UnsupportedRepresentation {
                representation: "cosine distance",
                suggestion: "request SKDBSCANAlgorithm::BruteForce or keep Auto for the cosine metric",
            });
        }
        Ok(SKDBSCANConfiguration {
            eps: self.eps,
            minimum_samples: self.minimum_samples,
            metric: self.metric,
            algorithm: self.algorithm,
            leaf_size: self.leaf_size,
            backend: self.backend,
            execution_mode: self.state.execution_intent(),
        })
    }
}

impl<F: SKFloat> Default for SKDBSCANBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKDBSCAN<F>> for SKDBSCANBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKDBSCAN<F>, SKError> {
        let configuration = self.validated_configuration()?;
        Ok(SKDBSCAN::from_configuration(configuration))
    }
}
