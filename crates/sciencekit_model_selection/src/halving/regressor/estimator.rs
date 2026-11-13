//! Configured regressor successive-halving grid search
//! (spec `halving-grid-search`).

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::grid_search::{SKErrorScore, SKRegressorScoring};

use super::super::SKHalvingResource;
use super::super::builder::SKHalvingGridSearchRegressorBuilder;
use super::super::configuration::SKHalvingGridSearchConfiguration;

/// Successive-halving hyperparameter search over regressor candidates,
/// constructible only through [`SKHalvingGridSearchRegressorBuilder`].
pub struct SKHalvingGridSearchRegressor<F: SKFloat, E> {
    pub(crate) configuration: SKHalvingGridSearchConfiguration<F, E>,
    pub(crate) scoring: SKRegressorScoring,
}

impl<F: SKFloat, E> SKHalvingGridSearchRegressor<F, E> {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKHalvingGridSearchRegressorBuilder<F, E> {
        SKHalvingGridSearchRegressorBuilder::new()
    }

    /// The number of configured candidates.
    pub fn number_of_candidates(&self) -> usize {
        self.configuration.number_of_candidates()
    }

    /// The elimination factor (`3` by default).
    pub fn factor(&self) -> usize {
        self.configuration.factor
    }

    /// The initial resource budget (`1` by default).
    pub fn minimum_resource(&self) -> usize {
        self.configuration.minimum_resource
    }

    /// The explicit maximum resource cap, or `None` when bounded by all samples.
    pub fn maximum_resource(&self) -> Option<usize> {
        self.configuration.maximum_resource
    }

    /// The resource axis scaled across rounds.
    pub fn resource(&self) -> SKHalvingResource {
        self.configuration.resource
    }

    /// Whether early aggressive elimination is enabled.
    pub fn aggressive_elimination(&self) -> bool {
        self.configuration.aggressive_elimination
    }

    /// The deterministic subsampling seed (`0` by default).
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// Whether the winning candidate is refitted on all rows (`true` by default).
    pub fn refit(&self) -> bool {
        self.configuration.refit
    }

    /// The candidate failure policy.
    pub fn error_score(&self) -> SKErrorScore<F> {
        self.configuration.error_score
    }

    /// The optional group labels forwarded to group-aware splitters.
    pub fn groups(&self) -> Option<ArrayView1<'_, i64>> {
        self.configuration
            .groups
            .as_ref()
            .map(|groups| groups.view())
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }

    /// The backend kind recorded in every operation span (`Faer` by default).
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// The regression scoring policy.
    pub fn scoring(&self) -> SKRegressorScoring {
        self.scoring
    }
}
