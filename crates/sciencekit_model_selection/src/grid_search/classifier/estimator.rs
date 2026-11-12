//! The configured classifier grid search (spec `grid-search`).

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::grid_search::builder::SKGridSearchCVClassifierBuilder;
use crate::grid_search::configuration::SKGridSearchConfiguration;
use crate::grid_search::error_score::SKErrorScore;
use crate::grid_search::scoring::SKClassifierScoring;

/// Hyperparameter grid search over classifier candidates, constructed only
/// through its builder.
///
/// Fitting evaluates every configured candidate on every fold of the splitter,
/// selects the highest mean score (ties by candidate index) and, when refit is
/// enabled, trains the winner on all rows.
pub struct SKGridSearchCVClassifier<F: SKFloat, E> {
    pub(crate) configuration: SKGridSearchConfiguration<F, E>,
    pub(crate) scoring: SKClassifierScoring,
}

impl<F: SKFloat, E> SKGridSearchCVClassifier<F, E> {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKGridSearchCVClassifierBuilder<F, E> {
        SKGridSearchCVClassifierBuilder::new()
    }

    /// The number of configured candidates.
    pub fn number_of_candidates(&self) -> usize {
        self.configuration.number_of_candidates()
    }

    /// Whether the winner is refitted on all rows.
    pub fn refit(&self) -> bool {
        self.configuration.refit
    }

    /// The configured failure policy.
    pub fn error_score(&self) -> SKErrorScore<F> {
        self.configuration.error_score
    }

    /// The configured groups, when present.
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

    /// The backend kind recorded by every operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// The configured scoring policy.
    pub fn scoring(&self) -> SKClassifierScoring {
        self.scoring
    }
}
