//! The fitted imputer model: owned per-column state (spec `simple-imputer`).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::missing_indicator::SKMissingIndicatorModel;
use crate::simple_imputer::state::SKImputationState;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// The learned state of an [`crate::simple_imputer::SKSimpleImputer`]: the
/// per-column statistic, the columns dropped as all-missing and the carried
/// backend kind. The sole bearer of the fitted state, `Send + Sync` by
/// construction.
pub struct SKSimpleImputerModel<F> {
    pub(super) statistics: Array1<F>,
    pub(super) dropped_columns: Vec<usize>,
    pub(super) number_of_input_columns: usize,
    pub(super) missing_values: F,
    pub(super) strategy: SKImputationStrategy,
    pub(super) fill_value: Option<F>,
    pub(super) keep_empty_features: bool,
    pub(super) state: SKImputationState<F>,
    pub(super) indicator: Option<SKMissingIndicatorModel<F>>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKSimpleImputerModel<F> {
    /// Build the model from its fitted state; the sole construction path used
    /// by `fit` and `partial_fit`.
    pub(in crate::simple_imputer) fn from_state(
        statistics: Array1<F>,
        dropped_columns: Vec<usize>,
        number_of_input_columns: usize,
        state: SKImputationState<F>,
        indicator: Option<SKMissingIndicatorModel<F>>,
        estimator: &super::super::core_implementation::SKSimpleImputer<F>,
    ) -> Self {
        Self {
            statistics,
            dropped_columns,
            number_of_input_columns,
            missing_values: estimator.missing_values,
            strategy: estimator.strategy,
            fill_value: estimator.fill_value,
            keep_empty_features: estimator.keep_empty_features,
            state,
            indicator,
            backend: estimator.backend,
            execution_intent: estimator.execution_intent(),
        }
    }

    /// The per-column fill statistic, one entry per retained column.
    pub fn statistics(&self) -> ArrayView1<'_, F> {
        self.statistics.view()
    }

    /// The indices of the input columns dropped as all-missing, ascending.
    pub fn dropped_columns(&self) -> &[usize] {
        &self.dropped_columns
    }

    /// The number of columns the estimator was fitted on.
    pub fn number_of_input_columns(&self) -> usize {
        self.number_of_input_columns
    }

    /// The number of columns the transform will emit: the fitted columns
    /// minus the ones dropped as all-missing.
    pub fn number_of_output_features(&self) -> usize {
        self.number_of_input_columns - self.dropped_columns.len()
    }

    /// The fitted missingness indicator stacked by the transform, when
    /// `add_indicator` was requested.
    pub fn indicator(&self) -> Option<&SKMissingIndicatorModel<F>> {
        self.indicator.as_ref()
    }

    /// The missing-value marker the model recognizes.
    pub fn missing_values(&self) -> F {
        self.missing_values
    }

    /// The strategy the statistics were computed with.
    pub fn strategy(&self) -> SKImputationStrategy {
        self.strategy
    }

    /// The backend kind this model records in its operation spans.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent carried from the estimator at fit time.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
