//! The fitted nearest-neighbor imputer model
//! (spec `nearest-neighbor-imputation`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::knn::distance_block::sk_distance_block_rows;
use crate::knn::weights::SKImputationWeights;
use crate::missing_indicator::SKMissingIndicatorModel;

/// The learned state of a [`crate::knn::SKKNNImputer`]: the training design
/// with missing cells normalized to `NaN`, its missingness mask, the donor
/// rows of every column, the retained columns and the observed column means.
///
/// The model owns immutable state only and is `Send + Sync` by construction.
pub struct SKKNNImputerModel<F> {
    pub(super) training: Array2<F>,
    pub(super) missing_mask: Array2<bool>,
    pub(super) column_donors: Vec<Vec<usize>>,
    pub(super) kept_columns: Vec<usize>,
    pub(super) dropped_columns: Vec<usize>,
    pub(super) column_means: Array1<F>,
    pub(super) number_of_input_columns: usize,
    pub(super) missing_values: F,
    pub(super) nearest_neighbors_count: usize,
    pub(super) weights: SKImputationWeights,
    pub(super) indicator: Option<SKMissingIndicatorModel<F>>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKKNNImputerModel<F> {
    /// The retained training design, with every missing cell normalized to
    /// `NaN` so the NaN-aware kernel ignores it.
    pub fn training_features(&self) -> ArrayView2<'_, F> {
        self.training.view()
    }

    /// The training missingness mask: `true` exactly at the missing cells.
    pub fn missing_mask(&self) -> ArrayView2<'_, bool> {
        self.missing_mask.view()
    }

    /// The training rows that observed `column`, ascending.
    pub fn donor_indices(&self, column: usize) -> &[usize] {
        &self.column_donors[column]
    }

    /// The input columns the transform emits, in ascending order.
    pub fn kept_columns(&self) -> &[usize] {
        &self.kept_columns
    }

    /// The input columns dropped as all-missing, ascending.
    pub fn dropped_columns(&self) -> &[usize] {
        &self.dropped_columns
    }

    /// The observed mean of every input column; `NaN` for an all-missing one.
    pub fn column_means(&self) -> ArrayView1<'_, F> {
        self.column_means.view()
    }

    /// The number of training rows the fit retained.
    pub fn training_rows(&self) -> usize {
        self.training.nrows()
    }

    /// The number of input columns the estimator was fitted on.
    pub fn number_of_input_columns(&self) -> usize {
        self.number_of_input_columns
    }

    /// The number of feature columns the transform emits before any indicator.
    pub fn number_of_output_features(&self) -> usize {
        self.kept_columns.len()
    }

    /// The number of receiver rows processed per distance block.
    ///
    /// The width comes from a fixed byte budget, so the distance matrix never
    /// grows with the receiver count.
    pub fn distance_block_rows(&self) -> usize {
        sk_distance_block_rows(self.training_rows(), size_of::<F>())
    }

    /// The missing-value marker the model recognizes.
    pub fn missing_values(&self) -> F {
        self.missing_values
    }

    /// The number of donors consulted per missing cell.
    pub fn nearest_neighbors_count(&self) -> usize {
        self.nearest_neighbors_count
    }

    /// The donor weighting scheme.
    pub fn weights(&self) -> SKImputationWeights {
        self.weights
    }

    /// The fitted missingness indicator, when `add_indicator` was requested.
    pub fn indicator(&self) -> Option<&SKMissingIndicatorModel<F>> {
        self.indicator.as_ref()
    }

    /// The backend kind recorded in this model's operation spans.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent carried from the estimator at fit time.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
