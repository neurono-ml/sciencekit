//! The criterion dispatch: one scan entry point over the per-criterion sweeps.

use sciencekit_common::SKFloat;

use super::absolute::sk_scan_absolute_error_feature;
use super::classification::sk_scan_classification_feature;
use super::squared::sk_scan_squared_error_feature;
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;
use crate::splitter::SKSplitCandidate;

/// Whether the scan keeps the best threshold or draws one at random.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SKThresholdChoice {
    /// Evaluate every valid boundary and keep the best.
    Best,
    /// Draw one valid boundary from the seeded stream.
    Random(u64),
}

/// The criterion-specific inputs a scan needs.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SKScanContext<'a, F> {
    /// Classification statistics: canonical class index and weight per sample.
    Classification {
        /// The canonical class index of every training row.
        class_indices: &'a [usize],
        /// The sample weight of every training row.
        sample_weights: &'a [F],
        /// The number of fitted classes.
        number_of_classes: usize,
        /// Gini or entropy.
        criterion: SKTreeCriterion,
    },
    /// Regression statistics: the continuous response of every training row.
    Regression {
        /// The response of every training row.
        responses: &'a [F],
        /// Squared error or absolute error.
        criterion: SKTreeCriterion,
    },
}

/// Scan one feature and return its best (or single random) candidate.
pub(crate) fn sk_scan_feature<F: SKFloat>(
    design: &SKDesign<'_, F>,
    sample_indices: &[usize],
    feature: usize,
    minimum_samples_leaf: usize,
    context: SKScanContext<'_, F>,
    choice: SKThresholdChoice,
) -> Option<SKSplitCandidate<F>> {
    match context {
        SKScanContext::Classification {
            class_indices,
            sample_weights,
            number_of_classes,
            criterion,
        } => sk_scan_classification_feature(
            design,
            sample_indices,
            feature,
            minimum_samples_leaf,
            class_indices,
            sample_weights,
            number_of_classes,
            criterion,
            choice,
        ),
        SKScanContext::Regression {
            responses,
            criterion: SKTreeCriterion::SquaredError,
        } => sk_scan_squared_error_feature(
            design,
            sample_indices,
            feature,
            minimum_samples_leaf,
            responses,
            choice,
        ),
        SKScanContext::Regression {
            responses,
            criterion: SKTreeCriterion::AbsoluteError,
        } => sk_scan_absolute_error_feature(
            design,
            sample_indices,
            feature,
            minimum_samples_leaf,
            responses,
            choice,
        ),
        SKScanContext::Regression { .. } => None,
    }
}
