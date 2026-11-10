//! Per-feature split scans for every supported criterion.
//!
//! This is a **pure dispatcher** module: `mod.rs` declares the submodules and
//! re-exports the scan entry point; the scans live in their own files.

mod absolute;
mod classification;
mod ranked;
mod squared;

use sciencekit_common::SKFloat;

use super::SKSplitCandidate;
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;

pub(crate) use classification::{sk_classification_impurity, sk_scan_classification_feature};
pub(crate) use squared::sk_scan_squared_error_feature;

use absolute::sk_scan_absolute_error_feature;

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

mod selection;

pub(crate) use selection::{
    sk_better_candidate, sk_choose_boundary, sk_midpoint, sk_summary_for_indices,
};
