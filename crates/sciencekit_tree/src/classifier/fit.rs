//! Classifier fitting (the implementation lands with task 4.2).

use sciencekit_common::{SKDataView, SKError, SKFloat, SKTargetView};

use super::core_implementation::SKDecisionTreeClassifier;
use super::model::SKDecisionTreeClassifierModel;

/// Fit the classifier (placeholder until the growth-backed implementation).
pub(super) fn sk_fit_classifier<F: SKFloat>(
    _estimator: &SKDecisionTreeClassifier<F>,
    _features: SKDataView<'_, F>,
    _targets: SKTargetView<'_, F>,
) -> Result<SKDecisionTreeClassifierModel<F>, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "fit",
        reason: "the classifier growth is not implemented yet".to_string(),
    })
}
