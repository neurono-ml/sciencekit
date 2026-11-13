//! The classifier arms of the native conversion table
//! (spec `native-model-import`, requirement "Safetensors native import").

use sciencekit_common::{SKFloat, SKModelState, SKParameterValue, sk_string_parameter};
use sciencekit_ensemble::random_forest::{
    SK_RANDOM_FOREST_CLASSIFIER_ESTIMATOR_NAME, SKRandomForestClassifierModel,
};
use sciencekit_linear_model::logistic_regression::SKLogisticRegressionModel;
use sciencekit_linear_model::logistic_regression::model::SK_LOGISTIC_REGRESSION_ESTIMATOR_NAME;
use sciencekit_linear_model::sgd_classifier::SKSGDClassifierModel;
use sciencekit_linear_model::sgd_classifier::model::SK_SGD_CLASSIFIER_ESTIMATOR_NAME;
use sciencekit_tree::SKDecisionTreeClassifierModel;
use sciencekit_tree::classifier::SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME;

use super::dispatch::{descriptor, import_model, not_retrainable};
use super::tensor_names;
use super::validation::{validate_forest_tensors, validate_tensor_names, validate_tree_tensors};
use crate::native::conversion::result::SKNativeImport;
use crate::native::descriptor::SKRetrainingSupport;
use crate::native::error::SKNativeImportError;
use crate::native::model::SKNativeClassifier;

/// Convert a classifier state, or return `None` when the name is not one.
pub(crate) fn convert_classifier<F: SKFloat>(
    state: &SKModelState<F>,
    estimator_name: &str,
) -> Result<Option<SKNativeImport<F>>, SKNativeImportError> {
    let (model, retraining) = match estimator_name {
        SK_LOGISTIC_REGRESSION_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::LOGISTIC_REGRESSION)?;
            validate_class_table(state)?;
            let model: SKLogisticRegressionModel<F> = import_model(state)?;
            (
                SKNativeClassifier::LogisticRegression(model),
                SKRetrainingSupport::Supported,
            )
        }
        SK_SGD_CLASSIFIER_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::SGD_CLASSIFIER)?;
            validate_class_table(state)?;
            let model: SKSGDClassifierModel<F> = import_model(state)?;
            (
                SKNativeClassifier::StochasticGradient(model),
                SKRetrainingSupport::Supported,
            )
        }
        SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME => {
            validate_tree_tensors(state, tensor_names::DECISION_TREE_REQUIRED)?;
            validate_class_table(state)?;
            let model: SKDecisionTreeClassifierModel<F> = import_model(state)?;
            (
                SKNativeClassifier::DecisionTree(model),
                not_retrainable("a fitted tree is inference-only"),
            )
        }
        SK_RANDOM_FOREST_CLASSIFIER_ESTIMATOR_NAME => {
            validate_forest_tensors(state)?;
            validate_class_table(state)?;
            let model: SKRandomForestClassifierModel<F> = import_model(state)?;
            (
                SKNativeClassifier::RandomForest(model),
                not_retrainable("a fitted forest is inference-only"),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(SKNativeImport::Classifier {
        model,
        descriptor: descriptor(state, retraining),
    }))
}

/// Reject a class table whose length disagrees with the declared class count.
///
/// A classifier whose labels and score heads disagree cannot map an index to a
/// label; accepting it would produce a model that predicts the wrong class
/// silently, so the conversion stops here.
fn validate_class_table<F: SKFloat>(state: &SKModelState<F>) -> Result<(), SKNativeImportError> {
    let Some(labels) = state.metadata().class_labels() else {
        return Ok(());
    };
    let declared = state
        .metadata()
        .parameter("number_of_classes")
        .and_then(SKParameterValue::as_integer);
    let heads = state
        .tensor("logistic.coefficients")
        .or_else(|| state.tensor("sgd_classifier.coefficients"))
        .and_then(|tensor| tensor.shape().first().copied());
    let strategy = sk_string_parameter(state.metadata(), "strategy", "binary");
    let expected = declared
        .map(|declared| declared.max(0) as usize)
        .or_else(|| {
            (strategy != "binary")
                .then_some(heads)
                .flatten()
                .or(Some(2))
        })
        .unwrap_or(2);
    if expected != labels.len() {
        return Err(SKNativeImportError::ShapeMismatch {
            tensor: "class_labels".to_string(),
            expected: vec![expected],
            found: vec![labels.len()],
        });
    }
    Ok(())
}
