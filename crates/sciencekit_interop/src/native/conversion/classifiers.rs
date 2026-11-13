//! The classifier arms of the native conversion table
//! (spec `native-model-import`, requirement "Safetensors native import").

use sciencekit_common::{SKFloat, SKModelState};
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
            let model: SKLogisticRegressionModel<F> = import_model(state)?;
            (
                SKNativeClassifier::LogisticRegression(model),
                SKRetrainingSupport::Supported,
            )
        }
        SK_SGD_CLASSIFIER_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::SGD_CLASSIFIER)?;
            let model: SKSGDClassifierModel<F> = import_model(state)?;
            (
                SKNativeClassifier::StochasticGradient(model),
                SKRetrainingSupport::Supported,
            )
        }
        SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME => {
            validate_tree_tensors(state, tensor_names::DECISION_TREE_REQUIRED)?;
            let model: SKDecisionTreeClassifierModel<F> = import_model(state)?;
            (
                SKNativeClassifier::DecisionTree(model),
                not_retrainable("a fitted tree is inference-only"),
            )
        }
        SK_RANDOM_FOREST_CLASSIFIER_ESTIMATOR_NAME => {
            validate_forest_tensors(state)?;
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
