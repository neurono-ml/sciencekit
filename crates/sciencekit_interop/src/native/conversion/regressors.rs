//! The regressor arms of the native conversion table
//! (spec `native-model-import`, requirement "Safetensors native import").

use sciencekit_common::{SKFloat, SKModelState};
use sciencekit_ensemble::random_forest::{
    SK_RANDOM_FOREST_REGRESSOR_ESTIMATOR_NAME, SKRandomForestRegressorModel,
};
use sciencekit_linear_model::linear_regression::model::{
    SK_LINEAR_REGRESSION_ESTIMATOR_NAME, SKLinearRegressionModel,
};
use sciencekit_linear_model::ridge::SKRidgeModel;
use sciencekit_linear_model::ridge::model::SK_RIDGE_ESTIMATOR_NAME;
use sciencekit_linear_model::sgd_regressor::SKSGDRegressorModel;
use sciencekit_linear_model::sgd_regressor::model::SK_SGD_REGRESSOR_ESTIMATOR_NAME;
use sciencekit_tree::SKDecisionTreeRegressorModel;
use sciencekit_tree::regressor::SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME;

use super::dispatch::{descriptor, import_model, not_retrainable};
use super::tensor_names;
use super::validation::{validate_forest_tensors, validate_tensor_names, validate_tree_tensors};
use crate::native::conversion::result::SKNativeImport;
use crate::native::descriptor::SKRetrainingSupport;
use crate::native::error::SKNativeImportError;
use crate::native::model::SKNativeRegressor;

/// Convert a regressor state, or return `None` when the name is not a regressor.
pub(crate) fn convert_regressor<F: SKFloat>(
    state: &SKModelState<F>,
    estimator_name: &str,
) -> Result<Option<SKNativeImport<F>>, SKNativeImportError> {
    let (model, retraining) = match estimator_name {
        SK_LINEAR_REGRESSION_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::LINEAR_REGRESSION)?;
            let model: SKLinearRegressionModel<F> = import_model(state)?;
            (
                SKNativeRegressor::LinearRegression(model),
                SKRetrainingSupport::Supported,
            )
        }
        SK_RIDGE_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::RIDGE)?;
            let model: SKRidgeModel<F> = import_model(state)?;
            (
                SKNativeRegressor::Ridge(model),
                not_retrainable("a ridge fit re-solves the full design"),
            )
        }
        SK_SGD_REGRESSOR_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::SGD_REGRESSOR)?;
            let model: SKSGDRegressorModel<F> = import_model(state)?;
            (
                SKNativeRegressor::StochasticGradient(model),
                SKRetrainingSupport::Supported,
            )
        }
        SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME => {
            validate_tree_tensors(state, tensor_names::DECISION_TREE_REQUIRED)?;
            let model: SKDecisionTreeRegressorModel<F> = import_model(state)?;
            (
                SKNativeRegressor::DecisionTree(model),
                not_retrainable("a fitted tree is inference-only"),
            )
        }
        SK_RANDOM_FOREST_REGRESSOR_ESTIMATOR_NAME => {
            validate_forest_tensors(state)?;
            let model: SKRandomForestRegressorModel<F> = import_model(state)?;
            (
                SKNativeRegressor::RandomForest(model),
                not_retrainable("a fitted forest is inference-only"),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(SKNativeImport::Regressor {
        model,
        descriptor: descriptor(state, retraining),
    }))
}
