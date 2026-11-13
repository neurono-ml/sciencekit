//! Format-agnostic state export/import for [`SKSGDRegressorModel`].

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::SKSGDRegressorModel;
use crate::sgd::export_helpers::{read_sgd_config, with_sgd_config};
use crate::sgd::{SKLearningState, SKSGDRegressorLoss};

/// Stable estimator identifier for exported SGD-regressor states.
pub const SK_SGD_REGRESSOR_ESTIMATOR_NAME: &str = "SKSGDRegressor";
/// The tensor carrying the learned coefficients.
pub const SK_SGD_REGRESSOR_COEFFICIENTS_TENSOR: &str = "sgd_regressor.coefficients";
/// The tensor carrying the learned intercept.
pub const SK_SGD_REGRESSOR_INTERCEPT_TENSOR: &str = "sgd_regressor.intercept";

impl<F: SKFloat> SKExportable<F> for SKSGDRegressorModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = with_sgd_config(
            SKModelMetadata::new(
                SK_SGD_REGRESSOR_ESTIMATOR_NAME,
                SK_MODEL_FORMAT_VERSION,
                sk_model_tensor_data_type::<F>(),
                self.coefficients.len(),
            )?
            .with_parameter("loss", SKParameterValue::from(loss_name(self.loss)))?
            .with_parameter(
                "epsilon",
                SKParameterValue::from(self.epsilon.to_f64().unwrap_or(0.1)),
            )?
            .with_parameter("converged", SKParameterValue::from(self.converged))?
            .with_parameter(
                "epochs_taken",
                SKParameterValue::from(self.epochs_taken as i64),
            )?
            .with_parameter("backend", sk_backend_parameter(self.backend))?,
            &self.config,
        )?;
        let coefficients = SKTensor::floating(
            SK_SGD_REGRESSOR_COEFFICIENTS_TENSOR,
            vec![self.coefficients.len()],
            self.coefficients.to_vec(),
        )?;
        let intercept = SKTensor::floating(
            SK_SGD_REGRESSOR_INTERCEPT_TENSOR,
            Vec::new(),
            vec![self.intercept],
        )?;
        SKModelState::new(metadata, vec![coefficients, intercept])
    }
}

impl<F: SKFloat> SKImportable<F> for SKSGDRegressorModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_SGD_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (_, coefficients) =
            sk_read_floating_tensor(state, SK_SGD_REGRESSOR_COEFFICIENTS_TENSOR)?;
        if coefficients.len() != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![feature_count],
                found: vec![coefficients.len()],
            });
        }
        let (_, intercept_values) =
            sk_read_floating_tensor(state, SK_SGD_REGRESSOR_INTERCEPT_TENSOR)?;
        let &[intercept] = intercept_values.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1],
                found: vec![intercept_values.len()],
            });
        };
        let metadata = state.metadata();
        let config = read_sgd_config::<F>(metadata)?;
        let learning_state = SKLearningState::from_exported(
            Array1::from_vec(coefficients.clone()),
            intercept,
            &config,
        );
        Ok(SKSGDRegressorModel {
            coefficients: Array1::from_vec(coefficients),
            intercept,
            state: learning_state,
            config,
            loss: loss_from_name(sk_string_parameter(metadata, "loss", "squared_error")),
            epsilon: F::from(sk_float_parameter(metadata, "epsilon", 0.1))
                .ok_or(SKError::Conversion("epsilon is not finite".to_string()))?,
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
            converged: sk_boolean_parameter(metadata, "converged", true),
            epochs_taken: sk_integer_parameter(metadata, "epochs_taken", 0).max(0) as usize,
        })
    }
}

fn loss_name(loss: SKSGDRegressorLoss) -> &'static str {
    match loss {
        SKSGDRegressorLoss::SquaredError => "squared_error",
        SKSGDRegressorLoss::Huber => "huber",
        SKSGDRegressorLoss::EpsilonInsensitive => "epsilon_insensitive",
        SKSGDRegressorLoss::SquaredEpsilonInsensitive => "squared_epsilon_insensitive",
    }
}

fn loss_from_name(name: &str) -> SKSGDRegressorLoss {
    match name {
        "huber" => SKSGDRegressorLoss::Huber,
        "epsilon_insensitive" => SKSGDRegressorLoss::EpsilonInsensitive,
        "squared_epsilon_insensitive" => SKSGDRegressorLoss::SquaredEpsilonInsensitive,
        _ => SKSGDRegressorLoss::SquaredError,
    }
}
