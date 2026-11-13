//! Format-agnostic state export/import for [`SKSGDClassifierModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_canonicalize_labels, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::SKSGDClassifierModel;
use crate::class_weight::SKClassWeight;
use crate::sgd::export_helpers::{read_sgd_config, with_sgd_config};
use crate::sgd::{SKLearningState, SKSGDClassifierLoss};

/// Stable estimator identifier for exported SGD-classifier states.
pub const SK_SGD_CLASSIFIER_ESTIMATOR_NAME: &str = "SKSGDClassifier";
/// The tensor carrying the per-head coefficients.
pub const SK_SGD_CLASSIFIER_COEFFICIENTS_TENSOR: &str = "sgd_classifier.coefficients";
/// The tensor carrying the per-head intercepts.
pub const SK_SGD_CLASSIFIER_INTERCEPTS_TENSOR: &str = "sgd_classifier.intercepts";

impl<F: SKFloat> SKExportable<F> for SKSGDClassifierModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let feature_count = self.coefficients.ncols();
        let mut metadata = with_sgd_config(
            SKModelMetadata::new(
                SK_SGD_CLASSIFIER_ESTIMATOR_NAME,
                SK_MODEL_FORMAT_VERSION,
                sk_model_tensor_data_type::<F>(),
                feature_count,
            )?
            .with_class_labels(self.classes.labels().map(str::to_string).collect())?
            .with_parameter("loss", SKParameterValue::from(loss_name(self.loss)))?
            .with_parameter("converged", SKParameterValue::from(self.converged))?
            .with_parameter(
                "epochs_taken",
                SKParameterValue::from(self.epochs_taken as i64),
            )?
            .with_parameter("backend", sk_backend_parameter(self.backend))?,
            &self.config,
        )?;
        metadata = match &self.class_weight {
            SKClassWeight::None => {
                metadata.with_parameter("class_weight_policy", SKParameterValue::from("none"))?
            }
            SKClassWeight::Balanced => metadata
                .with_parameter("class_weight_policy", SKParameterValue::from("balanced"))?,
            SKClassWeight::Explicit(weights) => metadata
                .with_parameter("class_weight_policy", SKParameterValue::from("explicit"))?
                .with_parameter(
                    "class_weight",
                    SKParameterValue::List(
                        weights
                            .iter()
                            .map(|weight| SKParameterValue::from(weight.to_f64().unwrap_or(1.0)))
                            .collect(),
                    ),
                )?,
        };
        let coefficients = SKTensor::floating(
            SK_SGD_CLASSIFIER_COEFFICIENTS_TENSOR,
            vec![self.coefficients.nrows(), feature_count],
            self.coefficients.iter().copied().collect(),
        )?;
        let intercepts = SKTensor::floating(
            SK_SGD_CLASSIFIER_INTERCEPTS_TENSOR,
            vec![self.intercepts.len()],
            self.intercepts.to_vec(),
        )?;
        SKModelState::new(metadata, vec![coefficients, intercepts])
    }
}

impl<F: SKFloat> SKImportable<F> for SKSGDClassifierModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_SGD_CLASSIFIER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (coefficient_shape, coefficient_values) =
            sk_read_floating_tensor(state, SK_SGD_CLASSIFIER_COEFFICIENTS_TENSOR)?;
        let &[heads, columns] = coefficient_shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: coefficient_shape,
            });
        };
        if columns != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![heads, feature_count],
                found: vec![heads, columns],
            });
        }
        let coefficients = Array2::from_shape_vec((heads, columns), coefficient_values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let (_, intercepts) = sk_read_floating_tensor(state, SK_SGD_CLASSIFIER_INTERCEPTS_TENSOR)?;
        if intercepts.len() != heads {
            return Err(SKError::ShapeMismatch {
                expected: vec![heads],
                found: vec![intercepts.len()],
            });
        }
        let metadata = state.metadata();
        let config = read_sgd_config::<F>(metadata)?;
        let labels: Vec<String> = metadata
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        let states = coefficients
            .rows()
            .into_iter()
            .enumerate()
            .map(|(head, row)| {
                SKLearningState::from_exported(
                    row.to_owned(),
                    intercepts.get(head).copied().unwrap_or_else(F::zero),
                    &config,
                )
            })
            .collect();
        Ok(SKSGDClassifierModel {
            coefficients,
            intercepts: Array1::from_vec(intercepts),
            classes,
            states,
            config,
            loss: loss_from_name(sk_string_parameter(metadata, "loss", "hinge")),
            class_weight: class_weight_from_state(state)?,
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
            converged: sk_boolean_parameter(metadata, "converged", true),
            epochs_taken: sk_integer_parameter(metadata, "epochs_taken", 0).max(0) as usize,
        })
    }
}

fn class_weight_from_state<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<SKClassWeight<F>, SKError> {
    match sk_string_parameter(state.metadata(), "class_weight_policy", "none") {
        "balanced" => Ok(SKClassWeight::Balanced),
        "explicit" => {
            let values = state
                .metadata()
                .parameter("class_weight")
                .and_then(SKParameterValue::as_list)
                .ok_or(SKError::InvalidHyperparameter {
                    name: "class_weight",
                    reason: "an explicit policy must carry its weights".to_string(),
                })?;
            Ok(SKClassWeight::Explicit(
                values
                    .iter()
                    .filter_map(SKParameterValue::as_float)
                    .map(|weight| F::from(weight).unwrap_or_else(F::one))
                    .collect(),
            ))
        }
        _ => Ok(SKClassWeight::None),
    }
}

fn loss_name(loss: SKSGDClassifierLoss) -> &'static str {
    match loss {
        SKSGDClassifierLoss::Hinge => "hinge",
        SKSGDClassifierLoss::LogLoss => "log_loss",
        SKSGDClassifierLoss::ModifiedHuber => "modified_huber",
        SKSGDClassifierLoss::SquaredHinge => "squared_hinge",
        SKSGDClassifierLoss::Perceptron => "perceptron",
    }
}

fn loss_from_name(name: &str) -> SKSGDClassifierLoss {
    match name {
        "log_loss" => SKSGDClassifierLoss::LogLoss,
        "modified_huber" => SKSGDClassifierLoss::ModifiedHuber,
        "squared_hinge" => SKSGDClassifierLoss::SquaredHinge,
        "perceptron" => SKSGDClassifierLoss::Perceptron,
        _ => SKSGDClassifierLoss::Hinge,
    }
}
