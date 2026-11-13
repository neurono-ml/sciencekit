//! Format-agnostic state export/import for [`SKLinearSVCModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKBuilderState, SKError, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_canonicalize_labels, sk_float_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_read_integer_tensor,
    sk_string_parameter, sk_validate_import_metadata,
};

use super::SKLinearSVCHead;
use super::SKLinearSVCLoss;
use super::model::SKLinearSVCModel;

/// Stable estimator identifier for exported LinearSVC states.
pub const SK_LINEAR_SVC_ESTIMATOR_NAME: &str = "SKLinearSVC";
/// The tensor carrying the per-head coefficients.
pub const SK_LINEAR_SVC_COEFFICIENTS_TENSOR: &str = "linear_svc.heads.coefficients";
/// The tensor carrying the per-head intercepts.
pub const SK_LINEAR_SVC_INTERCEPTS_TENSOR: &str = "linear_svc.heads.intercepts";
/// The tensor carrying the per-head sweep counts.
pub const SK_LINEAR_SVC_ITERATIONS_TENSOR: &str = "linear_svc.heads.iterations";

impl<F: SKFloat> SKExportable<F> for SKLinearSVCModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_LINEAR_SVC_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_features,
        )?
        .with_class_labels(self.classes.labels().map(str::to_string).collect())?
        .with_parameter("loss", SKParameterValue::from(loss_name(self.loss)))?
        .with_parameter(
            "regularization_c",
            SKParameterValue::from(self.regularization_c.to_f64().unwrap_or(1.0)),
        )?
        .with_parameter(
            "include_intercept",
            SKParameterValue::from(self.include_intercept),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let heads = self.heads.len();
        let coefficients = SKTensor::floating(
            SK_LINEAR_SVC_COEFFICIENTS_TENSOR,
            vec![heads, self.number_of_features],
            self.heads
                .iter()
                .flat_map(|head| head.coefficients.iter().copied())
                .collect(),
        )?;
        let intercepts = SKTensor::floating(
            SK_LINEAR_SVC_INTERCEPTS_TENSOR,
            vec![heads],
            self.heads.iter().map(|head| head.intercept).collect(),
        )?;
        let iterations = SKTensor::integer(
            SK_LINEAR_SVC_ITERATIONS_TENSOR,
            vec![heads],
            self.heads
                .iter()
                .map(|head| head.iterations as i64)
                .collect(),
        )?;
        SKModelState::new(metadata, vec![coefficients, intercepts, iterations])
    }
}

impl<F: SKFloat> SKImportable<F> for SKLinearSVCModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_LINEAR_SVC_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (shape, values) = sk_read_floating_tensor(state, SK_LINEAR_SVC_COEFFICIENTS_TENSOR)?;
        let &[heads, columns] = shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: shape,
            });
        };
        let coefficients = Array2::from_shape_vec((heads, columns), values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let (_, intercepts) = sk_read_floating_tensor(state, SK_LINEAR_SVC_INTERCEPTS_TENSOR)?;
        let iterations: Vec<usize> =
            sk_read_integer_tensor(state, SK_LINEAR_SVC_ITERATIONS_TENSOR)?
                .into_iter()
                .map(|value| value.max(0) as usize)
                .collect();
        if intercepts.len() != heads || iterations.len() != heads {
            return Err(SKError::ShapeMismatch {
                expected: vec![heads],
                found: vec![intercepts.len().max(iterations.len())],
            });
        }
        let metadata = state.metadata();
        let labels: Vec<String> = metadata
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        let heads = coefficients
            .rows()
            .into_iter()
            .zip(intercepts.iter())
            .zip(iterations.iter())
            .map(|((row, &intercept), &iterations)| SKLinearSVCHead {
                coefficients: Array1::from_iter(row.iter().copied()),
                intercept,
                iterations,
            })
            .collect();
        Ok(SKLinearSVCModel {
            classes,
            heads,
            loss: loss_from_name(sk_string_parameter(metadata, "loss", "squared_hinge")),
            regularization_c: F::from(sk_float_parameter(metadata, "regularization_c", 1.0))
                .ok_or(SKError::Conversion(
                    "regularization_c is not finite".to_string(),
                ))?,
            number_of_features: feature_count,
            include_intercept: sk_boolean_parameter(metadata, "include_intercept", true),
            backend: sk_parse_backend_parameter(metadata),
            builder_state: SKBuilderState::new(),
        })
    }
}

fn loss_name(loss: SKLinearSVCLoss) -> &'static str {
    match loss {
        SKLinearSVCLoss::Hinge => "hinge",
        SKLinearSVCLoss::SquaredHinge => "squared_hinge",
    }
}

fn loss_from_name(name: &str) -> SKLinearSVCLoss {
    match name {
        "hinge" => SKLinearSVCLoss::Hinge,
        _ => SKLinearSVCLoss::SquaredHinge,
    }
}
