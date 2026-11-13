//! Format-agnostic state export/import for [`SKSVCModel`].
//!
//! Every one-versus-one pair is a nested binary state: its support vectors keep
//! their dense/CSR representation, and its dual coefficients, bias, iteration
//! count and optional Platt calibration travel beside them.

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKBuilderState, SKError, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_canonicalize_labels, sk_float_parameter, sk_integer_parameter,
    sk_model_tensor_data_type, sk_parse_backend_parameter, sk_read_floating_tensor,
    sk_read_integer_tensor, sk_validate_import_metadata,
};

use super::{SKBinarySVCModel, SKSVCModel};
use crate::export_support::{
    export_support_vectors, import_support_vectors, read_kernel_parameters, with_kernel_parameters,
};
use crate::probability::SKPlattParameters;

/// Stable estimator identifier for exported SVC states.
pub const SK_SVC_ESTIMATOR_NAME: &str = "SKSVC";

impl<F: SKFloat> SKExportable<F> for SKSVCModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = with_kernel_parameters(
            SKModelMetadata::new(
                SK_SVC_ESTIMATOR_NAME,
                SK_MODEL_FORMAT_VERSION,
                sk_model_tensor_data_type::<F>(),
                self.number_of_features,
            )?
            .with_class_labels(self.classes.labels().map(str::to_string).collect())?
            .with_parameter(
                "probability_enabled",
                SKParameterValue::from(self.probability_enabled),
            )?
            .with_parameter(
                "number_of_pairs",
                SKParameterValue::from(self.pairwise_models.len() as i64),
            )?
            .with_parameter("backend", sk_backend_parameter(self.backend))?,
            &self.kernel_parameters,
        )?;
        let mut tensors = Vec::new();
        let mut representations = Vec::new();
        for (index, pair) in self.pairwise_models.iter().enumerate() {
            let prefix = format!("svc.pair.{index}");
            let (pair_tensors, representation) =
                export_support_vectors(&pair.support_vectors, &prefix)?;
            representations.push(SKParameterValue::from(representation));
            tensors.extend(pair_tensors);
            tensors.push(SKTensor::floating(
                format!("{prefix}.dual_coefficients"),
                vec![pair.dual_coefficients.len()],
                pair.dual_coefficients.to_vec(),
            )?);
            tensors.push(SKTensor::floating(
                format!("{prefix}.bias"),
                Vec::new(),
                vec![pair.bias],
            )?);
            tensors.push(SKTensor::integer(
                format!("{prefix}.support_training_indices"),
                vec![pair.support_training_indices.len()],
                pair.support_training_indices
                    .iter()
                    .map(|&row| row as i64)
                    .collect(),
            )?);
            metadata = metadata
                .with_parameter(
                    format!("{prefix}.class_negative"),
                    SKParameterValue::from(pair.class_negative as i64),
                )?
                .with_parameter(
                    format!("{prefix}.class_positive"),
                    SKParameterValue::from(pair.class_positive as i64),
                )?
                .with_parameter(
                    format!("{prefix}.iterations"),
                    SKParameterValue::from(pair.iterations as i64),
                )?
                .with_parameter(
                    format!("{prefix}.has_platt"),
                    SKParameterValue::from(pair.platt_parameters.is_some()),
                )?;
            if let Some(platt) = pair.platt_parameters {
                tensors.push(SKTensor::floating(
                    format!("{prefix}.platt_slope"),
                    Vec::new(),
                    vec![platt.slope],
                )?);
                tensors.push(SKTensor::floating(
                    format!("{prefix}.platt_intercept"),
                    Vec::new(),
                    vec![platt.intercept],
                )?);
            }
        }
        metadata = metadata.with_parameter(
            "pair_representations",
            SKParameterValue::List(representations),
        )?;
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKSVCModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_SVC_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let number_of_pairs = sk_integer_parameter(metadata, "number_of_pairs", 0).max(0) as usize;
        let representations = metadata
            .parameter("pair_representations")
            .and_then(SKParameterValue::as_list)
            .map(<[SKParameterValue]>::to_vec)
            .ok_or(SKError::InvalidHyperparameter {
                name: "pair_representations",
                reason: "the exported state does not carry the pair representations".to_string(),
            })?;
        if representations.len() != number_of_pairs {
            return Err(SKError::ShapeMismatch {
                expected: vec![number_of_pairs],
                found: vec![representations.len()],
            });
        }
        let mut pairwise_models = Vec::with_capacity(number_of_pairs);
        for (index, representation) in representations.iter().enumerate() {
            let prefix = format!("svc.pair.{index}");
            let representation =
                representation
                    .as_string()
                    .ok_or(SKError::InvalidHyperparameter {
                        name: "pair_representations",
                        reason: "a representation is not a string".to_string(),
                    })?;
            let support_vectors = import_support_vectors(state, &prefix, representation)?;
            let (_, dual_coefficients) =
                sk_read_floating_tensor(state, &format!("{prefix}.dual_coefficients"))?;
            let (_, bias_values) = sk_read_floating_tensor(state, &format!("{prefix}.bias"))?;
            let &[bias] = bias_values.as_slice() else {
                return Err(SKError::ShapeMismatch {
                    expected: vec![1],
                    found: vec![bias_values.len()],
                });
            };
            let support_training_indices: Vec<usize> =
                sk_read_integer_tensor(state, &format!("{prefix}.support_training_indices"))?
                    .into_iter()
                    .map(|row| row.max(0) as usize)
                    .collect();
            let platt_parameters =
                if sk_boolean_parameter(metadata, &format!("{prefix}.has_platt"), false) {
                    let (_, slope) =
                        sk_read_floating_tensor(state, &format!("{prefix}.platt_slope"))?;
                    let (_, intercept) =
                        sk_read_floating_tensor(state, &format!("{prefix}.platt_intercept"))?;
                    let (&slope_value, &intercept_value) = (
                        slope.first().unwrap_or(&F::zero()),
                        intercept.first().unwrap_or(&F::zero()),
                    );
                    Some(SKPlattParameters::new(slope_value, intercept_value))
                } else {
                    None
                };
            pairwise_models.push(SKBinarySVCModel {
                class_negative: sk_integer_parameter(
                    metadata,
                    &format!("{prefix}.class_negative"),
                    0,
                )
                .max(0) as usize,
                class_positive: sk_integer_parameter(
                    metadata,
                    &format!("{prefix}.class_positive"),
                    1,
                )
                .max(0) as usize,
                support_vectors,
                support_training_indices,
                dual_coefficients: Array1::from_vec(dual_coefficients),
                bias,
                iterations: sk_integer_parameter(metadata, &format!("{prefix}.iterations"), 0)
                    .max(0) as usize,
                platt_parameters,
            });
        }
        let labels: Vec<String> = metadata
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        Ok(SKSVCModel {
            classes,
            pairwise_models,
            kernel_parameters: read_kernel_parameters(metadata)?,
            number_of_features: feature_count,
            backend: sk_parse_backend_parameter(metadata),
            builder_state: SKBuilderState::new(),
            probability_enabled: sk_boolean_parameter(metadata, "probability_enabled", false),
        })
    }
}

/// Read a floating metadata parameter, defaulting when absent.
#[allow(dead_code)]
fn platt_from_metadata<F: SKFloat>(
    metadata: &SKModelMetadata,
    prefix: &str,
) -> Option<SKPlattParameters<F>> {
    let slope = sk_float_parameter(metadata, &format!("{prefix}.platt_slope"), 0.0);
    let intercept = sk_float_parameter(metadata, &format!("{prefix}.platt_intercept"), 0.0);
    Some(SKPlattParameters::new(F::from(slope)?, F::from(intercept)?))
}
