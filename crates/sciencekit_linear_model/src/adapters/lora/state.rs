//! Export and import of an adapter through the format-agnostic
//! [`SKModelState`](sciencekit_common::SKModelState) contract (spec
//! `lora-adapters`, decision 4).

use ndarray::Array2;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::artifact::SKLoraAdapterArtifact;
use super::attached_model::SKLoraAttachedModel;
use super::builder::SKLoraAdapterBuilder;
use super::config::SKLoraTargetModule;

/// The stable estimator name every exported adapter state carries.
pub const SK_LORA_ESTIMATOR_NAME: &str = "SKLoraAdapter";
/// The format version of the exported adapter state.
pub const SK_LORA_FORMAT_VERSION: u32 = SK_MODEL_FORMAT_VERSION;

/// The stable tensor name of the `A` factor for a target module.
pub fn sk_lora_a_tensor_name(target: SKLoraTargetModule) -> String {
    format!("lora.{}.a", target.tensor_segment())
}

/// The stable tensor name of the `B` factor for a target module.
pub fn sk_lora_b_tensor_name(target: SKLoraTargetModule) -> String {
    format!("lora.{}.b", target.tensor_segment())
}

impl<F: SKFloat> SKExportable<F> for SKLoraAttachedModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let target = self.adapter().target();
        let metadata = SKModelMetadata::new(
            SK_LORA_ESTIMATOR_NAME,
            SK_LORA_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.feature_count(),
        )?
        .with_parameter(
            "identifier",
            SKParameterValue::from(self.adapter().identifier()),
        )?
        .with_parameter("rank", SKParameterValue::from(self.adapter().rank() as i64))?
        .with_parameter(
            "alpha",
            SKParameterValue::from(self.adapter().alpha().to_f64().unwrap_or(f64::NAN)),
        )?
        .with_parameter("target", SKParameterValue::from(target.parameter_name()))?
        .with_parameter(
            "heads",
            SKParameterValue::from(self.number_of_heads() as i64),
        )?
        .with_parameter("fitted", SKParameterValue::from(self.is_fitted()))?
        .with_parameter(
            "learning_rate",
            SKParameterValue::from(self.adapter().learning_rate().to_f64().unwrap_or(f64::NAN)),
        )?
        .with_parameter(
            "maximum_number_of_iterations",
            SKParameterValue::from(self.adapter().maximum_number_of_iterations() as i64),
        )?
        .with_parameter(
            "tolerance",
            SKParameterValue::from(self.adapter().tolerance().to_f64().unwrap_or(f64::NAN)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.adapter().backend()))?;
        let a_tensor = SKTensor::floating(
            sk_lora_a_tensor_name(target),
            self.a_matrix().shape().to_vec(),
            self.a_matrix().iter().copied().collect(),
        )?;
        let b_tensor = SKTensor::floating(
            sk_lora_b_tensor_name(target),
            self.b_matrix().shape().to_vec(),
            self.b_matrix().iter().copied().collect(),
        )?;
        SKModelState::new(metadata, vec![a_tensor, b_tensor])
    }
}

impl<F: SKFloat> SKImportable<F> for SKLoraAdapterArtifact<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let metadata = state.metadata();
        sk_validate_import_metadata(
            state,
            SK_LORA_ESTIMATOR_NAME,
            SK_LORA_FORMAT_VERSION,
            metadata.feature_count(),
        )?;
        let target_name = sk_string_parameter(metadata, "target", "linear.coefficients");
        let target =
            SKLoraTargetModule::parse(target_name).ok_or(SKError::InvalidHyperparameter {
                name: "target",
                reason: format!("unknown adapter target `{target_name}`"),
            })?;
        let rank = sk_integer_parameter(metadata, "rank", 0);
        if rank <= 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "rank",
                reason: "a persisted rank must be at least one".to_string(),
            });
        }
        let rank = rank as usize;
        let alpha = sk_float_parameter(metadata, "alpha", 1.0);
        let learning_rate = sk_float_parameter(metadata, "learning_rate", 0.5);
        let maximum_number_of_iterations =
            sk_integer_parameter(metadata, "maximum_number_of_iterations", 200);
        let tolerance = sk_float_parameter(metadata, "tolerance", 1e-8);
        let backend = sk_parse_backend_parameter(metadata);
        let adapter = SKLoraAdapterBuilder::<F>::new()
            .identifier(sk_string_parameter(metadata, "identifier", "default"))
            .rank(rank)
            .alpha(F::from(alpha).ok_or(SKError::InvalidHyperparameter {
                name: "alpha",
                reason: "the persisted alpha is not representable in this scalar".to_string(),
            })?)
            .target(target)
            .backend(backend)
            .execution_mode(SKExecutionMode::Automatic)
            .learning_rate(
                F::from(learning_rate).ok_or(SKError::InvalidHyperparameter {
                    name: "learning_rate",
                    reason: "the persisted learning rate is not representable in this scalar"
                        .to_string(),
                })?,
            )
            .maximum_number_of_iterations(maximum_number_of_iterations.max(1) as usize)
            .tolerance(F::from(tolerance).ok_or(SKError::InvalidHyperparameter {
                name: "tolerance",
                reason: "the persisted tolerance is not representable in this scalar".to_string(),
            })?)
            .build()?;
        let fitted = sk_boolean_parameter(metadata, "fitted", false);
        let heads = sk_integer_parameter(metadata, "heads", 1).max(1) as usize;
        let feature_count = metadata.feature_count();
        let a_matrix = read_factor(state, &sk_lora_a_tensor_name(target))?;
        let b_matrix = read_factor(state, &sk_lora_b_tensor_name(target))?;
        if a_matrix.dim() != (rank, feature_count) {
            return Err(SKError::ShapeMismatch {
                expected: vec![rank, feature_count],
                found: a_matrix.shape().to_vec(),
            });
        }
        if b_matrix.dim() != (heads, rank) {
            return Err(SKError::ShapeMismatch {
                expected: vec![heads, rank],
                found: b_matrix.shape().to_vec(),
            });
        }
        sk_validate_finite_factor(&a_matrix)?;
        sk_validate_finite_factor(&b_matrix)?;
        Ok(SKLoraAdapterArtifact::new(
            adapter, a_matrix, b_matrix, fitted,
        ))
    }
}

/// Reject a stored factor that carries a non-finite value, naming its flat
/// position so the caller can locate the corrupt entry.
fn sk_validate_finite_factor<F: SKFloat>(factor: &Array2<F>) -> Result<(), SKError> {
    match factor.iter().position(|value| !value.is_finite()) {
        Some(position) => Err(SKError::NonFiniteValue { position }),
        None => Ok(()),
    }
}

/// Read one factor tensor and rebuild it as a matrix.
fn read_factor<F: SKFloat>(state: &SKModelState<F>, name: &str) -> Result<Array2<F>, SKError> {
    let (shape, values) = sk_read_floating_tensor(state, name)?;
    let rows = shape.first().copied().unwrap_or(0);
    let columns = shape.get(1).copied().unwrap_or(0);
    Array2::from_shape_vec((rows, columns), values).map_err(|_| SKError::ShapeMismatch {
        expected: vec![rows, columns],
        found: shape,
    })
}
