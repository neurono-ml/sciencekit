//! Nesting helpers for composite models.
//!
//! A composite model (a pipeline, an ensemble member, a composed imputer)
//! embeds another model's whole state under a namespace. These helpers copy a
//! state's metadata parameters and tensors in and out, keeping the inner names
//! stable and the outer state a single `SKModelState`.

use crate::SKError;
use crate::interop::{SKModelMetadata, SKModelState, SKTensor};
use crate::sk_float::SKFloat;

/// The parameter prefix a nested state's metadata uses.
pub fn sk_nested_parameter_prefix(namespace: &str) -> String {
    format!("{namespace}.parameter.")
}

/// The tensor prefix a nested state's tensors use.
pub fn sk_nested_tensor_prefix(namespace: &str) -> String {
    format!("{namespace}.tensor.")
}

/// Copy a nested state's parameters and tensors into an outer metadata/tensor
/// pair. The caller supplies the outer metadata and tensor list.
///
/// # Errors
///
/// Returns [`SKError::InvalidHyperparameter`] when copying a parameter or
/// tensor fails.
pub fn sk_embed_state<F: SKFloat>(
    mut metadata: SKModelMetadata,
    tensors: &mut Vec<SKTensor<F>>,
    namespace: &str,
    inner: &SKModelState<F>,
) -> Result<SKModelMetadata, SKError> {
    let parameter_prefix = sk_nested_parameter_prefix(namespace);
    let tensor_prefix = sk_nested_tensor_prefix(namespace);
    for (name, value) in inner.metadata().parameters() {
        metadata = metadata.with_parameter(format!("{parameter_prefix}{name}"), value.clone())?;
    }
    for (name, tensor) in inner.tensors() {
        tensors.push(SKTensor::new(
            format!("{tensor_prefix}{name}"),
            tensor.data_type(),
            tensor.shape().to_vec(),
            tensor.values().clone(),
        )?);
    }
    Ok(metadata)
}

/// Rebuild a nested state from the outer state's namespaced entries.
///
/// # Errors
///
/// Returns the first [`SKError`] raised while rebuilding the state.
pub fn sk_extract_state<F: SKFloat>(
    outer: &SKModelState<F>,
    namespace: &str,
    estimator_name: &str,
    feature_count: usize,
) -> Result<SKModelState<F>, SKError> {
    let parameter_prefix = sk_nested_parameter_prefix(namespace);
    let tensor_prefix = sk_nested_tensor_prefix(namespace);
    let mut metadata = SKModelMetadata::new(
        estimator_name,
        crate::interop::SK_MODEL_FORMAT_VERSION,
        crate::interop::sk_model_tensor_data_type::<F>(),
        feature_count,
    )?;
    for (name, value) in outer.metadata().parameters() {
        if let Some(inner) = name.strip_prefix(&parameter_prefix) {
            metadata = metadata.with_parameter(inner, value.clone())?;
        }
    }
    if let Some(labels) = outer.metadata().class_labels() {
        metadata = metadata.with_class_labels(labels.to_vec())?;
    }
    let mut tensors = Vec::new();
    for (name, tensor) in outer.tensors() {
        if let Some(inner) = name.strip_prefix(&tensor_prefix) {
            tensors.push(SKTensor::new(
                inner,
                tensor.data_type(),
                tensor.shape().to_vec(),
                tensor.values().clone(),
            )?);
        }
    }
    SKModelState::new(metadata, tensors)
}
