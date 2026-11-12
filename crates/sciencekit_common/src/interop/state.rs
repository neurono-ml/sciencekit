//! The format-agnostic model state container (spec `model-export-contracts`).

use std::collections::BTreeMap;

use crate::SKError;
use crate::sk_float::SKFloat;

use super::{SKModelMetadata, SKParameterValue, SKTensor};

/// Everything a format adapter needs to persist a fitted model.
///
/// The state pairs descriptive [`SKModelMetadata`] with a deterministic map of
/// named tensors. Tensor names are unique and every floating tensor matches the
/// metadata's scalar data_type; both are validated at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct SKModelState<F: SKFloat> {
    metadata: SKModelMetadata,
    tensors: BTreeMap<String, SKTensor<F>>,
}

impl<F: SKFloat> SKModelState<F> {
    /// Build a state from metadata and tensors, rejecting duplicate tensor names
    /// and floating tensors whose data_type disagrees with the metadata scalar.
    pub fn new(metadata: SKModelMetadata, tensors: Vec<SKTensor<F>>) -> Result<Self, SKError> {
        let mut named_tensors = BTreeMap::new();
        for tensor in tensors {
            if tensor.data_type() != metadata.scalar_data_type() && tensor.data_type().is_floating()
            {
                return Err(SKError::InvalidHyperparameter {
                    name: "tensor_data_type",
                    reason: format!(
                        "tensor `{}` uses {} but the model scalar is {}",
                        tensor.name(),
                        tensor.data_type().as_str(),
                        metadata.scalar_data_type().as_str()
                    ),
                });
            }
            let name = tensor.name().to_string();
            if named_tensors.insert(name.clone(), tensor).is_some() {
                return Err(SKError::InvalidHyperparameter {
                    name: "tensor_names",
                    reason: format!("tensor name `{name}` is not unique"),
                });
            }
        }
        Ok(SKModelState {
            metadata,
            tensors: named_tensors,
        })
    }

    /// The descriptive metadata.
    pub fn metadata(&self) -> &SKModelMetadata {
        &self.metadata
    }

    /// Every tensor keyed by its stable name, in deterministic order.
    pub fn tensors(&self) -> &BTreeMap<String, SKTensor<F>> {
        &self.tensors
    }

    /// A single tensor by its stable name.
    pub fn tensor(&self, name: &str) -> Option<&SKTensor<F>> {
        self.tensors.get(name)
    }

    /// The number of exported tensors.
    pub fn number_of_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// The typed hyperparameters carried by the metadata.
    pub fn parameters(&self) -> &BTreeMap<String, SKParameterValue> {
        self.metadata.parameters()
    }
}
