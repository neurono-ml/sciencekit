//! Tensor-set validation before a family importer runs
//! (spec `native-model-import`, requirement "Validation and concurrency").

use sciencekit_common::{SKFloat, SKModelState};

use super::tensor_names::FOREST_IMPORTANCE;
use crate::native::error::SKNativeImportError;

/// Reject a tensor set that differs from the documented one in either direction.
pub(crate) fn validate_tensor_names<F: SKFloat>(
    state: &SKModelState<F>,
    expected: &[&str],
) -> Result<(), SKNativeImportError> {
    for name in expected {
        if state.tensor(name).is_none() {
            return Err(SKNativeImportError::MissingTensor {
                name: (*name).to_string(),
            });
        }
    }
    for name in state.tensors().keys() {
        if !expected.contains(&name.as_str()) {
            return Err(SKNativeImportError::UnexpectedTensor { name: name.clone() });
        }
    }
    Ok(())
}

/// Validate a tree state: node tensors share one prefix, importances are exact.
pub(crate) fn validate_tree_tensors<F: SKFloat>(
    state: &SKModelState<F>,
    required: &[&str],
) -> Result<(), SKNativeImportError> {
    for name in required {
        if state.tensor(name).is_none() {
            return Err(SKNativeImportError::MissingTensor {
                name: (*name).to_string(),
            });
        }
    }
    for name in state.tensors().keys() {
        let node_tensor = name
            .strip_prefix("tree.")
            .is_some_and(|tail| tail.starts_with("nodes."));
        if !node_tensor && !required.contains(&name.as_str()) {
            return Err(SKNativeImportError::UnexpectedTensor { name: name.clone() });
        }
    }
    Ok(())
}

/// Validate a forest state: the forest tensors plus per-tree embedded states.
pub(crate) fn validate_forest_tensors<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<(), SKNativeImportError> {
    if state.tensor(FOREST_IMPORTANCE).is_none() {
        return Err(SKNativeImportError::MissingTensor {
            name: FOREST_IMPORTANCE.to_string(),
        });
    }
    for name in state.tensors().keys() {
        if name == FOREST_IMPORTANCE {
            continue;
        }
        if !name.starts_with("forest.tree.") {
            return Err(SKNativeImportError::UnexpectedTensor { name: name.clone() });
        }
    }
    Ok(())
}
