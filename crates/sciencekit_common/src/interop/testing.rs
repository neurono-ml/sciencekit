//! Shared export/import round-trip assertions used by every model family.
//!
//! # Stable tensor names
//!
//! Every model family names its tensors with a dotted path. The first segment
//! is a short scope (`standard_scaler`, `minmax_scaler`, `linear`, `tree`,
//! `svm`, `kmeans`, `pca`, `pipeline`, …) and the remaining segments name the
//! learned field. Nested collections append an index segment:
//!
//! ```text
//! standard_scaler.mean
//! minmax_scaler.minimum
//! one_hot.categories.<column>
//! linear.coefficients
//! tree.nodes.threshold
//! forest.trees.<index>.nodes.value
//! pipeline.stage.<index>.<field>
//! ```
//!
//! The names are part of the compatibility promise: renaming a tensor requires
//! a format-version bump. Two exports of the same fitted model must always
//! produce the same name set, which is what [`sk_assert_export_round_trip`]
//! checks.
//!
//! These helpers are public so every model crate can share one definition of
//! "a valid round trip"; they perform no file I/O and never depend on a format
//! adapter.

use crate::SKError;
use crate::sk_float::SKFloat;

use super::{SKExportable, SKImportable, SKModelState, SKTensorValues};

/// Export `model` twice, assert the two states are byte-for-byte equal, import
/// the state and assert the imported metadata is preserved.
///
/// The helper returns the imported model so the caller can compare predictions
/// or transformations within its documented tolerance. It is deliberately
/// state-level: predictions are family-specific, so each family's companion
/// tests own that comparison.
///
/// # Errors
///
/// Returns the first [`SKError`] raised by export or import, or
/// [`SKError::InvalidHyperparameter`] when the exported state carries no
/// tensors or a tensor name is not a dotted path.
pub fn sk_assert_export_round_trip<F, M>(model: &M) -> Result<M, SKError>
where
    F: SKFloat,
    M: SKExportable<F> + SKImportable<F>,
{
    let first_state = model.export_state()?;
    let second_state = model.export_state()?;
    if !sk_states_equal(&first_state, &second_state) {
        return Err(SKError::InvalidHyperparameter {
            name: "export_determinism",
            reason: "two exports of the same fitted model produced different states".to_string(),
        });
    }
    sk_assert_state_names(&first_state)?;
    let imported = M::import_state(&first_state)?;
    let imported_state = imported.export_state()?;
    if imported_state.metadata() != first_state.metadata() {
        return Err(SKError::InvalidHyperparameter {
            name: "import_metadata",
            reason: "the imported model does not preserve the exported metadata".to_string(),
        });
    }
    if imported_state
        .tensors()
        .keys()
        .ne(first_state.tensors().keys())
    {
        return Err(SKError::InvalidHyperparameter {
            name: "import_tensor_names",
            reason: "the imported model re-exports a different tensor-name set".to_string(),
        });
    }
    Ok(imported)
}

/// Compare two states, treating two NaNs as equal.
///
/// Missing-value markers and imputation statistics legitimately store NaN; a
/// derived `PartialEq` would report two identical exports as different.
pub fn sk_states_equal<F: SKFloat>(first: &SKModelState<F>, second: &SKModelState<F>) -> bool {
    if first.metadata() != second.metadata() || first.tensors().len() != second.tensors().len() {
        return false;
    }
    first.tensors().iter().all(|(name, left)| {
        let Some(right) = second.tensor(name) else {
            return false;
        };
        if left.data_type() != right.data_type() || left.shape() != right.shape() {
            return false;
        }
        match (left.values(), right.values()) {
            (SKTensorValues::Floating(a), SKTensorValues::Floating(b)) => a
                .iter()
                .zip(b.iter())
                .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())),
            (SKTensorValues::Integer(a), SKTensorValues::Integer(b)) => a == b,
            (SKTensorValues::Boolean(a), SKTensorValues::Boolean(b)) => a == b,
            _ => false,
        }
    })
}

/// Assert a state carries at least one tensor and every name is a dotted path.
///
/// # Errors
///
/// Returns [`SKError::InvalidHyperparameter`] when the state is empty or a name
/// lacks the `scope.field` shape.
pub fn sk_assert_state_names<F: SKFloat>(state: &SKModelState<F>) -> Result<(), SKError> {
    if state.number_of_tensors() == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "tensor_names",
            reason: "an exported state must carry at least one tensor".to_string(),
        });
    }
    for name in state.tensors().keys() {
        if name.split('.').count() < 2 || name.split('.').any(str::is_empty) {
            return Err(SKError::InvalidHyperparameter {
                name: "tensor_name",
                reason: format!("tensor name `{name}` is not a dotted `scope.field` path"),
            });
        }
    }
    Ok(())
}
