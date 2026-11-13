//! The export/import trait pair and shared import validation
//! (spec `model-export-contracts`).

use crate::SKError;
use crate::sk_float::SKFloat;

use super::SKModelState;

/// A model that describes its learned state in a format-agnostic way.
pub trait SKExportable<F: SKFloat> {
    /// Produce the model state a format adapter can persist.
    fn export_state(&self) -> Result<SKModelState<F>, SKError>;
}

/// A model that reconstructs itself from an exported state.
pub trait SKImportable<F: SKFloat>: Sized {
    /// Rebuild the model, validating the state before accepting it.
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError>;
}

/// Validate the estimator identity, format version and feature count of a state.
///
/// Importers call this before trusting any tensor. A state produced by another
/// estimator or format version is rejected as an unsupported representation; a
/// feature-count disagreement is a shape mismatch.
pub fn sk_validate_import_metadata<F: SKFloat>(
    state: &SKModelState<F>,
    expected_estimator_name: &str,
    expected_format_version: u32,
    expected_feature_count: usize,
) -> Result<(), SKError> {
    let metadata = state.metadata();
    if metadata.estimator_name() != expected_estimator_name {
        return Err(SKError::UnsupportedRepresentation {
            representation: "model state estimator",
            suggestion: "import a state exported by the same estimator",
        });
    }
    if metadata.format_version() != expected_format_version {
        return Err(SKError::UnsupportedRepresentation {
            representation: "model state format version",
            suggestion: "re-export the model with the current library version",
        });
    }
    if metadata.feature_count() != expected_feature_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![expected_feature_count],
            found: vec![metadata.feature_count()],
        });
    }
    Ok(())
}
