//! The export/import trait pair and shared import validation
//! (spec `model-export-contracts`).

use crate::SKError;
use crate::sk_float::SKFloat;

use super::SKModelState;

/// A model that describes its learned state in a format-agnostic way.
///
/// Every fitted model type implements this trait (spec `model-export-coverage`,
/// requirement "Every fitted model exports state"). Most produce a state from
/// `export_state`; a model whose learned state cannot be described at all — the
/// type-erased graph and streaming pipelines — returns `UnsupportedOperation`
/// with guidance instead of a partial state.
pub trait SKExportable<F: SKFloat> {
    /// Produce the model state a format adapter can persist.
    ///
    /// Export is read-only and deterministic: two calls on the same fitted model
    /// produce equal states. A model that cannot describe its state returns
    /// `UnsupportedOperation` explaining why.
    fn export_state(&self) -> Result<SKModelState<F>, SKError>;
}

/// A model that reconstructs itself from an exported state.
///
/// Import is **conditional** (spec `model-export-coverage`, requirement "Native
/// import where reconstructible"): only a model whose exported state is
/// sufficient to rebuild its configuration and learned values round-trips.
/// Models that export state but cannot be reconstructed — bagging, grid-search
/// and successive-halving models, typed pipelines and feature unions — still
/// implement this trait and return `UnsupportedOperation` with guidance, never a
/// partial model. Of the 47 fitted model types, 35 import and reproduce the
/// original output bit-for-bit.
pub trait SKImportable<F: SKFloat>: Sized {
    /// Rebuild the model, validating the state before accepting it.
    ///
    /// Returns `UnsupportedOperation` when the state is valid but does not
    /// carry enough information to reconstruct the model.
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
