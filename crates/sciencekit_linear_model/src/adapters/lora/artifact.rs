//! A persisted adapter before it is attached to a base (spec `lora-adapters`,
//! decision 4).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::SKFloat;

use super::attached_model::SKLoraAttachedModel;
use super::base_model::SKLoraBaseModel;
use super::core_implementation::SKLoraAdapter;

/// An imported adapter state: configuration plus the stored `A`/`B` factors.
///
/// An adapter is stored without its base — the base is the frozen model the
/// consumer already owns — so the state boundary produces this artifact and
/// [`Self::attach`] pairs it with a base, validating that the stored shapes fit.
#[derive(Debug, Clone, PartialEq)]
pub struct SKLoraAdapterArtifact<F> {
    adapter: SKLoraAdapter<F>,
    a_matrix: Array2<F>,
    b_matrix: Array2<F>,
    fitted: bool,
}

impl<F: SKFloat> SKLoraAdapterArtifact<F> {
    /// Assemble an artifact (used by the importer and by round-trip tests).
    pub(crate) fn new(
        adapter: SKLoraAdapter<F>,
        a_matrix: Array2<F>,
        b_matrix: Array2<F>,
        fitted: bool,
    ) -> SKLoraAdapterArtifact<F> {
        SKLoraAdapterArtifact {
            adapter,
            a_matrix,
            b_matrix,
            fitted,
        }
    }

    /// The adapter configuration.
    pub fn adapter(&self) -> &SKLoraAdapter<F> {
        &self.adapter
    }

    /// The stored `rank × features` factor `A`.
    pub fn a_matrix(&self) -> ArrayView2<'_, F> {
        self.a_matrix.view()
    }

    /// The stored `heads × rank` factor `B`.
    pub fn b_matrix(&self) -> ArrayView2<'_, F> {
        self.b_matrix.view()
    }

    /// Whether the adapter was fitted when it was exported.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Pair the stored adapter with a base, validating every stored shape.
    ///
    /// # Errors
    ///
    /// Returns [`sciencekit_common::SKError::ShapeMismatch`] when the stored
    /// factors do not fit the base, and [`sciencekit_common::SKError::UnsupportedOperation`]
    /// when the target or family cannot be adapted.
    pub fn attach(
        &self,
        base: SKLoraBaseModel<F>,
    ) -> Result<SKLoraAttachedModel<F>, sciencekit_common::SKError> {
        let mut attached = self.adapter.attach(base)?;
        let (rows, columns) = attached.a_matrix().dim();
        if self.a_matrix.dim() != (rows, columns) {
            return Err(sciencekit_common::SKError::ShapeMismatch {
                expected: vec![rows, columns],
                found: self.a_matrix.shape().to_vec(),
            });
        }
        let (rows, columns) = attached.b_matrix().dim();
        if self.b_matrix.dim() != (rows, columns) {
            return Err(sciencekit_common::SKError::ShapeMismatch {
                expected: vec![rows, columns],
                found: self.b_matrix.shape().to_vec(),
            });
        }
        attached.set_factors(self.a_matrix.clone(), self.b_matrix.clone());
        if self.fitted {
            attached.mark_fitted();
        }
        Ok(attached)
    }
}
