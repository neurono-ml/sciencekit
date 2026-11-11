//! Accessors on [`super::SKFeatureUnionModel`] describing member count,
//! per-member widths, total feature count and recorded metadata (spec
//! `feature-union`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::model::SKFeatureUnionModel;
use super::stages::SKFeatureUnionStages;

impl<F, Stages> SKFeatureUnionModel<F, Stages>
where
    F: SKFloat,
    Stages: SKFeatureUnionStages<F>,
{
    /// The number of fitted member transformers in the union.
    pub fn number_of_members(&self) -> usize {
        self.member_widths.len()
    }

    /// Output column width of each member, in declaration order.
    pub fn member_widths(&self) -> &[usize] {
        &self.member_widths
    }

    /// Total concatenated output feature count (equal to the sum of
    /// [`Self::member_widths`]).
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// Alias of [`Self::number_of_features`] naming the concatenated output
    /// width explicitly.
    pub fn number_of_output_features(&self) -> usize {
        self.number_of_features
    }

    /// Number of input features observed when the union was fitted.
    pub fn number_of_input_features(&self) -> usize {
        self.number_of_input_features
    }

    /// Borrow the configured member stages.
    pub fn stages(&self) -> &Stages {
        &self.stages
    }

    /// The backend kind recorded in every operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent recorded in every operation span.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
