//! Fitted [`SKFeatureUnionModel`] and fit/transform execution (spec
//! `feature-union`).

use std::sync::Arc;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use crate::SKFeatureBlock;

use super::builder::SKFeatureUnion;
use super::concatenate::sk_concatenate_blocks;
use super::errors::SKFeatureUnionMemberError;
use super::stages::SKFeatureUnionStages;

/// Fitted feature union owning the configured member stages, their fitted
/// twins, and the learned column-width metadata.
pub struct SKFeatureUnionModel<F, Stages>
where
    F: SKFloat,
    Stages: SKFeatureUnionStages<F>,
{
    pub(super) stages: Arc<Stages>,
    pub(super) fitted_stages: Stages::FittedStages,
    pub(super) member_widths: Vec<usize>,
    pub(super) number_of_input_features: usize,
    pub(super) number_of_features: usize,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<Stages> SKFeatureUnion<Stages> {
    /// Fit every member transformer on `block` and record per-member output
    /// widths.
    pub fn fit<F>(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureUnionModel<F, Stages>, SKError>
    where
        F: SKFloat,
        Stages: SKFeatureUnionStages<F>,
    {
        self.fit_checked(block).map_err(Into::into)
    }

    /// Fit every member transformer on `block`, returning any failure paired
    /// with the zero-based member index when a member fails.
    pub fn fit_checked<F>(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureUnionModel<F, Stages>, SKFeatureUnionMemberError>
    where
        F: SKFloat,
        Stages: SKFeatureUnionStages<F>,
    {
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: block.number_of_rows(),
            columns_count: block.number_of_columns(),
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        let mut captured_error: Option<SKFeatureUnionMemberError> = None;
        let outcome =
            sk_run_operation(attributes, || match self.stages.fit_stages_checked(block) {
                Ok(pair) => Ok(pair),
                Err(member_error) => {
                    let placeholder = SKError::Conversion("member fit failed".to_string());
                    captured_error = Some(member_error);
                    Err(placeholder)
                }
            });
        match outcome {
            Ok((fitted_stages, member_widths)) => {
                let number_of_features = member_widths.iter().copied().sum();
                Ok(SKFeatureUnionModel {
                    stages: Arc::clone(&self.stages),
                    fitted_stages,
                    member_widths,
                    number_of_input_features: block.number_of_columns(),
                    number_of_features,
                    backend: self.backend,
                    execution_intent: self.execution_intent,
                })
            }
            Err(fallback) => {
                Err(captured_error
                    .unwrap_or_else(|| SKFeatureUnionMemberError::union_level(fallback)))
            }
        }
    }
}

impl<F, Stages> SKFeatureUnionModel<F, Stages>
where
    F: SKFloat,
    Stages: SKFeatureUnionStages<F>,
{
    /// Borrow the fitted models of every member in declaration order.
    pub fn fitted_stages(&self) -> &Stages::FittedStages {
        &self.fitted_stages
    }

    /// Transform `block` with every fitted member and concatenate their outputs
    /// column-wise in member order.
    pub fn transform(&self, block: &SKFeatureBlock<F>) -> Result<SKFeatureBlock<F>, SKError> {
        self.transform_checked(block).map_err(Into::into)
    }

    /// Transform `block`, returning any failure paired with the zero-based
    /// member index when a member fails.
    pub fn transform_checked(
        &self,
        block: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKFeatureUnionMemberError> {
        let rows_count = block.number_of_rows();
        let columns_count = block.number_of_columns();
        if columns_count != self.number_of_input_features {
            return Err(SKFeatureUnionMemberError::union_level(
                SKError::ShapeMismatch {
                    expected: vec![rows_count, self.number_of_input_features],
                    found: vec![rows_count, columns_count],
                },
            ));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        let mut captured_error: Option<SKFeatureUnionMemberError> = None;
        let outcome = sk_run_operation(attributes, || {
            let outputs = match self
                .stages
                .transform_stages_checked(&self.fitted_stages, block)
            {
                Ok(outputs) => outputs,
                Err(member_error) => {
                    let placeholder = SKError::Conversion("member transform failed".to_string());
                    captured_error = Some(member_error);
                    return Err(placeholder);
                }
            };
            sk_concatenate_blocks(
                &outputs,
                rows_count,
                &self.member_widths,
                self.number_of_features,
            )
            .map_err(|error| {
                let placeholder = SKError::Conversion("concatenation failed".to_string());
                captured_error = Some(SKFeatureUnionMemberError::union_level(error));
                placeholder
            })
        });
        outcome.map_err(|fallback| {
            captured_error.unwrap_or_else(|| SKFeatureUnionMemberError::union_level(fallback))
        })
    }
}
