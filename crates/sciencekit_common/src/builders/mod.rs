//! Mandatory builder foundation (spec `base-builders`, PRD §2.1, §5.3).
//!
//! Every public estimator, transformer and algorithm is constructed through a
//! typed builder ([`SKBuilder`]) whose direct constructor stays private. The
//! shared [`SKBuilderState`] carries the execution intent (defaulting to
//! `Automatic`) and composes the automatic execution decision, so algorithm
//! crates reuse the intent plumbing instead of re-implementing it.

use crate::SKError;
use crate::execution::{
    SKExecutionContext, SKExecutionMode, SKExecutionPlan, sk_resolve_execution_plan,
};

mod reference_estimator;

pub use reference_estimator::{SKReferenceEstimator, SKReferenceEstimatorBuilder};

/// The mandatory builder contract for estimators, transformers and algorithms.
///
/// The execution intent defaults to [`SKExecutionMode::Automatic`] through
/// [`SKBuilderState`]; `build` validates the accumulated configuration and
/// returns the constructed model through the central error taxonomy, never
/// panicking.
pub trait SKBuilder<Model> {
    /// Set the execution intent; the default is `Automatic`.
    fn execution_mode(&mut self, mode: SKExecutionMode) -> &mut Self;

    /// Construct the model, validating the accumulated configuration.
    fn build(self) -> Result<Model, SKError>;
}

/// Shared execution-intent storage and validation state for builders.
///
/// Algorithm builders embed an [`SKBuilderState`] (defaulting to
/// [`SKExecutionMode::Automatic`]) and compose
/// [`SKBuilderState::resolve_plan`] to turn the accumulated intent into a
/// concrete [`SKExecutionPlan`] at operation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKBuilderState {
    execution_intent: SKExecutionMode,
}

impl SKBuilderState {
    /// A fresh state carrying the automatic execution intent.
    pub fn new() -> Self {
        SKBuilderState {
            execution_intent: SKExecutionMode::Automatic,
        }
    }

    /// Override the execution intent (trait-compatible `&mut self` form).
    pub fn execution_mode(&mut self, mode: SKExecutionMode) -> &mut Self {
        self.execution_intent = mode;
        self
    }

    /// The currently accumulated execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }

    /// Resolve the accumulated intent against a context into a concrete plan.
    ///
    /// Pure and deterministic: an explicit mode incompatible with the declared
    /// access pattern fails with [`SKError::ExecutionModeIncompatible`];
    /// automatic intent never does.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        sk_resolve_execution_plan(self.execution_intent, context)
    }
}

impl Default for SKBuilderState {
    fn default() -> Self {
        SKBuilderState::new()
    }
}

/// Validate a named hyperparameter, failing with
/// [`SKError::InvalidHyperparameter`] when the condition is false.
///
/// Shared by every builder so invalid configuration surfaces as an error at
/// `build()` time instead of panicking.
pub fn sk_validate_hyperparameter(
    name: &'static str,
    valid: bool,
    reason: impl Into<String>,
) -> Result<(), SKError> {
    if valid {
        Ok(())
    } else {
        Err(SKError::InvalidHyperparameter {
            name,
            reason: reason.into(),
        })
    }
}

#[cfg(test)]
mod builder_tests;
