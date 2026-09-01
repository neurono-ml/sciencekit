//! Shared execution-intent storage and validation state (spec `base-builders`).

use crate::SKError;
use crate::execution::{
    SKExecutionContext, SKExecutionMode, SKExecutionPlan, sk_resolve_execution_plan,
};

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
