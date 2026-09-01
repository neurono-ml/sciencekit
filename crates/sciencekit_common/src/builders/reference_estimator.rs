//! Reference estimator: canonical exemplar of the mandatory builder pattern
//! (spec `base-builders`).
//!
//! Phase 0.4 predates the Wave-1 algorithms, so this small estimator exists to
//! exercise the shared builder machinery in real code before any algorithm
//! crate lands. Its direct constructor is crate-private: construction flows
//! exclusively through [`SKReferenceEstimatorBuilder`], which inherits the
//! execution-intent plumbing of [`SKBuilderState`]. Wave-1 algorithm crates
//! follow exactly this shape.

use crate::SKError;
use crate::builders::{SKBuilder, SKBuilderState, sk_validate_hyperparameter};
use crate::execution::{SKExecutionContext, SKExecutionMode, SKExecutionPlan};
use crate::observability::{SKOperationAttributes, sk_run_operation};

/// The reference estimator demonstrating the mandatory builder pattern.
///
/// Not an algorithm: it resolves execution plans and emits observability
/// spans, providing a complete, compilable example of the construction flow
/// every Wave-1 estimator reuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKReferenceEstimator {
    execution_intent: SKExecutionMode,
    minimum_samples: usize,
}

impl SKReferenceEstimator {
    /// Crate-private constructor: consumers build through the builder only.
    pub(crate) fn new(execution_intent: SKExecutionMode, minimum_samples: usize) -> Self {
        SKReferenceEstimator {
            execution_intent,
            minimum_samples,
        }
    }

    /// The execution intent this estimator carries.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }

    /// The minimum-samples hyperparameter this estimator was built with.
    pub fn minimum_samples(&self) -> usize {
        self.minimum_samples
    }

    /// Resolve this estimator's execution plan for the given context.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        crate::execution::sk_resolve_execution_plan(self.execution_intent, context)
    }

    /// A representative public operation emitting an observability span.
    ///
    /// Resolves the estimator's plan under a `fit` span recording the input
    /// shape, the execution mode and the backend.
    pub fn fit(
        &self,
        shape: (usize, usize),
        context: &SKExecutionContext,
    ) -> Result<SKExecutionPlan, SKError> {
        let attributes = SKOperationAttributes {
            operation: "fit",
            rows: shape.0,
            columns: shape.1,
            execution_mode: self.execution_intent,
            backend: "reference",
        };
        sk_run_operation(attributes, || self.resolve_plan(context))
    }
}

/// The builder for [`SKReferenceEstimator`]; the only construction path.
#[derive(Debug, Clone)]
pub struct SKReferenceEstimatorBuilder {
    state: SKBuilderState,
    minimum_samples: usize,
}

impl SKReferenceEstimatorBuilder {
    /// Start building with the reference defaults.
    pub fn new() -> Self {
        SKReferenceEstimatorBuilder {
            state: SKBuilderState::new(),
            minimum_samples: 4,
        }
    }

    /// Set the minimum-samples hyperparameter.
    pub fn minimum_samples(mut self, value: usize) -> Self {
        self.minimum_samples = value;
        self
    }

    /// Set the execution intent (ergonomic by-value form for the PRD §2.1
    /// chain); the trait additionally exposes the `&mut self` form.
    pub fn execution_mode(mut self, mode: SKExecutionMode) -> Self {
        self.state.execution_mode(mode);
        self
    }
}

impl Default for SKReferenceEstimatorBuilder {
    fn default() -> Self {
        SKReferenceEstimatorBuilder::new()
    }
}

impl SKBuilder<SKReferenceEstimator> for SKReferenceEstimatorBuilder {
    fn execution_mode(&mut self, mode: SKExecutionMode) -> &mut Self {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKReferenceEstimator, SKError> {
        sk_validate_hyperparameter(
            "minimum_samples",
            self.minimum_samples > 0,
            "must be at least 1",
        )?;
        Ok(SKReferenceEstimator::new(
            self.state.execution_intent(),
            self.minimum_samples,
        ))
    }
}
