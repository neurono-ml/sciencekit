//! Builder and configuration surface for [`SKFeatureUnion`] (spec
//! `feature-union`).

use std::sync::Arc;

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::append::SKAppendUnionStage;
use super::stages::SKFeatureUnionArity;

/// Parallel composition of transformer stages fitted on the same input block and
/// concatenated column-wise.
///
/// Constructed exclusively through [`SKFeatureUnion::build`] (or
/// [`SKFeatureUnion::start`]). Direct constructors are private.
pub struct SKFeatureUnion<Stages = ()> {
    pub(super) stages: Arc<Stages>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl SKFeatureUnion<()> {
    /// Start a feature-union builder with default `Faer` backend and `Automatic`
    /// execution mode.
    pub fn build() -> SKFeatureUnionBuilder<()> {
        SKFeatureUnionBuilder::new(())
    }

    /// Alias of [`Self::build`] mirroring [`crate::SKPipeline::start`].
    pub fn start() -> SKFeatureUnionBuilder<()> {
        SKFeatureUnionBuilder::new(())
    }
}

impl<Stages> SKFeatureUnion<Stages> {
    /// Borrow the configured member collection.
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

impl<Stages: SKFeatureUnionArity> SKFeatureUnion<Stages> {
    /// The number of configured member transformers.
    pub fn number_of_members(&self) -> usize {
        self.stages.number_of_members()
    }
}

/// Mandatory builder for [`SKFeatureUnion`].
pub struct SKFeatureUnionBuilder<Stages = ()> {
    pub(super) stages: Stages,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<Stages> SKFeatureUnionBuilder<Stages> {
    pub(super) fn new(stages: Stages) -> Self {
        Self {
            stages,
            backend: SKBackendKind::Faer,
            execution_intent: SKExecutionMode::Automatic,
        }
    }

    /// Replace the member collection with `stages` (a tuple of up to eight
    /// stages or a dynamic boxed list).
    pub fn stages<NextStages>(self, stages: NextStages) -> SKFeatureUnionBuilder<NextStages> {
        SKFeatureUnionBuilder {
            stages,
            backend: self.backend,
            execution_intent: self.execution_intent,
        }
    }

    /// Append one member stage to a flat tuple (up to eight members).
    pub fn with_stage<Next>(self, stage: Next) -> SKFeatureUnionBuilder<Stages::Extended>
    where
        Stages: SKAppendUnionStage<Next>,
    {
        SKFeatureUnionBuilder {
            stages: self.stages.append_union_stage(stage),
            backend: self.backend,
            execution_intent: self.execution_intent,
        }
    }

    /// Set the backend kind recorded in every operation span (default `Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Set the execution intent recorded in every operation span (default
    /// `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        if let Ok(resolved) = mode.try_into() {
            self.execution_intent = resolved;
        }
        self
    }
}

impl<Stages: SKFeatureUnionArity> SKFeatureUnionBuilder<Stages> {
    /// Validate that at least one member is present and construct the union.
    pub fn build(self) -> Result<SKFeatureUnion<Stages>, SKError> {
        if self.stages.number_of_members() == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "stages",
                reason: "a feature union requires at least one member transformer".to_string(),
            });
        }
        Ok(SKFeatureUnion {
            stages: Arc::new(self.stages),
            backend: self.backend,
            execution_intent: self.execution_intent,
        })
    }
}

impl<Stages: SKFeatureUnionArity> SKBuilder<SKFeatureUnion<Stages>>
    for SKFeatureUnionBuilder<Stages>
{
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        if let Ok(resolved) = mode.try_into() {
            self.execution_intent = resolved;
        }
        self
    }

    fn build(self) -> Result<SKFeatureUnion<Stages>, SKError> {
        SKFeatureUnionBuilder::build(self)
    }
}
