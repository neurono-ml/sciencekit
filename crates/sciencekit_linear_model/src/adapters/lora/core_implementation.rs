//! The immutable adapter configuration and its accessors (spec
//! `lora-adapters`).
//!
//! [`SKLoraAdapter`] carries what a user chose: the identifier, the rank, the
//! scaling numerator `alpha`, the target module and the optimizer budget. The
//! learned factors `A`/`B` live on the attached model, because their shapes
//! depend on the base parameter they modify.

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::config::SKLoraTargetModule;

/// How many factors a freshly built adapter starts from.
pub(super) const SK_LORA_DEFAULT_RANK: usize = 4;
/// The default scaling numerator, giving `alpha / rank = 0.25` at the default rank.
pub(super) const SK_LORA_DEFAULT_ALPHA: f64 = 1.0;
/// The default gradient-descent step.
pub(super) const SK_LORA_DEFAULT_LEARNING_RATE: f64 = 0.5;
/// The default iteration budget.
pub(super) const SK_LORA_DEFAULT_MAXIMUM_NUMBER_OF_ITERATIONS: usize = 200;
/// The default relative-loss tolerance.
pub(super) const SK_LORA_DEFAULT_TOLERANCE: f64 = 1e-8;

/// The configuration of a low-rank adapter.
///
/// Build one through [`super::builder::SKLoraAdapterBuilder`]; the struct keeps
/// its fields private so no invalid configuration can be constructed directly.
/// The adapter is immutable configuration — attaching it to a base model yields
/// a [`super::SKLoraAttachedModel`] that owns the trainable `A`/`B` factors.
#[derive(Debug, Clone, PartialEq)]
pub struct SKLoraAdapter<F> {
    pub(super) identifier: String,
    pub(super) rank: usize,
    pub(super) alpha: F,
    pub(super) target: SKLoraTargetModule,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
    pub(super) learning_rate: F,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) tolerance: F,
}

impl<F: SKFloat> SKLoraAdapter<F> {
    /// The stable identifier that addresses this adapter in a registry and in
    /// exported state.
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    /// The factorization rank `r`.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The scaling numerator `alpha`.
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// The update scale `alpha / r`.
    pub fn scaling(&self) -> F {
        F::from(self.rank).map_or(F::one(), |rank| self.alpha / rank)
    }

    /// The base parameter block this adapter targets.
    pub fn target(&self) -> SKLoraTargetModule {
        self.target
    }

    /// The math backend kind recorded by every operation.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent carried from the builder.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_intent
    }

    /// The gradient-descent step the fit starts from.
    pub fn learning_rate(&self) -> F {
        self.learning_rate
    }

    /// The maximum number of optimizer iterations.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.maximum_number_of_iterations
    }

    /// The relative-loss tolerance that stops the optimizer early.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }
}
