//! The builder for [`SKLoraAdapter`] (spec `lora-adapters`, `base-builders`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilder, SKBuilderState},
};

use super::config::SKLoraTargetModule;
use super::core_implementation::{
    SKLoraAdapter, SK_LORA_DEFAULT_ALPHA, SK_LORA_DEFAULT_LEARNING_RATE,
    SK_LORA_DEFAULT_MAXIMUM_NUMBER_OF_ITERATIONS, SK_LORA_DEFAULT_RANK, SK_LORA_DEFAULT_TOLERANCE,
};

/// Builder for [`SKLoraAdapter`]: the only construction surface.
///
/// Defaults: identifier `"default"`, rank 4, alpha 1, target
/// [`SKLoraTargetModule::LinearCoefficients`], backend `Faer`, execution intent
/// `Automatic`, learning rate 0.5, 200 iterations and tolerance `1e-8`.
pub struct SKLoraAdapterBuilder<F> {
    state: SKBuilderState,
    identifier: String,
    rank: usize,
    alpha: F,
    target: SKLoraTargetModule,
    backend: SKBackendKind,
    learning_rate: F,
    maximum_number_of_iterations: usize,
    tolerance: F,
}

impl<F: SKFloat> SKLoraAdapterBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub fn new() -> SKLoraAdapterBuilder<F> {
        SKLoraAdapterBuilder {
            state: SKBuilderState::new(),
            identifier: String::from("default"),
            rank: SK_LORA_DEFAULT_RANK,
            alpha: F::from(SK_LORA_DEFAULT_ALPHA).unwrap_or_else(F::one),
            target: SKLoraTargetModule::default(),
            backend: SKBackendKind::default(),
            learning_rate: F::from(SK_LORA_DEFAULT_LEARNING_RATE).unwrap_or_else(F::one),
            maximum_number_of_iterations: SK_LORA_DEFAULT_MAXIMUM_NUMBER_OF_ITERATIONS,
            tolerance: F::from(SK_LORA_DEFAULT_TOLERANCE).unwrap_or_else(F::epsilon),
        }
    }

    /// The stable identifier that addresses this adapter in a registry and in
    /// exported state (default `"default"`).
    pub fn identifier(mut self, identifier: impl Into<String>) -> Self {
        self.identifier = identifier.into();
        self
    }

    /// The factorization rank `r` (default 4, must be at least one).
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// The scaling numerator `alpha`; the applied scale is `alpha / rank`
    /// (default 1).
    pub fn alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// The base parameter block to adapt (default linear coefficients).
    pub fn target(mut self, target: SKLoraTargetModule) -> Self {
        self.target = target;
        self
    }

    /// The math backend kind recorded by every operation (default `Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The execution intent (default `SKExecutionMode::Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    /// The initial gradient-descent step (default 0.5).
    pub fn learning_rate(mut self, learning_rate: F) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// The maximum number of optimizer iterations (default 200).
    pub fn maximum_number_of_iterations(mut self, iterations: usize) -> Self {
        self.maximum_number_of_iterations = iterations;
        self
    }

    /// The relative-loss tolerance that stops the optimizer early (default
    /// `1e-8`).
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Validate the configuration and produce the immutable adapter.
    ///
    /// # Errors
    ///
    /// Returns [`SKError::InvalidHyperparameter`] for an empty identifier, a
    /// zero rank, a non-positive or non-finite `alpha`, a non-positive or
    /// non-finite learning rate, a zero iteration budget or a non-positive or
    /// non-finite tolerance.
    pub fn build(self) -> Result<SKLoraAdapter<F>, SKError> {
        if self.identifier.trim().is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "identifier",
                reason: "an adapter identifier must not be empty".to_string(),
            });
        }
        if self.rank == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "rank",
                reason: "a low-rank adapter needs a rank of at least one".to_string(),
            });
        }
        if !self.alpha.is_finite() || self.alpha <= F::zero() {
            return Err(SKError::InvalidHyperparameter {
                name: "alpha",
                reason: String::from("alpha must be finite and positive"),
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= F::zero() {
            return Err(SKError::InvalidHyperparameter {
                name: "learning_rate",
                reason: String::from("the learning rate must be finite and positive"),
            });
        }
        if self.maximum_number_of_iterations == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "maximum_number_of_iterations",
                reason: "a fit needs at least one iteration".to_string(),
            });
        }
        if !self.tolerance.is_finite() || self.tolerance <= F::zero() {
            return Err(SKError::InvalidHyperparameter {
                name: "tolerance",
                reason: String::from("the tolerance must be finite and positive"),
            });
        }
        Ok(SKLoraAdapter {
            identifier: self.identifier,
            rank: self.rank,
            alpha: self.alpha,
            target: self.target,
            backend: self.backend,
            execution_intent: self.state.execution_intent(),
            learning_rate: self.learning_rate,
            maximum_number_of_iterations: self.maximum_number_of_iterations,
            tolerance: self.tolerance,
        })
    }
}

impl<F: SKFloat> Default for SKLoraAdapterBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKLoraAdapter<F>> for SKLoraAdapterBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKLoraAdapter<F>, SKError> {
        SKLoraAdapterBuilder::build(self)
    }
}
