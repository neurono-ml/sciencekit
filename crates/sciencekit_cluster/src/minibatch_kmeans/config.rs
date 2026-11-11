//! MiniBatchKMeans configuration vocabulary and validation (spec
//! `mini-batch-k-means`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, sk_validate_hyperparameter,
};

use super::builder::SKMiniBatchKMeansBuilder;
use crate::kmeans::SKKMeansInitialization;

/// The validated configuration carried by a built [`super::SKMiniBatchKMeans`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKMiniBatchKMeansConfiguration<F: SKFloat> {
    /// Requested cluster count.
    pub(crate) number_of_clusters: usize,
    /// Centroid initialization strategy.
    pub(crate) initialization: SKKMeansInitialization,
    /// Number of independent seeded initializations.
    pub(crate) number_of_initializations: usize,
    /// Mini-batch row count for in-memory chunking.
    pub(crate) batch_size: usize,
    /// Maximum sample size drawn for centroid initialization.
    pub(crate) initialization_batch_size: usize,
    /// Maximum number of passes over in-memory training data.
    pub(crate) maximum_number_of_iterations: usize,
    /// Maximum number of streaming passes over a reopenable source provider.
    pub(crate) maximum_number_of_epochs: usize,
    /// Convergence tolerance (`0` disables early stopping).
    pub(crate) tolerance: F,
    /// Low-count center reassignment threshold fraction in `[0, 1]`.
    pub(crate) reassignment_ratio: F,
    /// Deterministic global seed.
    pub(crate) random_seed: u64,
    /// Math backend recorded in operation spans.
    pub(crate) backend: SKBackendKind,
    /// Execution intent.
    pub(crate) execution_mode: SKExecutionMode,
}

impl<F: SKFloat> SKMiniBatchKMeansConfiguration<F> {
    /// Validates accumulated builder fields and constructs the configuration.
    pub(super) fn from_builder(builder: &SKMiniBatchKMeansBuilder<F>) -> Result<Self, SKError> {
        sk_validate_hyperparameter(
            "number_of_clusters",
            builder.number_of_clusters > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "number_of_initializations",
            builder.number_of_initializations > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "batch_size",
            builder.batch_size > 0,
            "must be strictly positive",
        )?;
        let initialization_batch_size = match builder.initialization_batch_size {
            Some(explicit) => {
                sk_validate_hyperparameter(
                    "initialization_batch_size",
                    explicit >= builder.number_of_clusters,
                    "must be at least number_of_clusters",
                )?;
                explicit
            }
            None => builder
                .batch_size
                .saturating_mul(3)
                .max(builder.number_of_clusters),
        };
        sk_validate_hyperparameter(
            "maximum_number_of_iterations",
            builder.maximum_number_of_iterations > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "maximum_number_of_epochs",
            builder.maximum_number_of_epochs > 0,
            "must be strictly positive",
        )?;
        sk_validate_hyperparameter(
            "tolerance",
            builder.tolerance.is_finite() && builder.tolerance >= F::zero(),
            "must be finite and non-negative",
        )?;
        sk_validate_hyperparameter(
            "reassignment_ratio",
            builder.reassignment_ratio.is_finite()
                && builder.reassignment_ratio >= F::zero()
                && builder.reassignment_ratio <= F::one(),
            "must be finite and in [0, 1]",
        )?;
        Ok(Self {
            number_of_clusters: builder.number_of_clusters,
            initialization: builder.initialization,
            number_of_initializations: builder.number_of_initializations,
            batch_size: builder.batch_size,
            initialization_batch_size,
            maximum_number_of_iterations: builder.maximum_number_of_iterations,
            maximum_number_of_epochs: builder.maximum_number_of_epochs,
            tolerance: builder.tolerance,
            reassignment_ratio: builder.reassignment_ratio,
            random_seed: builder.random_seed,
            backend: builder.backend,
            execution_mode: builder.state.execution_intent(),
        })
    }
}
