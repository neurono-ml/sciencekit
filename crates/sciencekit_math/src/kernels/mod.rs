//! Higher-order operation kernels built on `azip!`/`par_azip!`/`zip_mut_with`
//! (spec `higher-order-kernels`, PRD §4.2).
//!
//! Every kernel is a pure function over [`SKFloat`] and honours the
//! higher-order-function mandate: no manual index loops anywhere. Layout-aware
//! reductions iterate over contiguous rows on row-major inputs.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod config;
mod elementwise;
mod grain;
mod gram;
mod linear;
mod min_max;
mod reductions;
mod scaling;
mod validation;

pub use config::{
    SKKernel, SKKernelGamma, SKKernelParameters, sk_resolve_kernel_gamma,
    sk_resolve_kernel_parameters, sk_validate_kernel_hyperparameters,
};
pub use elementwise::{sk_binary_combine, sk_elementwise_transform};
pub use grain::{
    SK_AXIS_SUM_GRAIN, SK_BINARY_COMBINE_GRAIN, SK_ELEMENTWISE_TRANSFORM_GRAIN,
    SK_SCALE_IN_PLACE_GRAIN,
};
pub use gram::{sk_gram_matrix, sk_gram_vector};
pub use linear::sk_linear_kernel_matrix;
pub use min_max::sk_axis_min_max;
pub use reductions::{sk_axis_sum, sk_axis_sum_squares};
pub use scaling::sk_scale_in_place;
pub use validation::sk_feature_population_variance;

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod grain_tests;
#[cfg(test)]
mod gram_tests;
#[cfg(test)]
mod kernels_tests;
#[cfg(test)]
mod linear_tests;
