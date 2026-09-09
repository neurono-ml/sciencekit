//! Higher-order operation kernels built on `azip!`/`par_azip!`/`zip_mut_with`
//! (spec `higher-order-kernels`, PRD §4.2).
//!
//! Every kernel is a pure function over [`SKFloat`] and honours the
//! higher-order-function mandate: no manual index loops anywhere. Layout-aware
//! reductions iterate over contiguous rows on row-major inputs.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod elementwise;
mod reductions;
mod scaling;

pub use elementwise::{sk_binary_combine, sk_elementwise_transform};
pub use reductions::sk_axis_sum;
pub use scaling::sk_scale_in_place;

#[cfg(test)]
mod kernels_tests;
