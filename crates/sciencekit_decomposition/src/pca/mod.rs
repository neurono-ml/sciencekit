//! Principal Component Analysis (`SKPCA`, spec `principal-component-analysis`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! its submodules; every implementation lives in its own file.

mod builder;
mod centering;
mod config;
mod core_implementation;
mod solvers;

pub use builder::SKPCABuilder;
pub use config::SKPCASolver;
pub use core_implementation::SKPCA;

#[cfg(test)]
mod pca_builder_tests;
#[cfg(test)]
mod pca_centering_tests;
#[cfg(test)]
mod pca_covariance_tests;
#[cfg(test)]
mod pca_full_tests;
#[cfg(test)]
mod pca_randomized_tests;
#[cfg(test)]
mod pca_solver_tests;
#[cfg(test)]
mod test_tracing;
