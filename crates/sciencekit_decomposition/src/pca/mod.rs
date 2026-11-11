//! Principal Component Analysis (`SKPCA`, spec `principal-component-analysis`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! its submodules; every implementation lives in its own file.

mod centering;

#[cfg(test)]
mod pca_centering_tests;
#[cfg(test)]
mod test_tracing;
