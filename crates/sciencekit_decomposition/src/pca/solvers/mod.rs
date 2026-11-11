//! PCA solver portfolio: full SVD, randomized SVD, covariance eigensolver and
//! deterministic `Auto` resolution.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! its submodules; every implementation lives in its own file.

mod full;
mod state;

pub(super) use full::sk_solve_full_svd;
