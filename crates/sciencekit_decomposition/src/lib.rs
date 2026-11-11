//! `sciencekit_decomposition` — matrix decomposition and dimensionality-reduction
//! estimators for sciencekit (PRD §12, Phase 5).
//!
//! The crate opens with `SKPCA`, Principal Component Analysis over a solver
//! portfolio (full SVD, seeded randomized SVD with implicit sparse centering,
//! and symmetric covariance eigendecomposition) with optional whitening and an
//! explicit densification policy.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod pca;

pub use pca::{SKPCA, SKPCABuilder, SKPCAModel, SKPCASolver};
