//! `sciencekit_decomposition` — matrix decomposition and dimensionality-reduction
//! estimators for sciencekit (PRD §12, Phase 5).
//!
//! The crate opens with [`SKPCA`], Principal Component Analysis over a solver
//! portfolio (full SVD, seeded randomized SVD with implicit sparse centering,
//! and symmetric covariance eigendecomposition) with optional whitening and an
//! explicit densification policy.
//!
//! # Example
//!
//! ```
//! use ndarray::array;
//! use sciencekit_common::{SKBuilder, SKUnsupervisedFit};
//! use sciencekit_decomposition::SKPCA;
//!
//! let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let model = SKPCA::<f64>::build()
//!     .number_of_components(1)
//!     .build()
//!     .unwrap()
//!     .fit(&data)
//!     .unwrap();
//! let projected = model.transform(&data).unwrap();
//! assert_eq!(projected.dim(), (4, 1));
//! ```
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod incremental_pca;
pub mod pca;

pub use pca::{SKPCA, SKPCABuilder, SKPCAModel, SKPCASolver};

#[cfg(test)]
mod acceptance_tests;
