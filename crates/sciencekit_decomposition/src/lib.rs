//! `sciencekit_decomposition` — matrix decomposition and dimensionality-reduction
//! estimators for sciencekit (PRD §12, Phase 5).
//!
//! The crate provides:
//! - [`SKPCA`], Principal Component Analysis over a solver portfolio (full SVD,
//!   seeded randomized SVD with implicit sparse centering, and symmetric
//!   covariance eigendecomposition) with optional whitening and an explicit
//!   densification policy;
//! - [`SKIncrementalPCA`], out-of-core Principal Component Analysis via the
//!   Chan-Golub-LeVeque scatter merge over `partial_fit` batches and
//!   `fit_streaming` providers without retaining rows or densifying CSR inputs;
//! - [`SKTruncatedSVD`], uncentered low-rank Singular Value Decomposition via
//!   the seeded randomized range finder, operating directly on dense or sparse
//!   CSR matrices without centering or densification (latent semantic analysis
//!   and sparse dimensionality reduction).
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use sciencekit_common::{SKBuilder, SKUnsupervisedFit};
//! use sciencekit_decomposition::{SKIncrementalPCA, SKPCA, SKTruncatedSVD};
//! use sprs::TriMat;
//!
//! let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
//! let pca_model = SKPCA::<f64>::build()
//!     .number_of_components(1)
//!     .build()
//!     .unwrap()
//!     .fit(&data)
//!     .unwrap();
//! assert_eq!(pca_model.transform(&data).unwrap().dim(), (4, 1));
//!
//! let incremental_model = SKIncrementalPCA::<f64>::build()
//!     .number_of_components(1)
//!     .batch_size(2)
//!     .build()
//!     .unwrap()
//!     .fit(&data)
//!     .unwrap();
//! assert_eq!(incremental_model.transform(&data).unwrap().dim(), (4, 1));
//!
//! let mut triplet = TriMat::new((4, 5));
//! triplet.add_triplet(0, 0, 2.0_f64);
//! triplet.add_triplet(1, 2, 4.0);
//! triplet.add_triplet(2, 0, 6.0);
//! triplet.add_triplet(3, 4, 3.0);
//! let sparse = triplet.to_csr();
//! let svd_model = SKTruncatedSVD::<f64>::build()
//!     .number_of_components(2)
//!     .random_seed(7)
//!     .build()
//!     .unwrap()
//!     .fit(sparse.view())
//!     .unwrap();
//! assert_eq!(svd_model.transform(sparse.view()).unwrap().dim(), (4, 2));
//! ```
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod incremental_pca;
pub mod pca;
pub mod truncated_svd;

pub use incremental_pca::{SKIncrementalPCA, SKIncrementalPCABuilder, SKIncrementalPCAModel};
pub use pca::{SKPCA, SKPCABuilder, SKPCAModel, SKPCASolver};
pub use truncated_svd::{
    SK_TRUNCATED_SVD_PARITY_TOLERANCE, SKTruncatedSVD, SKTruncatedSVDAlgorithm,
    SKTruncatedSVDBuilder, SKTruncatedSVDModel,
};

#[cfg(test)]
mod acceptance_tests;
