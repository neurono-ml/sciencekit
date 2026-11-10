//! Sparse pairwise-distance kernels (spec `pairwise-distances`).
//!
//! The kernels consume borrowed `sprs` CSR views and return owned dense
//! distance matrices. They never densify an input: squared Euclidean, Cosine,
//! Manhattan and sparse×sparse Chebyshev walk only the stored entries, while
//! sparse×dense Chebyshev scans the dense rows (documented as the one
//! non-`O(nnz)` path).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod row_merge;
mod row_statistics;
mod squared_euclidean;

pub use squared_euclidean::{
    sk_sparse_euclidean_distance_matrix, sk_sparse_squared_euclidean_distance_matrix,
};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod row_merge_tests;
#[cfg(test)]
mod squared_euclidean_tests;
