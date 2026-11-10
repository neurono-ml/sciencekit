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

#[cfg(test)]
mod row_merge_tests;
