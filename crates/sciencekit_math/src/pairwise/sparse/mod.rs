//! Sparse pairwise-distance kernels (spec `pairwise-distances`).
//!
//! The kernels consume borrowed `sprs` CSR views and return owned dense
//! distance matrices. They never densify an input: squared Euclidean, Cosine,
//! Manhattan and sparse×sparse Chebyshev walk only the stored entries, while
//! sparse×dense Chebyshev scans the dense rows (documented as the one
//! non-`O(nnz)` path).
//!
//! The `mod.rs` files in this folder only declare and re-export submodules;
//! every implementation lives in its own file.

mod chebyshev;
mod cosine;
mod manhattan;
mod parallel;
mod row_merge;
mod row_statistics;
mod squared_euclidean;

pub use chebyshev::sk_sparse_chebyshev_distance_matrix;
pub use cosine::sk_sparse_cosine_distance_matrix;
pub use manhattan::sk_sparse_manhattan_distance_matrix;
pub use squared_euclidean::{
    sk_sparse_euclidean_distance_matrix, sk_sparse_squared_euclidean_distance_matrix,
};

pub(crate) use chebyshev::{
    sk_dense_sparse_chebyshev_distance_matrix, sk_sparse_dense_chebyshev_distance_matrix,
};
pub(crate) use cosine::{
    sk_dense_sparse_cosine_distance_matrix, sk_sparse_dense_cosine_distance_matrix,
};
pub(crate) use manhattan::{
    sk_dense_sparse_manhattan_distance_matrix, sk_sparse_dense_manhattan_distance_matrix,
};
pub(crate) use squared_euclidean::{
    sk_dense_sparse_euclidean_distance_matrix, sk_dense_sparse_squared_euclidean_distance_matrix,
    sk_sparse_dense_euclidean_distance_matrix, sk_sparse_dense_squared_euclidean_distance_matrix,
};

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod chebyshev_tests;
#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
mod cosine_tests;
#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
mod manhattan_tests;
#[cfg(test)]
mod mixed_tests;
#[cfg(test)]
mod parallel_tests;
#[cfg(test)]
mod row_merge_tests;
#[cfg(test)]
mod squared_euclidean_tests;
