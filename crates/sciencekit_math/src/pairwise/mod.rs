//! Zero-copy pairwise-distance kernels (spec `pairwise-distances`).
//!
//! Distances are computed between **rows** of two feature matrices using the
//! SIMD-friendly norm-expansion for squared Euclidean:
//! `‖a−b‖² = ‖a‖² − 2 a·b + ‖b‖²`. This avoids a per-pair subtract loop and
//! lets the inner product run over contiguous row memory.
//!
//! Dense kernels take `&ArrayView2<F>`, sparse kernels take borrowed `CsMatView`
//! inputs, and [`SKPairwiseMetric`] / [`sk_pairwise_distance_matrix`] select the
//! right kernel for any of the four dense/sparse combinations. Every kernel is
//! pure and thread-safe; no sparse input is densified.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod chebyshev;
mod cosine;
mod dispatcher;
mod manhattan;
mod nan_euclidean;
mod simd_dot;
mod sparse;
mod squared_euclidean;

pub use chebyshev::sk_chebyshev_distance_matrix;
pub use cosine::sk_cosine_distance_matrix;
pub use dispatcher::{SKPairwiseMetric, sk_pairwise_distance_matrix};
pub use manhattan::sk_manhattan_distance_matrix;
pub use nan_euclidean::sk_nan_euclidean_distance_matrix;
pub use sparse::{
    sk_sparse_chebyshev_distance_matrix, sk_sparse_cosine_distance_matrix,
    sk_sparse_euclidean_distance_matrix, sk_sparse_manhattan_distance_matrix,
    sk_sparse_squared_euclidean_distance_matrix,
};
pub(crate) use simd_dot::dot_product_simd;
pub(crate) use sparse::sk_sparse_dot_product;
pub use squared_euclidean::{sk_euclidean_distance_matrix, sk_squared_euclidean_distance_matrix};

#[cfg(test)]
mod chebyshev_tests;
#[cfg(test)]
mod dispatcher_representation_tests;
#[cfg(test)]
mod dispatcher_tests;
#[cfg(test)]
mod nan_euclidean_tests;
#[cfg(test)]
mod pairwise_tests;
