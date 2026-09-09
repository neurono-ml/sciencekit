//! Zero-copy pairwise-distance kernels (spec `pairwise-distances`).
//!
//! Distances are computed between **rows** of two feature matrices using the
//! SIMD-friendly norm-expansion for squared Euclidean:
//! `‖a−b‖² = ‖a‖² − 2 a·b + ‖b‖²`. This avoids a per-pair subtract loop and
//! lets the inner product run over contiguous row memory.
//!
//! All kernels take `&ArrayView2<F>` inputs and return owned `Array2<F>`
//! outputs; they are pure and thread-safe.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod cosine;
mod manhattan;
mod simd_dot;
mod squared_euclidean;

pub use cosine::sk_cosine_distance_matrix;
pub use manhattan::sk_manhattan_distance_matrix;
pub use squared_euclidean::{sk_euclidean_distance_matrix, sk_squared_euclidean_distance_matrix};

#[cfg(test)]
mod pairwise_tests;
