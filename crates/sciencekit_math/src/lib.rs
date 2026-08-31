//! `sciencekit_math` — computational kernels and the BLAS/LAPACK interface for
//! sciencekit (PRD §4, §6.2).
//!
//! This crate is the numeric substrate every downstream algorithm crate builds
//! on. It provides:
//!
//! - [`kernels`] — reusable higher-order operation kernels (`azip!`/`par_azip!`
//!   transforms, binary combines, axis reductions, in-place scaling).
//! - [`layout`] — memory-layout helpers (C/F-contiguity detection, forcing
//!   contiguity before hot loops).
//! - [`pairwise`] — zero-copy pairwise distance kernels (squared Euclidean,
//!   Manhattan, Cosine) with a SIMD-friendly hot path.
//! - [`sparse_ops`] — sparse (`sprs`) product kernels (sparse×dense,
//!   sparse×sparse) over zero-copy views.
//! - [`backend`] — the [`SKMathBackend`](backend::SKMathBackend) abstraction,
//!   [`SKFaerBackend`](backend::SKFaerBackend) (pure-Rust default) and the opt-in
//!   `blas-backend` path.
//!
//! **Naming:** public items follow PRD §3.4 — structs and traits are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Files stay at
//! or under 200 lines; companion `*_tests.rs` modules hold the TDD suites.
//!
//! **Safety:** this is the performance crate. `unsafe` is confined to the SIMD
//! and BLAS hot kernels (PRD §4) and never appears in the higher-order or
//! layout layers.

#![warn(missing_docs)]

pub mod backend;
pub mod kernels;
pub mod layout;
pub mod pairwise;
pub mod sparse_ops;

pub use backend::{SKFaerBackend, SKMathBackend, SKMatrixMultiplyBackend, sk_default_math_backend};
pub use kernels::{sk_axis_sum, sk_binary_combine, sk_elementwise_transform, sk_scale_in_place};
pub use layout::{sk_force_contiguous, sk_is_c_contiguous, sk_is_f_contiguous, sk_memory_layout};
pub use pairwise::{
    sk_cosine_distance_matrix, sk_euclidean_distance_matrix, sk_manhattan_distance_matrix,
    sk_squared_euclidean_distance_matrix,
};
pub use sparse_ops::{sk_csr_dense_product, sk_sparse_product};

pub mod prelude {
    //! Convenience re-exports for downstream algorithm crates.

    pub use crate::backend::{
        SKFaerBackend, SKMathBackend, SKMatrixMultiplyBackend, sk_default_math_backend,
    };
    pub use crate::layout::{
        sk_force_contiguous, sk_is_c_contiguous, sk_is_f_contiguous, sk_memory_layout,
    };
}
