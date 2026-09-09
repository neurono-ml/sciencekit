//! Sparse (`sprs`) product kernels (spec `dense-sparse-products`, PRD §4.5).
//!
//! These kernels consume zero-copy views ([`CsMatView`]) and never densify the
//! sparse operand: the CSR×dense product walks the sparse structure directly,
//! while sparse×sparse delegates to sprs' SMMP product.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod csr_dense;
mod sparse_sparse;

pub use csr_dense::sk_csr_dense_product;
pub use sparse_sparse::sk_sparse_product;

#[cfg(test)]
mod sparse_ops_tests;
