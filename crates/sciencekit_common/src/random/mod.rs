//! Seeded randomness shared by every stochastic estimator (spec
//! `deterministic-randomness`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and
//! re-exports submodules; the implementations live in their own files.
//!
//! # Reproducibility scope
//!
//! The generator is bit-deterministic for the same library version and seed:
//! it uses only wrapping integer arithmetic, so the raw stream is identical on
//! every platform. Floating-point draws (`next_f64`, `uniform`, `normal`) rely
//! on IEEE-754 basics plus `sqrt`/`ln`/`cos`; the same build and platform
//! reproduce them exactly, while cross-platform or cross-backend results are
//! tolerated only within the estimator's documented tolerance. Estimators
//! additionally require the same input and backend for identical results.

mod generator;
mod sub_seed;

pub use generator::SKRandomGenerator;
pub use sub_seed::{
    SK_RANDOM_DOMAIN_FOLD, SK_RANDOM_DOMAIN_INITIALIZATION, SK_RANDOM_DOMAIN_PROJECTION,
    SK_RANDOM_DOMAIN_STAGE, SK_RANDOM_DOMAIN_TREE, sk_derive_seed,
};

#[cfg(test)]
mod generator_tests;
#[cfg(test)]
mod sub_seed_tests;
