//! Pure-Rust Sequential Minimal Optimization (SMO) solver substrate
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod cache;
mod gradient;
mod stopping;
mod working_set;

pub use cache::SKKernelRowCache;
pub use gradient::SKSmoState;
pub use stopping::{
    SKSmoSolution, SKSmoTracker, sk_maximal_kkt_violation, sk_recover_smo_bias,
};
pub use working_set::{
    SKSmoWorkingPair, sk_select_first_working_index, sk_select_working_pair,
};

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod stopping_tests;
#[cfg(test)]
mod working_set_tests;
