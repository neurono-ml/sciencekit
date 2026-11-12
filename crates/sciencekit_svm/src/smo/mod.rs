//! Pure-Rust Sequential Minimal Optimization (SMO) solver substrate
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod cache;
mod gradient;
mod problems;
mod shrinking;
mod solver;
mod step;
mod stopping;
mod working_set;

pub use cache::SKKernelRowCache;
pub use gradient::SKSmoState;
pub use problems::{
    SKBinarySvcSmoResult, SKEpsilonSvrSmoResult, sk_solve_binary_c_svc, sk_solve_epsilon_svr,
};
pub use shrinking::{sk_reconstruct_full_gradients, sk_shrink_active_set};
pub use solver::{SKSmoOptions, sk_solve_smo_quadratic_program};
pub use stopping::{SKSmoSolution, SKSmoTracker, sk_maximal_kkt_violation, sk_recover_smo_bias};
pub use working_set::{SKSmoWorkingPair, sk_select_first_working_index, sk_select_working_pair};

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod stopping_tests;
#[cfg(test)]
mod svc_loop_tests;
#[cfg(test)]
mod svr_loop_tests;
#[cfg(test)]
mod working_set_tests;
