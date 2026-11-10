//! The brute-force exact searcher (spec `nearest-neighbor-estimators`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod k_nearest;
mod radius;
mod searcher;

pub use searcher::SKBruteForceSearcher;

pub(crate) use k_nearest::sk_brute_force_k_nearest;
#[cfg(test)]
pub(crate) use k_nearest::sk_brute_force_k_nearest_with_budget;

#[cfg(test)]
mod brute_force_k_nearest_tests;
#[cfg(test)]
mod brute_force_radius_tests;
#[cfg(test)]
mod brute_force_validation_tests;
