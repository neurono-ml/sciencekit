//! Platt scaling and pairwise coupling probability calibration
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod coupling;
mod cross_validation;
mod platt;

pub use coupling::{
    sk_couple_pairwise_probabilities, sk_couple_pairwise_probabilities_with_budget,
};
pub use cross_validation::sk_calibrate_binary_platt;
pub use platt::{SKPlattParameters, sk_fit_platt_sigmoid};

#[cfg(test)]
mod coupling_tests;
#[cfg(test)]
mod platt_tests;
