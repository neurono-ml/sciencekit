//! Platt scaling and pairwise coupling probability calibration
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod cv_decisions;
mod platt;

pub use cv_decisions::sk_calibrate_binary_platt;
pub use platt::{SKPlattParameters, sk_fit_platt_sigmoid};

#[cfg(test)]
mod platt_tests;
