//! Pure-Rust Sequential Minimal Optimization (SMO) solver substrate
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod gradient;
mod working_set;

pub use gradient::SKSmoState;
pub use working_set::{
    SKSmoWorkingPair, sk_select_first_working_index, sk_select_working_pair,
};

#[cfg(test)]
mod working_set_tests;
