//! Epsilon-Support-Vector Regressor (`SKSVR`)
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod builder;
mod core_implementation;

pub use builder::SKSVRBuilder;
pub use core_implementation::SKSVR;

#[cfg(test)]
mod svr_builder_tests;
