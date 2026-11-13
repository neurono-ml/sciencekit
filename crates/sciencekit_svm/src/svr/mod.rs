//! Epsilon-Support-Vector Regressor (`SKSVR`)
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod builder;
mod core_implementation;
mod fit;
mod model;
mod predict;

pub use builder::SKSVRBuilder;
pub use core_implementation::SKSVR;
pub use model::SKSVRModel;

#[cfg(test)]
mod svr_builder_tests;
#[cfg(test)]
mod svr_tests;
