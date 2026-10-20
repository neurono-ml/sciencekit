//! Incremental fit state (spec `simple-imputer`).
//!
//! Pure dispatcher module: declares and re-exports its submodules; every
//! implementation lives in its own file under this folder.

mod core_implementation;
mod extend;

pub(in crate::simple_imputer) use core_implementation::SKImputationState;

#[cfg(test)]
mod state_tests;
