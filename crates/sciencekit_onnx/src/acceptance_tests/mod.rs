//! Acceptance helpers shared by the exporter tests.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod fixtures;
mod interface;

pub(crate) use fixtures::write_fixture;
pub(crate) use interface::{assert_classifier_interface, assert_regressor_interface};

#[cfg(test)]
mod identity_tests;
