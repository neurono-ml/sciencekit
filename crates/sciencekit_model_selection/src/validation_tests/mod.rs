//! Cross-cutting validation tests (spec `model-selection-splits`, requirement
//! `Validation`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules; the
//! scenarios live in their own files.

mod finiteness_tests;
mod partition_tests;
