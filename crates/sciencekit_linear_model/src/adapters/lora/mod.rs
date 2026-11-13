//! Low-rank adaptation of a frozen fitted model (spec `lora-adapters`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder;
mod config;
mod core_implementation;

pub use builder::SKLoraAdapterBuilder;
pub use config::SKLoraTargetModule;
pub use core_implementation::SKLoraAdapter;

#[cfg(test)]
mod lora_adapter_tests;
