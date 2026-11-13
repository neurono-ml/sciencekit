//! Low-rank adaptation of a frozen fitted model (spec `lora-adapters`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod attached_model;
mod base_model;
mod builder;
mod config;
mod core_implementation;
mod fitting;
mod prediction;

pub use attached_model::SKLoraAttachedModel;
pub use base_model::SKLoraBaseModel;
pub use builder::SKLoraAdapterBuilder;
pub use config::SKLoraTargetModule;
pub use core_implementation::SKLoraAdapter;

#[cfg(test)]
mod lora_adapter_tests;
#[cfg(test)]
mod lora_attach_tests;
#[cfg(test)]
mod lora_fit_tests;
