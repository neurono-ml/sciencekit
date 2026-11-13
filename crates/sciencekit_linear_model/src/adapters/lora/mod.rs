//! Low-rank adaptation of a frozen fitted model (spec `lora-adapters`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod artifact;
mod attached_model;
mod base_model;
mod builder;
mod config;
mod core_implementation;
mod fitting;
mod merge;
mod prediction;
mod registry;
mod state;

pub use artifact::SKLoraAdapterArtifact;
pub use attached_model::SKLoraAttachedModel;
pub use base_model::SKLoraBaseModel;
pub use builder::SKLoraAdapterBuilder;
pub use config::SKLoraTargetModule;
pub use core_implementation::SKLoraAdapter;
pub use merge::SKLoraMergedModel;
pub use registry::SKLoraAdapterRegistry;
pub use state::{
    SK_LORA_ESTIMATOR_NAME, SK_LORA_FORMAT_VERSION, sk_lora_a_tensor_name, sk_lora_b_tensor_name,
};

#[cfg(test)]
mod lora_acceptance_tests;
#[cfg(test)]
mod lora_adapter_tests;
#[cfg(test)]
mod lora_attach_tests;
#[cfg(test)]
mod lora_fit_tests;
#[cfg(test)]
mod lora_logistic_tests;
#[cfg(test)]
mod lora_merge_tests;
#[cfg(test)]
mod lora_predict_tests;
#[cfg(test)]
mod lora_state_tests;
#[cfg(test)]
mod lora_validation_tests;
