//! Pure dispatcher for the Isolation Forest module (`isolation-forest`).

mod builder;
mod config;
mod core_implementation;
mod score;
mod sparse;
mod subsample;
mod tree;

pub use builder::SKIsolationForestBuilder;
pub use config::{
    SK_ISOLATION_AUTO_OFFSET, SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, SKContamination,
    SKIsolationMaximumSamples,
};
pub use core_implementation::SKIsolationForest;

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod isolation_builder_tests;
#[cfg(test)]
mod isolation_path_tests;
#[cfg(test)]
mod isolation_representation_tests;
#[cfg(test)]
mod isolation_tree_tests;
