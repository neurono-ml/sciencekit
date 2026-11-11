//! Pure dispatcher for the Isolation Forest module (`isolation-forest`).

mod config;
mod score;
mod subsample;
mod tree;

pub use config::{
    SK_ISOLATION_AUTO_OFFSET, SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, SKContamination,
    SKIsolationMaximumSamples,
};

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod isolation_path_tests;
#[cfg(test)]
mod isolation_tree_tests;
