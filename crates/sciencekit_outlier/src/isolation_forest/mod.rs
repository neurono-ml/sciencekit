//! Pure dispatcher for the Isolation Forest module (`isolation-forest`).

mod config;

pub use config::{
    SK_ISOLATION_AUTO_OFFSET, SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, SKContamination,
    SKIsolationMaximumSamples,
};

#[cfg(test)]
mod config_tests;
