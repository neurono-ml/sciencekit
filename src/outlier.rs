//! The `outlier` family: isolation forests.
//!
//! Every item here resolves to `sciencekit_outlier`. `SKOutlierLabel` and
//! `SKOutlierPredictor` are owned by `sciencekit_common` and re-exported here
//! too, so they are the same types as the ones at the umbrella root.

pub use sciencekit_outlier::isolation_forest;
pub use sciencekit_outlier::{
    SK_ISOLATION_AUTO_OFFSET, SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, SKContamination,
    SKIsolationForest, SKIsolationForestBuilder, SKIsolationForestModel, SKIsolationMaximumSamples,
    SKOutlierLabel, SKOutlierPredictor,
};
