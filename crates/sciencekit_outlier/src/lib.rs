//! `sciencekit_outlier` — anomaly and outlier detection for `sciencekit`
//! (specs `outlier-prediction` and `isolation-forest`).
//!
//! The crate ships [`SKIsolationForest`], constructed exclusively through
//! [`SKIsolationForestBuilder`]. Fitting grows a deterministic ensemble of
//! random isolation trees over seeded subsamples using the shared
//! `sciencekit_tree::SKTree` node layout, without densifying sparse CSR inputs.
//!
//! # Example
//!
//! ```
//! use ndarray::array;
//! use sciencekit_outlier::{SKIsolationForest, SKOutlierLabel};
//!
//! let features = array![
//!     [0.0_f64, 0.1],
//!     [0.1, -0.1],
//!     [-0.1, 0.0],
//!     [0.05, 0.05],
//!     [25.0, -30.0]
//! ];
//! let model = SKIsolationForest::<f64>::build()
//!     .number_of_trees(64)
//!     .random_seed(42)
//!     .build()
//!     .unwrap()
//!     .fit(&features)
//!     .unwrap();
//! let labels = model.predict(&features).unwrap();
//! assert_eq!(labels[4], SKOutlierLabel::Outlier);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod isolation_forest;

pub use isolation_forest::{
    SK_ISOLATION_AUTO_OFFSET, SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES, SKContamination,
    SKIsolationForest, SKIsolationForestBuilder, SKIsolationForestModel, SKIsolationMaximumSamples,
};
pub use sciencekit_common::{SKOutlierLabel, SKOutlierPredictor};

#[cfg(test)]
mod acceptance_tests;
