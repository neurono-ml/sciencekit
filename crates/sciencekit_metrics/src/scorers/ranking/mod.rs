//! Ranking and probability scorer adapters.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod auc;
mod average_precision;
mod column;
mod label;
mod log_loss;

pub use auc::SKAucScorer;
pub use average_precision::SKAveragePrecisionScorer;
pub use label::{SKBalancedAccuracyScorer, SKMatthewsCorrelationScorer};
pub use log_loss::SKLogLossScorer;
