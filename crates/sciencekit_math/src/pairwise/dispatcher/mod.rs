//! Pairwise dispatch (spec `pairwise-distances`).
//!
//! This is a **pure dispatcher** module: `mod.rs` declares and re-exports the
//! metric catalogue, the single public entry point and the internal routing
//! helpers, while every implementation lives in its own file.

mod metric;
mod routing;

pub use metric::{SKPairwiseMetric, sk_pairwise_distance_matrix};
