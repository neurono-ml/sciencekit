//! `sciencekit_common` — core traits, types and errors shared across the
//! `sciencekit` sub-crates (PRD §3.3).
//!
//! This crate freezes the vocabulary every future crate compiles against:
//! scalar typing (`sk_float`), the central error taxonomy (`errors`), the
//! data/target boundary (`data_view`, `target_view`), label canonicalization
//! (`label_table`), fit/transformation contracts (`fit_traits`), scoring
//! contracts (`scorer_traits`), execution planning (`execution`) and streaming
//! batches (`batching`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod batching;
pub mod data_view;
pub mod errors;
pub mod execution;
pub mod fit_traits;
pub mod label_table;
pub mod scorer_traits;
pub mod sk_float;
pub mod target_view;

pub use batching::{SKDataBatch, SKLazySource, SKMappableSource};
pub use data_view::SKDataView;
pub use errors::SKError;
pub use execution::{SKExecutionContext, SKExecutionMode, SKExecutionPlan};
pub use fit_traits::{SKFeatureTransformer, SKPredictor, SKSupervisedFit, SKUnsupervisedFit};
pub use label_table::SKLabelTable;
pub use scorer_traits::{SKSupervisedScorer, SKUnsupervisedScorer};
pub use sk_float::SKFloat;
pub use target_view::SKTargetView;
