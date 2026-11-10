//! `sciencekit_common` — core traits, types and errors shared across the
//! `sciencekit` sub-crates (PRD §3.3).
//!
//! This crate freezes the vocabulary every future crate compiles against:
//! scalar typing (`sk_float`), the central error taxonomy (`errors`), the
//! data/target boundary (`data_view`, `target_view`), label canonicalization
//! (`label_table`), fit/transformation contracts (`fit_traits`), scoring
//! contracts (`scorer_traits`), execution planning (`execution`), streaming
//! batches (`batching`), the mandatory builder foundation (`builders`), the
//! automatic execution decision + observability (`observability`), the
//! format-agnostic model export/import contracts (`interop`) and the optional
//! global-allocator selection (`allocator`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod allocator;
pub mod batching;
pub mod builders;
pub mod data_view;
pub mod errors;
pub mod execution;
pub mod fit_traits;
pub mod interop;
pub mod label_table;
pub mod observability;
pub mod random;
pub mod scorer_traits;
pub mod sk_float;
pub mod target_view;

pub use batching::{
    SKDataBatch, SKFeatureBlock, SKIntoFeatureBlock, SKLazySource, SKMappableSource,
    SKReopenableSource, SKSourceProvider, SKSupervisedBatch, SKSupervisedLazySource,
    SKSupervisedSourceProvider,
};
pub use builders::{SKBuilder, SKBuilderState, sk_validate_hyperparameter};
pub use data_view::{
    SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKDataView, SKNominalMatrixView,
};
pub use errors::SKError;
pub use execution::{SKExecutionContext, SKExecutionMode, SKExecutionPlan};
pub use fit_traits::{
    SKClassifierPredictor, SKClusterIdentifier, SKClusterLabel, SKClustererPredictor,
    SKFeatureTransformer, SKRegressorPredictor, SKSupervisedFit, SKUnsupervisedFit,
    sk_validate_cluster_assignments,
};
pub use interop::{
    SKExportable, SKImportable, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    SKTensorDataType, SKTensorValues, sk_validate_import_metadata,
};
pub use label_table::{SKLabelTable, sk_canonicalize_labels};
pub use observability::{
    SKBackendKind, SKOperationAttributes, SKOperationKind, SKOperationObservation, sk_run_operation,
};
pub use random::{
    SK_RANDOM_DOMAIN_FOLD, SK_RANDOM_DOMAIN_INITIALIZATION, SK_RANDOM_DOMAIN_PROJECTION,
    SK_RANDOM_DOMAIN_STAGE, SK_RANDOM_DOMAIN_TREE, SKRandomGenerator, sk_derive_seed,
};
pub use scorer_traits::{
    SKLabelScorer, SKProbabilityScorer, SKSupervisedScorer, SKUnsupervisedScorer,
    sk_validate_probability_shape,
};
pub use sk_float::SKFloat;
pub use target_view::SKTargetView;
