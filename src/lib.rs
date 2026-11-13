//! `sciencekit` — Machine Learning library in Rust reimplementing all of
//! scikit-learn, with extreme performance, zero-copy and native out-of-core.
//!
//! This is the umbrella crate: one dependency that re-exports the stable public
//! surface of every `sciencekit_*` sub-crate under documented module paths. It
//! adds **no logic** — every item resolves to the crate that owns it.
//!
//! # Getting started
//!
//! ```text
//! use ndarray::array;
//! use sciencekit::prelude::*;
//!
//! let features = array![[0.0_f64, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
//! let targets = array![0.5_f64, 1.5, 2.5, 3.5];
//!
//! let model = SKLinearRegression::build()
//!     .build()
//!     .unwrap()
//!     .fit(features.view(), targets.view())
//!     .unwrap();
//! let predictions = model.predict(features.view()).unwrap();
//!
//! assert_eq!(predictions.len(), features.nrows());
//! ```
//!
//! # Layout
//!
//! - [`math`] — kernels and the math backend abstraction.
//! - [`preprocessing`] — scaling, encoding and feature expansion.
//! - [`impute`] — missing-data transformers.
//! - [`linear_model`] — linear, regularized, logistic and SGD models.
//! - [`neighbors`] — nearest-neighbor search and estimators.
//! - [`svm`] — support-vector machines.
//! - [`tree`], [`ensemble`] and [`outlier`] — tree-based learning.
//! - [`cluster`] and [`decomposition`] — unsupervised learning.
//! - [`model_selection`] — splitting and cross-validation.
//! - [`pipeline`] — typed pipelines and pipeline graphs.
//! - [`metrics`] — scores and streaming accumulators.
//! - [`interop`] — dataset interop, model formats and ONNX.
//! - [`accelerators`] — device-compute contracts and the OpenCL backend.
//! - [`telemetry`] — opt-in OpenTelemetry export.
//! - [`ffi`] — the stable C ABI.
//!
//! [`prelude`] gathers the traits, types and estimators needed by typical code.
//!
//! # Documentation
//!
//! The full book — API, guides and the algorithms explained from scratch — lives
//! at <https://neurono-ml.github.io/sciencekit/>.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod accelerators;
pub mod cluster;
pub mod decomposition;
pub mod ensemble;
pub mod ffi;
pub mod impute;
pub mod interop;
pub mod linear_model;
pub mod math;
pub mod metrics;
pub mod model_selection;
pub mod neighbors;
pub mod outlier;
pub mod pipeline;
pub mod preprocessing;
pub mod svm;
pub mod telemetry;
pub mod tree;

// The shared vocabulary is re-exported at the root: a user reaching for
// `sciencekit::SKError` or `sciencekit::SKDataView` should not have to know
// which sub-crate owns the type.
pub use sciencekit_common::batching::{
    SKDataBatch, SKFeatureBlock, SKIntoFeatureBlock, SKLazySource, SKMappableSource,
    SKReopenableSource, SKSourceProvider, SKSupervisedBatch, SKSupervisedLazySource,
    SKSupervisedSourceProvider,
};
pub use sciencekit_common::builders::{SKBuilder, SKBuilderState, sk_validate_hyperparameter};
pub use sciencekit_common::data_view::{
    SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKDataView, SKNominalMatrixView,
};
pub use sciencekit_common::device::{
    SK_HOST_FALLBACK_BACKEND, SKDeviceCapabilities, SKDeviceContext, SKDeviceInfo,
    SKDevicePreference, SKFallbackReason, SKGpuKernel, sk_clear_device_inventory,
    sk_register_device_inventory, sk_registered_device_inventory, sk_resolve_operation_backend,
    sk_select_device,
};
pub use sciencekit_common::errors::SKError;
pub use sciencekit_common::execution::{SKExecutionContext, SKExecutionMode, SKExecutionPlan};
pub use sciencekit_common::fit_traits::{
    SKClassifierPredictor, SKClusterIdentifier, SKClusterLabel, SKClustererPredictor,
    SKFeatureTransformer, SKOutlierLabel, SKOutlierPredictor, SKRegressorPredictor,
    SKSupervisedFit, SKUnsupervisedFit, sk_validate_cluster_assignments,
};
pub use sciencekit_common::interop::{
    SK_MODEL_FORMAT_VERSION, SKExportable, SKImportable, SKModelMetadata, SKModelState,
    SKParameterValue, SKTensor, SKTensorDataType, SKTensorValues, sk_assert_export_round_trip,
    sk_assert_state_names, sk_backend_parameter, sk_boolean_parameter, sk_embed_state,
    sk_extract_state, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_name, sk_parse_backend_parameter, sk_read_boolean_tensor,
    sk_read_floating_tensor, sk_read_integer_tensor, sk_states_equal, sk_string_parameter,
    sk_validate_import_metadata,
};
pub use sciencekit_common::label_table::{SKLabelTable, sk_canonicalize_labels};
pub use sciencekit_common::observability::{
    SKBackendKind, SKOperationAttributes, SKOperationKind, SKOperationObservation, sk_run_operation,
};
pub use sciencekit_common::random::{
    SK_RANDOM_DOMAIN_FOLD, SK_RANDOM_DOMAIN_INITIALIZATION, SK_RANDOM_DOMAIN_PROJECTION,
    SK_RANDOM_DOMAIN_STAGE, SK_RANDOM_DOMAIN_TREE, SKRandomGenerator, sk_derive_seed,
};
pub use sciencekit_common::scorer_traits::{
    SKLabelScorer, SKProbabilityScorer, SKSupervisedScorer, SKUnsupervisedScorer,
    sk_validate_probability_shape,
};
pub use sciencekit_common::sk_float::SKFloat;
pub use sciencekit_common::target_view::SKTargetView;

#[cfg(test)]
mod export_coverage_tests;
#[cfg(test)]
mod reexports_tests;
