//! `sciencekit_pipeline` — compile-time typed linear pipelines, parallel
//! feature unions and validated runtime DAG pipelines (specs `typed-pipeline`,
//! `feature-union`, `pipeline-graph`).
//!
//! A linear pipeline is composed through [`SKPipeline::start`], parallel
//! feature branches through [`SKFeatureUnion::build`], and arbitrary branching
//! and merging directed acyclic graphs through [`SKPipelineGraph::start`].
//! Every composition layer exchanges [`SKFeatureBlock`] values without
//! densifying sparse CSR blocks.
//!
//! # Example
//!
//! ```
//! use ndarray::{Array1, Array2};
//! use sciencekit_common::SKBuilder;
//! use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
//! use sciencekit_pipeline::SKPipeline;
//! use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
//!
//! let features = Array2::from_shape_vec(
//!     (4, 2),
//!     vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
//! )
//! .unwrap();
//! let labels = Array1::from_vec(vec![0_i64, 0, 1, 1]);
//! let scaler = SKStandardScaler::build().build().unwrap();
//! let classifier = SKLogisticRegression::build().build().unwrap();
//! let pipeline = SKPipeline::start::<f64>()
//!     .then(scaler)
//!     .finalize_classifier(classifier)
//!     .unwrap()
//!     .fit(&features, labels.view())
//!     .unwrap();
//! assert_eq!(pipeline.number_of_stages(), 1);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod block;
pub mod feature_union;
pub mod graph;
pub mod pipeline;
mod stage;
mod stages;

pub use block::{SKFeatureBlock, SKIntoFeatureBlock};
pub use feature_union::{
    SKAppendUnionStage, SKBlockTransformer, SKBoxedFittedStage, SKFeatureUnion,
    SKFeatureUnionArity, SKFeatureUnionBuilder, SKFeatureUnionMemberError, SKFeatureUnionModel,
    SKFeatureUnionStages,
};
pub use graph::{
    SKBoxedFinalModel, SKNodeBlockStore, SKNodeIdentifier, SKPipelineGraph, SKPipelineGraphBuilder,
    SKPipelineGraphClassifier, SKPipelineGraphModel, SKPipelineGraphNodeError,
    SKPipelineGraphNodeKind, SKPipelineGraphRegressor,
};
pub use pipeline::{
    SKPipeline, SKPipelineBuilder, SKPipelineClassifier, SKPipelineClassifierBuilder,
    SKPipelineRegressor, SKPipelineRegressorBuilder,
};
pub use stage::SKPipelineTransformer;
pub use stages::SKConfiguredStages;

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod block_tests;
#[cfg(test)]
mod stage_tests;

#[cfg(test)]
mod streaming_coverage;
