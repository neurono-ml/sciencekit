//! The `pipeline` family: typed pipelines, feature unions and pipeline graphs.
//!
//! Every item here resolves to `sciencekit_pipeline`. `SKFeatureBlock` and
//! `SKIntoFeatureBlock` are owned by `sciencekit_common`; the pipeline creates
//! no second block type, and the umbrella re-exports the same identity here and
//! at the root.

pub use sciencekit_pipeline::feature_union::{
    SKAppendUnionStage, SKBlockTransformer, SKBoxedFittedStage, SKFeatureUnion,
    SKFeatureUnionArity, SKFeatureUnionBuilder, SKFeatureUnionMemberError, SKFeatureUnionModel,
    SKFeatureUnionStages,
};
pub use sciencekit_pipeline::graph::{
    SKBoxedFinalModel, SKNodeBlockStore, SKNodeIdentifier, SKPipelineGraph, SKPipelineGraphBuilder,
    SKPipelineGraphClassifier, SKPipelineGraphModel, SKPipelineGraphNodeError,
    SKPipelineGraphNodeKind, SKPipelineGraphRegressor,
};
pub use sciencekit_pipeline::pipeline::{
    SKPipeline, SKPipelineBuilder, SKPipelineClassifier, SKPipelineClassifierBuilder,
    SKPipelineRegressor, SKPipelineRegressorBuilder,
};
pub use sciencekit_pipeline::streaming::{
    SKFittedStreamingPipeline, SKStreamingFitReport, SKStreamingPipeline,
    SKStreamingPipelineBuilder, SKStreamingPipelineConfigured, SKStreamingTransformer,
};
pub use sciencekit_pipeline::{
    SKConfiguredStages, SKFeatureBlock, SKIntoFeatureBlock, SKPipelineTransformer,
};
pub use sciencekit_pipeline::{export, feature_union, graph, pipeline, streaming};
