//! Node identifiers, node kinds and graph payload variants for
//! [`super::SKPipelineGraph`] (spec `pipeline-graph`).

use std::fmt;

use sciencekit_common::SKFloat;

use crate::SKBlockTransformer;

use super::final_estimator::GraphFinalEstimator;

/// Opaque zero-based identifier of a node inside [`super::SKPipelineGraphBuilder`]
/// and [`super::SKPipelineGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SKNodeIdentifier(usize);

impl SKNodeIdentifier {
    pub(super) const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Zero-based index of this node in insertion order.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for SKNodeIdentifier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "node {}", self.0)
    }
}

/// Kind of stage stored at a node of [`super::SKPipelineGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKPipelineGraphNodeKind {
    /// An unsupervised block transformer stage.
    Transformer,
    /// A parallel [`crate::SKFeatureUnion`] stage.
    FeatureUnion,
    /// An explicit horizontal concatenation merge stage.
    Merge,
    /// The unique supervised final estimator sink stage.
    FinalEstimator,
}

impl SKPipelineGraphNodeKind {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Transformer => "transformer",
            Self::FeatureUnion => "feature_union",
            Self::Merge => "merge",
            Self::FinalEstimator => "final_estimator",
        }
    }
}

pub(super) enum GraphNodePayload<F: SKFloat> {
    Transformer(Box<dyn SKBlockTransformer<F>>),
    FeatureUnion(Box<dyn SKBlockTransformer<F>>),
    Merge,
    FinalEstimator(Box<dyn GraphFinalEstimator<F>>),
}

impl<F: SKFloat> GraphNodePayload<F> {
    pub(super) fn kind(&self) -> SKPipelineGraphNodeKind {
        match self {
            Self::Transformer(_) => SKPipelineGraphNodeKind::Transformer,
            Self::FeatureUnion(_) => SKPipelineGraphNodeKind::FeatureUnion,
            Self::Merge => SKPipelineGraphNodeKind::Merge,
            Self::FinalEstimator(_) => SKPipelineGraphNodeKind::FinalEstimator,
        }
    }
}
