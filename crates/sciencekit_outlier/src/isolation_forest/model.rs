//! Fitted `SKIsolationForestModel` state and inspection accessors.

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};
use sciencekit_tree::SKTree;

/// Fitted Isolation Forest model holding the isolation-tree ensemble and score offset.
#[derive(Debug, Clone)]
pub struct SKIsolationForestModel<F> {
    pub(super) trees: Vec<SKTree<F>>,
    pub(super) subsample_size: usize,
    pub(super) feature_count: usize,
    pub(super) offset: F,
    pub(super) backend: SKBackendKind,
    pub(super) execution_mode: SKExecutionMode,
}

impl<F: SKFloat> SKIsolationForestModel<F> {
    /// Number of fitted isolation trees in the ensemble.
    pub fn number_of_trees(&self) -> usize {
        self.trees.len()
    }

    /// Number of input features seen during fitting.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Resolved per-tree subsample size `n` used to normalize path lengths via `c(n)`.
    pub fn subsample_size(&self) -> usize {
        self.subsample_size
    }

    /// Decision threshold offset subtracted from `score_samples`.
    pub fn offset(&self) -> F {
        self.offset
    }

    /// Borrows the underlying `SKTree` at `tree_index`, if in bounds.
    pub fn tree(&self, tree_index: usize) -> Option<&SKTree<F>> {
        self.trees.get(tree_index)
    }

    /// Compute backend transferred from the configured estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent transferred from the configured estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }
}
