//! The `tree` family: CART decision trees.
//!
//! Every item here resolves to `sciencekit_tree`. `SKClassWeight` and
//! `SKMaximumFeatures` are the tree-owned definitions; the `ensemble` module
//! re-exports the same types so ensembles can be configured without depending
//! on this crate directly.

pub use sciencekit_tree::classifier::{
    SKDecisionTreeClassifier, SKDecisionTreeClassifierBuilder, SKDecisionTreeClassifierModel,
};
pub use sciencekit_tree::node::{SKTree, SKTreeNode};
pub use sciencekit_tree::regressor::{
    SKDecisionTreeRegressor, SKDecisionTreeRegressorBuilder, SKDecisionTreeRegressorModel,
};
pub use sciencekit_tree::structure::{
    SKStructureConstraints, SKStructureConstraintsBuilder, SKTreeStructure,
};
pub use sciencekit_tree::{SKClassWeight, SKTreeCriterion};
pub use sciencekit_tree::{SKMaximumFeatures, SKSplitCandidate, SKSplitter};
pub use sciencekit_tree::{classifier, export_helpers, impurity, node, regressor, structure};
