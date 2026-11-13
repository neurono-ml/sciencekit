//! The pipeline medium: the owned dense/sparse block that flows between stages.
//!
//! [`SKFeatureBlock`] is re-exported from `sciencekit_common`, where the
//! streaming regime already uses it. The pipeline adds no representation of its
//! own: a stage receives a block, transforms it according to its own contract
//! and returns a block, so a dense or sparse representation flows through the
//! chain without an implicit densification step.

pub use sciencekit_common::{SKFeatureBlock, SKIntoFeatureBlock};
