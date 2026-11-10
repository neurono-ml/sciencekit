//! Entry of the impute outputs into the owned pipeline block medium (spec
//! `typed-pipeline`).
//!
//! [`SKImputedOutput`] and [`SKIndicatorOutput`] are local types, so this crate
//! is the one allowed to implement [`SKIntoFeatureBlock`] for them. Each
//! conversion is a variant-to-variant move: the representation the transform
//! produced is the representation the pipeline receives, and a sparse output is
//! never densified at the boundary.

use sciencekit_common::{SKFeatureBlock, SKIntoFeatureBlock};

use crate::missing_indicator::SKIndicatorOutput;
use crate::simple_imputer::SKImputedOutput;

impl<F> SKIntoFeatureBlock<F> for SKImputedOutput<F> {
    fn into_feature_block(self) -> SKFeatureBlock<F> {
        match self {
            SKImputedOutput::Dense(matrix) => SKFeatureBlock::Dense(matrix),
            SKImputedOutput::Sparse(matrix) => SKFeatureBlock::Sparse(matrix),
        }
    }
}

impl<F> SKIntoFeatureBlock<F> for SKIndicatorOutput<F> {
    fn into_feature_block(self) -> SKFeatureBlock<F> {
        match self {
            SKIndicatorOutput::Dense(matrix) => SKFeatureBlock::Dense(matrix),
            SKIndicatorOutput::Sparse(matrix) => SKFeatureBlock::Sparse(matrix),
        }
    }
}
