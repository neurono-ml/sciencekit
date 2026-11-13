//! The stage adapter over the existing transformer contract.
//!
//! A pipeline stage is anything that can fit on a block and transform a block
//! with the fitted model. A blanket implementation reuses every estimator that
//! already honours [`SKUnsupervisedFit`] and returns a model implementing
//! [`SKFeatureTransformer`]: its output only has to enter the block medium
//! through [`SKIntoFeatureBlock`], which keeps the representation the
//! transformer produced.

use sciencekit_common::{
    SKError, SKFeatureTransformer, SKFloat, SKIntoFeatureBlock, SKUnsupervisedFit,
};

use crate::SKFeatureBlock;

/// One configured pipeline stage: a transformer plus the fitting rule the
/// pipeline drives.
///
/// `Fitted` is the stage's own model type. The pipeline stores a tuple of these
/// fitted models, so prediction never refits a stage.
pub trait SKPipelineTransformer<F: SKFloat> {
    /// The fitted stage model.
    type Fitted;

    /// Fit the stage on the current training block.
    fn fit_stage(&self, block: &SKFeatureBlock<F>) -> Result<Self::Fitted, SKError>;

    /// Transform a block with the fitted stage model, preserving whichever
    /// representation the stage produces.
    fn transform_stage(
        &self,
        fitted: &Self::Fitted,
        block: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError>;
}

/// Every unsupervised transformer is a pipeline stage.
///
/// The output conversion is the only requirement beyond the existing contracts:
/// a stage that cannot enter the block medium cannot be composed, and that
/// failure surfaces at compile time as an unsatisfied bound.
impl<F, T> SKPipelineTransformer<F> for T
where
    F: SKFloat,
    T: SKUnsupervisedFit<F>,
    T::Error: Into<SKError>,
    T::Model: SKFeatureTransformer<F>,
    <T::Model as SKFeatureTransformer<F>>::Output: SKIntoFeatureBlock<F>,
    <T::Model as SKFeatureTransformer<F>>::Error: Into<SKError>,
{
    type Fitted = T::Model;

    fn fit_stage(&self, block: &SKFeatureBlock<F>) -> Result<Self::Fitted, SKError> {
        self.fit(block).map_err(Into::into)
    }

    fn transform_stage(
        &self,
        fitted: &Self::Fitted,
        block: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, SKError> {
        fitted
            .transform(block)
            .map(SKIntoFeatureBlock::into_feature_block)
            .map_err(Into::into)
    }
}
