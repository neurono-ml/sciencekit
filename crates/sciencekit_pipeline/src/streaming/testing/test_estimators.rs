//! Test-only batch finalizers.

use sciencekit_common::{
    SKDataView, SKError, SKFeatureBlock, SKFeatureTransformer, SKFloat, SKIntoFeatureBlock,
    SKUnsupervisedFit,
};

/// A finalizer that validates the accumulated block and learns its width.
pub(crate) struct SKTestIdentityEstimator;

impl<F: SKFloat> SKUnsupervisedFit<F> for SKTestIdentityEstimator {
    type Model = SKTestIdentityModel;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let columns = match view {
            SKDataView::Dense(matrix) => matrix.ncols(),
            SKDataView::Sparse(matrix) => matrix.cols(),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "unknown",
                    suggestion: "convert the features to a dense matrix or a CSR matrix",
                });
            }
        };
        Ok(SKTestIdentityModel { columns })
    }
}

/// The learned width of [`SKTestIdentityEstimator`].
pub(crate) struct SKTestIdentityModel {
    columns: usize,
}

impl<F: SKFloat> SKFeatureTransformer<F> for SKTestIdentityModel {
    type Output = SKTestIdentityOutput<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let block = SKFeatureBlock::from(view);
        if block.number_of_columns() != self.columns {
            return Err(SKError::shape_mismatch_2d(
                block.number_of_rows(),
                self.columns,
                block.number_of_rows(),
                block.number_of_columns(),
            ));
        }
        Ok(SKTestIdentityOutput(block))
    }
}

/// The representation-preserving output of [`SKTestIdentityModel`].
pub(crate) struct SKTestIdentityOutput<F>(SKFeatureBlock<F>);

impl<F> SKIntoFeatureBlock<F> for SKTestIdentityOutput<F> {
    fn into_feature_block(self) -> SKFeatureBlock<F> {
        self.0
    }
}
