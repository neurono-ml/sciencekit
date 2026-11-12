//! A representation-preserving test streaming transformer.

use sciencekit_common::execution::{
    SKStreamDecision, SKStreamingPassPolicy, sk_run_streaming_provider,
};
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat, SKSourceProvider};

use super::test_execution_plan::sk_test_execution_plan;
use crate::streaming::SKStreamingTransformer;

pub(crate) struct SKTestColumnScaling<F> {
    factor: F,
}

impl<F: SKFloat> SKTestColumnScaling<F> {
    pub(crate) fn new(factor: F) -> Self {
        SKTestColumnScaling { factor }
    }
}

/// The learned state of [`SKTestColumnScaling`]: the observed row count.
pub(crate) struct SKTestColumnScalingModel<F> {
    factor: F,
    _number_of_rows_seen: usize,
}

impl<F: SKFloat> SKStreamingTransformer<F> for SKTestColumnScaling<F> {
    type Model = SKTestColumnScalingModel<F>;
    type Error = SKError;

    fn fit_streaming<S: SKSourceProvider<F>>(
        &self,
        source: &mut S,
        epochs: usize,
    ) -> Result<Self::Model, Self::Error> {
        if epochs == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "epochs",
                reason: "a streaming fit requires at least one epoch".to_string(),
            });
        }
        let mut rows_seen = 0;
        sk_run_streaming_provider(
            source,
            &mut rows_seen,
            |batch, _parallelism, state| {
                *state += batch.number_of_rows();
                SKStreamDecision::Continue
            },
            &sk_test_execution_plan(),
            SKStreamingPassPolicy::Iterative {
                maximum_number_of_passes: epochs,
            },
        )?;
        Ok(SKTestColumnScalingModel {
            factor: self.factor,
            _number_of_rows_seen: rows_seen,
        })
    }

    fn transform_batch(
        &self,
        model: &Self::Model,
        batch: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, Self::Error> {
        let factor = model.factor;
        match batch {
            SKFeatureBlock::Dense(matrix) => {
                Ok(SKFeatureBlock::Dense(matrix.mapv(|value| value * factor)))
            }
            SKFeatureBlock::Sparse(matrix) => {
                Ok(SKFeatureBlock::Sparse(matrix.map(|value| *value * factor)))
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            }),
        }
    }
}
