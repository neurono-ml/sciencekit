//! Test-only streaming transformers.
//!
//! [`SKTestMeanCentering`] learns a running mean and centers dense batches — the
//! hand-checkable stage for the fit and transform tests. [`SKTestColumnScaling`]
//! multiplies every stored value by a constant, which lets a sparse batch stay
//! CSR. Both drive their fit through the shared prefetch executor exactly like
//! the production streaming estimators.

use std::marker::PhantomData;

use ndarray::{Array1, Axis};
use sciencekit_common::execution::{
    SKStreamDecision, SKStreamingPassPolicy, sk_run_streaming_provider,
};
use sciencekit_common::{
    SKDataView, SKError, SKExecutionPlan, SKFeatureBlock, SKFloat, SKSourceProvider,
};

use crate::streaming::SKStreamingTransformer;

fn sk_test_execution_plan() -> SKExecutionPlan {
    SKExecutionPlan {
        mode: sciencekit_common::SKExecutionMode::OutOfCoreStreaming,
        parallelism: 1,
        batch_size: None,
        buffer_depth: 1,
        available_memory_bytes: u64::MAX,
        backend: sciencekit_common::SKBackendKind::Faer,
        fallback_reason: None,
    }
}

/// A dense-only streaming transformer that learns the column means.
pub(crate) struct SKTestMeanCentering<F> {
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKTestMeanCentering<F> {
    pub(crate) fn new() -> Self {
        SKTestMeanCentering {
            marker: PhantomData,
        }
    }
}

/// The learned column means of [`SKTestMeanCentering`].
pub(crate) struct SKTestMeanCenteringModel<F> {
    means: Array1<F>,
}

struct SKMeanAccumulator<F> {
    sums: Array1<F>,
    rows_count: usize,
    columns_count: usize,
    error: Option<SKError>,
}

impl<F: SKFloat> SKStreamingTransformer<F> for SKTestMeanCentering<F> {
    type Model = SKTestMeanCenteringModel<F>;
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
        let mut state = SKMeanAccumulator::<F> {
            sums: Array1::zeros(0),
            rows_count: 0,
            columns_count: 0,
            error: None,
        };
        sk_run_streaming_provider(
            source,
            &mut state,
            |batch, _parallelism, state| {
                let view = batch.view();
                let (rows, columns) = match view {
                    SKDataView::Dense(matrix) => matrix.dim(),
                    SKDataView::Sparse(matrix) => (matrix.rows(), matrix.cols()),
                    _ => {
                        state.error = Some(SKError::UnsupportedRepresentation {
                            representation: "unknown",
                            suggestion: "convert the features to a dense matrix or a CSR matrix",
                        });
                        return SKStreamDecision::Stop;
                    }
                };
                if state.columns_count == 0 {
                    state.columns_count = columns;
                    state.sums = Array1::zeros(columns);
                }
                if state.columns_count != columns {
                    state.error = Some(SKError::shape_mismatch_2d(
                        rows,
                        state.columns_count,
                        rows,
                        columns,
                    ));
                    return SKStreamDecision::Stop;
                }
                match view {
                    SKDataView::Dense(matrix) => {
                        state.sums = &state.sums + &matrix.sum_axis(Axis(0))
                    }
                    SKDataView::Sparse(matrix) => {
                        for row in matrix.outer_iterator() {
                            for (&column, &value) in row.indices().iter().zip(row.data()) {
                                state.sums[column] = state.sums[column] + value;
                            }
                        }
                    }
                    _ => unreachable!("the representation check rejected every other variant"),
                }
                state.rows_count += rows;
                SKStreamDecision::Continue
            },
            &sk_test_execution_plan(),
            SKStreamingPassPolicy::Iterative {
                maximum_number_of_passes: epochs,
            },
        )?;
        if let Some(error) = state.error {
            return Err(error);
        }
        if state.rows_count == 0 {
            return Err(SKError::EmptyInput {
                operation: "streaming_transformer_fit",
            });
        }
        let rows = F::from(state.rows_count).expect("a row count converts to the scalar");
        Ok(SKTestMeanCenteringModel {
            means: state.sums.mapv(|value| value / rows),
        })
    }

    fn transform_batch(
        &self,
        model: &Self::Model,
        batch: &SKFeatureBlock<F>,
    ) -> Result<SKFeatureBlock<F>, Self::Error> {
        match batch {
            SKFeatureBlock::Dense(matrix) => {
                if matrix.ncols() != model.means.len() {
                    return Err(SKError::shape_mismatch_2d(
                        matrix.nrows(),
                        model.means.len(),
                        matrix.nrows(),
                        matrix.ncols(),
                    ));
                }
                Ok(SKFeatureBlock::Dense(matrix - &model.means))
            }
            SKFeatureBlock::Sparse(_) => Err(SKError::UnsupportedRepresentation {
                representation: "sparse",
                suggestion: "center a sparse batch only after densifying it explicitly",
            }),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            }),
        }
    }
}

/// A representation-preserving streaming transformer that scales every value.
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
    rows_seen: usize,
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
            rows_seen,
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
