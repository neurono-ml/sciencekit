//! Incremental learning: one more batch retained and the whole design re-solved
//! (spec `coordinate-descent-solver`, Decision 7).

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKTargetView,
    sk_run_operation,
};
use sprs::{CsMat, TriMat};

use super::definition::SKCoordinateDescentModelCore;
use crate::coordinate_descent::accumulator::SKCoordinateDescentState;

impl<F: SKFloat> SKCoordinateDescentModelCore<F> {
    /// Continue learning from one more batch, retaining the batch and re-solving
    /// the whole accumulated design.
    ///
    /// Returns the model itself for chaining. An empty batch is a no-op; a
    /// column count differing from the fitted design is rejected with
    /// `ShapeMismatch`. Dense and sparse batches are both accepted, whichever
    /// the model was fitted with.
    pub(crate) fn partial_fit<'a, X, T>(
        &mut self,
        features: X,
        targets: T,
    ) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?.as_continuous()?.into_owned();
        let (rows_count, columns_count) = match data {
            SKDataView::Dense(dense) => dense.dim(),
            SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
            _ => return Err(unsupported_representation()),
        };
        let expected_columns = self.feature_count();
        if columns_count != expected_columns {
            return Err(SKError::ShapeMismatch {
                expected: vec![expected_columns],
                found: vec![columns_count],
            });
        }
        if rows_count == 0 {
            return Ok(self);
        }
        if target.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![target.len()],
            });
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            self.accumulate(data, target.view())?;
            let solution = self.state.solve(
                &self.alphas,
                self.l1_ratio,
                self.include_intercept,
                &self.config,
            )?;
            self.coefficients = solution.coefficients;
            self.intercepts = solution.intercepts;
            self.screened_count = solution.screened_count;
            Ok(())
        })?;
        Ok(self)
    }

    /// Append one batch to the retained design, converting the representation
    /// when the batch and the accumulator disagree.
    fn accumulate(&mut self, data: SKDataView<'_, F>, target: ArrayView1<F>) -> Result<(), SKError> {
        let state_is_dense = matches!(self.state, SKCoordinateDescentState::Dense { .. });
        match data {
            SKDataView::Dense(dense) => {
                if state_is_dense {
                    self.state.accumulate_dense(dense, target);
                } else {
                    let sparse = sk_dense_to_csr(dense);
                    self.state.accumulate_sparse(sparse.view(), target);
                }
            }
            SKDataView::Sparse(sparse) => {
                if state_is_dense {
                    let dense_owned = sparse.to_dense();
                    self.state.accumulate_dense(dense_owned.view(), target);
                } else {
                    self.state.accumulate_sparse(sparse, target);
                }
            }
            _ => return Err(unsupported_representation()),
        }
        Ok(())
    }
}

/// The error for a feature representation the incremental update cannot consume.
fn unsupported_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}

/// Convert a dense batch to an equivalent CSR matrix, dropping exact zeros.
fn sk_dense_to_csr<F: SKFloat>(dense: ArrayView2<'_, F>) -> CsMat<F> {
    let mut triplets = TriMat::new((dense.nrows(), dense.ncols()));
    dense.indexed_iter().for_each(|((row, column), &value)| {
        if value != F::zero() {
            triplets.add_triplet(row, column, value);
        }
    });
    triplets.to_csr()
}
