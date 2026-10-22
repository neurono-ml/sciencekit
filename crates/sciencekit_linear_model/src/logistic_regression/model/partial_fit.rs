//! Incremental learning for the fitted logistic-regression model (spec
//! `logistic-regression`).

use ndarray::ArrayView2;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKTargetView,
    sk_run_operation,
};
use sprs::{CsMat, TriMat};

use super::SKLogisticRegressionModel;
use crate::logistic::SKLogisticState;
use crate::logistic::sk_canonicalize_against;
use crate::logistic_regression::core_implementation::sk_dimensions;

impl<F: SKFloat> SKLogisticRegressionModel<F> {
    /// Continue learning from one more batch, retaining the batch and re-solving
    /// the whole accumulated design.
    ///
    /// Returns the model itself for chaining. An empty batch is a no-op; a
    /// column count differing from the fitted design is rejected with
    /// `ShapeMismatch`, and a label the fit never saw is rejected.
    pub fn partial_fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?;
        let labels = sk_canonicalize_against(&self.classes, target)?;
        let (rows_count, columns_count) = sk_dimensions(&data);
        if columns_count != self.feature_count() {
            return Err(SKError::ShapeMismatch {
                expected: vec![self.feature_count()],
                found: vec![columns_count],
            });
        }
        if rows_count == 0 {
            return Ok(self);
        }
        if labels.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![labels.len()],
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
            self.accumulate(data, &labels)?;
            let solution =
                self.state
                    .solve(&self.config, self.strategy, self.number_of_classes, None)?;
            self.coefficients = solution.coefficients;
            self.intercepts = solution.intercepts;
            Ok(())
        })?;
        Ok(self)
    }

    /// Append one batch to the retained design, converting the representation
    /// when the batch and the accumulator disagree.
    fn accumulate(&mut self, data: SKDataView<'_, F>, labels: &[usize]) -> Result<(), SKError> {
        let state_is_dense = matches!(self.state, SKLogisticState::Dense { .. });
        match data {
            SKDataView::Dense(dense) => {
                if state_is_dense {
                    self.state.accumulate_dense(dense, labels);
                } else {
                    let sparse = sk_dense_to_csr(dense);
                    self.state.accumulate_sparse(sparse.view(), labels);
                }
            }
            SKDataView::Sparse(sparse) => {
                if state_is_dense {
                    let dense_owned = sparse.to_dense();
                    self.state.accumulate_dense(dense_owned.view(), labels);
                } else {
                    self.state.accumulate_sparse(sparse, labels);
                }
            }
            _ => return Err(unsupported()),
        }
        Ok(())
    }
}

/// The error for a feature representation the incremental update cannot consume.
fn unsupported() -> SKError {
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
