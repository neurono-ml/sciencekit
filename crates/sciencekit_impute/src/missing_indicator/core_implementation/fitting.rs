//! Batch fitting: the representation dispatch and the scope learned from each
//! representation (spec `missing-indicator`).

use ndarray::{ArrayView2, Axis};
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};
use sprs::CsMatView;

use super::SKMissingIndicator;
use crate::missing_indicator::model::SKMissingIndicatorModel;
use crate::missing_indicator::scope::SKMissingFeatureScope;
use crate::missingness::{sk_is_missing, sk_sparse_marks_implicit_zeros};

impl<F: SKFloat> SKMissingIndicator<F> {
    /// Fit the dense representation.
    fn fit_dense(&self, dense: ArrayView2<'_, F>) -> Result<SKMissingIndicatorModel<F>, SKError> {
        let (rows_count, columns_count) = validated_shape(&dense)?;
        let scope = sk_run_operation(self.fit_attributes(rows_count, columns_count), || {
            let plan = resolve_fit_plan(self, dense.len() as u64)?;
            debug_assert!(plan.parallelism >= 1);
            Ok::<_, SKError>(self.dense_scope(&dense, columns_count))
        })?;
        Ok(self.finish_fit(scope, columns_count))
    }

    /// Fit the sparse representation; the structural zeros join the observed
    /// set under the `0.0` marker.
    fn fit_sparse(&self, sparse: CsMatView<'_, F>) -> Result<SKMissingIndicatorModel<F>, SKError> {
        let (rows_count, columns_count) = validated_sparse_shape(&sparse)?;
        let scope = sk_run_operation(self.fit_attributes(rows_count, columns_count), || {
            let plan = resolve_fit_plan(self, sparse.nnz() as u64)?;
            debug_assert!(plan.parallelism >= 1);
            Ok::<_, SKError>(self.sparse_scope(&sparse, columns_count))
        })?;
        Ok(self.finish_fit(scope, columns_count))
    }

    /// The mask scope of a dense batch: every missing column, or every column
    /// under `All`.
    fn dense_scope(&self, dense: &ArrayView2<'_, F>, columns_count: usize) -> Vec<usize> {
        if self.features == SKMissingFeatureScope::All {
            return (0..columns_count).collect();
        }
        dense
            .axis_iter(Axis(1))
            .into_par_iter()
            .enumerate()
            .filter_map(|(column, values)| {
                values
                    .iter()
                    .any(|&value| sk_is_missing(value, self.missing_values))
                    .then_some(column)
            })
            .collect()
    }

    /// The mask scope of a sparse batch: explicit missing entries, plus the
    /// implicit zeros when the marker claims them.
    fn sparse_scope(&self, sparse: &CsMatView<'_, F>, columns_count: usize) -> Vec<usize> {
        if self.features == SKMissingFeatureScope::All {
            return (0..columns_count).collect();
        }
        let mut missing = vec![false; columns_count];
        let mut explicit_counts = vec![0usize; columns_count];
        for row in sparse.outer_iterator() {
            for (column, &value) in row.iter() {
                explicit_counts[column] += 1;
                if sk_is_missing(value, self.missing_values) {
                    missing[column] = true;
                }
            }
        }
        if sk_sparse_marks_implicit_zeros(self.missing_values) {
            for (column, count) in explicit_counts.iter().enumerate() {
                if *count < sparse.rows() {
                    missing[column] = true;
                }
            }
        }
        missing
            .into_iter()
            .enumerate()
            .filter_map(|(column, is_missing)| is_missing.then_some(column))
            .collect()
    }

    /// The span attributes of a fit, carrying the configured execution mode
    /// and backend kind.
    fn fit_attributes(&self, rows_count: usize, columns_count: usize) -> SKOperationAttributes {
        SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        }
    }

    /// Assemble the fitted model from the learned scope.
    fn finish_fit(&self, scope: Vec<usize>, columns_count: usize) -> SKMissingIndicatorModel<F> {
        SKMissingIndicatorModel::from_state(scope, columns_count, self)
    }
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKMissingIndicator<F> {
    type Model = SKMissingIndicatorModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        match features.try_into()? {
            SKDataView::Dense(dense) => self.fit_dense(dense),
            SKDataView::Sparse(sparse) => self.fit_sparse(sparse),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the input to a dense or CSR matrix",
            }),
        }
    }
}

/// Reject empty dense inputs before any element is processed.
fn validated_shape<F>(features: &ArrayView2<F>) -> Result<(usize, usize), SKError> {
    let (rows_count, columns_count) = features.dim();
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    Ok((rows_count, columns_count))
}

/// Reject empty sparse inputs before any element is processed.
fn validated_sparse_shape<F>(features: &CsMatView<'_, F>) -> Result<(usize, usize), SKError> {
    let (rows_count, columns_count) = features.shape();
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    Ok((rows_count, columns_count))
}

/// Build a fresh execution context per operation, declare the reduction grain
/// and the `Sequential` access pattern, and resolve the plan.
fn resolve_fit_plan<F: SKFloat>(
    estimator: &SKMissingIndicator<F>,
    stored_elements: u64,
) -> Result<SKExecutionPlan, SKError> {
    let mut context = sciencekit_common::SKExecutionContext::real();
    context.dataset_size_bytes = stored_elements * size_of::<F>() as u64;
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = Some(stored_elements);
    context.access_pattern = sciencekit_common::execution::SKAccessPattern::Sequential;
    context.grain = sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;
    estimator.resolve_plan(&context)
}
