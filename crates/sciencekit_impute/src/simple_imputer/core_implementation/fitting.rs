//! Batch fitting: the representation dispatch, the state seeding and the
//! fitted model assembly (spec `simple-imputer`).

use ndarray::ArrayView2;
use sciencekit_common::{
    SKDataView, SKError, SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};
use sprs::CsMatView;

use super::SKSimpleImputer;
use crate::simple_imputer::model::SKSimpleImputerModel;
use crate::simple_imputer::state::SKImputationState;
use crate::simple_imputer::statistics::{dense_column_statistics, sparse_column_statistics};
use crate::simple_imputer::strategy::SKImputationStrategy;

impl<F: SKFloat> SKSimpleImputer<F> {
    /// Fit the dense representation; the observed set is the non-missing cells.
    fn fit_dense(&self, dense: ArrayView2<'_, F>) -> Result<SKSimpleImputerModel<F>, SKError> {
        let (rows_count, columns_count) = validated_shape(&dense)?;
        let (statistics, state) = self.fitted_state(
            rows_count,
            columns_count,
            dense.len() as u64,
            |state| state.extend_dense(&dense, self.strategy, self.missing_values),
            || dense_column_statistics(&dense, self.strategy, self.missing_values, self.fill_value),
        )?;
        Ok(self.finish_fit(statistics, columns_count, state))
    }

    /// Fit the sparse representation; the observed set is the explicit
    /// non-missing entries plus the structural zeros the marker observes.
    fn fit_sparse(&self, sparse: CsMatView<'_, F>) -> Result<SKSimpleImputerModel<F>, SKError> {
        let (rows_count, columns_count) = validated_sparse_shape(&sparse)?;
        let (statistics, state) = self.fitted_state(
            rows_count,
            columns_count,
            sparse.nnz() as u64,
            |state| state.extend_sparse(&sparse, self.strategy, self.missing_values),
            || {
                sparse_column_statistics(
                    &sparse,
                    self.strategy,
                    self.missing_values,
                    self.fill_value,
                )
            },
        )?;
        Ok(self.finish_fit(statistics, columns_count, state))
    }

    /// Seed the incremental state with the fit batch and derive the exact
    /// statistics for it.
    ///
    /// The mean, mode and constant statistics come from the state itself,
    /// which is exact; the median keeps the exact order statistic of the batch
    /// (the sketch approximation is confined to the incremental path), while
    /// the sketch is still seeded so `partial_fit` can continue from it.
    fn fitted_state(
        &self,
        rows_count: usize,
        columns_count: usize,
        stored_elements: u64,
        seed_state: impl FnOnce(&mut SKImputationState<F>),
        exact_statistics: impl FnOnce() -> ndarray::Array1<F>,
    ) -> Result<(ndarray::Array1<F>, SKImputationState<F>), SKError> {
        sk_run_operation(self.fit_attributes(rows_count, columns_count), || {
            let plan = resolve_fit_plan(self, stored_elements)?;
            debug_assert!(plan.parallelism >= 1);
            let mut state = SKImputationState::new(columns_count, self.strategy);
            seed_state(&mut state);
            let statistics = if self.strategy == SKImputationStrategy::Median {
                exact_statistics()
            } else {
                state.statistics(self.strategy, self.fill_value)
            };
            Ok::<_, SKError>((statistics, state))
        })
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

    /// Assemble the fitted model from the statistics, the seeded state and the
    /// empty-column policy.
    fn finish_fit(
        &self,
        statistics: ndarray::Array1<F>,
        columns_count: usize,
        state: SKImputationState<F>,
    ) -> SKSimpleImputerModel<F> {
        let dropped_columns = empty_columns(&statistics, self.keep_empty_features);
        SKSimpleImputerModel::from_state(statistics, dropped_columns, columns_count, state, self)
    }
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKSimpleImputer<F> {
    type Model = SKSimpleImputerModel<F>;
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

/// The indices of the columns that produced no statistic, kept only when the
/// estimator retains empty features.
pub(in crate::simple_imputer) fn empty_columns<F: SKFloat>(
    statistics: &ndarray::Array1<F>,
    keep_empty_features: bool,
) -> Vec<usize> {
    if keep_empty_features {
        return Vec::new();
    }
    statistics
        .iter()
        .enumerate()
        .filter(|(_, statistic)| statistic.is_nan())
        .map(|(index, _)| index)
        .collect()
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
    estimator: &SKSimpleImputer<F>,
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
