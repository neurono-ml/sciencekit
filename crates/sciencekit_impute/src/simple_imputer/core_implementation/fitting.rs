//! Batch fitting: the representation dispatch, the state seeding and the
//! fitted model assembly (spec `simple-imputer`).

use ndarray::ArrayView2;
use sciencekit_common::{
    SKBuilder, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};
use sprs::CsMatView;

use super::SKSimpleImputer;
use super::validation::{empty_columns, resolve_fit_plan, validated_shape, validated_sparse_shape};
use crate::missing_indicator::{SKMissingIndicator, SKMissingIndicatorModel};
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
        self.finish_fit(statistics, columns_count, state)
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
        self.finish_fit(statistics, columns_count, state)
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

    /// The indicator estimator composed by `add_indicator`: the same marker and
    /// backend kind, tolerating missingness unseen at fit so it never fails a
    /// transform the imputer can handle.
    pub(in crate::simple_imputer) fn indicator_estimator(
        &self,
    ) -> Result<SKMissingIndicator<F>, SKError> {
        SKMissingIndicator::build()
            .missing_values(self.missing_values)
            .error_on_new(false)
            .backend(self.backend)
            .execution_mode(self.execution_intent())
            .build()
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
    /// empty-column policy, composing the indicator from the columns the state
    /// saw missing.
    fn finish_fit(
        &self,
        statistics: ndarray::Array1<F>,
        columns_count: usize,
        state: SKImputationState<F>,
    ) -> Result<SKSimpleImputerModel<F>, SKError> {
        let dropped_columns = empty_columns(&statistics, state.counts(), self.keep_empty_features);
        let indicator = if self.add_indicator {
            let scope: Vec<usize> = state
                .missing_columns()
                .iter()
                .enumerate()
                .filter_map(|(column, missing)| missing.then_some(column))
                .collect();
            Some(SKMissingIndicatorModel::from_scope(
                scope,
                columns_count,
                &self.indicator_estimator()?,
            ))
        } else {
            None
        };
        Ok(SKSimpleImputerModel::from_state(
            statistics,
            dropped_columns,
            columns_count,
            state,
            indicator,
            self,
        ))
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
