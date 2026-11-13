//! Incremental batch updates (`partial_fit`) on [`SKIncrementalPCAModel`]
//! (spec `incremental-pca`, decisions 1, 2 and 3).

use sciencekit_common::{
    SKDataView, SKError, SKOperationAttributes, SKOperationKind, sk_run_operation,
};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};

use super::finalize::sk_finalize_accumulator;
use super::fit::sk_resolve_batch_plan;
use super::merge::sk_update_dense;
use super::model::SKIncrementalPCAModel;
use super::sparse::sk_update_sparse;
use super::validation::sk_validate_partial_batch;
use crate::pca::sk_view_dimensions;

impl<F: SKBackendSelection> SKIncrementalPCAModel<F> {
    /// Merges one dense or CSR batch into the running Chan-Golub-LeVeque
    /// covariance state without retaining prior rows and refreshes the
    /// finalized principal components.
    pub fn partial_fit<'a, X>(&mut self, features: X) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&view).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend(),
        };
        sk_run_operation(attributes, || -> Result<(), SKError> {
            let (validated_rows, validated_columns) = sk_validate_partial_batch(
                view,
                self.number_of_features(),
                self.number_of_components(),
                "partial_fit",
            )?;
            let plan = sk_resolve_batch_plan(
                &self.configuration.builder_state,
                &view,
                validated_rows,
                validated_columns,
            )?;
            let backend = sk_math_backend_for_kind::<F>(self.configuration.backend);
            match view {
                SKDataView::Dense(dense) => {
                    sk_update_dense(
                        &mut self.accumulator,
                        backend.as_ref(),
                        dense,
                        plan.parallelism,
                    );
                }
                SKDataView::Sparse(sparse) => {
                    sk_update_sparse(&mut self.accumulator, sparse);
                }
                _ => {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: view.representation(),
                        suggestion: "convert the input to a supported dense or CSR view",
                    });
                }
            }
            let decomposition = sk_finalize_accumulator(
                &self.accumulator,
                backend.as_ref(),
                self.number_of_components(),
            )?;
            self.refresh_decomposition(decomposition);
            Ok(())
        })?;
        Ok(self)
    }
}
