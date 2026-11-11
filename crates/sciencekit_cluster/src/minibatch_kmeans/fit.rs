//! In-memory chunked `fit` and single-batch `fit_batch` bootstrap for
//! [`SKMiniBatchKMeans`] (spec `mini-batch-k-means`, decision 5).

use ndarray::{Array1, Axis};
use sciencekit_common::{
    SKClusterLabel, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};
use sprs::{CsMat, CsMatView};

use super::assign::sk_assign_batch_rows;
use super::core_implementation::SKMiniBatchKMeans;
use super::init::sk_initialize_minibatch_centers;
use super::model::SKMiniBatchKMeansModel;
use super::state::SKMiniBatchState;
use super::validation::sk_validate_initial_batch;
use crate::kmeans::distance::sk_view_dimensions;

impl<F: SKFloat> SKUnsupervisedFit<F> for SKMiniBatchKMeans<F> {
    type Model = SKMiniBatchKMeansModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || sk_fit_in_memory(self, features))
    }
}

impl<F: SKFloat> SKMiniBatchKMeans<F> {
    /// Initializes centers from a single batch and performs one mini-batch
    /// update, returning a model ready for incremental continuation.
    pub fn fit_batch<'a, X>(&self, features: X) -> Result<SKMiniBatchKMeansModel<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::PartialFit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            sk_validate_initial_batch(
                features,
                self.configuration.number_of_clusters,
                "partial_fit",
            )?;
            let (centers, _) = sk_initialize_minibatch_centers(features, &self.configuration)?;
            let mut state = SKMiniBatchState::new(centers, self.configuration);
            state.apply_batch(features)?;
            let inertia = state.last_batch_inertia;
            Ok(SKMiniBatchKMeansModel::from_state(state, None, inertia, 1))
        })
    }

    /// Fits the estimator in memory and returns the typed training assignments.
    pub fn fit_predict<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::FitPredict,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            let model = sk_fit_in_memory(self, features)?;
            Ok(model
                .labels
                .expect("in-memory fit always populates training assignments"))
        })
    }
}

pub(crate) fn sk_fit_in_memory<F: SKFloat>(
    estimator: &SKMiniBatchKMeans<F>,
    features: SKDataView<'_, F>,
) -> Result<SKMiniBatchKMeansModel<F>, SKError> {
    let configuration = estimator.configuration;
    sk_validate_initial_batch(features, configuration.number_of_clusters, "fit")?;
    let (centers, _) = sk_initialize_minibatch_centers(features, &configuration)?;
    let mut state = SKMiniBatchState::new(centers, configuration);

    let mut epochs_consumed = 0_usize;
    for _ in 0..configuration.maximum_number_of_iterations {
        epochs_consumed += 1;
        let converged = run_epoch_chunks(&mut state, features, configuration.batch_size)?;
        if converged {
            break;
        }
    }

    let (raw_labels, squared_distances) = sk_assign_batch_rows(features, &state.centers.view())?;
    let inertia = squared_distances
        .iter()
        .fold(F::zero(), |accumulator, &value| accumulator + value);
    let labels = Array1::from_iter(
        raw_labels
            .iter()
            .map(|&cluster| SKClusterLabel::cluster(cluster as u32)),
    );
    Ok(SKMiniBatchKMeansModel::from_state(
        state,
        Some(labels),
        inertia,
        epochs_consumed,
    ))
}

fn run_epoch_chunks<F: SKFloat>(
    state: &mut SKMiniBatchState<F>,
    features: SKDataView<'_, F>,
    batch_size: usize,
) -> Result<bool, SKError> {
    match features {
        SKDataView::Dense(dense) => {
            for chunk in dense.axis_chunks_iter(Axis(0), batch_size) {
                if state.apply_batch(SKDataView::Dense(chunk))? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        SKDataView::Sparse(sparse) => {
            let total_rows = sparse.rows();
            let starts = (0..total_rows).step_by(batch_size);
            for start in starts {
                let end = (start + batch_size).min(total_rows);
                let chunk = slice_csr_rows(&sparse, start, end);
                if state.apply_batch(SKDataView::Sparse(chunk.view()))? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}

pub(crate) fn slice_csr_rows<F: SKFloat>(
    source: &CsMatView<'_, F>,
    start_row: usize,
    end_row: usize,
) -> CsMat<F> {
    let row_count = end_row - start_row;
    let mut indptr = Vec::with_capacity(row_count + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    indptr.push(0_usize);
    (start_row..end_row).for_each(|row_index| {
        if let Some(stored) = source.outer_view(row_index) {
            columns.extend_from_slice(stored.indices());
            values.extend_from_slice(stored.data());
        }
        indptr.push(values.len());
    });
    CsMat::new((row_count, source.cols()), indptr, columns, values)
}
