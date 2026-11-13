//! Format-agnostic state export and import (`SKExportable` / `SKImportable`)
//! for [`SKMiniBatchKMeansModel`].

use std::mem::size_of;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExportable, SKFloat, SKImportable, SKModelMetadata,
    SKModelState, SKParameterValue, SKTensor, SKTensorDataType, SKTensorValues,
    sk_validate_import_metadata,
};

use super::SKMiniBatchKMeans;
use super::model::SKMiniBatchKMeansModel;
use super::state::SKMiniBatchState;

/// Stable estimator identifier for exported MiniBatchKMeans states.
pub const SK_MINIBATCH_KMEANS_ESTIMATOR_NAME: &str = "SKMiniBatchKMeans";
/// Schema version for exported MiniBatchKMeans states.
pub const SK_MINIBATCH_KMEANS_FORMAT_VERSION: u32 = 1;

const CENTERS_TENSOR: &str = "minibatch_kmeans.centers";
const COUNTS_TENSOR: &str = "minibatch_kmeans.counts";
const INERTIA_TENSOR: &str = "minibatch_kmeans.inertia";

impl<F: SKFloat> SKExportable<F> for SKMiniBatchKMeansModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let dtype = if size_of::<F>() == size_of::<f32>() {
            SKTensorDataType::F32
        } else {
            SKTensorDataType::F64
        };
        let metadata = SKModelMetadata::new(
            SK_MINIBATCH_KMEANS_ESTIMATOR_NAME,
            SK_MINIBATCH_KMEANS_FORMAT_VERSION,
            dtype,
            self.number_of_features(),
        )?
        .with_parameter(
            "number_of_clusters",
            SKParameterValue::from(self.number_of_clusters() as i64),
        )?
        .with_parameter(
            "batches_processed",
            SKParameterValue::from(self.batches_processed as i64),
        )?
        .with_parameter(
            "epochs_consumed",
            SKParameterValue::from(self.epochs_consumed as i64),
        )?
        .with_parameter("converged", SKParameterValue::from(self.converged))?
        .with_parameter(
            "backend",
            SKParameterValue::from(format!("{:?}", self.configuration.backend)),
        )?;

        let centers_tensor = SKTensor::floating(
            CENTERS_TENSOR,
            vec![self.number_of_clusters(), self.number_of_features()],
            self.centers.iter().copied().collect(),
        )?;
        let counts_tensor = SKTensor::integer(
            COUNTS_TENSOR,
            vec![self.number_of_clusters()],
            self.counts.iter().map(|&count| count as i64).collect(),
        )?;
        let inertia_tensor = SKTensor::floating(INERTIA_TENSOR, Vec::new(), vec![self.inertia])?;
        SKModelState::new(
            metadata,
            vec![centers_tensor, counts_tensor, inertia_tensor],
        )
    }
}

impl<F: SKFloat> SKImportable<F> for SKMiniBatchKMeansModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_MINIBATCH_KMEANS_ESTIMATOR_NAME,
            SK_MINIBATCH_KMEANS_FORMAT_VERSION,
            feature_count,
        )?;

        let (center_shape, center_values) = read_floating(state, CENTERS_TENSOR)?;
        let &[clusters, columns] = center_shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: center_shape,
            });
        };
        if columns != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![clusters, feature_count],
                found: vec![clusters, columns],
            });
        }
        let centers = Array2::from_shape_vec((clusters, columns), center_values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;

        let count_values = read_integer(state, COUNTS_TENSOR)?;
        if count_values.len() != clusters {
            return Err(SKError::ShapeMismatch {
                expected: vec![clusters],
                found: vec![count_values.len()],
            });
        }
        let counts = Array1::from_iter(count_values.iter().map(|&value| value.max(0) as usize));

        let (_, inertia_values) = read_floating(state, INERTIA_TENSOR)?;
        let &[inertia] = inertia_values.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1],
                found: vec![inertia_values.len()],
            });
        };

        let metadata = state.metadata();
        let integer_param = |key, default| {
            metadata
                .parameter(key)
                .and_then(SKParameterValue::as_integer)
                .unwrap_or(default)
                .max(0) as usize
        };
        let backend = metadata
            .parameter("backend")
            .and_then(SKParameterValue::as_string)
            .map(parse_backend)
            .unwrap_or(SKBackendKind::Faer);
        let estimator = SKMiniBatchKMeans::<F>::build()
            .number_of_clusters(clusters)
            .backend(backend)
            .build()?;
        let mut inner = SKMiniBatchState::new(centers, estimator.configuration);
        inner.counts = counts;
        inner.batches_processed = integer_param("batches_processed", 0);
        inner.converged = metadata
            .parameter("converged")
            .and_then(SKParameterValue::as_bool)
            .unwrap_or(false);
        inner.last_batch_inertia = inertia;
        Ok(SKMiniBatchKMeansModel::from_state(
            inner,
            None,
            inertia,
            integer_param("epochs_consumed", 1),
        ))
    }
}

fn read_floating<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
) -> Result<(Vec<usize>, Vec<F>), SKError> {
    let tensor = state.tensor(name).ok_or(SKError::InvalidHyperparameter {
        name,
        reason: "missing exported tensor".to_string(),
    })?;
    match tensor.values() {
        SKTensorValues::Floating(values) => Ok((tensor.shape().to_vec(), values.clone())),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-floating tensor",
            suggestion: "export tensor as floating",
        }),
    }
}

fn read_integer<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
) -> Result<Vec<i64>, SKError> {
    let tensor = state.tensor(name).ok_or(SKError::InvalidHyperparameter {
        name,
        reason: "missing exported tensor".to_string(),
    })?;
    match tensor.values() {
        SKTensorValues::Integer(values) => Ok(values.clone()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-integer tensor",
            suggestion: "export tensor as integer",
        }),
    }
}

fn parse_backend(name: &str) -> SKBackendKind {
    name.parse::<SKBackendKind>().unwrap_or(match name {
        "MatrixMultiply" => SKBackendKind::MatrixMultiply,
        "NdArrayLinalg" => SKBackendKind::NdArrayLinalg,
        "OpenCl" => SKBackendKind::OpenCl,
        "Cuda" => SKBackendKind::Cuda,
        "Rocm" => SKBackendKind::Rocm,
        _ => SKBackendKind::Faer,
    })
}
