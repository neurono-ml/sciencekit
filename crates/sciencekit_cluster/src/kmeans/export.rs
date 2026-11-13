//! Format-agnostic state export/import for [`SKKMeansModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKClusterLabel, SKError, SKExecutionMode, SKExportable, SKFloat,
    SKImportable, SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_read_integer_tensor,
    sk_string_parameter, sk_validate_import_metadata,
};

use super::SKKMeansAlgorithm;
use super::model::SKKMeansModel;

/// Stable estimator identifier for exported KMeans states.
pub const SK_KMEANS_ESTIMATOR_NAME: &str = "SKKMeans";
/// The tensor carrying the cluster centers.
pub const SK_KMEANS_CENTERS_TENSOR: &str = "kmeans.cluster_centers";
/// The tensor carrying the typed training assignment.
pub const SK_KMEANS_LABELS_TENSOR: &str = "kmeans.labels";

impl<F: SKFloat> SKExportable<F> for SKKMeansModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_KMEANS_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_features(),
        )?
        .with_parameter(
            "number_of_clusters",
            SKParameterValue::from(self.number_of_clusters() as i64),
        )?
        .with_parameter(
            "inertia",
            SKParameterValue::from(self.inertia.to_f64().unwrap_or(0.0)),
        )?
        .with_parameter("iterations", SKParameterValue::from(self.iterations as i64))?
        .with_parameter("converged", SKParameterValue::from(self.converged))?
        .with_parameter(
            "algorithm",
            SKParameterValue::from(algorithm_name(self.algorithm)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let centers = SKTensor::floating(
            SK_KMEANS_CENTERS_TENSOR,
            vec![self.number_of_clusters(), self.number_of_features()],
            self.centers.iter().copied().collect(),
        )?;
        let labels = SKTensor::integer(
            SK_KMEANS_LABELS_TENSOR,
            vec![self.labels.len()],
            self.labels.iter().map(|label| label.to_i64()).collect(),
        )?;
        SKModelState::new(metadata, vec![centers, labels])
    }
}

impl<F: SKFloat> SKImportable<F> for SKKMeansModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_KMEANS_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (shape, values) = sk_read_floating_tensor(state, SK_KMEANS_CENTERS_TENSOR)?;
        let &[clusters, columns] = shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: shape,
            });
        };
        if columns != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![clusters, feature_count],
                found: vec![clusters, columns],
            });
        }
        let centers = Array2::from_shape_vec((clusters, columns), values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let labels: Vec<SKClusterLabel> = sk_read_integer_tensor(state, SK_KMEANS_LABELS_TENSOR)?
            .into_iter()
            .map(SKClusterLabel::from_i64)
            .collect::<Result<_, _>>()?;
        let metadata = state.metadata();
        Ok(SKKMeansModel {
            centers,
            labels: Array1::from_vec(labels),
            inertia: F::from(sk_float_parameter(metadata, "inertia", 0.0))
                .ok_or(SKError::Conversion("inertia is not finite".to_string()))?,
            iterations: sk_integer_parameter(metadata, "iterations", 0).max(0) as usize,
            converged: sciencekit_common::sk_boolean_parameter(metadata, "converged", true),
            algorithm: algorithm_from_name(sk_string_parameter(metadata, "algorithm", "lloyd")),
            backend: sk_parse_backend_parameter(metadata),
            execution_mode: SKExecutionMode::Automatic,
        })
    }
}

fn algorithm_name(algorithm: SKKMeansAlgorithm) -> &'static str {
    match algorithm {
        SKKMeansAlgorithm::Auto => "auto",
        SKKMeansAlgorithm::Lloyd => "lloyd",
        SKKMeansAlgorithm::Elkan => "elkan",
    }
}

fn algorithm_from_name(name: &str) -> SKKMeansAlgorithm {
    match name {
        "auto" => SKKMeansAlgorithm::Auto,
        "elkan" => SKKMeansAlgorithm::Elkan,
        _ => SKKMeansAlgorithm::Lloyd,
    }
}
