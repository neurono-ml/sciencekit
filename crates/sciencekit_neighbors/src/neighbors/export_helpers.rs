//! Shared state encoding for the nearest-neighbor estimators.
//!
//! A fitted neighbor model owns its training design, so export must preserve
//! the representation: a dense design stays a two-dimensional floating tensor,
//! a CSR design travels as its `indptr`/`indices`/`data` triple and is rebuilt
//! as CSR on import — never densified.

use ndarray::Array2;
use sciencekit_common::{
    SKError, SKExecutionMode, SKFloat, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    sk_backend_parameter, sk_integer_parameter, sk_parse_backend_parameter,
    sk_read_floating_tensor, sk_read_integer_tensor, sk_string_parameter,
};
use sprs::CsMat;

use crate::neighbors::{
    SKNeighborsAlgorithm, SKNeighborsConfiguration, SKNeighborsMetric, SKNeighborsWeights,
    SKTrainingData,
};

/// The metadata parameter naming the stored representation.
pub const SK_TRAINING_REPRESENTATION_PARAMETER: &str = "training_representation";

/// Attach the search configuration to metadata.
pub fn with_neighbors_configuration(
    mut metadata: SKModelMetadata,
    configuration: &SKNeighborsConfiguration,
) -> Result<SKModelMetadata, SKError> {
    metadata = metadata
        .with_parameter(
            "nearest_neighbors_count",
            SKParameterValue::from(configuration.nearest_neighbors_count as i64),
        )?
        .with_parameter(
            "weights",
            SKParameterValue::from(weights_name(configuration.weights)),
        )?
        .with_parameter(
            "algorithm",
            SKParameterValue::from(algorithm_name(configuration.algorithm)),
        )?
        .with_parameter(
            "metric",
            SKParameterValue::from(metric_name(configuration.metric)),
        )?
        .with_parameter(
            "leaf_size",
            SKParameterValue::from(configuration.leaf_size as i64),
        )?
        .with_parameter("backend", sk_backend_parameter(configuration.backend))?;
    Ok(metadata)
}

/// Decode the search configuration from metadata.
pub fn read_neighbors_configuration(
    metadata: &SKModelMetadata,
) -> Result<SKNeighborsConfiguration, SKError> {
    let configuration = SKNeighborsConfiguration {
        nearest_neighbors_count: sk_integer_parameter(metadata, "nearest_neighbors_count", 1).max(1)
            as usize,
        weights: weights_from_name(sk_string_parameter(metadata, "weights", "uniform")),
        algorithm: algorithm_from_name(sk_string_parameter(metadata, "algorithm", "auto")),
        metric: metric_from_name(sk_string_parameter(metadata, "metric", "euclidean")),
        leaf_size: sk_integer_parameter(metadata, "leaf_size", 30).max(1) as usize,
        backend: sk_parse_backend_parameter(metadata),
        execution_mode: SKExecutionMode::Automatic,
    };
    configuration
        .validate()
        .map_err(|error| SKError::Conversion(error.to_string()))?;
    Ok(configuration)
}

/// Encode the owned training design under a family prefix.
pub fn export_training_data<F: SKFloat>(
    training: &SKTrainingData<F>,
    prefix: &str,
) -> Result<(Vec<SKTensor<F>>, &'static str), SKError> {
    match training {
        SKTrainingData::Dense(dense) => {
            let tensor = SKTensor::floating(
                format!("{prefix}.training_dense"),
                vec![dense.nrows(), dense.ncols()],
                dense.iter().copied().collect(),
            )?;
            Ok((vec![tensor], "dense"))
        }
        SKTrainingData::Sparse(sparse) => {
            let indptr: Vec<i64> = sparse
                .indptr()
                .raw_storage()
                .iter()
                .map(|&value| value as i64)
                .collect();
            let indices: Vec<i64> = sparse.indices().iter().map(|&value| value as i64).collect();
            let data = sparse.data().to_vec();
            let tensors = vec![
                SKTensor::integer(
                    format!("{prefix}.training_indptr"),
                    vec![sparse.rows() + 1],
                    indptr,
                )?,
                SKTensor::integer(
                    format!("{prefix}.training_indices"),
                    vec![indices.len()],
                    indices,
                )?,
                SKTensor::floating(format!("{prefix}.training_data"), vec![data.len()], data)?,
            ];
            Ok((tensors, "csr"))
        }
    }
}

/// Rebuild the owned training design from a family state.
pub fn import_training_data<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
    representation: &str,
) -> Result<SKTrainingData<F>, SKError> {
    if representation == "csr" {
        let indptr: Vec<usize> =
            sk_read_integer_tensor(state, &format!("{prefix}.training_indptr"))?
                .into_iter()
                .map(|value| value.max(0) as usize)
                .collect();
        let indices: Vec<usize> =
            sk_read_integer_tensor(state, &format!("{prefix}.training_indices"))?
                .into_iter()
                .map(|value| value.max(0) as usize)
                .collect();
        let (_, data) = sk_read_floating_tensor(state, &format!("{prefix}.training_data"))?;
        let rows = indptr.len().saturating_sub(1);
        let columns = state.metadata().feature_count();
        let matrix = CsMat::new((rows, columns), indptr, indices, data);
        Ok(SKTrainingData::Sparse(matrix))
    } else {
        let (shape, values) = sk_read_floating_tensor(state, &format!("{prefix}.training_dense"))?;
        let &[rows, columns] = shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, state.metadata().feature_count()],
                found: shape,
            });
        };
        let dense = Array2::from_shape_vec((rows, columns), values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        Ok(SKTrainingData::Dense(dense))
    }
}

fn weights_name(weights: SKNeighborsWeights) -> &'static str {
    match weights {
        SKNeighborsWeights::Uniform => "uniform",
        SKNeighborsWeights::Distance => "distance",
    }
}

fn weights_from_name(name: &str) -> SKNeighborsWeights {
    match name {
        "distance" => SKNeighborsWeights::Distance,
        _ => SKNeighborsWeights::Uniform,
    }
}

fn algorithm_name(algorithm: SKNeighborsAlgorithm) -> &'static str {
    match algorithm {
        SKNeighborsAlgorithm::Auto => "auto",
        SKNeighborsAlgorithm::KdTree => "kd_tree",
        SKNeighborsAlgorithm::BallTree => "ball_tree",
        SKNeighborsAlgorithm::BruteForce => "brute_force",
    }
}

fn algorithm_from_name(name: &str) -> SKNeighborsAlgorithm {
    match name {
        "kd_tree" => SKNeighborsAlgorithm::KdTree,
        "ball_tree" => SKNeighborsAlgorithm::BallTree,
        "brute_force" => SKNeighborsAlgorithm::BruteForce,
        _ => SKNeighborsAlgorithm::Auto,
    }
}

fn metric_name(metric: SKNeighborsMetric) -> &'static str {
    match metric {
        SKNeighborsMetric::Euclidean => "euclidean",
        SKNeighborsMetric::Manhattan => "manhattan",
        SKNeighborsMetric::Chebyshev => "chebyshev",
        SKNeighborsMetric::Cosine => "cosine",
    }
}

fn metric_from_name(name: &str) -> SKNeighborsMetric {
    match name {
        "manhattan" => SKNeighborsMetric::Manhattan,
        "chebyshev" => SKNeighborsMetric::Chebyshev,
        "cosine" => SKNeighborsMetric::Cosine,
        _ => SKNeighborsMetric::Euclidean,
    }
}
