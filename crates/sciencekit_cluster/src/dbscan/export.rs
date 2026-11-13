//! Format-agnostic state export/import for [`SKDBSCANModel`].
//!
//! The nearest-core assignment extension needs the core-sample matrix, its
//! per-core cluster labels and its metric. They travel as a representation-
//! preserving design (dense or CSR) plus the label table; the optional tree
//! index is rebuilt as a brute-force searcher on import, which changes speed
//! but not predictions.

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKClusterLabel, SKError, SKExecutionMode, SKExportable, SKFloat,
    SKImportable, SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_integer_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};
use sciencekit_neighbors::SKBruteForceSearcher;
use sciencekit_neighbors::neighbors::export_helpers::{export_training_data, import_training_data};

use super::config::{SKDBSCANMetric, SKDBSCANSearchStrategy};
use super::model::{SKDBSCANCoreSearch, SKDBSCANModel};

/// Stable estimator identifier for exported DBSCAN states.
pub const SK_DBSCAN_ESTIMATOR_NAME: &str = "SKDBSCAN";
/// The tensor carrying the typed training assignment.
pub const SK_DBSCAN_LABELS_TENSOR: &str = "dbscan.labels";
/// The tensor carrying the sorted core-sample row indices.
pub const SK_DBSCAN_CORE_INDICES_TENSOR: &str = "dbscan.core_sample_indices";
/// The tensor carrying each core sample's cluster label.
pub const SK_DBSCAN_CORE_LABELS_TENSOR: &str = "dbscan.core_labels";

impl<F: SKFloat> SKExportable<F> for SKDBSCANModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = SKModelMetadata::new(
            SK_DBSCAN_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_features,
        )?
        .with_parameter(
            "number_of_clusters",
            SKParameterValue::from(self.number_of_clusters as i64),
        )?
        .with_parameter(
            "eps",
            SKParameterValue::from(self.eps.to_f64().unwrap_or(0.0)),
        )?
        .with_parameter(
            "minimum_samples",
            SKParameterValue::from(self.minimum_samples as i64),
        )?
        .with_parameter("metric", SKParameterValue::from(metric_name(self.metric)))?
        .with_parameter(
            "search_strategy",
            SKParameterValue::from(strategy_name(self.search_strategy)),
        )?
        .with_parameter(
            "training_representation",
            SKParameterValue::from(self.training_representation),
        )?
        .with_parameter(
            "has_core_search",
            SKParameterValue::from(self.core_search.is_some()),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut tensors = vec![
            SKTensor::integer(
                SK_DBSCAN_LABELS_TENSOR,
                vec![self.labels.len()],
                self.labels.iter().map(|label| label.to_i64()).collect(),
            )?,
            SKTensor::integer(
                SK_DBSCAN_CORE_INDICES_TENSOR,
                vec![self.core_sample_indices.len()],
                self.core_sample_indices
                    .iter()
                    .map(|&row| row as i64)
                    .collect(),
            )?,
        ];
        if let Some(core_search) = self.core_search.as_ref() {
            tensors.push(SKTensor::integer(
                SK_DBSCAN_CORE_LABELS_TENSOR,
                vec![core_search.labels.len()],
                core_search
                    .labels
                    .iter()
                    .map(|label| label.to_i64())
                    .collect(),
            )?);
            let (core_tensors, representation) =
                export_training_data(core_search.brute_force.training_data(), "dbscan.core")?;
            metadata = metadata.with_parameter(
                "core_representation",
                SKParameterValue::from(representation),
            )?;
            tensors.extend(core_tensors);
        }
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKDBSCANModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_DBSCAN_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let labels: Vec<SKClusterLabel> = sk_read_integer_tensor(state, SK_DBSCAN_LABELS_TENSOR)?
            .into_iter()
            .map(SKClusterLabel::from_i64)
            .collect::<Result<_, _>>()?;
        let core_sample_indices: Vec<usize> =
            sk_read_integer_tensor(state, SK_DBSCAN_CORE_INDICES_TENSOR)?
                .into_iter()
                .map(|row| row.max(0) as usize)
                .collect();
        let metric = metric_from_name(sk_string_parameter(metadata, "metric", "euclidean"));
        let core_search = if sk_boolean_parameter(metadata, "has_core_search", false) {
            let core_labels: Vec<SKClusterLabel> =
                sk_read_integer_tensor(state, SK_DBSCAN_CORE_LABELS_TENSOR)?
                    .into_iter()
                    .map(SKClusterLabel::from_i64)
                    .collect::<Result<_, _>>()?;
            let representation = sk_string_parameter(metadata, "core_representation", "dense");
            let training = import_training_data(state, "dbscan.core", representation)?;
            let brute_force = SKBruteForceSearcher::build(training, metric.neighbors_metric())?;
            Some(SKDBSCANCoreSearch {
                labels: core_labels,
                tree: None,
                brute_force,
            })
        } else {
            None
        };
        Ok(SKDBSCANModel {
            labels: Array1::from_vec(labels),
            core_sample_indices: Array1::from_vec(core_sample_indices),
            number_of_clusters: sk_integer_parameter(metadata, "number_of_clusters", 0).max(0)
                as usize,
            number_of_features: feature_count,
            eps: F::from(sk_float_parameter(metadata, "eps", 0.0))
                .ok_or(SKError::Conversion("eps is not finite".to_string()))?,
            minimum_samples: sk_integer_parameter(metadata, "minimum_samples", 1).max(1) as usize,
            metric,
            search_strategy: strategy_from_name(sk_string_parameter(
                metadata,
                "search_strategy",
                "brute_force",
            )),
            training_representation: "dense",
            core_search,
            backend: sk_parse_backend_parameter(metadata),
            execution_mode: SKExecutionMode::Automatic,
        })
    }
}

fn metric_name(metric: SKDBSCANMetric) -> &'static str {
    match metric {
        SKDBSCANMetric::Euclidean => "euclidean",
        SKDBSCANMetric::Manhattan => "manhattan",
        SKDBSCANMetric::Chebyshev => "chebyshev",
        SKDBSCANMetric::Cosine => "cosine",
    }
}

fn metric_from_name(name: &str) -> SKDBSCANMetric {
    match name {
        "manhattan" => SKDBSCANMetric::Manhattan,
        "chebyshev" => SKDBSCANMetric::Chebyshev,
        "cosine" => SKDBSCANMetric::Cosine,
        _ => SKDBSCANMetric::Euclidean,
    }
}

fn strategy_name(strategy: SKDBSCANSearchStrategy) -> &'static str {
    match strategy {
        SKDBSCANSearchStrategy::KdTree => "kd_tree",
        SKDBSCANSearchStrategy::BallTree => "ball_tree",
        SKDBSCANSearchStrategy::BruteForce => "brute_force",
    }
}

fn strategy_from_name(name: &str) -> SKDBSCANSearchStrategy {
    match name {
        "kd_tree" => SKDBSCANSearchStrategy::KdTree,
        "ball_tree" => SKDBSCANSearchStrategy::BallTree,
        _ => SKDBSCANSearchStrategy::BruteForce,
    }
}
