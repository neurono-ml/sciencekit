//! Format-agnostic state export/import for [`SKKNNImputerModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_boolean_tensor, sk_read_floating_tensor,
    sk_read_integer_tensor, sk_string_parameter, sk_validate_import_metadata,
};

use super::SKKNNImputerModel;
use super::weights::SKImputationWeights;
use crate::missing_indicator::SKMissingIndicatorModel;
use crate::missing_indicator::model::export::SK_MISSING_INDICATOR_ESTIMATOR_NAME;

/// Stable estimator identifier for exported KNN-imputer states.
pub const SK_KNN_IMPUTER_ESTIMATOR_NAME: &str = "SKKNNImputer";
/// The tensor carrying the training design with missing cells normalized to NaN.
pub const SK_KNN_IMPUTER_TRAINING_TENSOR: &str = "knn_imputer.training";
/// The tensor carrying the training missingness mask.
pub const SK_KNN_IMPUTER_MASK_TENSOR: &str = "knn_imputer.missing_mask";
/// The tensor carrying the flattened donor lists.
pub const SK_KNN_IMPUTER_DONORS_TENSOR: &str = "knn_imputer.column_donors";
/// The tensor carrying each column's donor-list start offset plus the final length.
pub const SK_KNN_IMPUTER_DONOR_OFFSETS_TENSOR: &str = "knn_imputer.donor_offsets";
/// The tensor carrying the retained input columns.
pub const SK_KNN_IMPUTER_KEPT_TENSOR: &str = "knn_imputer.kept_columns";
/// The tensor carrying the dropped input columns.
pub const SK_KNN_IMPUTER_DROPPED_TENSOR: &str = "knn_imputer.dropped_columns";
/// The tensor carrying the observed mean of every input column.
pub const SK_KNN_IMPUTER_MEANS_TENSOR: &str = "knn_imputer.column_means";

const INDICATOR_PARAMETER_PREFIX: &str = "knn_imputer.indicator.parameter.";
const INDICATOR_TENSOR_PREFIX: &str = "knn_imputer.indicator.tensor.";

impl<F: SKFloat> SKExportable<F> for SKKNNImputerModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = SKModelMetadata::new(
            SK_KNN_IMPUTER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_input_columns,
        )?
        .with_parameter(
            "missing_values",
            missing_marker_parameter(self.missing_values),
        )?
        .with_parameter(
            "nearest_neighbors_count",
            SKParameterValue::from(self.nearest_neighbors_count as i64),
        )?
        .with_parameter(
            "weights",
            SKParameterValue::from(weights_name(self.weights)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?
        .with_parameter(
            "indicator_present",
            SKParameterValue::from(self.indicator.is_some()),
        )?;
        let mut tensors = vec![
            SKTensor::floating(
                SK_KNN_IMPUTER_TRAINING_TENSOR,
                vec![self.training.nrows(), self.training.ncols()],
                self.training.iter().copied().collect(),
            )?,
            SKTensor::boolean(
                SK_KNN_IMPUTER_MASK_TENSOR,
                vec![self.missing_mask.nrows(), self.missing_mask.ncols()],
                self.missing_mask.iter().copied().collect(),
            )?,
            SKTensor::integer(
                SK_KNN_IMPUTER_KEPT_TENSOR,
                vec![self.kept_columns.len()],
                self.kept_columns
                    .iter()
                    .map(|&column| column as i64)
                    .collect(),
            )?,
            SKTensor::integer(
                SK_KNN_IMPUTER_DROPPED_TENSOR,
                vec![self.dropped_columns.len()],
                self.dropped_columns
                    .iter()
                    .map(|&column| column as i64)
                    .collect(),
            )?,
            SKTensor::floating(
                SK_KNN_IMPUTER_MEANS_TENSOR,
                vec![self.column_means.len()],
                self.column_means.to_vec(),
            )?,
        ];
        let mut flattened = Vec::new();
        let mut offsets = Vec::with_capacity(self.column_donors.len() + 1);
        offsets.push(0_i64);
        for donors in &self.column_donors {
            flattened.extend(donors.iter().map(|&row| row as i64));
            offsets.push(flattened.len() as i64);
        }
        tensors.push(SKTensor::integer(
            SK_KNN_IMPUTER_DONORS_TENSOR,
            vec![flattened.len()],
            flattened,
        )?);
        tensors.push(SKTensor::integer(
            SK_KNN_IMPUTER_DONOR_OFFSETS_TENSOR,
            vec![offsets.len()],
            offsets,
        )?);
        if let Some(indicator) = self.indicator.as_ref() {
            let indicator_state = indicator.export_state()?;
            for (name, value) in indicator_state.metadata().parameters() {
                metadata = metadata
                    .with_parameter(format!("{INDICATOR_PARAMETER_PREFIX}{name}"), value.clone())?;
            }
            for (name, tensor) in indicator_state.tensors() {
                tensors.push(SKTensor::new(
                    format!("{INDICATOR_TENSOR_PREFIX}{name}"),
                    tensor.data_type(),
                    tensor.shape().to_vec(),
                    tensor.values().clone(),
                )?);
            }
        }
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKKNNImputerModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let number_of_input_columns = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_KNN_IMPUTER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            number_of_input_columns,
        )?;
        let (training_shape, training_values) =
            sk_read_floating_tensor(state, SK_KNN_IMPUTER_TRAINING_TENSOR)?;
        let &[rows, columns] = training_shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, number_of_input_columns],
                found: training_shape,
            });
        };
        let training = Array2::from_shape_vec((rows, columns), training_values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let mask_values = sk_read_boolean_tensor(state, SK_KNN_IMPUTER_MASK_TENSOR)?;
        let missing_mask = Array2::from_shape_vec((rows, columns), mask_values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let flattened = sk_read_integer_tensor(state, SK_KNN_IMPUTER_DONORS_TENSOR)?;
        let offsets = sk_read_integer_tensor(state, SK_KNN_IMPUTER_DONOR_OFFSETS_TENSOR)?;
        let mut column_donors = Vec::with_capacity(offsets.len().saturating_sub(1));
        for window in offsets.windows(2) {
            let start = window[0].max(0) as usize;
            let end = window[1].max(0) as usize;
            let donors = flattened.get(start..end).ok_or(SKError::ShapeMismatch {
                expected: vec![flattened.len()],
                found: vec![end],
            })?;
            column_donors.push(donors.iter().map(|&row| row.max(0) as usize).collect());
        }
        let kept_columns: Vec<usize> = sk_read_integer_tensor(state, SK_KNN_IMPUTER_KEPT_TENSOR)?
            .into_iter()
            .map(|column| column.max(0) as usize)
            .collect();
        let dropped_columns: Vec<usize> =
            sk_read_integer_tensor(state, SK_KNN_IMPUTER_DROPPED_TENSOR)?
                .into_iter()
                .map(|column| column.max(0) as usize)
                .collect();
        let (_, column_means) = sk_read_floating_tensor(state, SK_KNN_IMPUTER_MEANS_TENSOR)?;
        let metadata = state.metadata();
        let indicator = if sk_boolean_parameter(metadata, "indicator_present", false) {
            Some(import_indicator(state, number_of_input_columns)?)
        } else {
            None
        };
        Ok(SKKNNImputerModel {
            training,
            missing_mask,
            column_donors,
            kept_columns,
            dropped_columns,
            column_means: Array1::from_vec(column_means),
            number_of_input_columns,
            missing_values: read_missing_marker::<F>(metadata),
            nearest_neighbors_count: sk_integer_parameter(metadata, "nearest_neighbors_count", 1)
                .max(1) as usize,
            weights: weights_from_name(sk_string_parameter(metadata, "weights", "uniform")),
            indicator,
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}

/// Rebuild the composed indicator from its embedded namespace.
fn import_indicator<F: SKFloat>(
    state: &SKModelState<F>,
    feature_count: usize,
) -> Result<SKMissingIndicatorModel<F>, SKError> {
    let mut metadata = SKModelMetadata::new(
        SK_MISSING_INDICATOR_ESTIMATOR_NAME,
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        feature_count,
    )?;
    for (name, value) in state.metadata().parameters() {
        if let Some(inner) = name.strip_prefix(INDICATOR_PARAMETER_PREFIX) {
            metadata = metadata.with_parameter(inner, value.clone())?;
        }
    }
    let mut tensors = Vec::new();
    for (name, tensor) in state.tensors() {
        if let Some(inner) = name.strip_prefix(INDICATOR_TENSOR_PREFIX) {
            tensors.push(SKTensor::new(
                inner,
                tensor.data_type(),
                tensor.shape().to_vec(),
                tensor.values().clone(),
            )?);
        }
    }
    SKMissingIndicatorModel::import_state(&SKModelState::new(metadata, tensors)?)
}

fn missing_marker_parameter<F: SKFloat>(marker: F) -> SKParameterValue {
    if marker.is_nan() {
        SKParameterValue::from("nan")
    } else {
        SKParameterValue::from(marker.to_f64().expect("a finite marker fits in f64"))
    }
}

fn read_missing_marker<F: SKFloat>(metadata: &SKModelMetadata) -> F {
    match metadata.parameter("missing_values") {
        Some(SKParameterValue::String(text)) if text == "nan" => F::nan(),
        Some(value) => value.as_float().and_then(F::from).unwrap_or_else(F::nan),
        None => F::nan(),
    }
}

fn weights_name(weights: SKImputationWeights) -> &'static str {
    match weights {
        SKImputationWeights::Uniform => "uniform",
        SKImputationWeights::Distance => "distance",
    }
}

fn weights_from_name(name: &str) -> SKImputationWeights {
    match name {
        "distance" => SKImputationWeights::Distance,
        _ => SKImputationWeights::Uniform,
    }
}
