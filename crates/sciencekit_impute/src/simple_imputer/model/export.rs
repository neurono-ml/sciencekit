//! Format-agnostic state export/import for [`SKSimpleImputerModel`].
//!
//! The imputer carries an optional composed missing-indicator model. Its whole
//! exported state is embedded here under a `simple_imputer.indicator.*`
//! namespace, so the composed model rides the same contract instead of a
//! private sub-format.

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_float_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_boolean_tensor, sk_read_floating_tensor,
    sk_read_integer_tensor, sk_string_parameter, sk_validate_import_metadata,
};

use super::SKSimpleImputerModel;
use crate::missing_indicator::SKMissingIndicatorModel;
use crate::missing_indicator::model::export::SK_MISSING_INDICATOR_ESTIMATOR_NAME;
use crate::simple_imputer::state::SKImputationState;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// Stable estimator identifier for exported simple-imputer states.
pub const SK_SIMPLE_IMPUTER_ESTIMATOR_NAME: &str = "SKSimpleImputer";
/// The tensor carrying the per-column fill statistics.
pub const SK_SIMPLE_IMPUTER_STATISTICS_TENSOR: &str = "simple_imputer.statistics";
/// The tensor carrying the observed count of every column.
pub const SK_SIMPLE_IMPUTER_COUNTS_TENSOR: &str = "simple_imputer.counts";
/// The tensor carrying the all-missing columns dropped from the output.
pub const SK_SIMPLE_IMPUTER_DROPPED_TENSOR: &str = "simple_imputer.dropped_columns";
/// The tensor flagging the columns that carried a missing cell.
pub const SK_SIMPLE_IMPUTER_MISSING_TENSOR: &str = "simple_imputer.missing_columns";

/// The metadata prefix that embeds the composed indicator's parameters.
const INDICATOR_PARAMETER_PREFIX: &str = "simple_imputer.indicator.parameter.";
/// The tensor prefix that embeds the composed indicator's tensors.
const INDICATOR_TENSOR_PREFIX: &str = "simple_imputer.indicator.tensor.";

impl<F: SKFloat> SKExportable<F> for SKSimpleImputerModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = SKModelMetadata::new(
            SK_SIMPLE_IMPUTER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.number_of_input_columns,
        )?
        .with_parameter(
            "strategy",
            SKParameterValue::from(strategy_name(self.strategy)),
        )?
        .with_parameter(
            "missing_values",
            missing_marker_parameter(self.missing_values),
        )?
        .with_parameter(
            "keep_empty_features",
            SKParameterValue::from(self.keep_empty_features),
        )?
        .with_parameter(
            "fill_value_present",
            SKParameterValue::from(self.fill_value.is_some()),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?
        .with_parameter(
            "indicator_present",
            SKParameterValue::from(self.indicator.is_some()),
        )?;
        if let Some(fill_value) = self.fill_value {
            metadata = metadata.with_parameter(
                "fill_value",
                SKParameterValue::from(
                    fill_value
                        .to_f64()
                        .expect("a finite fill value fits in f64"),
                ),
            )?;
        }
        let mut tensors = vec![
            SKTensor::floating(
                SK_SIMPLE_IMPUTER_STATISTICS_TENSOR,
                vec![self.statistics.len()],
                self.statistics.to_vec(),
            )?,
            SKTensor::integer(
                SK_SIMPLE_IMPUTER_COUNTS_TENSOR,
                vec![self.state.counts().len()],
                self.state
                    .counts()
                    .iter()
                    .map(|&count| count as i64)
                    .collect(),
            )?,
            SKTensor::integer(
                SK_SIMPLE_IMPUTER_DROPPED_TENSOR,
                vec![self.dropped_columns.len()],
                self.dropped_columns
                    .iter()
                    .map(|&column| column as i64)
                    .collect(),
            )?,
            SKTensor::boolean(
                SK_SIMPLE_IMPUTER_MISSING_TENSOR,
                vec![self.state.missing_columns().len()],
                self.state.missing_columns().to_vec(),
            )?,
        ];
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

impl<F: SKFloat> SKImportable<F> for SKSimpleImputerModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let number_of_input_columns = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_SIMPLE_IMPUTER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            number_of_input_columns,
        )?;
        let metadata = state.metadata();
        let strategy = strategy_from_name(sk_string_parameter(metadata, "strategy", "mean"));
        let (_, statistics) = sk_read_floating_tensor(state, SK_SIMPLE_IMPUTER_STATISTICS_TENSOR)?;
        if statistics.len() != number_of_input_columns {
            return Err(SKError::ShapeMismatch {
                expected: vec![number_of_input_columns],
                found: vec![statistics.len()],
            });
        }
        let counts: Vec<u64> = sk_read_integer_tensor(state, SK_SIMPLE_IMPUTER_COUNTS_TENSOR)?
            .into_iter()
            .map(|count| count.max(0) as u64)
            .collect();
        let missing_columns = sk_read_boolean_tensor(state, SK_SIMPLE_IMPUTER_MISSING_TENSOR)?;
        if counts.len() != number_of_input_columns
            || missing_columns.len() != number_of_input_columns
        {
            return Err(SKError::ShapeMismatch {
                expected: vec![number_of_input_columns],
                found: vec![counts.len().max(missing_columns.len())],
            });
        }
        let dropped_columns: Vec<usize> =
            sk_read_integer_tensor(state, SK_SIMPLE_IMPUTER_DROPPED_TENSOR)?
                .into_iter()
                .map(|column| column.max(0) as usize)
                .collect();
        let indicator = if sk_boolean_parameter(metadata, "indicator_present", false) {
            Some(import_indicator(state, number_of_input_columns)?)
        } else {
            None
        };
        let missing_values = read_missing_marker::<F>(metadata);
        let fill_value = if sk_boolean_parameter(metadata, "fill_value_present", false) {
            Some(
                F::from(sk_float_parameter(metadata, "fill_value", 0.0))
                    .ok_or(SKError::Conversion("fill_value is not finite".to_string()))?,
            )
        } else {
            None
        };
        Ok(SKSimpleImputerModel {
            statistics: Array1::from_vec(statistics),
            dropped_columns,
            number_of_input_columns,
            missing_values,
            strategy,
            fill_value,
            keep_empty_features: sk_boolean_parameter(metadata, "keep_empty_features", false),
            state: SKImputationState::from_exported(counts, missing_columns, strategy),
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
    let indicator_state = SKModelState::new(metadata, tensors)?;
    SKMissingIndicatorModel::import_state(&indicator_state)
}

/// Encode a missing marker, spelling NaN as a string so two exports compare equal.
fn missing_marker_parameter<F: SKFloat>(marker: F) -> SKParameterValue {
    if marker.is_nan() {
        SKParameterValue::from("nan")
    } else {
        SKParameterValue::from(marker.to_f64().expect("a finite marker fits in f64"))
    }
}

/// Read a missing marker, restoring the NaN spelling.
fn read_missing_marker<F: SKFloat>(metadata: &SKModelMetadata) -> F {
    match metadata.parameter("missing_values") {
        Some(SKParameterValue::String(text)) if text == "nan" => F::nan(),
        Some(value) => value.as_float().and_then(F::from).unwrap_or_else(F::nan),
        None => F::nan(),
    }
}

/// The stable string for a strategy.
fn strategy_name(strategy: SKImputationStrategy) -> &'static str {
    match strategy {
        SKImputationStrategy::Mean => "mean",
        SKImputationStrategy::Median => "median",
        SKImputationStrategy::MostFrequent => "most_frequent",
        SKImputationStrategy::Constant => "constant",
    }
}

/// The strategy for its stable string, defaulting to the mean.
fn strategy_from_name(name: &str) -> SKImputationStrategy {
    match name {
        "median" => SKImputationStrategy::Median,
        "most_frequent" => SKImputationStrategy::MostFrequent,
        "constant" => SKImputationStrategy::Constant,
        _ => SKImputationStrategy::Mean,
    }
}
