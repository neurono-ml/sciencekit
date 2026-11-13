//! Format-agnostic state export/import for [`SKOneHotEncoderModel`].
//!
//! The per-column category tables are string or integer codecs, not floating
//! tensors, so they travel in the metadata as typed lists while a small
//! integer tensor records each column's cardinality. Import rebuilds the
//! tables in index order and recomputes the output offsets deterministically.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_canonicalize_labels, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_integer_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::SKOneHotEncoderModel;
use crate::one_hot_encoder::categories::{SKIntegerCategoryTable, SKOneHotColumnCategories};
use crate::one_hot_encoder::unknown_policy::SKUnknownPolicy;

/// Stable estimator identifier for exported one-hot-encoder states.
pub const SK_ONE_HOT_ENCODER_ESTIMATOR_NAME: &str = "SKOneHotEncoder";
/// The tensor carrying each input column's category cardinality.
pub const SK_ONE_HOT_CATEGORY_COUNTS_TENSOR: &str = "one_hot.category_counts";

impl<F: SKFloat> SKExportable<F> for SKOneHotEncoderModel {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = SKModelMetadata::new(
            SK_ONE_HOT_ENCODER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.columns.len(),
        )?
        .with_parameter("sparse_output", SKParameterValue::from(self.sparse_output))?
        .with_parameter(
            "handle_unknown",
            SKParameterValue::from(match self.handle_unknown {
                SKUnknownPolicy::Reject => "reject",
                SKUnknownPolicy::Ignore => "ignore",
            }),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut kinds = Vec::with_capacity(self.columns.len());
        for (column, categories) in self.columns.iter().enumerate() {
            match categories {
                SKOneHotColumnCategories::Nominal(table) => {
                    kinds.push(SKParameterValue::from("nominal"));
                    let labels = table.labels().map(SKParameterValue::from).collect();
                    metadata = metadata.with_parameter(
                        format!("one_hot.categories.{column}"),
                        SKParameterValue::List(labels),
                    )?;
                }
                SKOneHotColumnCategories::Integer(table) => {
                    kinds.push(SKParameterValue::from("integer"));
                    let codes = (0..table.number_of_categories())
                        .filter_map(|index| table.code_of(index))
                        .map(SKParameterValue::from)
                        .collect();
                    metadata = metadata.with_parameter(
                        format!("one_hot.integer_categories.{column}"),
                        SKParameterValue::List(codes),
                    )?;
                }
            }
        }
        metadata =
            metadata.with_parameter("one_hot.column_kinds", SKParameterValue::List(kinds))?;
        let counts = self
            .columns
            .iter()
            .map(|categories| categories.number_of_categories() as i64)
            .collect();
        let counts_tensor = SKTensor::integer(
            SK_ONE_HOT_CATEGORY_COUNTS_TENSOR,
            vec![self.columns.len()],
            counts,
        )?;
        SKModelState::new(metadata, vec![counts_tensor])
    }
}

impl<F: SKFloat> SKImportable<F> for SKOneHotEncoderModel {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_ONE_HOT_ENCODER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata().clone();
        let counts = sk_read_integer_tensor(state, SK_ONE_HOT_CATEGORY_COUNTS_TENSOR)?;
        if counts.len() != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![feature_count],
                found: vec![counts.len()],
            });
        }
        let kinds: Vec<SKParameterValue> = state
            .parameters()
            .get("one_hot.column_kinds")
            .and_then(|value| value.as_list())
            .map(<[SKParameterValue]>::to_vec)
            .ok_or(SKError::InvalidHyperparameter {
                name: "one_hot.column_kinds",
                reason: "the exported state does not carry the column kinds".to_string(),
            })?;
        if kinds.len() != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![feature_count],
                found: vec![kinds.len()],
            });
        }
        let mut columns = Vec::with_capacity(feature_count);
        for (column, kind) in kinds.iter().enumerate() {
            let kind = kind.as_string().ok_or(SKError::InvalidHyperparameter {
                name: "one_hot.column_kinds",
                reason: "a column kind is not a string".to_string(),
            })?;
            match kind {
                "nominal" => {
                    let labels = list_strings(state, &format!("one_hot.categories.{column}"))?;
                    let references: Vec<&str> = labels.iter().map(String::as_str).collect();
                    let (_, table) = sk_canonicalize_labels(&references);
                    columns.push(SKOneHotColumnCategories::Nominal(table));
                }
                "integer" => {
                    let codes =
                        list_integers(state, &format!("one_hot.integer_categories.{column}"))?;
                    columns.push(SKOneHotColumnCategories::Integer(
                        SKIntegerCategoryTable::from_codes(codes),
                    ));
                }
                _ => {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: "unknown category column kind",
                        suggestion: "export a nominal or integer category column",
                    });
                }
            }
        }
        let policy = match sk_string_parameter(&metadata, "handle_unknown", "reject") {
            "ignore" => SKUnknownPolicy::Ignore,
            _ => SKUnknownPolicy::Reject,
        };
        Ok(SKOneHotEncoderModel::from_columns(
            columns,
            sk_boolean_parameter(&metadata, "sparse_output", false),
            policy,
            sk_parse_backend_parameter(&metadata),
            SKExecutionMode::Automatic,
        ))
    }
}

/// Read a metadata parameter as a list of strings.
fn list_strings<F: SKFloat>(state: &SKModelState<F>, name: &str) -> Result<Vec<String>, SKError> {
    let value = state
        .parameters()
        .get(name)
        .ok_or(SKError::InvalidHyperparameter {
            name: "one_hot.categories",
            reason: format!("the exported state does not carry `{name}`"),
        })?;
    let values = value.as_list().ok_or(SKError::InvalidHyperparameter {
        name: "one_hot.categories",
        reason: format!("`{name}` is not a list"),
    })?;
    Ok(values
        .iter()
        .filter_map(SKParameterValue::as_string)
        .map(str::to_string)
        .collect())
}

/// Read a metadata parameter as a list of integer codes.
fn list_integers<F: SKFloat>(state: &SKModelState<F>, name: &str) -> Result<Vec<i64>, SKError> {
    let value = state
        .parameters()
        .get(name)
        .ok_or(SKError::InvalidHyperparameter {
            name: "one_hot.integer_categories",
            reason: format!("the exported state does not carry `{name}`"),
        })?;
    let values = value.as_list().ok_or(SKError::InvalidHyperparameter {
        name: "one_hot.integer_categories",
        reason: format!("`{name}` is not a list"),
    })?;
    Ok(values
        .iter()
        .filter_map(SKParameterValue::as_integer)
        .collect())
}
