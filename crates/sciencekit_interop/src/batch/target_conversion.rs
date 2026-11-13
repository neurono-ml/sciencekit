//! Conversion of a selected target column into an owned target extraction
//! (spec `dataset-interop-core`).

use ndarray::Array1;
use sciencekit_common::{SK_CATEGORICAL_MISSING_CODE, SKError, SKFloat, sk_canonicalize_labels};

use super::{SKTabularBatch, SKTargetExtraction};
use crate::missing::SKMissingValuePolicy;
use crate::schema::SKLogicalType;
use crate::selection::SKColumnSelection;

impl SKTabularBatch {
    /// Extract the selected target column under the declared policy.
    ///
    /// # Errors
    /// Returns [`SKError::UnsupportedOperation`] when the selection has no
    /// target, and propagates null-policy and representability failures.
    pub fn extract_target<F: SKFloat>(
        &self,
        selection: &SKColumnSelection,
        policy: SKMissingValuePolicy,
    ) -> Result<SKTargetExtraction<F>, SKError> {
        let resolved = selection.resolve(self.schema())?;
        let target_index =
            resolved
                .target_index()
                .ok_or_else(|| SKError::UnsupportedOperation {
                    operation: "target_extraction",
                    reason: "the selection does not declare a target column".to_string(),
                })?;
        let column_schema = &self.schema().columns()[target_index];
        let column = &self.columns()[target_index];

        match column_schema.logical_type() {
            SKLogicalType::Float32 | SKLogicalType::Float64 => {
                let values = crate::columns::sk_coerce_numeric_column::<F>(
                    column_schema.name(),
                    column,
                    policy,
                )?;
                Ok(SKTargetExtraction::Continuous(values))
            }
            SKLogicalType::Integer64 | SKLogicalType::Boolean => {
                if column.has_nulls() {
                    match policy {
                        SKMissingValuePolicy::NumericNaN => {
                            let values = crate::columns::sk_coerce_numeric_column::<F>(
                                column_schema.name(),
                                column,
                                policy,
                            )?;
                            Ok(SKTargetExtraction::Continuous(values))
                        }
                        SKMissingValuePolicy::ConstantFill(_) => {
                            let values = crate::columns::sk_coerce_numeric_column::<F>(
                                column_schema.name(),
                                column,
                                policy,
                            )?;
                            Ok(SKTargetExtraction::Continuous(values))
                        }
                        _ => Err(SKError::NullValueInColumn {
                            column: column_schema.name().to_string(),
                            policy: policy.name(),
                        }),
                    }
                } else {
                    Ok(SKTargetExtraction::Integer(sk_integer_target(column)?))
                }
            }
            SKLogicalType::CategoryCode => Ok(SKTargetExtraction::Integer(sk_category_codes(
                column_schema.name(),
                column,
                policy,
            )?)),
            SKLogicalType::Nominal => sk_nominal_target(column_schema.name(), column, policy),
        }
    }
}

fn sk_integer_target(column: &crate::columns::SKColumnData) -> Result<Array1<i64>, SKError> {
    match column {
        crate::columns::SKColumnData::Integer64(values) => Ok(values
            .iter()
            .map(|value| value.unwrap_or_default())
            .collect()),
        crate::columns::SKColumnData::Boolean(values) => Ok(values
            .iter()
            .map(|value| if value.unwrap_or_default() { 1 } else { 0 })
            .collect()),
        _ => Err(SKError::SchemaMismatch {
            expected: "an integer or boolean target column".to_string(),
            found: column.logical_type().to_string(),
        }),
    }
}

fn sk_category_codes(
    column_name: &str,
    column: &crate::columns::SKColumnData,
    policy: SKMissingValuePolicy,
) -> Result<Array1<i64>, SKError> {
    let crate::columns::SKColumnData::CategoryCode(values) = column else {
        return Err(SKError::SchemaMismatch {
            expected: "a category-coded target column".to_string(),
            found: column.logical_type().to_string(),
        });
    };
    values
        .iter()
        .map(|value| match value {
            Some(code) => Ok(*code),
            None => match policy {
                SKMissingValuePolicy::CategoricalMissingCode => Ok(SK_CATEGORICAL_MISSING_CODE),
                _ => Err(SKError::NullValueInColumn {
                    column: column_name.to_string(),
                    policy: policy.name(),
                }),
            },
        })
        .collect()
}

fn sk_nominal_target<F: SKFloat>(
    column_name: &str,
    column: &crate::columns::SKColumnData,
    policy: SKMissingValuePolicy,
) -> Result<SKTargetExtraction<F>, SKError> {
    let crate::columns::SKColumnData::Nominal(labels) = column else {
        return Err(SKError::SchemaMismatch {
            expected: "a nominal target column".to_string(),
            found: column.logical_type().to_string(),
        });
    };

    if labels.iter().any(Option::is_none)
        && !matches!(policy, SKMissingValuePolicy::CategoricalMissingCode)
    {
        return Err(SKError::NullValueInColumn {
            column: column_name.to_string(),
            policy: policy.name(),
        });
    }

    let present: Vec<&str> = labels.iter().filter_map(|label| label.as_deref()).collect();
    let (canonical, table) = sk_canonicalize_labels(&present);
    let mut canonical_iter = canonical.into_iter();
    let indices: Vec<i64> = labels
        .iter()
        .map(|label| match label {
            Some(_) => canonical_iter.next().unwrap_or_default() as i64,
            None => SK_CATEGORICAL_MISSING_CODE,
        })
        .collect();

    Ok(SKTargetExtraction::Nominal {
        indices: Array1::from_vec(indices),
        labels: labels.clone(),
        table,
    })
}
