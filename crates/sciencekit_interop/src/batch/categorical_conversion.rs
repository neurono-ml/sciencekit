//! Conversion of a selected categorical feature block into an owned extraction
//! (spec `dataset-interop-core`).

use ndarray::Array2;
use sciencekit_common::{
    SK_CATEGORICAL_MISSING_CODE, SKError, SKLabelTable, sk_canonicalize_labels,
};

use super::{SKCategoricalExtraction, SKTabularBatch};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::schema::SKLogicalType;
use crate::selection::SKColumnSelection;

impl SKTabularBatch {
    /// Extract the selected categorical features under the declared policy.
    ///
    /// # Errors
    /// Returns [`SKError::UnsupportedColumnType`] when a selected column is
    /// numeric, and [`SKError::NullValueInColumn`] when the policy cannot
    /// handle a null.
    pub fn extract_categorical(
        &self,
        selection: &SKColumnSelection,
        policy: SKMissingValuePolicy,
    ) -> Result<SKCategoricalExtraction, SKError> {
        let resolved = selection.resolve(self.schema())?;
        let indices = resolved.feature_indices();

        for index in indices {
            let column = &self.schema().columns()[*index];
            if !column.logical_type().is_categorical() {
                return Err(SKError::UnsupportedColumnType {
                    column: column.name().to_string(),
                    logical_type: column.logical_type().to_string(),
                    suggestion:
                        "select only nominal or category-coded columns for the categorical view"
                            .to_string(),
                });
            }
        }

        let needs_codes = indices.iter().any(|index| {
            matches!(
                self.schema().columns()[*index].logical_type(),
                SKLogicalType::CategoryCode
            ) || self.columns()[*index].has_nulls()
        });

        if !needs_codes {
            return Ok(SKCategoricalExtraction::Nominal {
                rows: self.nominal_rows(indices),
                number_of_columns: indices.len(),
            });
        }

        let mut codes = Array2::zeros((self.number_of_rows(), indices.len()));
        let mut label_tables = Vec::with_capacity(indices.len());
        for (offset, index) in indices.iter().enumerate() {
            let column_schema = &self.schema().columns()[*index];
            let column = &self.columns()[*index];
            let (column_codes, table) = match column {
                SKColumnData::CategoryCode(values) => {
                    (sk_code_column(column_schema.name(), values, policy)?, None)
                }
                SKColumnData::Nominal(values) => {
                    let (values, table) = sk_nominal_codes(column_schema.name(), values, policy)?;
                    (values, Some(table))
                }
                _ => {
                    return Err(SKError::UnsupportedColumnType {
                        column: column_schema.name().to_string(),
                        logical_type: column_schema.logical_type().to_string(),
                        suggestion:
                            "select only nominal or category-coded columns for the categorical view"
                                .to_string(),
                    });
                }
            };
            for (row, code) in column_codes.iter().enumerate() {
                codes[[row, offset]] = *code;
            }
            label_tables.push(table);
        }

        Ok(SKCategoricalExtraction::Integer {
            codes,
            label_tables,
        })
    }

    fn nominal_rows(&self, indices: &[usize]) -> Vec<Vec<String>> {
        (0..self.number_of_rows())
            .map(|row| {
                indices
                    .iter()
                    .map(|index| match &self.columns()[*index] {
                        SKColumnData::Nominal(values) => values[row].clone().unwrap_or_default(),
                        _ => String::new(),
                    })
                    .collect()
            })
            .collect()
    }
}

fn sk_code_column(
    column_name: &str,
    values: &[Option<i64>],
    policy: SKMissingValuePolicy,
) -> Result<Vec<i64>, SKError> {
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

fn sk_nominal_codes(
    column_name: &str,
    values: &[Option<String>],
    policy: SKMissingValuePolicy,
) -> Result<(Vec<i64>, SKLabelTable), SKError> {
    if values.iter().any(Option::is_none)
        && !matches!(policy, SKMissingValuePolicy::CategoricalMissingCode)
    {
        return Err(SKError::NullValueInColumn {
            column: column_name.to_string(),
            policy: policy.name(),
        });
    }

    let present: Vec<&str> = values.iter().filter_map(|value| value.as_deref()).collect();
    let (canonical, table) = sk_canonicalize_labels(&present);
    let mut canonical_iter = canonical.into_iter();
    let codes = values
        .iter()
        .map(|value| match value {
            Some(_) => canonical_iter.next().unwrap_or_default() as i64,
            None => SK_CATEGORICAL_MISSING_CODE,
        })
        .collect();
    Ok((codes, table))
}
