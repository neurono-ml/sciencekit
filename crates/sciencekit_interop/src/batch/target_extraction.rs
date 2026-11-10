//! Extraction of the selected target column through the library's target view
//! (spec `dataset-interop-core`).

use ndarray::{Array1, ArrayView1};
use num_traits::cast;
use sciencekit_common::{
    SK_CATEGORICAL_MISSING_CODE, SKError, SKFloat, SKLabelTable, SKTargetView,
    sk_canonicalize_labels,
};

use super::SKTabularBatch;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::schema::SKLogicalType;

/// An owned target column, ready to be exposed through [`SKTargetView`].
///
/// Continuous and integer targets are stored as scalar vectors; a nominal
/// target keeps both its deterministic label table and the canonical indices,
/// so a classifier can round-trip labels without ambiguity.
#[derive(Debug, Clone, PartialEq)]
pub enum SKTargetExtraction<F: SKFloat> {
    /// Continuous response values in the model scalar.
    Continuous(Array1<F>),
    /// Integer-valued targets, including category codes.
    Integer(Array1<i64>),
    /// Nominal labels with their canonical index mapping.
    Nominal {
        /// Canonical zero-based class indices; nulls carry the documented
        /// missing code when the policy allows them.
        indices: Array1<i64>,
        /// The original labels per row; `None` marks a missing entry.
        labels: Vec<Option<String>>,
        /// The deterministic label-to-index table.
        table: SKLabelTable,
    },
}

impl<F: SKFloat> SKTargetExtraction<F> {
    /// The number of targets.
    pub fn len(&self) -> usize {
        match self {
            SKTargetExtraction::Continuous(values) => values.len(),
            SKTargetExtraction::Integer(values) => values.len(),
            SKTargetExtraction::Nominal { indices, .. } => indices.len(),
        }
    }

    /// Whether the extraction is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The deterministic label table, when the target is nominal.
    pub fn label_table(&self) -> Option<&SKLabelTable> {
        match self {
            SKTargetExtraction::Nominal { table, .. } => Some(table),
            _ => None,
        }
    }

    /// The canonical indices, when the target is nominal.
    pub fn canonical_indices(&self) -> Option<ArrayView1<'_, i64>> {
        match self {
            SKTargetExtraction::Nominal { indices, .. } => Some(indices.view()),
            _ => None,
        }
    }

    /// Run a consumer with the borrowed [`SKTargetView`].
    ///
    /// A nominal target with missing entries cannot be represented as borrowed
    /// text, so it is exposed as its canonical integer view instead.
    pub fn with_view<R>(&self, consumer: impl FnOnce(SKTargetView<'_, F>) -> R) -> R {
        match self {
            SKTargetExtraction::Continuous(values) => {
                consumer(SKTargetView::Continuous(values.view()))
            }
            SKTargetExtraction::Integer(values) => {
                consumer(SKTargetView::Integer(values.view()))
            }
            SKTargetExtraction::Nominal { indices, labels, .. } => {
                if labels.iter().all(Option::is_some) {
                    let borrowed: Vec<&str> = labels
                        .iter()
                        .filter_map(|label| label.as_deref())
                        .collect();
                    consumer(SKTargetView::Nominal(&borrowed))
                } else {
                    consumer(SKTargetView::Integer(indices.view()))
                }
            }
        }
    }

    /// Convert the target to owned model-scalar values.
    ///
    /// # Errors
    /// Returns [`SKError::Conversion`] when an integer target is outside the
    /// scalar's exact range.
    pub fn to_scalar_values(&self) -> Result<Array1<F>, SKError> {
        match self {
            SKTargetExtraction::Continuous(values) => Ok(values.clone()),
            SKTargetExtraction::Integer(values) => values
                .iter()
                .map(|value| {
                    cast::<i64, F>(*value).ok_or_else(|| {
                        SKError::Conversion(format!(
                            "target value {value} is not representable in the model scalar"
                        ))
                    })
                })
                .collect(),
            SKTargetExtraction::Nominal { indices, .. } => indices
                .iter()
                .map(|value| {
                    cast::<i64, F>(*value).ok_or_else(|| {
                        SKError::Conversion(format!(
                            "target index {value} is not representable in the model scalar"
                        ))
                    })
                })
                .collect(),
        }
    }
}

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
        let target_index = resolved
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
            SKLogicalType::CategoryCode => {
                Ok(SKTargetExtraction::Integer(sk_category_codes(
                    column_schema.name(),
                    column,
                    policy,
                )?))
            }
            SKLogicalType::Nominal => sk_nominal_target(column_schema.name(), column, policy),
        }
    }
}

fn sk_integer_target(
    column: &crate::columns::SKColumnData,
) -> Result<Array1<i64>, SKError> {
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
                SKMissingValuePolicy::CategoricalMissingCode => {
                    Ok(SK_CATEGORICAL_MISSING_CODE)
                }
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
