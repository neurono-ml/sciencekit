//! Extraction of selected categorical columns through the library's
//! categorical view (spec `dataset-interop-core`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{
    SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKError, SKLabelTable, SKNominalMatrixView,
    sk_canonicalize_labels,
};

use super::SKTabularBatch;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::schema::SKLogicalType;

/// An owned categorical selection, ready to be exposed through
/// [`SKCategoricalView`].
///
/// Nominal columns whose every entry is present surface as the nominal view.
/// A category-coded column, or a nominal column carrying a missing entry under
/// the categorical policy, surfaces as integer codes with the documented
/// missing sentinel.
#[derive(Debug, Clone, PartialEq)]
pub enum SKCategoricalExtraction {
    /// Nominal labels, all present, in row-major order.
    Nominal {
        /// The labels per row.
        rows: Vec<Vec<String>>,
        /// The number of columns (zero when there is no row).
        number_of_columns: usize,
    },
    /// Integer category codes, including the missing sentinel.
    Integer {
        /// The row-major code matrix.
        codes: Array2<i64>,
        /// The label table per selected column, when the column was nominal.
        label_tables: Vec<Option<SKLabelTable>>,
    },
}

impl SKCategoricalExtraction {
    /// The `(rows, columns)` shape.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            SKCategoricalExtraction::Nominal {
                rows,
                number_of_columns,
            } => (rows.len(), *number_of_columns),
            SKCategoricalExtraction::Integer { codes, .. } => codes.dim(),
        }
    }

    /// The number of rows.
    pub fn number_of_rows(&self) -> usize {
        self.shape().0
    }

    /// The number of columns.
    pub fn number_of_columns(&self) -> usize {
        self.shape().1
    }

    /// Whether the extraction is empty.
    pub fn is_empty(&self) -> bool {
        let (rows_count, columns_count) = self.shape();
        rows_count == 0 || columns_count == 0
    }

    /// The integer codes, when the extraction carries them.
    pub fn codes(&self) -> Option<ArrayView2<'_, i64>> {
        match self {
            SKCategoricalExtraction::Integer { codes, .. } => Some(codes.view()),
            _ => None,
        }
    }

    /// The per-column label tables, empty for a purely nominal extraction.
    pub fn label_tables(&self) -> &[Option<SKLabelTable>] {
        match self {
            SKCategoricalExtraction::Nominal { .. } => &[],
            SKCategoricalExtraction::Integer { label_tables, .. } => label_tables,
        }
    }

    /// Run a consumer with the borrowed [`SKCategoricalView`].
    pub fn with_view<R>(&self, consumer: impl FnOnce(SKCategoricalView<'_>) -> R) -> R {
        match self {
            SKCategoricalExtraction::Nominal {
                rows,
                number_of_columns,
            } => {
                if rows.is_empty() || *number_of_columns == 0 {
                    let borrowed: Vec<Vec<&str>> = Vec::new();
                    let row_views: Vec<&[&str]> =
                        borrowed.iter().map(Vec::as_slice).collect();
                    return consumer(SKCategoricalView::Nominal(SKNominalMatrixView::new(
                        &row_views,
                    )));
                }
                let borrowed: Vec<Vec<&str>> = rows
                    .iter()
                    .map(|row| row.iter().map(String::as_str).collect())
                    .collect();
                let row_views: Vec<&[&str]> = borrowed.iter().map(Vec::as_slice).collect();
                consumer(SKCategoricalView::Nominal(SKNominalMatrixView::new(
                    &row_views,
                )))
            }
            SKCategoricalExtraction::Integer { codes, .. } => {
                consumer(SKCategoricalView::Integer(codes.view()))
            }
        }
    }
}

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
                SKColumnData::CategoryCode(values) => (
                    sk_code_column(column_schema.name(), values, policy)?,
                    None,
                ),
                SKColumnData::Nominal(values) => {
                    let (values, table) =
                        sk_nominal_codes(column_schema.name(), values, policy)?;
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
                        SKColumnData::Nominal(values) => {
                            values[row].clone().unwrap_or_default()
                        }
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

    let present: Vec<&str> = values
        .iter()
        .filter_map(|value| value.as_deref())
        .collect();
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
