//! The owned nominal block backing a tabular dataset (spec `tabular-sources`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{
    SK_CATEGORICAL_MISSING_CODE, SKCategoricalView, SKError, SKLabelTable, sk_canonicalize_labels,
};

use crate::batch::SKCategoricalExtraction;

/// An owned categorical matrix of integer codes plus one label table per column.
///
/// Missing entries use [`sciencekit_common::SK_CATEGORICAL_MISSING_CODE`]. The
/// block owns its code matrix and label tables, so it outlives the source that
/// produced it and can be shared across threads.
#[derive(Debug, Clone, PartialEq)]
pub struct SKNominalMatrixOwned {
    codes: Array2<i64>,
    label_tables: Vec<Option<SKLabelTable>>,
}

impl SKNominalMatrixOwned {
    /// Build an owned block from a code matrix and its per-column label tables.
    ///
    /// # Errors
    /// Returns [`SKError::SchemaMismatch`] when the number of label tables does
    /// not match the number of code columns, or when a code cannot be decoded by
    /// its column's label table (a code outside `[0, number_of_classes)` other
    /// than the missing sentinel).
    pub fn new(
        codes: Array2<i64>,
        label_tables: Vec<Option<SKLabelTable>>,
    ) -> Result<Self, SKError> {
        if label_tables.len() != codes.ncols() {
            return Err(SKError::SchemaMismatch {
                expected: format!("{} label tables", codes.ncols()),
                found: format!("{} label tables", label_tables.len()),
            });
        }
        for (column, label_table) in label_tables.iter().enumerate() {
            let Some(label_table) = label_table else {
                continue;
            };
            let number_of_classes = label_table.number_of_classes() as i64;
            let undecodable = codes.column(column).iter().copied().find(|code| {
                *code != SK_CATEGORICAL_MISSING_CODE && (*code < 0 || *code >= number_of_classes)
            });
            if let Some(code) = undecodable {
                return Err(SKError::SchemaMismatch {
                    expected: format!(
                        "codes within the column's label table (0 to {}) or the missing sentinel",
                        number_of_classes.saturating_sub(1)
                    ),
                    found: format!("code {code}"),
                });
            }
        }
        Ok(SKNominalMatrixOwned {
            codes,
            label_tables,
        })
    }

    /// The `(rows, columns)` shape.
    pub fn shape(&self) -> (usize, usize) {
        self.codes.dim()
    }

    /// The number of rows.
    pub fn number_of_rows(&self) -> usize {
        self.shape().0
    }

    /// The number of categorical columns.
    pub fn number_of_columns(&self) -> usize {
        self.shape().1
    }

    /// The code matrix, with the missing sentinel in missing entries.
    pub fn codes(&self) -> ArrayView2<'_, i64> {
        self.codes.view()
    }

    /// The label table of one column, `None` for a category-coded column.
    pub fn label_table(&self, column: usize) -> Option<&SKLabelTable> {
        self.label_tables.get(column).and_then(Option::as_ref)
    }

    /// Run a consumer with the borrowed categorical view.
    pub fn with_view<R>(&self, consumer: impl FnOnce(SKCategoricalView<'_>) -> R) -> R {
        consumer(SKCategoricalView::Integer(self.codes.view()))
    }

    /// Build the owned block from a batch's categorical extraction.
    ///
    /// A nominal extraction is canonicalized column by column so the owned
    /// block always carries codes and a label table; an integer extraction is
    /// adopted as-is.
    pub(crate) fn from_extraction(extraction: SKCategoricalExtraction) -> Result<Self, SKError> {
        match extraction {
            SKCategoricalExtraction::Integer {
                codes,
                label_tables,
            } => SKNominalMatrixOwned::new(codes, label_tables),
            SKCategoricalExtraction::Nominal {
                rows,
                number_of_columns,
            } => {
                let mut codes = Array2::zeros((rows.len(), number_of_columns));
                let mut label_tables = Vec::with_capacity(number_of_columns);
                for column in 0..number_of_columns {
                    let labels: Vec<&str> = rows
                        .iter()
                        .map(|row| row.get(column).map(String::as_str).unwrap_or_default())
                        .collect();
                    let (column_codes, table) = sk_canonicalize_labels(&labels);
                    for (row, code) in column_codes.into_iter().enumerate() {
                        codes[[row, column]] = code as i64;
                    }
                    label_tables.push(Some(table));
                }
                SKNominalMatrixOwned::new(codes, label_tables)
            }
        }
    }
}
