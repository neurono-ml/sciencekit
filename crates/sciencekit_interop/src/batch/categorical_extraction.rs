//! Extraction of selected categorical columns through the library's
//! categorical view (spec `dataset-interop-core`).

use ndarray::{Array2, ArrayView2};
use sciencekit_common::{SKCategoricalView, SKLabelTable, SKNominalMatrixView};

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
                    let row_views: Vec<&[&str]> = borrowed.iter().map(Vec::as_slice).collect();
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
