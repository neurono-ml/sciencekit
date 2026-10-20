//! The `SKCategoricalView` enum and its inherent methods.

use ndarray::ArrayView2;

use super::nominal_matrix::SKNominalMatrixView;

/// A zero-copy view over categorical features — the categorical counterpart of
/// `SKDataView`.
///
/// `#[non_exhaustive]` so new categorical representations can be added later
/// without breaking consumers that match with a wildcard.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SKCategoricalView<'a> {
    /// Nominal (text) columns, row-major, borrowed.
    Nominal(SKNominalMatrixView<'a>),
    /// Integer-coded columns; see [`crate::SK_CATEGORICAL_MISSING_CODE`] for the
    /// missing sentinel.
    Integer(ArrayView2<'a, i64>),
}

impl SKCategoricalView<'_> {
    /// The representation label, used for one-time dispatch and diagnostics.
    pub fn representation(&self) -> &'static str {
        match self {
            SKCategoricalView::Nominal(_) => "nominal",
            SKCategoricalView::Integer(_) => "integer",
        }
    }

    /// The matrix shape `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        match self {
            SKCategoricalView::Nominal(view) => view.shape(),
            SKCategoricalView::Integer(view) => view.dim(),
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

    /// Whether the matrix has no rows or no columns.
    pub fn is_empty(&self) -> bool {
        let (rows_count, columns_count) = self.shape();
        rows_count == 0 || columns_count == 0
    }
}

impl<'a> From<SKNominalMatrixView<'a>> for SKCategoricalView<'a> {
    fn from(view: SKNominalMatrixView<'a>) -> Self {
        SKCategoricalView::Nominal(view)
    }
}

impl<'a> From<ArrayView2<'a, i64>> for SKCategoricalView<'a> {
    fn from(view: ArrayView2<'a, i64>) -> Self {
        SKCategoricalView::Integer(view)
    }
}
