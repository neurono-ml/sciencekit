//! Zero-copy views for inputs that already match the row-major layout (spec
//! `dataset-interop-core`).

use ndarray::ArrayView2;
use sciencekit_common::{SKDataView, SKError};

/// A borrowed, contiguous row-major matrix of the model scalar.
///
/// This is the only input shape the seam can alias: it already matches
/// [`SKDataView::Dense`]. Construction rejects a non-standard (for example
/// column-major) layout instead of silently copying it.
#[derive(Debug, Clone, Copy)]
pub struct SKRowMajorFeatureView<'a, F> {
    matrix: ArrayView2<'a, F>,
}

impl<'a, F> SKRowMajorFeatureView<'a, F> {
    /// Wrap a contiguous row-major matrix without copying.
    ///
    /// # Errors
    /// Returns [`SKError::Conversion`] when the view is not in standard
    /// row-major layout, because aliasing it would return wrong values.
    pub fn new(matrix: ArrayView2<'a, F>) -> Result<Self, SKError> {
        if !matrix.is_standard_layout() {
            return Err(SKError::Conversion(
                "the dense matrix is not contiguous row-major, so it cannot be borrowed as a feature view; reorder it first"
                    .to_string(),
            ));
        }
        Ok(SKRowMajorFeatureView { matrix })
    }

    /// The matrix shape `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        self.matrix.dim()
    }

    /// Borrow the view as an [`SKDataView`] — the same buffer, no copy.
    pub fn data_view(&self) -> SKDataView<'a, F> {
        SKDataView::Dense(self.matrix)
    }
}

/// A borrowed, contiguous single numeric column used as a one-column matrix.
///
/// Row-major one-column matrices and contiguous numeric columns share the exact
/// memory layout, so the column is reshaped without copying.
#[derive(Debug, Clone, Copy)]
pub struct SKNumericColumnView<'a, F> {
    values: &'a [F],
}

impl<'a, F> SKNumericColumnView<'a, F> {
    /// Wrap a contiguous numeric column without copying.
    pub fn new(values: &'a [F]) -> Self {
        SKNumericColumnView { values }
    }

    /// The number of rows.
    pub fn number_of_rows(&self) -> usize {
        self.values.len()
    }

    /// Reshape the column to an `(rows, 1)` matrix view over the same buffer.
    pub fn matrix_view(&self) -> ArrayView2<'a, F> {
        ArrayView2::from_shape((self.values.len(), 1), self.values)
            .expect("a slice always reshapes to a one-column matrix")
    }

    /// Borrow the column as an [`SKDataView`] — the same buffer, no copy.
    pub fn data_view(&self) -> SKDataView<'a, F> {
        SKDataView::Dense(self.matrix_view())
    }

    /// The number of columns, always one.
    pub fn number_of_columns(&self) -> usize {
        1
    }
}
