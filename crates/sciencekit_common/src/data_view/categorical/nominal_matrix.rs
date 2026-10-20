//! `SKNominalMatrixView`: a zero-copy, row-major matrix of borrowed labels.

/// A zero-copy view over a row-major matrix of borrowed text labels.
///
/// The outer slice indexes rows; each inner slice holds that row's column
/// labels. Labels are borrowed (`&'a str`) — no text is copied.
#[derive(Debug, Clone, Copy)]
pub struct SKNominalMatrixView<'a> {
    rows: &'a [&'a [&'a str]],
}

impl<'a> SKNominalMatrixView<'a> {
    /// Wrap a row-major matrix of borrowed labels.
    pub fn new(rows: &'a [&'a [&'a str]]) -> Self {
        SKNominalMatrixView { rows }
    }

    /// The matrix shape `(rows, columns)`; columns come from the first row
    /// (zero when there is no row).
    pub fn shape(&self) -> (usize, usize) {
        let rows_count = self.rows.len();
        let columns_count = self.rows.first().map_or(0, |row| row.len());
        (rows_count, columns_count)
    }

    /// The number of rows.
    pub fn number_of_rows(&self) -> usize {
        self.rows.len()
    }

    /// The number of columns (from the first row; zero when empty).
    pub fn number_of_columns(&self) -> usize {
        self.rows.first().map_or(0, |row| row.len())
    }

    /// The label at `row`/`column`.
    ///
    /// # Panics
    /// Panics when the indices are out of bounds.
    pub fn value(&self, row: usize, column: usize) -> &'a str {
        self.rows[row][column]
    }

    /// The row slice at `row`.
    ///
    /// # Panics
    /// Panics when `row` is out of bounds.
    pub fn row(&self, row: usize) -> &'a [&'a str] {
        self.rows[row]
    }

    /// Whether the matrix has no rows or no columns.
    pub fn is_empty(&self) -> bool {
        self.number_of_rows() == 0 || self.number_of_columns() == 0
    }
}
