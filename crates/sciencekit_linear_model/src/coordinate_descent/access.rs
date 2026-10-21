//! The representation-agnostic column access the coordinate-descent iteration
//! drives (spec `coordinate-descent-solver`, Decision 6).
//!
//! Dense and sparse designs share one coordinate loop through this trait: the
//! iteration only asks for a column's squared norm, its signed dot with the
//! residual and a residual update. A sparse design is never densified — its
//! column entries are the stored triplets, gathered once into per-column
//! index/value lists.

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, azip};
use sciencekit_common::SKFloat;
use sprs::CsMatView;

/// The per-coordinate operations the coordinate-descent iteration needs.
pub(crate) trait SKColumnAccess<F: SKFloat> {
    /// The number of rows of the design.
    fn rows_count(&self) -> usize;

    /// The number of columns of the design.
    fn columns_count(&self) -> usize;

    /// The squared Euclidean norm `Σ xᵢⱼ²` of one column.
    fn column_squared_norm(&self, column: usize) -> F;

    /// The signed dot `Σ xᵢⱼ rᵢ` of one column with the residual.
    fn signed_column_dot(&self, column: usize, residual: ArrayView1<F>) -> F;

    /// Subtract `delta · xⱼ` from the residual.
    fn update_residual(&self, column: usize, delta: F, residual: ArrayViewMut1<F>);
}

/// Dense column access over a borrowed row-major design.
pub(crate) struct SKDenseColumns<'a, F> {
    view: ArrayView2<'a, F>,
}

impl<'a, F: SKFloat> SKDenseColumns<'a, F> {
    /// Wrap a borrowed dense design.
    pub(crate) fn new(view: ArrayView2<'a, F>) -> Self {
        Self { view }
    }
}

impl<F: SKFloat> SKColumnAccess<F> for SKDenseColumns<'_, F> {
    fn rows_count(&self) -> usize {
        self.view.nrows()
    }

    fn columns_count(&self) -> usize {
        self.view.ncols()
    }

    fn column_squared_norm(&self, column: usize) -> F {
        let values = self.view.column(column);
        values.dot(&values)
    }

    fn signed_column_dot(&self, column: usize, residual: ArrayView1<F>) -> F {
        self.view.column(column).dot(&residual)
    }

    fn update_residual(&self, column: usize, delta: F, mut residual: ArrayViewMut1<F>) {
        azip!((entry in &mut residual, &value in &self.view.column(column)) {
            *entry = *entry - delta * value;
        });
    }
}

/// Sparse column access over a borrowed CSR design, gathered into per-column
/// `(row, value)` lists without ever densifying the design.
pub(crate) struct SKSparseColumns<F> {
    columns: Vec<Vec<(usize, F)>>,
    rows_count: usize,
}

impl<F: SKFloat> SKSparseColumns<F> {
    /// Gather a borrowed CSR design's column entries.
    pub(crate) fn new(view: CsMatView<'_, F>) -> Self {
        let rows_count = view.rows();
        let mut columns: Vec<Vec<(usize, F)>> = vec![Vec::new(); view.cols()];
        for (row_index, sparse_row) in view.outer_iterator().enumerate() {
            for (&column_index, &value) in
                sparse_row.indices().iter().zip(sparse_row.data())
            {
                columns[column_index].push((row_index, value));
            }
        }
        Self {
            columns,
            rows_count,
        }
    }
}

impl<F: SKFloat> SKColumnAccess<F> for SKSparseColumns<F> {
    fn rows_count(&self) -> usize {
        self.rows_count
    }

    fn columns_count(&self) -> usize {
        self.columns.len()
    }

    fn column_squared_norm(&self, column: usize) -> F {
        self.columns[column]
            .iter()
            .map(|(_, value)| *value * *value)
            .fold(F::zero(), |accumulator, term| accumulator + term)
    }

    fn signed_column_dot(&self, column: usize, residual: ArrayView1<F>) -> F {
        self.columns[column]
            .iter()
            .map(|(row, value)| *value * residual[*row])
            .fold(F::zero(), |accumulator, term| accumulator + term)
    }

    fn update_residual(&self, column: usize, delta: F, mut residual: ArrayViewMut1<F>) {
        for (row, value) in &self.columns[column] {
            residual[*row] = residual[*row] - delta * *value;
        }
    }
}
