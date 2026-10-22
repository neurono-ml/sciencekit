//! The representation-agnostic design abstraction the logistic solvers drive
//! (spec `logistic-solver`).
//!
//! Dense and sparse designs share one interface: a full matrix–vector product,
//! a single row's dot product and a scaled row addition. A sparse design is
//! never densified — only the `O(rows)` linear predictor is retained.

use ndarray::{ArrayView1, ArrayViewMut1, azip};
use sciencekit_common::SKFloat;
use sciencekit_math::sk_csr_dense_product;
use sprs::CsMatView;

/// The per-row operations the logistic solvers need from a design.
pub(crate) trait SKLogisticDesign<F: SKFloat> {
    /// The number of rows.
    fn rows_count(&self) -> usize;
    /// The number of columns (features).
    fn columns_count(&self) -> usize;
    /// Write `out[i] = Σⱼ xᵢⱼ wⱼ` for every row.
    fn dot_all(&self, weights: ArrayView1<F>, out: ArrayViewMut1<F>);
    /// The dot product of one row with the weights.
    fn row_dot(&self, row: usize, weights: ArrayView1<F>) -> F;
    /// Add `scale · xᵣ` to a column vector.
    fn add_row_scaled(&self, row: usize, scale: F, out: ArrayViewMut1<F>);
    /// The squared Euclidean norm `Σⱼ xᵢⱼ²` of one row.
    fn row_squared_norm(&self, row: usize) -> F;
    /// The dot product of one column with a vector.
    fn column_dot(&self, column: usize, vector: ArrayView1<F>) -> F;
    /// The weighted squared norm `Σᵢ cᵢ xᵢⱼ²` of one column.
    fn weighted_column_squared_norm(&self, column: usize, weights: ArrayView1<F>) -> F;
    /// Add `scale · xⱼ` to a row vector.
    fn add_column_scaled(&self, column: usize, scale: F, out: ArrayViewMut1<F>);
}

/// Dense design access over a borrowed row-major matrix.
pub(crate) struct SKDenseDesign<'a, F> {
    view: ndarray::ArrayView2<'a, F>,
}

impl<'a, F: SKFloat> SKDenseDesign<'a, F> {
    /// Wrap a borrowed dense design.
    pub(crate) fn new(view: ndarray::ArrayView2<'a, F>) -> Self {
        Self { view }
    }
}

impl<F: SKFloat> SKLogisticDesign<F> for SKDenseDesign<'_, F> {
    fn rows_count(&self) -> usize {
        self.view.nrows()
    }

    fn columns_count(&self) -> usize {
        self.view.ncols()
    }

    fn dot_all(&self, weights: ArrayView1<F>, mut out: ArrayViewMut1<F>) {
        out.assign(&self.view.dot(&weights));
    }

    fn row_dot(&self, row: usize, weights: ArrayView1<F>) -> F {
        self.view.row(row).dot(&weights)
    }

    fn add_row_scaled(&self, row: usize, scale: F, mut out: ArrayViewMut1<F>) {
        azip!((entry in &mut out, &value in &self.view.row(row)) {
            *entry = *entry + scale * value;
        });
    }

    fn row_squared_norm(&self, row: usize) -> F {
        let values = self.view.row(row);
        values.dot(&values)
    }

    fn column_dot(&self, column: usize, vector: ArrayView1<F>) -> F {
        self.view.column(column).dot(&vector)
    }

    fn weighted_column_squared_norm(&self, column: usize, weights: ArrayView1<F>) -> F {
        let values = self.view.column(column);
        values
            .iter()
            .zip(weights)
            .fold(F::zero(), |accumulator, (&value, &weight)| {
                accumulator + weight * value * value
            })
    }

    fn add_column_scaled(&self, column: usize, scale: F, mut out: ArrayViewMut1<F>) {
        azip!((entry in &mut out, &value in &self.view.column(column)) {
            *entry = *entry + scale * value;
        });
    }
}

/// Sparse design access over a borrowed CSR matrix, never densified.
pub(crate) struct SKSparseDesign<'a, F> {
    view: CsMatView<'a, F>,
}

impl<'a, F: SKFloat> SKSparseDesign<'a, F> {
    /// Wrap a borrowed CSR design.
    pub(crate) fn new(view: CsMatView<'a, F>) -> Self {
        Self { view }
    }
}

impl<F: SKFloat> SKLogisticDesign<F> for SKSparseDesign<'_, F> {
    fn rows_count(&self) -> usize {
        self.view.rows()
    }

    fn columns_count(&self) -> usize {
        self.view.cols()
    }

    fn dot_all(&self, weights: ArrayView1<F>, mut out: ArrayViewMut1<F>) {
        let column = weights.to_owned().insert_axis(ndarray::Axis(1));
        let product = sk_csr_dense_product(&self.view, &column.view());
        out.assign(&product.column(0));
    }

    fn row_dot(&self, row: usize, weights: ArrayView1<F>) -> F {
        let sparse_row = self.view.outer_view(row).expect("row within bounds");
        sparse_row
            .indices()
            .iter()
            .zip(sparse_row.data())
            .map(|(&column, &value)| value * weights[column])
            .fold(F::zero(), |accumulator, term| accumulator + term)
    }

    fn add_row_scaled(&self, row: usize, scale: F, mut out: ArrayViewMut1<F>) {
        let sparse_row = self.view.outer_view(row).expect("row within bounds");
        sparse_row
            .indices()
            .iter()
            .zip(sparse_row.data())
            .for_each(|(&column, &value)| {
                out[column] = out[column] + scale * value;
            });
    }

    fn row_squared_norm(&self, row: usize) -> F {
        let sparse_row = self.view.outer_view(row).expect("row within bounds");
        sparse_row
            .data()
            .iter()
            .fold(F::zero(), |accumulator, value| {
                accumulator + *value * *value
            })
    }

    fn column_dot(&self, column: usize, vector: ArrayView1<F>) -> F {
        let mut total = F::zero();
        for (row, sparse_row) in self.view.outer_iterator().enumerate() {
            if let Ok(position) = sparse_row.indices().binary_search(&column) {
                total = total + sparse_row.data()[position] * vector[row];
            }
        }
        total
    }

    fn weighted_column_squared_norm(&self, column: usize, weights: ArrayView1<F>) -> F {
        let mut total = F::zero();
        for (row, sparse_row) in self.view.outer_iterator().enumerate() {
            if let Ok(position) = sparse_row.indices().binary_search(&column) {
                let value = sparse_row.data()[position];
                total = total + weights[row] * value * value;
            }
        }
        total
    }

    fn add_column_scaled(&self, column: usize, scale: F, mut out: ArrayViewMut1<F>) {
        for (row, sparse_row) in self.view.outer_iterator().enumerate() {
            if let Ok(position) = sparse_row.indices().binary_search(&column) {
                out[row] = out[row] + scale * sparse_row.data()[position];
            }
        }
    }
}
