//! Owned result types returned by every searcher (spec
//! `nearest-neighbor-search`).

use ndarray::{Array2, ArrayView2};

/// The exact `k`-nearest-neighbor result of a query batch.
///
/// Distances and training row indices are row-major: row `i` belongs to query
/// row `i`, and column `j` is the `j`-th nearest neighbour (distance ascending,
/// ties broken by the smallest training row index).
#[derive(Debug, Clone, PartialEq)]
pub struct SKNeighborQuery<F> {
    distances: Array2<F>,
    indices: Array2<usize>,
}

impl<F> SKNeighborQuery<F> {
    /// Wrap an already ordered distance/index pair.
    pub(crate) fn new(distances: Array2<F>, indices: Array2<usize>) -> Self {
        debug_assert_eq!(distances.shape(), indices.shape());
        SKNeighborQuery { distances, indices }
    }

    /// The number of query rows in the batch.
    pub fn rows(&self) -> usize {
        self.distances.nrows()
    }

    /// The number of neighbours returned per query row.
    pub fn neighbors_count(&self) -> usize {
        self.distances.ncols()
    }

    /// The neighbour distances, shape `(rows, neighbors_count)`.
    pub fn distances(&self) -> ArrayView2<'_, F> {
        self.distances.view()
    }

    /// The training row indices, shape `(rows, neighbors_count)`.
    pub fn indices(&self) -> ArrayView2<'_, usize> {
        self.indices.view()
    }
}

/// The radius-neighbour result of a query batch.
///
/// Each row carries every training row within the inclusive radius, ordered by
/// distance ascending and ties by the smallest training row index. An empty
/// neighbourhood is an empty slice, never an error.
#[derive(Debug, Clone, PartialEq)]
pub struct SKRadiusNeighbors<F> {
    distances: Vec<Vec<F>>,
    indices: Vec<Vec<usize>>,
}

impl<F> SKRadiusNeighbors<F> {
    /// Wrap per-row ordered distance/index lists.
    pub(crate) fn new(distances: Vec<Vec<F>>, indices: Vec<Vec<usize>>) -> Self {
        debug_assert_eq!(distances.len(), indices.len());
        SKRadiusNeighbors { distances, indices }
    }

    /// The number of query rows in the batch.
    pub fn rows(&self) -> usize {
        self.distances.len()
    }

    /// The number of neighbours found for query row `row`.
    pub fn neighbors_count(&self, row: usize) -> usize {
        self.distances[row].len()
    }

    /// The ordered distances of query row `row`.
    pub fn distances(&self, row: usize) -> &[F] {
        &self.distances[row]
    }

    /// The ordered training row indices of query row `row`.
    pub fn indices(&self, row: usize) -> &[usize] {
        &self.indices[row]
    }
}
