//! The retained-design accumulator behind the incremental and out-of-core
//! regimes (spec `coordinate-descent-solver`, Decision 7).
//!
//! Because the L1 objective is **not** a function of the Gram matrix, the
//! accumulator cannot keep cross products the way `crate::ridge` does: it
//! retains the design — dense batches or stacked CSR batches — with the targets
//! and re-solves at read time. The memory trade-off is documented in the
//! design.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKError, SKFloat};
use sprs::{CsMat, CsMatView};

use super::access::{SKDenseColumns, SKSparseColumns};
use super::config::SKCoordinateDescentConfig;
use super::solution::SKCoordinateDescentSolution;
use super::solver::sk_solve_all_alphas;

/// The retained design and targets of an incremental or streaming fit.
#[derive(Debug, Clone)]
pub(crate) enum SKCoordinateDescentState<F> {
    /// Dense batches, concatenated at solve time.
    Dense {
        /// The accumulated dense batches.
        batches: Vec<Array2<F>>,
        /// The accumulated targets.
        targets: Array1<F>,
    },
    /// Sparse CSR batches, stacked at solve time.
    Sparse {
        /// The accumulated sparse batches.
        batches: Vec<CsMat<F>>,
        /// The accumulated targets.
        targets: Array1<F>,
        /// The feature count (zero until the first batch is seen).
        columns_count: usize,
    },
}

impl<F: SKFloat> SKCoordinateDescentState<F> {
    /// A fresh empty dense accumulator.
    pub(crate) fn new_dense() -> Self {
        Self::Dense {
            batches: Vec::new(),
            targets: Array1::zeros(0),
        }
    }

    /// A fresh empty sparse accumulator.
    pub(crate) fn new_sparse() -> Self {
        Self::Sparse {
            batches: Vec::new(),
            targets: Array1::zeros(0),
            columns_count: 0,
        }
    }

    /// The number of feature columns seen so far.
    pub(crate) fn columns_count(&self) -> usize {
        match self {
            Self::Dense { batches, .. } => batches.first().map_or(0, Array2::ncols),
            Self::Sparse { columns_count, .. } => *columns_count,
        }
    }

    /// The number of rows accumulated across every batch.
    pub(crate) fn row_count(&self) -> u64 {
        match self {
            Self::Dense { batches, .. } => batches.iter().map(Array2::nrows).sum::<usize>() as u64,
            Self::Sparse { batches, .. } => batches.iter().map(CsMat::rows).sum::<usize>() as u64,
        }
    }

    /// Whether no row has been accumulated.
    pub(crate) fn is_empty(&self) -> bool {
        self.row_count() == 0
    }

    /// Append one dense batch, borrowing the caller's data into owned storage.
    pub(crate) fn accumulate_dense(&mut self, features: ArrayView2<F>, targets: ArrayView1<F>) {
        if let Self::Dense {
            batches,
            targets: accumulated,
        } = self
        {
            batches.push(features.to_owned());
            *accumulated = concatenate_targets(accumulated, &targets);
        }
    }

    /// Append one sparse batch (owned copy) and its targets.
    pub(crate) fn accumulate_sparse(&mut self, features: CsMatView<'_, F>, targets: ArrayView1<F>) {
        if let Self::Sparse {
            batches,
            targets: accumulated,
            columns_count,
        } = self
        {
            if *columns_count == 0 {
                *columns_count = features.cols();
            }
            batches.push(features.to_owned());
            *accumulated = concatenate_targets(accumulated, &targets);
        }
    }

    /// Solve the accumulated design once per `alpha`.
    pub(crate) fn solve(
        &self,
        alphas: &[F],
        l1_ratio: F,
        include_intercept: bool,
        config: &SKCoordinateDescentConfig<F>,
    ) -> Result<SKCoordinateDescentSolution<F>, SKError> {
        if self.is_empty() {
            return Err(SKError::Conversion(
                "coordinate descent received no rows to solve".into(),
            ));
        }
        match self {
            Self::Dense { batches, targets } => {
                let views: Vec<ArrayView2<F>> = batches.iter().map(|batch| batch.view()).collect();
                let design = ndarray::concatenate(Axis(0), &views)
                    .map_err(|_| SKError::Conversion("dense batch concatenation failed".into()))?;
                let access = SKDenseColumns::new(design.view());
                sk_solve_all_alphas(
                    &access,
                    targets.view(),
                    alphas,
                    l1_ratio,
                    include_intercept,
                    config,
                )
            }
            Self::Sparse {
                batches,
                targets,
                columns_count,
            } => {
                let stacked = sk_stack_sparse(batches, *columns_count);
                let access = SKSparseColumns::new(stacked.view());
                sk_solve_all_alphas(
                    &access,
                    targets.view(),
                    alphas,
                    l1_ratio,
                    include_intercept,
                    config,
                )
            }
        }
    }
}

/// Append `incoming` targets to an accumulated target vector.
fn concatenate_targets<F: SKFloat>(accumulated: &Array1<F>, incoming: &ArrayView1<F>) -> Array1<F> {
    if accumulated.is_empty() {
        return incoming.to_owned();
    }
    ndarray::concatenate(Axis(0), &[accumulated.view(), *incoming])
        .expect("target batches carry one column each")
}

/// Vertically stack sparse batches into one CSR design, without requiring
/// `F: Default` (which `sprs::vstack` demands).
fn sk_stack_sparse<F: SKFloat>(batches: &[CsMat<F>], columns_count: usize) -> CsMat<F> {
    let total_rows: usize = batches.iter().map(CsMat::rows).sum();
    let mut triplets = sprs::TriMat::new((total_rows, columns_count));
    let mut row_offset = 0;
    for batch in batches {
        for (row_index, sparse_row) in batch.outer_iterator().enumerate() {
            for (&column_index, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
                triplets.add_triplet(row_offset + row_index, column_index, value);
            }
        }
        row_offset += batch.rows();
    }
    triplets.to_csr()
}
