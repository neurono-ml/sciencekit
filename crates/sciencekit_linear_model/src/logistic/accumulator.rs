//! The retained-design accumulator behind the incremental and out-of-core
//! logistic fits (spec `logistic-solver`, Decision 7).
//!
//! The logistic objective is not a function of the Gram matrix, so the
//! accumulator retains the design — dense rows or stacked CSR batches — with
//! the canonical labels and re-solves at read time.

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{SKError, SKFloat};
use sprs::{CsMat, CsMatView};

use super::class_weight::sk_expand_class_weights;
use super::config::SKLogisticConfig;
use super::design::{SKDenseDesign, SKSparseDesign};
use super::resolution::SKResolvedMultiClass;
use super::solver::{SKLogisticSolution, sk_solve_logistic};

/// The retained design and canonical labels of an incremental or streaming fit.
#[derive(Debug, Clone)]
pub(crate) enum SKLogisticState<F> {
    /// Dense batches, concatenated at solve time.
    Dense {
        /// The accumulated dense batches.
        batches: Vec<Array2<F>>,
        /// The accumulated canonical labels.
        labels: Vec<usize>,
    },
    /// Sparse CSR batches, stacked at solve time.
    Sparse {
        /// The accumulated sparse batches.
        batches: Vec<CsMat<F>>,
        /// The accumulated canonical labels.
        labels: Vec<usize>,
        /// The feature count (zero until the first batch is seen).
        columns_count: usize,
    },
}

impl<F: SKFloat> SKLogisticState<F> {
    /// A fresh empty dense accumulator.
    pub(crate) fn new_dense() -> Self {
        Self::Dense {
            batches: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// A fresh empty sparse accumulator.
    pub(crate) fn new_sparse() -> Self {
        Self::Sparse {
            batches: Vec::new(),
            labels: Vec::new(),
            columns_count: 0,
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

    /// The accumulated canonical labels.
    pub(crate) fn labels(&self) -> &[usize] {
        match self {
            Self::Dense { labels, .. } | Self::Sparse { labels, .. } => labels,
        }
    }

    /// Append one dense batch (owned copy) and its canonical labels.
    pub(crate) fn accumulate_dense(&mut self, features: ArrayView2<F>, labels: &[usize]) {
        if let Self::Dense {
            batches,
            labels: accumulated,
        } = self
        {
            batches.push(features.to_owned());
            accumulated.extend_from_slice(labels);
        }
    }

    /// Append one sparse batch (owned copy) and its canonical labels.
    pub(crate) fn accumulate_sparse(&mut self, features: CsMatView<'_, F>, labels: &[usize]) {
        if let Self::Sparse {
            batches,
            labels: accumulated,
            columns_count,
        } = self
        {
            if *columns_count == 0 {
                *columns_count = features.cols();
            }
            batches.push(features.to_owned());
            accumulated.extend_from_slice(labels);
        }
    }

    /// Solve the accumulated design for the resolved strategy.
    pub(crate) fn solve(
        &self,
        config: &SKLogisticConfig<F>,
        strategy: SKResolvedMultiClass,
        number_of_classes: usize,
        initial: Option<&SKLogisticSolution<F>>,
    ) -> Result<SKLogisticSolution<F>, SKError> {
        if self.is_empty() {
            return Err(SKError::Conversion(
                "logistic fit received no rows to solve".into(),
            ));
        }
        let weights =
            sk_expand_class_weights(&config.class_weight, self.labels(), number_of_classes)?;
        match self {
            Self::Dense { batches, labels } => {
                let views: Vec<ArrayView2<F>> = batches.iter().map(|batch| batch.view()).collect();
                let design = ndarray::concatenate(Axis(0), &views)
                    .map_err(|_| SKError::Conversion("dense batch concatenation failed".into()))?;
                sk_solve_logistic(
                    &SKDenseDesign::new(design.view()),
                    labels,
                    &weights,
                    number_of_classes,
                    strategy,
                    config,
                    initial,
                )
            }
            Self::Sparse {
                batches,
                labels,
                columns_count,
            } => {
                let stacked = sk_stack_sparse(batches, *columns_count);
                sk_solve_logistic(
                    &SKSparseDesign::new(stacked.view()),
                    labels,
                    &weights,
                    number_of_classes,
                    strategy,
                    config,
                    initial,
                )
            }
        }
    }
}

/// Vertically stack sparse batches into one CSR design.
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
