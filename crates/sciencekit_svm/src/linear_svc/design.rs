//! Dense design access for the linear SVC dual coordinate-descent solver
//! (spec `linear-support-vector-classifier`).
//!
//! The design abstracts over the feature representation so a single
//! coordinate-descent loop serves dense and borrowed-CSR inputs. Every access
//! is defined on the augmented row `[x_i, 1]` when an intercept is fitted; the
//! intercept component is not stored in the data.

use ndarray::ArrayView2;
use sprs::{CompressedStorage, CsMatView};

use sciencekit_common::{SKDataView, SKError, SKFloat};

/// Row-wise access the dual coordinate-descent solver needs from a design.
///
/// Coefficients are stored in the augmented layout: `number_of_columns()`
/// entries followed by one intercept entry when `has_intercept()` is true.
pub(crate) trait SKLinearSvcDesignAccess<F: SKFloat>: Sync {
    /// Number of design rows (`n`).
    fn number_of_rows(&self) -> usize;

    /// Number of stored feature columns (`p`).
    fn number_of_columns(&self) -> usize;

    /// Whether the design carries a constant (intercept) column.
    fn has_intercept(&self) -> bool;

    /// Dot product of the augmented row with the augmented coefficient slice.
    fn row_dot(&self, row_index: usize, coefficients: &[F]) -> F;

    /// Accumulate `scale · augmented_row` into `target`.
    fn add_scaled_row(&self, row_index: usize, scale: F, target: &mut [F]);

    /// Squared Euclidean norm of the augmented row (`Q_ii` diagonal).
    fn row_norm_squared(&self, row_index: usize) -> F;
}

/// Borrowed dense design.
#[derive(Debug)]
pub(crate) struct SKDenseLinearSvcDesign<'a, F: SKFloat> {
    features: ArrayView2<'a, F>,
    has_intercept: bool,
}

impl<'a, F: SKFloat> SKDenseLinearSvcDesign<'a, F> {
    pub(crate) fn new(features: ArrayView2<'a, F>, has_intercept: bool) -> Self {
        Self {
            features,
            has_intercept,
        }
    }
}

impl<F: SKFloat> SKLinearSvcDesignAccess<F> for SKDenseLinearSvcDesign<'_, F> {
    fn number_of_rows(&self) -> usize {
        self.features.nrows()
    }

    fn number_of_columns(&self) -> usize {
        self.features.ncols()
    }

    fn has_intercept(&self) -> bool {
        self.has_intercept
    }

    fn row_dot(&self, row_index: usize, coefficients: &[F]) -> F {
        let columns = self.number_of_columns();
        let dot = self
            .features
            .row(row_index)
            .dot(&ndarray::ArrayView1::from(&coefficients[..columns]));
        if self.has_intercept {
            dot + coefficients[columns]
        } else {
            dot
        }
    }

    fn add_scaled_row(&self, row_index: usize, scale: F, target: &mut [F]) {
        let columns = self.number_of_columns();
        target[..columns]
            .iter_mut()
            .zip(self.features.row(row_index))
            .for_each(|(slot, &value)| *slot = *slot + scale * value);
        if self.has_intercept {
            target[columns] = target[columns] + scale;
        }
    }

    fn row_norm_squared(&self, row_index: usize) -> F {
        let row = self.features.row(row_index);
        let squared_norm = row.dot(&row);
        if self.has_intercept {
            squared_norm + F::one()
        } else {
            squared_norm
        }
    }
}

/// Borrowed CSR design that scans stored entries only.
#[derive(Debug)]
pub(crate) struct SKSparseLinearSvcDesign<'a, F: SKFloat> {
    features: CsMatView<'a, F>,
    has_intercept: bool,
}

impl<'a, F: SKFloat> SKSparseLinearSvcDesign<'a, F> {
    pub(crate) fn new(features: CsMatView<'a, F>, has_intercept: bool) -> Self {
        Self {
            features,
            has_intercept,
        }
    }
}

impl<F: SKFloat> SKLinearSvcDesignAccess<F> for SKSparseLinearSvcDesign<'_, F> {
    fn number_of_rows(&self) -> usize {
        self.features.rows()
    }

    fn number_of_columns(&self) -> usize {
        self.features.cols()
    }

    fn has_intercept(&self) -> bool {
        self.has_intercept
    }

    fn row_dot(&self, row_index: usize, coefficients: &[F]) -> F {
        let stored = self
            .features
            .outer_view(row_index)
            .expect("row index bounded by number_of_rows");
        let dot = stored
            .indices()
            .iter()
            .zip(stored.data().iter())
            .fold(F::zero(), |accumulator, (&column, &value)| {
                accumulator + coefficients[column] * value
            });
        if self.has_intercept {
            dot + coefficients[self.number_of_columns()]
        } else {
            dot
        }
    }

    fn add_scaled_row(&self, row_index: usize, scale: F, target: &mut [F]) {
        let stored = self
            .features
            .outer_view(row_index)
            .expect("row index bounded by number_of_rows");
        stored
            .indices()
            .iter()
            .zip(stored.data().iter())
            .for_each(|(&column, &value)| {
                target[column] = target[column] + scale * value;
            });
        if self.has_intercept {
            let intercept = self.number_of_columns();
            target[intercept] = target[intercept] + scale;
        }
    }

    fn row_norm_squared(&self, row_index: usize) -> F {
        let stored = self
            .features
            .outer_view(row_index)
            .expect("row index bounded by number_of_rows");
        let squared_norm = stored
            .data()
            .iter()
            .fold(F::zero(), |accumulator, &value| accumulator + value * value);
        if self.has_intercept {
            squared_norm + F::one()
        } else {
            squared_norm
        }
    }
}

/// One-time representation dispatch for the solver.
#[derive(Debug)]
pub(crate) enum SKLinearSvcDesign<'a, F: SKFloat> {
    /// Dense design.
    Dense(SKDenseLinearSvcDesign<'a, F>),
    /// Borrowed CSR design.
    Sparse(SKSparseLinearSvcDesign<'a, F>),
}

impl<'a, F: SKFloat> SKLinearSvcDesign<'a, F> {
    /// Build the design for a dense or CSR feature view, dispatching once.
    pub(crate) fn dispatch(
        features: SKDataView<'a, F>,
        has_intercept: bool,
    ) -> Result<Self, SKError> {
        match features {
            SKDataView::Dense(dense) => Ok(Self::Dense(SKDenseLinearSvcDesign::new(
                dense,
                has_intercept,
            ))),
            SKDataView::Sparse(sparse) => {
                if sparse.storage() != CompressedStorage::CSR {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: "csc",
                        suggestion: "convert the sparse matrix to CSR",
                    });
                }
                Ok(Self::Sparse(SKSparseLinearSvcDesign::new(
                    sparse,
                    has_intercept,
                )))
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a dense matrix or a CSR matrix",
            }),
        }
    }
}

/// Hand the solver the active representation without per-row branching.
pub(crate) fn sk_linear_svc_design_access<'design, F: SKFloat>(
    design: &'design SKLinearSvcDesign<'_, F>,
) -> &'design (dyn SKLinearSvcDesignAccess<F> + Sync) {
    match design {
        SKLinearSvcDesign::Dense(dense) => dense,
        SKLinearSvcDesign::Sparse(sparse) => sparse,
    }
}
