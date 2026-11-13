//! Borrowed CSR design for the linear SVC coordinate-descent solver
//! (spec `linear-support-vector-classifier`).
//!
//! Every access scans stored entries only; the design never densifies.

use sprs::CsMatView;

use sciencekit_common::SKFloat;

use super::access::SKLinearSvcDesignAccess;

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
