//! Dense design for the linear SVC dual coordinate-descent solver
//! (spec `linear-support-vector-classifier`).

use ndarray::ArrayView2;

use sciencekit_common::SKFloat;

use super::access::SKLinearSvcDesignAccess;

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
