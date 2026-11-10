//! Batch fitting of the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKUnsupervisedFit,
    sk_run_operation,
};

use crate::knn::core_implementation::SKKNNImputer;
use crate::knn::donors::sk_column_donors;
use crate::knn::marker::{sk_canonicalize_missing, sk_dense_missing_mask};
use crate::knn::model::SKKNNImputerModel;
use crate::knn::validation::{sk_validated_shape, sk_validate_finite, sk_validate_neighbor_count};

impl<F: SKFloat> SKKNNImputer<F> {
    /// Fit the dense representation; sparse input has no missing marker and
    /// is rejected by [`SKUnsupervisedFit::fit`] before reaching here.
    fn fit_dense(&self, features: &ArrayView2<'_, F>) -> Result<SKKNNImputerModel<F>, SKError> {
        let (rows_count, columns_count) = sk_validated_shape(features, "fit")?;
        sk_validate_finite(features, self.missing_values)?;
        sk_validate_neighbor_count(self.nearest_neighbors_count, rows_count)?;

        let missing_mask = sk_dense_missing_mask(features, self.missing_values);
        let (column_counts, column_means) = observed_column_statistics(&missing_mask, features);
        let valid_columns = observed_valid_columns(&column_counts);
        let kept_columns = if self.keep_empty_features {
            (0..columns_count).collect()
        } else {
            valid_columns
        };
        let dropped_columns = (0..columns_count)
            .filter(|column| !kept_columns.contains(column))
            .collect();
        let training = sk_canonicalize_missing(features, self.missing_values);
        Ok(SKKNNImputerModel {
            training,
            column_donors: sk_column_donors(&missing_mask.view()),
            missing_mask,
            kept_columns,
            dropped_columns,
            column_means,
            number_of_input_columns: columns_count,
            missing_values: self.missing_values,
            nearest_neighbors_count: self.nearest_neighbors_count,
            weights: self.weights,
            indicator: None,
            backend: self.backend,
            execution_intent: self.execution_intent(),
        })
    }

    /// The span attributes of a fit, carrying the execution mode and the
    /// configured backend kind.
    pub(in crate::knn) fn fit_attributes(
        &self,
        rows_count: usize,
        columns_count: usize,
    ) -> SKOperationAttributes {
        SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        }
    }
}

/// Per-column observed counts and means over the non-missing cells.
///
/// An all-missing column has count zero and a `NaN` mean. Under a non-`NaN`
/// marker an observed `NaN` is a real value, so a column can hold data and
/// still carry a `NaN` mean.
fn observed_column_statistics<F: SKFloat>(
    missing_mask: &ndarray::Array2<bool>,
    features: &ArrayView2<'_, F>,
) -> (Vec<u64>, Array1<F>) {
    let columns_count = features.ncols();
    let mut counts = vec![0_u64; columns_count];
    let mut sums = vec![F::zero(); columns_count];
    for (row, feature_row) in features.rows().into_iter().enumerate() {
        for (column, value) in feature_row.iter().enumerate() {
            if !missing_mask[[row, column]] {
                counts[column] += 1;
                sums[column] = sums[column] + *value;
            }
        }
    }
    let means = Array1::from_shape_fn(columns_count, |column| {
        if counts[column] == 0 {
            F::nan()
        } else {
            sums[column] / F::from(counts[column]).expect("counts fit in the scalar")
        }
    });
    (counts, means)
}

/// The columns that observed at least one value.
fn observed_valid_columns(counts: &[u64]) -> Vec<usize> {
    counts
        .iter()
        .enumerate()
        .filter_map(|(column, count)| (*count > 0).then_some(column))
        .collect()
}

impl<F: SKFloat> SKUnsupervisedFit<F> for SKKNNImputer<F> {
    type Model = SKKNNImputerModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        match features.try_into()? {
            SKDataView::Dense(dense) => {
                let (rows_count, columns_count) = dense.dim();
                let attributes = self.fit_attributes(rows_count, columns_count);
                sk_run_operation(attributes, || self.fit_dense(&dense))
            }
            SKDataView::Sparse(_) => Err(SKError::UnsupportedRepresentation {
                representation: "csr",
                suggestion: "use SKSimpleImputer for sparse input: a CSR implicit zero is an observed zero, so it cannot carry the missing marker",
            }),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the input to a dense matrix",
            }),
        }
    }
}
