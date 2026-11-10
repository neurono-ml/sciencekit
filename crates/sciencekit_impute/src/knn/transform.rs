//! The dense transform of the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use ndarray::{Array2, ArrayView2, Axis};
use sciencekit_common::{
    SKDataView, SKError, SKFeatureTransformer, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use crate::knn::distance_block::sk_distance_block;
use crate::knn::donors::sk_impute_cell;
use crate::knn::marker::{sk_canonicalize_missing, sk_dense_missing_mask};
use crate::knn::model::SKKNNImputerModel;
use crate::knn::validation::{sk_validated_shape, sk_validate_finite};
use crate::missing_indicator::SKIndicatorOutput;
use crate::simple_imputer::SKImputedOutput;

impl<F: SKFloat> SKFeatureTransformer<F> for SKKNNImputerModel<F> {
    type Output = SKImputedOutput<F>;
    type Error = SKError;

    fn transform<'a, X>(&self, features: X) -> Result<Self::Output, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        match features.try_into()? {
            SKDataView::Dense(dense) => {
                let (rows_count, columns_count) = sk_validated_shape(&dense.view(), "transform")?;
                self.validate_columns(rows_count, columns_count)?;
                let attributes = self.transform_attributes(rows_count, columns_count);
                sk_run_operation(attributes, || {
                    Ok(SKImputedOutput::Dense(self.transform_dense(&dense)?))
                })
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

impl<F: SKFloat> SKKNNImputerModel<F> {
    /// Reject a column count that differs from the fitted one before any
    /// element is read.
    fn validate_columns(
        &self,
        rows_count: usize,
        columns_count: usize,
    ) -> Result<(), SKError> {
        if columns_count == self.number_of_input_columns {
            return Ok(());
        }
        Err(SKError::shape_mismatch_2d(
            rows_count,
            self.number_of_input_columns,
            rows_count,
            columns_count,
        ))
    }

    /// The span attributes of a transform, carrying the execution intent and
    /// backend kind the fit transferred into the model.
    fn transform_attributes(
        &self,
        rows_count: usize,
        columns_count: usize,
    ) -> SKOperationAttributes {
        SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        }
    }

    /// Fill every missing cell of the retained columns.
    fn transform_dense(&self, features: &ArrayView2<'_, F>) -> Result<Array2<F>, SKError> {
        sk_validate_finite(features, self.missing_values)?;
        let canonical = sk_canonicalize_missing(features, self.missing_values);
        let missing_mask = sk_dense_missing_mask(features, self.missing_values);
        let mut output = Array2::zeros((features.nrows(), self.kept_columns.len()));
        for row in 0..features.nrows() {
            let receiver = canonical.slice(ndarray::s![row..row + 1, ..]);
            let distance = sk_distance_block(&receiver, &self.training.view());
            for (target, &source) in self.kept_columns.iter().enumerate() {
                output[[row, target]] = if missing_mask[[row, source]] {
                    sk_impute_cell(
                        &distance.row(0),
                        &self.column_donors[source],
                        &self.training.column(source),
                        self.column_means[source],
                        self.nearest_neighbors_count,
                        self.weights,
                    )
                } else {
                    features[[row, source]]
                };
            }
        }
        let Some(indicator) = self.indicator.as_ref() else {
            return Ok(output);
        };
        let mask = match indicator.transform(*features)? {
            SKIndicatorOutput::Dense(mask) => mask,
            SKIndicatorOutput::Sparse(mask) => mask.to_dense(),
        };
        ndarray::concatenate(Axis(1), &[output.view(), mask.view()]).map_err(|_| {
            SKError::Conversion(
                "the imputed features and the missingness mask must share their row count"
                    .to_owned(),
            )
        })
    }
}
