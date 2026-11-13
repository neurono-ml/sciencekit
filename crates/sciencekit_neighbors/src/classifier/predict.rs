//! Label and probability prediction on the fitted classifier
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::model::SKKNeighborsClassifierModel;
use super::voting::{sk_argmax_label, sk_class_scores, sk_normalize_scores};
use crate::neighbors::{sk_validate_query_features, sk_view_dimensions};

impl<F: SKFloat> SKClassifierPredictor<F> for SKKNeighborsClassifierModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let scores = self.class_scores(features)?;
        Ok(Array1::from_iter(
            scores.iter().map(|row| sk_argmax_label(row)),
        ))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let scores = self.class_scores(features)?;
        let number_of_classes = self.number_of_classes();
        let mut probabilities = Array2::zeros((scores.len(), number_of_classes));
        for (row, scores) in scores.iter().enumerate() {
            for (class, probability) in sk_normalize_scores(scores).into_iter().enumerate() {
                probabilities[[row, class]] = probability;
            }
        }
        Ok(probabilities)
    }
}

impl<F: SKFloat> SKKNeighborsClassifierModel<F> {
    /// The weighted vote scores of every query row, one vector per row.
    fn class_scores(&self, features: SKDataView<'_, F>) -> Result<Vec<Vec<F>>, SKError> {
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend(),
        };
        sk_run_operation(attributes, || {
            sk_validate_query_features(features, self.number_of_features(), "predict")?;
            let neighbors =
                self.search
                    .k_nearest(self.metric(), features, self.nearest_neighbors_count())?;
            Ok((0..rows_count)
                .map(|row| {
                    sk_class_scores(
                        &neighbors.distances().row(row).to_vec(),
                        &neighbors.indices().row(row).to_vec(),
                        self.class_indices.view(),
                        self.number_of_classes(),
                        self.weights(),
                    )
                })
                .collect())
        })
    }
}
