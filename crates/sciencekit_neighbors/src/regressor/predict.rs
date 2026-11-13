//! Response prediction on the fitted regressor
//! (spec `nearest-neighbor-estimators`).

use ndarray::Array1;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    sk_run_operation,
};

use super::model::SKKNeighborsRegressorModel;
use crate::neighbors::{sk_validate_query_features, sk_view_dimensions, sk_weighted_average};

impl<F: SKFloat> SKRegressorPredictor<F> for SKKNeighborsRegressorModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
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
            let predictions: Vec<F> = (0..rows_count)
                .map(|row| {
                    let distances = neighbors.distances().row(row).to_vec();
                    let weights = self.weights().weights(&distances);
                    let responses: Vec<F> = neighbors
                        .indices()
                        .row(row)
                        .iter()
                        .map(|index| self.responses[*index])
                        .collect();
                    sk_weighted_average(&weights, &responses)
                })
                .collect();
            Ok(Array1::from_iter(predictions))
        })
    }
}
