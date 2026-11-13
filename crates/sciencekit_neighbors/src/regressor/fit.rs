//! The `SKSupervisedFit` implementation of the neighbor regressor
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use super::SKKNeighborsRegressor;
use super::model::SKKNeighborsRegressorModel;
use crate::neighbors::{sk_prepare_search, sk_view_dimensions};

impl<F: SKFloat> SKSupervisedFit<F> for SKKNeighborsRegressor<F> {
    type Model = SKKNeighborsRegressorModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let targets = targets.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || sk_fit_regressor(self, features, &targets))
    }
}

/// Elevate the responses, prepare the search and assemble the model.
fn sk_fit_regressor<F: SKFloat>(
    estimator: &SKKNeighborsRegressor<F>,
    features: SKDataView<'_, F>,
    targets: &SKTargetView<'_, F>,
) -> Result<SKKNeighborsRegressorModel<F>, SKError> {
    let responses = targets.as_continuous()?.to_owned();
    let search = sk_prepare_search(features, &estimator.configuration)?;
    if responses.len() != search.training_rows() {
        return Err(SKError::shape_mismatch_2d(
            search.training_rows(),
            1,
            responses.len(),
            1,
        ));
    }
    Ok(SKKNeighborsRegressorModel {
        search,
        configuration: estimator.configuration,
        responses,
    })
}
