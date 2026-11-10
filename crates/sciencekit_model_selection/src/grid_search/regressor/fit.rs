//! Fitting the regressor grid search (spec `grid-search`).

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    SKSupervisedFit, SKTargetView, sk_run_operation,
};

use super::SKGridSearchCVRegressor;
use super::model::SKGridSearchCVRegressorModel;
use crate::cross_validation::sk_fit_on_views;
use crate::grid_search::evaluation::sk_evaluate_candidates;
use crate::grid_search::observation::sk_view_dimensions;

impl<F, E> SKSupervisedFit<F> for SKGridSearchCVRegressor<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F> + Sync + Clone,
    E::Model: SKRegressorPredictor<F>,
    SKError: From<E::Error> + From<<E::Model as SKRegressorPredictor<F>>::Error>,
{
    type Model = SKGridSearchCVRegressorModel<F, E>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let targets = targets.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            let groups = self
                .configuration
                .groups
                .as_ref()
                .map(|groups| groups.view());
            let results = sk_evaluate_candidates(
                &self.configuration.candidates,
                self.configuration.splitter.as_ref(),
                features,
                targets,
                groups,
                self.configuration.error_score,
                &self.scoring,
            )?;

            let best_candidate = self.configuration.candidates[results.best_index()].clone();
            let best_model = if self.configuration.refit {
                Some(sk_fit_on_views(&best_candidate, features, targets)?)
            } else {
                None
            };

            Ok(SKGridSearchCVRegressorModel {
                results,
                best_candidate,
                best_model,
                scoring: self.scoring,
                execution_mode: self.configuration.execution_mode,
                backend: self.configuration.backend,
            })
        })
    }
}
