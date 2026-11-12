//! Fitting [`SKHalvingGridSearchRegressor`] (spec `halving-grid-search`).

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    SKSupervisedFit, SKTargetView, sk_run_operation,
};

use crate::cross_validation::sk_fit_on_views;
use crate::grid_search::observation::sk_view_dimensions;
use crate::halving::evaluation::{SKHalvingEvaluationRequest, sk_run_halving_search};

use super::SKHalvingGridSearchRegressor;
use super::model::SKHalvingGridSearchRegressorModel;

impl<F, E> SKSupervisedFit<F> for SKHalvingGridSearchRegressor<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F> + Sync + Clone,
    E::Model: SKRegressorPredictor<F>,
    SKError: From<E::Error> + From<<E::Model as SKRegressorPredictor<F>>::Error>,
{
    type Model = SKHalvingGridSearchRegressorModel<F, E>;
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
            let outcome = sk_run_halving_search(
                SKHalvingEvaluationRequest {
                    candidates: &self.configuration.candidates,
                    splitter: self.configuration.splitter.as_ref(),
                    features,
                    targets,
                    groups: self.groups(),
                    factor: self.configuration.factor,
                    minimum_resource: self.configuration.minimum_resource,
                    maximum_resource: self.configuration.maximum_resource,
                    resource: self.configuration.resource,
                    aggressive_elimination: self.configuration.aggressive_elimination,
                    random_seed: self.configuration.random_seed,
                    error_score: self.configuration.error_score,
                },
                &self.scoring,
            )?;

            let best_candidate =
                self.configuration.candidates[outcome.best_candidate_index].clone();
            let best_model = if self.configuration.refit {
                Some(sk_fit_on_views(&best_candidate, features, targets)?)
            } else {
                None
            };

            Ok(SKHalvingGridSearchRegressorModel {
                elimination_trace: outcome.rounds,
                best_candidate_index: outcome.best_candidate_index,
                best_score: outcome.best_score,
                best_candidate,
                best_model,
                scoring: self.scoring,
                execution_mode: self.configuration.execution_mode,
                backend: self.configuration.backend,
            })
        })
    }
}
