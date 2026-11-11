//! Pure dispatcher for the regressor halving search module.

mod estimator;
mod fit;
mod model;
mod predict;

pub use estimator::SKHalvingGridSearchRegressor;
pub use model::SKHalvingGridSearchRegressorModel;

#[cfg(test)]
mod halving_refit_tests;
