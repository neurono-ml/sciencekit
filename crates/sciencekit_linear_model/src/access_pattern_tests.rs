//! Declaration tests for the linear-model family (spec `access-pattern`).

use sciencekit_common::SKBuilder;
use sciencekit_common::execution::{SKAccessPattern, SKDeclaredAccessPattern};

use crate::elastic_net::SKElasticNet;
use crate::lasso::SKLasso;
use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::logistic_regression::SKLogisticRegression;
use crate::ridge::SKRidge;
use crate::sgd_classifier::SKSGDClassifier;
use crate::sgd_regressor::SKSGDRegressor;

#[test]
fn linear_regression_declares_iterative_access_pattern() {
    let estimator = SKLinearRegression::build().build().expect("build");
    assert_eq!(estimator.access_pattern(), SKAccessPattern::Iterative);
}

#[test]
fn ridge_declares_iterative_access_pattern() {
    let estimator = SKRidge::<f64>::build().build().expect("build");
    assert_eq!(estimator.access_pattern(), SKAccessPattern::Iterative);
}

#[test]
fn lasso_declares_iterative_access_pattern() {
    let estimator = SKLasso::<f64>::build().build().expect("build");
    assert_eq!(estimator.access_pattern(), SKAccessPattern::Iterative);
}

#[test]
fn elastic_net_declares_iterative_access_pattern() {
    let estimator = SKElasticNet::<f64>::build().build().expect("build");
    assert_eq!(estimator.access_pattern(), SKAccessPattern::Iterative);
}

#[test]
fn logistic_regression_declares_iterative_access_pattern() {
    let estimator = SKLogisticRegression::<f64>::build().build().expect("build");
    assert_eq!(estimator.access_pattern(), SKAccessPattern::Iterative);
}

#[test]
fn sgd_classifier_declares_sequential_access_pattern() {
    let estimator = SKSGDClassifier::<f64>::build().build().expect("build");
    assert_eq!(estimator.access_pattern(), SKAccessPattern::Sequential);
}

#[test]
fn sgd_regressor_declares_sequential_access_pattern() {
    let estimator = SKSGDRegressor::<f64>::build().build().expect("build");
    assert_eq!(estimator.access_pattern(), SKAccessPattern::Sequential);
}
