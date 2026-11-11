//! Compile-time supervision-axis contract tests for [`SKBaggingClassifier`]
//! and [`SKBaggingRegressor`].

use std::marker::PhantomData;

use sciencekit_common::SKSupervisedFit;
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_linear_model::ridge::SKRidge;
use sciencekit_neighbors::{SKKNeighborsClassifier, SKKNeighborsRegressor};
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor};

use crate::{SKBaggingClassifier, SKBaggingRegressor};

trait NotSupervisedFit {
    fn satisfies_fit(&self) -> bool {
        false
    }
}

impl<T> NotSupervisedFit for T {}

struct SupervisionProbe<T>(PhantomData<T>);

impl<T> SupervisionProbe<T> {
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: SKSupervisedFit<f64>> SupervisionProbe<T> {
    fn satisfies_fit(&self) -> bool {
        true
    }
}

fn assert_supervised_fit<E: SKSupervisedFit<f64>>() {}

#[test]
fn classifier_base_configurations_satisfy_bagging_classifier_contract() {
    assert_supervised_fit::<SKBaggingClassifier<SKDecisionTreeClassifier<f64>>>();
    assert_supervised_fit::<SKBaggingClassifier<SKLogisticRegression<f64>>>();
    assert_supervised_fit::<SKBaggingClassifier<SKKNeighborsClassifier<f64>>>();

    assert!(
        SupervisionProbe::<SKBaggingClassifier<SKDecisionTreeClassifier<f64>>>::new()
            .satisfies_fit()
    );
    assert!(
        SupervisionProbe::<SKBaggingClassifier<SKLogisticRegression<f64>>>::new().satisfies_fit()
    );
    assert!(
        SupervisionProbe::<SKBaggingClassifier<SKKNeighborsClassifier<f64>>>::new().satisfies_fit()
    );
}

#[test]
fn regressor_base_configurations_satisfy_bagging_regressor_contract() {
    assert_supervised_fit::<SKBaggingRegressor<SKDecisionTreeRegressor<f64>>>();
    assert_supervised_fit::<SKBaggingRegressor<SKLinearRegression>>();
    assert_supervised_fit::<SKBaggingRegressor<SKRidge<f64>>>();
    assert_supervised_fit::<SKBaggingRegressor<SKKNeighborsRegressor<f64>>>();

    assert!(
        SupervisionProbe::<SKBaggingRegressor<SKDecisionTreeRegressor<f64>>>::new().satisfies_fit()
    );
    assert!(SupervisionProbe::<SKBaggingRegressor<SKLinearRegression>>::new().satisfies_fit());
    assert!(SupervisionProbe::<SKBaggingRegressor<SKRidge<f64>>>::new().satisfies_fit());
}

#[test]
fn mismatched_supervision_axis_is_rejected_by_the_type_system() {
    assert!(
        !SupervisionProbe::<SKBaggingClassifier<SKDecisionTreeRegressor<f64>>>::new()
            .satisfies_fit()
    );
    assert!(!SupervisionProbe::<SKBaggingClassifier<SKLinearRegression>>::new().satisfies_fit());
    assert!(
        !SupervisionProbe::<SKBaggingRegressor<SKDecisionTreeClassifier<f64>>>::new()
            .satisfies_fit()
    );
    assert!(
        !SupervisionProbe::<SKBaggingRegressor<SKLogisticRegression<f64>>>::new().satisfies_fit()
    );
}
