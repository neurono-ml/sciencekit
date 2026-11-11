//! TDD tests for halving grid search builder defaults and validation
//! (spec `halving-grid-search`, requirements `Builder-only construction with documented defaults`
//! and `Validation and concurrency`).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{SKClassifierScoring, SKErrorScore, SKRegressorScoring};
use crate::halving::{
    SKHalvingGridSearchClassifier, SKHalvingGridSearchClassifierBuilder,
    SKHalvingGridSearchRegressor, SKHalvingGridSearchRegressorBuilder, SKHalvingResource,
};
use crate::split::kfold::SKKFold;

fn three_folds() -> SKKFold {
    SKKFold::build().number_of_splits(3).build().unwrap()
}

fn classifier_builder() -> SKHalvingGridSearchClassifierBuilder<f64, SKMockThresholdClassifier> {
    SKHalvingGridSearchClassifier::build()
        .candidates(vec![SKMockThresholdClassifier::new(0.5)])
        .splitter(three_folds())
}

fn regressor_builder() -> SKHalvingGridSearchRegressorBuilder<f64, SKMockConstantRegressor> {
    SKHalvingGridSearchRegressor::build()
        .candidates(vec![SKMockConstantRegressor::new(1.0)])
        .splitter(three_folds())
}

#[test]
fn halving_builder_classifier_and_regressor_defaults_match_specification() {
    let classifier = classifier_builder().build().unwrap();
    assert_eq!(classifier.factor(), 3);
    assert_eq!(classifier.minimum_resource(), 1);
    assert_eq!(classifier.maximum_resource(), None);
    assert_eq!(classifier.resource(), SKHalvingResource::NumberOfSamples);
    assert!(!classifier.aggressive_elimination());
    assert_eq!(classifier.random_seed(), 0);
    assert!(classifier.refit());
    assert_eq!(classifier.error_score(), SKErrorScore::Raise);
    assert_eq!(classifier.execution_mode(), SKExecutionMode::Automatic);
    assert_eq!(classifier.backend(), SKBackendKind::Faer);
    assert_eq!(classifier.scoring(), SKClassifierScoring::Accuracy);
    assert_eq!(classifier.number_of_candidates(), 1);
    assert!(classifier.groups().is_none());

    let regressor = regressor_builder().build().unwrap();
    assert_eq!(regressor.factor(), 3);
    assert_eq!(regressor.minimum_resource(), 1);
    assert_eq!(regressor.maximum_resource(), None);
    assert_eq!(
        regressor.scoring(),
        SKRegressorScoring::NegativeMeanSquaredError
    );
}

#[test]
fn halving_builder_rejects_factor_below_two() {
    for invalid_factor in [0, 1] {
        let Err(error) = classifier_builder().factor(invalid_factor).build() else {
            panic!("factor {invalid_factor} must be rejected");
        };
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter { name: "factor", .. }
        ));
    }
}

#[test]
fn halving_builder_rejects_invalid_resource_bounds() {
    let Err(zero_min) = classifier_builder().minimum_resource(0).build() else {
        panic!("minimum_resource 0 must be rejected");
    };
    assert!(matches!(
        zero_min,
        SKError::InvalidHyperparameter {
            name: "minimum_resource",
            ..
        }
    ));

    let Err(zero_max) = regressor_builder().maximum_resource(0).build() else {
        panic!("maximum_resource 0 must be rejected");
    };
    assert!(matches!(
        zero_max,
        SKError::InvalidHyperparameter {
            name: "maximum_resource",
            ..
        }
    ));

    let Err(inverted) = regressor_builder()
        .minimum_resource(50)
        .maximum_resource(20)
        .build()
    else {
        panic!("minimum_resource > maximum_resource must be rejected");
    };
    assert!(matches!(inverted, SKError::InvalidHyperparameter { .. }));
}

#[test]
fn halving_builder_rejects_empty_candidates_and_missing_splitter() {
    let Err(empty_candidates) =
        SKHalvingGridSearchClassifier::<f64, SKMockThresholdClassifier>::build()
            .splitter(three_folds())
            .build()
    else {
        panic!("empty candidates must be rejected");
    };
    assert!(matches!(
        empty_candidates,
        SKError::InvalidHyperparameter {
            name: "candidates",
            ..
        }
    ));

    let Err(missing_splitter) = SKHalvingGridSearchRegressor::<f64, _>::build()
        .candidates(vec![SKMockConstantRegressor::new(0.0)])
        .build()
    else {
        panic!("missing splitter must be rejected");
    };
    assert!(matches!(
        missing_splitter,
        SKError::InvalidHyperparameter {
            name: "splitter",
            ..
        }
    ));
}

#[test]
fn halving_builder_rejects_incompatible_scoring_policy_on_both_families() {
    let Err(classifier_error) = classifier_builder().scoring(SKRegressorScoring::R2).build() else {
        panic!("regressor scorer on classifier halving search must fail");
    };
    assert!(matches!(
        classifier_error,
        SKError::InvalidHyperparameter {
            name: "scoring",
            ..
        }
    ));

    let Err(regressor_error) = regressor_builder()
        .scoring(SKClassifierScoring::BalancedAccuracy)
        .build()
    else {
        panic!("classifier scorer on regressor halving search must fail");
    };
    assert!(matches!(
        regressor_error,
        SKError::InvalidHyperparameter {
            name: "scoring",
            ..
        }
    ));
}

#[test]
fn halving_builder_honors_every_optional_setter_and_trait_dispatch() {
    let mut builder = classifier_builder()
        .scoring(SKClassifierScoring::BalancedAccuracy)
        .factor(4)
        .minimum_resource(8)
        .maximum_resource(64)
        .resource(SKHalvingResource::NumberOfSamples)
        .aggressive_elimination(true)
        .random_seed(123)
        .refit(false)
        .error_score(SKErrorScore::value(-2.5))
        .groups(array![0, 0, 1, 1, 2, 2])
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous);

    <SKHalvingGridSearchClassifierBuilder<_, _> as SKBuilder<_>>::execution_mode(
        &mut builder,
        SKExecutionMode::InProcessAsynchronous,
    );

    let estimator = builder.build().unwrap();
    assert_eq!(estimator.scoring(), SKClassifierScoring::BalancedAccuracy);
    assert_eq!(estimator.factor(), 4);
    assert_eq!(estimator.minimum_resource(), 8);
    assert_eq!(estimator.maximum_resource(), Some(64));
    assert_eq!(estimator.resource(), SKHalvingResource::NumberOfSamples);
    assert!(estimator.aggressive_elimination());
    assert_eq!(estimator.random_seed(), 123);
    assert!(!estimator.refit());
    assert_eq!(estimator.error_score(), SKErrorScore::value(-2.5));
    assert_eq!(estimator.groups().unwrap().to_vec(), vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessAsynchronous
    );
}
