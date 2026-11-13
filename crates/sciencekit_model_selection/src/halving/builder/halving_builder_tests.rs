//! TDD tests for halving grid search builder defaults and validation
//! (spec `halving-grid-search`, requirements `Builder-only construction with documented defaults`
//! and `Validation and concurrency`).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::{SKClassifierScoring, SKErrorScore, SKRegressorScoring, SKScoringFamily};
use crate::halving::{
    SKHalvingGridSearchClassifier, SKHalvingGridSearchClassifierBuilder,
    SKHalvingGridSearchRegressor, SKHalvingGridSearchRegressorBuilder, SKHalvingResource,
    SKHalvingScoring,
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
    assert_eq!(regressor.backend(), SKBackendKind::Faer);
    assert_eq!(
        regressor.scoring(),
        SKRegressorScoring::NegativeMeanSquaredError
    );
    assert_eq!(
        SKHalvingScoring::from(SKClassifierScoring::Accuracy).family(),
        SKScoringFamily::Classifier
    );
    assert_eq!(
        SKHalvingScoring::from(SKRegressorScoring::R2).family(),
        SKScoringFamily::Regressor
    );
}

#[test]
fn halving_builder_rejects_factor_below_two() {
    for invalid_factor in [0, 1] {
        let error = classifier_builder().factor(invalid_factor).build().err();
        assert!(matches!(
            error,
            Some(SKError::InvalidHyperparameter { name: "factor", .. })
        ));
    }
}

#[test]
fn halving_builder_rejects_invalid_resource_bounds() {
    assert!(matches!(
        classifier_builder().minimum_resource(0).build().err(),
        Some(SKError::InvalidHyperparameter {
            name: "minimum_resource",
            ..
        })
    ));
    assert!(matches!(
        regressor_builder().maximum_resource(0).build().err(),
        Some(SKError::InvalidHyperparameter {
            name: "maximum_resource",
            ..
        })
    ));
    assert!(matches!(
        regressor_builder()
            .minimum_resource(50)
            .maximum_resource(20)
            .build()
            .err(),
        Some(SKError::InvalidHyperparameter {
            name: "minimum_resource",
            ..
        })
    ));
}

#[test]
fn halving_builder_rejects_empty_candidates_and_missing_splitter() {
    let empty_error = SKHalvingGridSearchClassifier::<f64, SKMockThresholdClassifier>::build()
        .splitter(three_folds())
        .build()
        .err();
    assert!(matches!(
        empty_error,
        Some(SKError::InvalidHyperparameter {
            name: "candidates",
            ..
        })
    ));

    let splitter_error = SKHalvingGridSearchRegressor::<f64, _>::build()
        .candidates(vec![SKMockConstantRegressor::new(0.0)])
        .build()
        .err();
    assert!(matches!(
        splitter_error,
        Some(SKError::InvalidHyperparameter {
            name: "splitter",
            ..
        })
    ));
}

#[test]
fn halving_builder_rejects_incompatible_scoring_policy_on_both_families() {
    assert!(matches!(
        classifier_builder()
            .scoring(SKRegressorScoring::R2)
            .build()
            .err(),
        Some(SKError::InvalidHyperparameter {
            name: "scoring",
            ..
        })
    ));
    assert!(matches!(
        regressor_builder()
            .scoring(SKClassifierScoring::BalancedAccuracy)
            .build()
            .err(),
        Some(SKError::InvalidHyperparameter {
            name: "scoring",
            ..
        })
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
