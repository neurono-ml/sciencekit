//! Builder defaults and construction-time validation (spec `grid-search`,
//! requirement `Builder-only construction with documented defaults`).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::{SKGridSearchCVClassifierBuilder, SKGridSearchCVRegressorBuilder};
use crate::grid_search::classifier::SKGridSearchCVClassifier;
use crate::grid_search::error_score::SKErrorScore;
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::grid_search::regressor::SKGridSearchCVRegressor;
use crate::grid_search::scoring::{
    SKClassifierScoring, SKRegressorScoring, SKScoringFamily, sk_validate_scoring_family,
};
use crate::split::kfold::SKKFold;

fn classifier_builder() -> SKGridSearchCVClassifierBuilder<f64, SKMockThresholdClassifier> {
    SKGridSearchCVClassifier::build().candidates(vec![SKMockThresholdClassifier::new(0.5)])
}

fn regressor_builder() -> SKGridSearchCVRegressorBuilder<f64, SKMockConstantRegressor> {
    SKGridSearchCVRegressor::build().candidates(vec![SKMockConstantRegressor::new(0.0)])
}

fn three_folds() -> SKKFold {
    SKKFold::build().number_of_splits(3).build().unwrap()
}

#[test]
fn grid_builder_classifier_defaults_are_documented() {
    let estimator = classifier_builder()
        .splitter(three_folds())
        .build()
        .expect("the default configuration builds");
    assert!(estimator.refit());
    assert_eq!(estimator.error_score(), SKErrorScore::Raise);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.scoring(), SKClassifierScoring::Accuracy);
    assert_eq!(estimator.number_of_candidates(), 1);
    assert!(estimator.groups().is_none());
}

#[test]
fn grid_builder_regressor_defaults_are_documented() {
    let estimator = regressor_builder()
        .splitter(three_folds())
        .build()
        .expect("the default configuration builds");
    assert!(estimator.refit());
    assert_eq!(estimator.error_score(), SKErrorScore::Raise);
    assert_eq!(
        estimator.scoring(),
        SKRegressorScoring::NegativeMeanSquaredError
    );
}

#[test]
fn grid_builder_rejects_an_empty_candidate_list() {
    let error = match SKGridSearchCVClassifier::<f64, SKMockThresholdClassifier>::build()
        .splitter(three_folds())
        .build()
    {
        Err(error) => error,
        Ok(_) => panic!("an empty candidate list is invalid"),
    };
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "candidates",
            ..
        }
    ));
}

#[test]
fn grid_builder_rejects_a_missing_splitter() {
    let error = match classifier_builder().build() {
        Err(error) => error,
        Ok(_) => panic!("a splitter is required"),
    };
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "splitter",
            ..
        }
    ));
}

#[test]
fn grid_builder_rejects_an_incompatible_scoring_family() {
    let error = sk_validate_scoring_family(SKScoringFamily::Regressor, SKScoringFamily::Classifier)
        .expect_err("a regressor policy cannot drive a classifier grid");
    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "scoring",
            ..
        }
    ));
    assert_eq!(
        SKClassifierScoring::Accuracy.family(),
        SKScoringFamily::Classifier
    );
    assert_eq!(SKRegressorScoring::R2.family(), SKScoringFamily::Regressor);
}

#[test]
fn grid_builder_honors_every_optional_setter() {
    let estimator = classifier_builder()
        .splitter(three_folds())
        .scoring(SKClassifierScoring::BalancedAccuracy)
        .refit(false)
        .error_score(SKErrorScore::value(-1.0))
        .groups(array![0, 0, 1, 1, 2, 2])
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("the fully configured estimator builds");

    assert!(!estimator.refit());
    assert_eq!(estimator.error_score(), SKErrorScore::value(-1.0));
    assert_eq!(estimator.scoring(), SKClassifierScoring::BalancedAccuracy);
    assert_eq!(estimator.groups().unwrap().to_vec(), vec![0, 0, 1, 1, 2, 2]);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn grid_builder_trait_setter_is_usable_through_generic_dispatch() {
    fn configure<B: SKBuilder<SKGridSearchCVClassifier<f64, SKMockThresholdClassifier>>>(
        builder: &mut B,
        mode: SKExecutionMode,
    ) -> &mut B {
        builder.execution_mode(mode)
    }
    let mut builder = classifier_builder().splitter(three_folds());
    configure(&mut builder, SKExecutionMode::InProcessAsynchronous);
    assert_eq!(
        builder.build().unwrap().execution_mode(),
        SKExecutionMode::InProcessAsynchronous
    );
}
