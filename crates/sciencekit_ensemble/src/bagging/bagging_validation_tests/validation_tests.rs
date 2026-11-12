//! Validation tests covering empty inputs, invalid hyperparameters, shape
//! mismatches, non-finite inputs and out-of-range base outputs.

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKExecutionMode, SKRegressorPredictor,
    SKSupervisedFit,
};
use sciencekit_tree::{SKDecisionTreeClassifier, SKMaximumFeatures};
use sprs::CsMat;

use crate::{SKBaggingClassifier, SKBaggingError, SKBaggingRegressor, SKMaximumSamples};

use super::fixtures::{InvalidOutputBase, default_classifier, default_regressor};

#[test]
fn zero_estimators_and_invalid_fractions_are_rejected_at_construction() {
    let tree = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    assert!(matches!(
        SKBaggingClassifier::build(tree.clone())
            .number_of_estimators(0)
            .build(),
        Err(SKError::InvalidHyperparameter {
            name: "number_of_estimators",
            ..
        })
    ));
    assert!(matches!(
        SKBaggingClassifier::build(tree.clone())
            .maximum_samples(SKMaximumSamples::Fraction(0.0))
            .build(),
        Err(SKError::InvalidHyperparameter {
            name: "maximum_samples",
            ..
        })
    ));
    assert!(matches!(
        SKBaggingClassifier::build(tree)
            .maximum_features(SKMaximumFeatures::Fraction(1.5))
            .build(),
        Err(SKError::InvalidHyperparameter {
            name: "maximum_features",
            ..
        })
    ));
}

#[test]
fn empty_inputs_and_feature_count_mismatches_return_central_errors() {
    let empty_dense = Array2::<f64>::zeros((0, 2));
    let empty_sparse: CsMat<f64> = CsMat::empty(sprs::CSR, 2);
    let (labels, responses) = (array![0_i64, 1], array![1.0, 2.0]);

    assert!(matches!(
        default_classifier().fit(&empty_dense, labels.view()),
        Err(SKBaggingError::Central(SKError::EmptyInput {
            operation: "fit"
        }))
    ));
    assert!(matches!(
        default_classifier().fit(empty_sparse.view(), labels.view()),
        Err(SKBaggingError::Central(SKError::EmptyInput {
            operation: "fit"
        }))
    ));
    assert!(matches!(
        default_regressor().fit(&empty_dense, responses.view()),
        Err(SKBaggingError::Central(SKError::EmptyInput {
            operation: "fit"
        }))
    ));

    let train = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
    let classifier_model = default_classifier()
        .fit(&train, array![0_i64, 0, 1, 1].view())
        .unwrap();
    let regressor_model = default_regressor()
        .fit(&train, array![1.0, 2.0, 3.0, 4.0].view())
        .unwrap();

    assert!(matches!(
        classifier_model.predict_labels(&empty_dense),
        Err(SKBaggingError::Central(SKError::EmptyInput {
            operation: "predict"
        }))
    ));
    assert!(matches!(
        regressor_model.predict(&empty_dense),
        Err(SKBaggingError::Central(SKError::EmptyInput {
            operation: "predict"
        }))
    ));
    assert!(matches!(
        classifier_model.predict_probabilities(array![[1.0, 2.0, 3.0]].view()),
        Err(SKBaggingError::Central(SKError::ShapeMismatch {
            expected,
            found
        })) if expected == vec![1, 2] && found == vec![1, 3]
    ));
    assert!(matches!(
        regressor_model.predict(array![[1.0]].view()),
        Err(SKBaggingError::Central(SKError::ShapeMismatch {
            expected,
            found
        })) if expected == vec![1, 2] && found == vec![1, 1]
    ));
}

#[test]
fn non_finite_values_target_errors_and_out_of_range_probabilities_are_rejected() {
    let (labels, responses) = (array![0_i64, 1], array![1.0, 2.0]);
    let bad_sparse = CsMat::new((2, 2), vec![0, 1, 2], vec![1, 0], vec![f64::NAN, 1.0]);

    assert!(matches!(
        default_classifier().fit(&array![[0.0, f64::NAN], [1.0, 2.0]], labels.view()),
        Err(SKBaggingError::Central(SKError::NonFiniteValue {
            position: 1
        }))
    ));
    assert!(matches!(
        default_regressor().fit(bad_sparse.view(), responses.view()),
        Err(SKBaggingError::Central(SKError::NonFiniteValue {
            position: 1
        }))
    ));
    assert!(matches!(
        default_regressor().fit(
            &array![[0.0, 1.0], [2.0, 3.0]],
            array![1.0, f64::INFINITY].view()
        ),
        Err(SKBaggingError::Central(SKError::NonFiniteValue {
            position: 1
        }))
    ));
    assert!(matches!(
        default_classifier().fit(&array![[0.0], [1.0], [2.0]], array![5_i64, 5, 5].view()),
        Err(SKBaggingError::Central(SKError::InvalidHyperparameter {
            name: "targets",
            ..
        }))
    ));

    let streaming =
        SKBaggingClassifier::build(SKDecisionTreeClassifier::<f64>::build().build().unwrap())
            .execution_mode(SKExecutionMode::OutOfCoreStreaming)
            .build()
            .unwrap();
    assert!(matches!(
        streaming.fit(&array![[0.0], [1.0]], labels.view()),
        Err(SKBaggingError::Central(
            SKError::ExecutionModeIncompatible {
                mode: "out-of-core-streaming",
                ..
            }
        ))
    ));

    let clean = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
    let out_of_range_model = SKBaggingClassifier::build(InvalidOutputBase(1.4))
        .number_of_estimators(2)
        .bootstrap(false)
        .build()
        .unwrap()
        .fit(&clean, array![0_i64, 0, 1, 1].view())
        .unwrap();
    assert!(matches!(
        out_of_range_model.predict_probabilities(&clean),
        Err(SKBaggingError::Central(SKError::ProbabilityOutOfRange {
            position: 0
        }))
    ));
    let nan_oob = SKBaggingRegressor::build(InvalidOutputBase(f64::NAN))
        .number_of_estimators(8)
        .oob_score(true)
        .random_seed(3)
        .build()
        .unwrap()
        .fit(&clean, array![1.0, 2.0, 3.0, 4.0].view());
    assert!(matches!(
        nan_oob,
        Err(SKBaggingError::Central(SKError::NonFiniteValue {
            position: 0
        }))
    ));
}
