//! Companion validation tests for [`SKBaggingClassifier`] and
//! [`SKBaggingRegressor`] asserting structured central [`SKError`] variants.

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKError, SKExecutionMode, SKRegressorPredictor,
    SKSupervisedFit, SKTargetView,
};
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};
use sprs::CsMat;

use crate::{SKBaggingClassifier, SKBaggingError, SKBaggingRegressor, SKMaximumSamples};

fn default_classifier() -> SKBaggingClassifier<SKDecisionTreeClassifier<f64>> {
    let tree = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    SKBaggingClassifier::build(tree).build().unwrap()
}

fn default_regressor() -> SKBaggingRegressor<SKDecisionTreeRegressor<f64>> {
    let tree = SKDecisionTreeRegressor::<f64>::build().build().unwrap();
    SKBaggingRegressor::build(tree).build().unwrap()
}

#[derive(Clone, Debug)]
struct InvalidProbabilityClassifier(f64);

impl SKSupervisedFit<f64> for InvalidProbabilityClassifier {
    type Model = Self;
    type Error = SKError;

    fn fit<'a, X, T>(&self, _features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        Ok(self.clone())
    }
}

impl SKClassifierPredictor<f64> for InvalidProbabilityClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let probabilities = self.predict_probabilities(features)?;
        Ok(Array1::zeros(probabilities.nrows()))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array2::from_shape_fn((rows, 2), |(_, column)| {
            if column == 0 { self.0 } else { 1.0 - self.0 }
        }))
    }
}

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
    let labels = array![0_i64, 1];
    let responses = array![1.0, 2.0];

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
    let labels = array![0_i64, 1];
    let responses = array![1.0, 2.0];
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

    let clean = array![[0.0, 1.0], [2.0, 3.0]];
    let out_of_range_model = SKBaggingClassifier::build(InvalidProbabilityClassifier(1.4))
        .number_of_estimators(2)
        .bootstrap(false)
        .build()
        .unwrap()
        .fit(&clean, labels.view())
        .unwrap();
    assert!(matches!(
        out_of_range_model.predict_probabilities(&clean),
        Err(SKBaggingError::Central(SKError::ProbabilityOutOfRange {
            position: 0
        }))
    ));
}
