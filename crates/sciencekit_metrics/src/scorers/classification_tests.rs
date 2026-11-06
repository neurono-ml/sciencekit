//! TDD tests for the classification scorer adapters (spec `evaluation-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKLabelScorer, SKTargetView,
};

use crate::classification::SKAveraging;

use super::{SKAccuracyScorer, SKF1Scorer, SKPrecisionScorer, SKRecallScorer};

/// A classifier returning fixed labels, ignoring the features.
struct FixedClassifier {
    labels: Vec<i64>,
}

impl<F: SKFloat> SKClassifierPredictor<F> for FixedClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array1::from_vec(self.labels.clone()))
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array2::zeros((0, 0)))
    }
}

/// A classifier whose inference must never run in the pure form.
struct PanickingClassifier;

impl<F: SKFloat> SKClassifierPredictor<F> for PanickingClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }
}

/// The pure form scores stored labels without running inference.
#[test]
fn pure_form_scores_without_inference() {
    let scorer = SKAccuracyScorer;
    let truth_labels = array![0_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let labels = array![0_i64, 1, 1];
    let score = <SKAccuracyScorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
        &scorer,
        truth,
        labels.view(),
    )
    .expect("valid input");
    assert!((score - 1.0).abs() < 1e-12);
}

/// The convenient form equals the pure form applied to the model's labels.
#[test]
fn convenient_form_equals_pure_form() {
    let scorer = SKAccuracyScorer;
    let model = FixedClassifier {
        labels: vec![0, 1, 1],
    };
    let features = Array2::<f64>::zeros((3, 2));
    let truth_labels = array![0_i64, 1, 0];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let convenient = scorer
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect("valid input");
    let predicted = array![0_i64, 1, 1];
    let pure = <SKAccuracyScorer as SKLabelScorer<f64, FixedClassifier>>::score_from_labels(
        &scorer,
        truth,
        predicted.view(),
    )
    .expect("valid input");
    assert!((convenient - pure).abs() < 1e-12);
}

/// The precision, recall and F1 adapters honour their averaging configuration.
#[test]
fn configured_adapters_honour_averaging() {
    let truth_labels = array![0_i64, 0, 0, 0, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let labels = array![0_i64, 0, 0, 1, 1];
    let averaging = SKAveraging::Binary { positive_label: 1 };

    let precision = SKPrecisionScorer::new(averaging);
    let value = <SKPrecisionScorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
        &precision,
        truth,
        labels.view(),
    )
    .expect("valid input");
    assert!((value - 0.5).abs() < 1e-12);

    let recall = SKRecallScorer::new(averaging);
    let value = <SKRecallScorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
        &recall,
        truth,
        labels.view(),
    )
    .expect("valid input");
    assert!((value - 1.0).abs() < 1e-12);

    let f1 = SKF1Scorer::new(averaging);
    let value = <SKF1Scorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
        &f1,
        truth,
        labels.view(),
    )
    .expect("valid input");
    assert!((value - 2.0 / 3.0).abs() < 1e-12);
}

/// An explicit label list is carried by the adapter.
#[test]
fn configured_adapters_carry_explicit_labels() {
    let truth_labels = array![0_i64, 0, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let labels = array![0_i64, 1, 0];
    let scorer = SKF1Scorer::new(SKAveraging::Macro).with_labels(&[0, 1, 2]);
    let value = <SKF1Scorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
        &scorer,
        truth,
        labels.view(),
    )
    .expect("valid input");
    // Classes 0, 1 and 2: F1 1/2, 0 and 0 -> macro mean 1/6.
    assert!((value - 1.0 / 6.0).abs() < 1e-12);
}

/// A finite continuous label outside the i64 range is a conversion failure.
#[test]
fn out_of_range_continuous_labels_are_rejected() {
    let truth_values = array![1.0e300_f64];
    let truth = SKTargetView::<f64>::Continuous(truth_values.view());
    let labels = array![0_i64];
    let error = <SKAccuracyScorer as SKLabelScorer<f64, PanickingClassifier>>::score_from_labels(
        &SKAccuracyScorer,
        truth,
        labels.view(),
    )
    .expect_err("the label does not fit i64");
    assert!(matches!(error, SKError::Conversion(_)));
}
