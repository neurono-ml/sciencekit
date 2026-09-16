//! Tests for scoring contracts (spec `scoring-contracts`).
use ndarray::{Array2, array};

use super::super::{SKLabelScorer, SKSupervisedScorer, SKUnsupervisedScorer};
use crate::SKError;
use crate::fit_traits::{SKClassifierPredictor, SKRegressorPredictor};

use super::example_contracts::{
    Accuracy, ConstantClassifier, ConstantRegressor, MajorityAgreement, MeanAbsoluteError,
    ZeroClassifier,
};
use crate::{SKDataView, SKTargetView};

// ---- Tests -----------------------------------------------------------------

/// A typed handle resolving the label scorer for a concrete classifier family.
#[allow(clippy::extra_unused_type_parameters)]
fn accuracy<F: crate::SKFloat, M: SKClassifierPredictor<F, Error = SKError>>() -> Accuracy {
    Accuracy
}

/// A typed handle resolving the continuous scorer for a regressor family.
#[allow(clippy::extra_unused_type_parameters)]
fn mean_absolute_error<F: crate::SKFloat, M: SKRegressorPredictor<F, Error = SKError>>()
-> MeanAbsoluteError {
    MeanAbsoluteError
}

/// A typed handle resolving the generic unsupervised scorer.
#[allow(clippy::extra_unused_type_parameters)]
fn majority<F: crate::SKFloat, M: SKRegressorPredictor<F, Error = SKError>>() -> MajorityAgreement {
    MajorityAgreement
}

/// Label pure form evaluates stored labels without re-inferring.
#[test]
fn label_pure_form_does_not_re_infer() {
    let scorer = accuracy::<f64, ConstantClassifier>();
    let truth = array![1_i64, 1, 0];
    let labels = array![1_i64, 1, 1];
    let score = <Accuracy as SKLabelScorer<f64, ConstantClassifier>>::score_from_labels(
        &scorer,
        SKTargetView::try_from(truth.view()).unwrap(),
        labels.view(),
    )
    .unwrap();
    // 2 of 3 correct (predicted 1,1,1 vs true 1,1,0).
    assert!((score - 2.0 / 3.0).abs() < 1e-9);
}

/// Label pure form works in f32 with integer truth.
#[test]
fn label_pure_form_in_f32() {
    let scorer = accuracy::<f32, ConstantClassifier>();
    let truth = array![1_i64, 0];
    let labels = array![1_i64, 1];
    let score = <Accuracy as SKLabelScorer<f32, ConstantClassifier>>::score_from_labels(
        &scorer,
        SKTargetView::try_from(truth.view()).unwrap(),
        labels.view(),
    )
    .unwrap();
    assert!((score - 0.5_f32).abs() < 1e-6);
}

/// Label convenient form runs inference then delegates to the pure form.
#[test]
fn label_convenient_form_runs_inference_and_delegates() {
    let scorer = accuracy::<f64, ConstantClassifier>();
    let model = ConstantClassifier;
    let features = Array2::from_shape_vec((3, 2), vec![0.0_f64; 6]).unwrap();
    let truth = array![1_i64, 1, 0];

    let via_convenient = scorer
        .score(
            &model,
            SKDataView::try_from(&features).unwrap(),
            SKTargetView::try_from(truth.view()).unwrap(),
        )
        .unwrap();
    // model predicts all 1 → pure form on those same labels.
    let labels = model.predict_labels(features.view()).unwrap();
    let via_pure = <Accuracy as SKLabelScorer<f64, ConstantClassifier>>::score_from_labels(
        &scorer,
        SKTargetView::try_from(truth.view()).unwrap(),
        labels.view(),
    )
    .unwrap();
    assert!((via_convenient - via_pure).abs() < 1e-12);
}

/// Continuous pure form stays in the model scalar (f32, no f64 detour).
#[test]
fn continuous_pure_form_preserves_scalar() {
    let scorer = mean_absolute_error::<f32, ConstantRegressor>();
    let truth = array![1.0_f32, 2.0, 4.0];
    let predictions = array![1.0_f32, 1.0, 1.0];
    let score =
        <MeanAbsoluteError as SKSupervisedScorer<f32, ConstantRegressor>>::score_from_predictions(
            &scorer,
            SKTargetView::try_from(truth.view()).unwrap(),
            predictions.view(),
        )
        .unwrap();
    // (|0| + |1| + |3|) / 3 = 4/3.
    assert!((score - 4.0_f32 / 3.0).abs() < 1e-6);
}

/// Continuous convenient form runs inference then delegates.
#[test]
fn continuous_convenient_form_runs_inference_and_delegates() {
    let scorer = mean_absolute_error::<f64, ConstantRegressor>();
    let model = ConstantRegressor;
    let features = Array2::from_shape_vec((3, 2), vec![0.0_f64; 6]).unwrap();
    let truth = array![1.0_f64, 1.0, 0.0];

    let via_convenient = scorer
        .score(
            &model,
            SKDataView::try_from(&features).unwrap(),
            SKTargetView::try_from(truth.view()).unwrap(),
        )
        .unwrap();
    let predictions = model.predict(features.view()).unwrap();
    let via_pure =
        <MeanAbsoluteError as SKSupervisedScorer<f64, ConstantRegressor>>::score_from_predictions(
            &scorer,
            SKTargetView::try_from(truth.view()).unwrap(),
            predictions.view(),
        )
        .unwrap();
    assert!((via_convenient - via_pure).abs() < 1e-12);
}

/// Incoherent inputs produce a structured taxonomy error, never a panic/sentinel.
#[test]
fn incoherent_inputs_produce_structured_error() {
    let scorer = accuracy::<f64, ConstantClassifier>();
    let truth = array![1_i64, 1, 0]; // 3 targets
    let short_labels = array![1_i64, 1]; // 2 labels → mismatch
    let result = <Accuracy as SKLabelScorer<f64, ConstantClassifier>>::score_from_labels(
        &scorer,
        SKTargetView::try_from(truth.view()).unwrap(),
        short_labels.view(),
    );
    match result {
        Err(SKError::ShapeMismatch { .. }) => {}
        other => panic!("expected shape mismatch, got {other:?}"),
    }
}

/// The same scorer evaluates models from distinct families.
#[test]
fn same_scorer_evaluates_distinct_families() {
    let scorer = accuracy::<f64, ConstantClassifier>();
    let truth = array![0_i64, 0, 0];
    let features = Array2::from_shape_vec((3, 2), vec![0.0_f64; 6]).unwrap();

    let constant = scorer
        .score(
            &ConstantClassifier,
            SKDataView::try_from(&features).unwrap(),
            SKTargetView::try_from(truth.view()).unwrap(),
        )
        .unwrap();
    let zero = <Accuracy as SKLabelScorer<f64, ZeroClassifier>>::score_from_labels(
        &accuracy::<f64, ZeroClassifier>(),
        SKTargetView::try_from(truth.view()).unwrap(),
        ZeroClassifier
            .predict_labels(features.view())
            .unwrap()
            .view(),
    )
    .unwrap();

    // ConstantClassifier predicts 1 → 0 accuracy on all-0 truth.
    assert!((constant - 0.0).abs() < 1e-9);
    // ZeroClassifier predicts 0 → 1.0 accuracy.
    assert!((zero - 1.0).abs() < 1e-9);
}

/// Unsupervised scorer: pure form over existing assignments.
#[test]
fn unsupervised_pure_form_over_assignments() {
    let scorer = majority::<f64, ConstantRegressor>();
    let features = Array2::from_shape_vec((4, 2), vec![0.0_f64; 8]).unwrap();
    let assignments = array![1_usize, 1, 0, 1];
    let score =
        <MajorityAgreement as SKUnsupervisedScorer<f64, ConstantRegressor>>::score_from_assignments(
            &scorer, SKDataView::try_from(&features).unwrap(), assignments.view(),
        )
        .unwrap();
    assert!((score - 0.75).abs() < 1e-9);
}

/// Unsupervised scorer: convenient form obtains model outputs then delegates.
#[test]
fn unsupervised_convenient_form_delegates() {
    let scorer = majority::<f64, ConstantRegressor>();
    let model = ConstantRegressor; // predicts all 1.0 → assignments all 1
    let features = Array2::from_shape_vec((4, 2), vec![0.0_f64; 8]).unwrap();
    let score = scorer
        .score(&model, SKDataView::try_from(&features).unwrap())
        .unwrap();
    assert!((score - 1.0).abs() < 1e-9);
}
