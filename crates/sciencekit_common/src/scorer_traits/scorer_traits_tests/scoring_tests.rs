//! Tests for scoring contracts (spec `scoring-contracts`).
use ndarray::{Array2, array};

use super::super::{SKSupervisedScorer, SKUnsupervisedScorer};
use crate::SKError;
use crate::fit_traits::SKPredictor;

use super::example_contracts::{Accuracy, ConstantPredictor, MajorityAgreement, ZeroPredictor};
use crate::{SKDataView, SKTargetView};

// ---- Tests -----------------------------------------------------------------

/// A typed handle resolving the generic scorer for a concrete model family.
#[allow(clippy::extra_unused_type_parameters)]
fn accuracy<F: crate::SKFloat, M: SKPredictor<F, Error = SKError>>() -> Accuracy {
    Accuracy
}
/// A typed handle resolving the generic unsupervised scorer.
#[allow(clippy::extra_unused_type_parameters)]
fn majority<F: crate::SKFloat, M: SKPredictor<F, Error = SKError>>() -> MajorityAgreement {
    MajorityAgreement
}

/// Pure form evaluates stored predictions without re-inferring.
#[test]
fn pure_form_does_not_re_infer() {
    let scorer = accuracy::<f64, ConstantPredictor>();
    let truth = array![1.0_f64, 1.0, 0.0];
    let predictions = array![1.0_f64, 1.0, 1.0];
    let score = <Accuracy as SKSupervisedScorer<f64, ConstantPredictor>>::score_from_predictions(
        &scorer,
        SKTargetView::try_from(truth.view()).unwrap(),
        predictions.view(),
    )
    .unwrap();
    // 2 of 3 correct (predicted 1,1,1 vs true 1,1,0).
    assert!((score - 2.0 / 3.0).abs() < 1e-9);
}

/// Convenient form runs inference then delegates to the pure form.
#[test]
fn convenient_form_runs_inference_and_delegates() {
    let scorer = accuracy::<f64, ConstantPredictor>();
    let model = ConstantPredictor;
    let features = Array2::from_shape_vec((3, 2), vec![0.0_f64; 6]).unwrap();
    let truth = array![1.0_f64, 1.0, 0.0];

    let via_convenient = scorer
        .score(
            &model,
            SKDataView::try_from(&features).unwrap(),
            SKTargetView::try_from(truth.view()).unwrap(),
        )
        .unwrap();
    // model.predict returns all 1.0 → pure form on those same predictions.
    let predictions = model.predict(features.view()).unwrap();
    let via_pure =
        <Accuracy as SKSupervisedScorer<f64, ConstantPredictor>>::score_from_predictions(
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
    let scorer = accuracy::<f64, ConstantPredictor>();
    let truth = array![1.0_f64, 1.0, 0.0]; // 3 targets
    let short_predictions = array![1.0_f64, 1.0]; // 2 predictions → mismatch
    let result = <Accuracy as SKSupervisedScorer<f64, ConstantPredictor>>::score_from_predictions(
        &scorer,
        SKTargetView::try_from(truth.view()).unwrap(),
        short_predictions.view(),
    );
    match result {
        Err(SKError::ShapeMismatch { .. }) => {}
        other => panic!("expected shape mismatch, got {other:?}"),
    }
}

/// The same scorer evaluates models from distinct families.
#[test]
fn same_scorer_evaluates_distinct_families() {
    let scorer = accuracy::<f64, ConstantPredictor>();
    let truth = array![0.0_f64, 0.0, 0.0];
    let features = Array2::from_shape_vec((3, 2), vec![0.0_f64; 6]).unwrap();

    let constant = scorer
        .score(
            &ConstantPredictor,
            SKDataView::try_from(&features).unwrap(),
            SKTargetView::try_from(truth.view()).unwrap(),
        )
        .unwrap();
    let zero = scorer
        .score(
            &ZeroPredictor,
            SKDataView::try_from(&features).unwrap(),
            SKTargetView::try_from(truth.view()).unwrap(),
        )
        .unwrap();

    // ConstantPredictor predicts 1 → 0 accuracy on all-0 truth.
    assert!((constant - 0.0).abs() < 1e-9);
    // ZeroPredictor predicts 0 → 1.0 accuracy.
    assert!((zero - 1.0).abs() < 1e-9);
}

/// Unsupervised scorer: pure form over existing assignments.
#[test]
fn unsupervised_pure_form_over_assignments() {
    let scorer = majority::<f64, ConstantPredictor>();
    let features = Array2::from_shape_vec((4, 2), vec![0.0_f64; 8]).unwrap();
    let assignments = array![1_usize, 1, 0, 1];
    let score =
        <MajorityAgreement as SKUnsupervisedScorer<f64, ConstantPredictor>>::score_from_assignments(
            &scorer, SKDataView::try_from(&features).unwrap(), assignments.view(),
        )
        .unwrap();
    assert!((score - 0.75).abs() < 1e-9);
}

/// Unsupervised scorer: convenient form obtains model outputs then delegates.
#[test]
fn unsupervised_convenient_form_delegates() {
    let scorer = majority::<f64, ConstantPredictor>();
    let model = ConstantPredictor; // predicts all 1.0 → assignments all 1
    let features = Array2::from_shape_vec((4, 2), vec![0.0_f64; 8]).unwrap();
    let score = scorer
        .score(&model, SKDataView::try_from(&features).unwrap())
        .unwrap();
    assert!((score - 1.0).abs() < 1e-9);
}
