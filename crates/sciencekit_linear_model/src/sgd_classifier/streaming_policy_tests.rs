//! Companion tests for the pass policy and the rejection paths of the
//! multi-epoch out-of-core classifier fit
//! (spec `stochastic-gradient-classifier`).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKReopenableSource};

use super::SKSGDClassifier;
use super::streaming_tests::split_batches;
use crate::test_datasets::{SKSupervisedFloatSource, separable_binary_design};

/// A one-shot source with a larger budget still takes exactly one pass and does
/// not raise under the strict policy.
#[test]
fn a_one_shot_source_with_a_larger_budget_takes_one_pass() {
    let (features, raw) = separable_binary_design();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &raw, 4));
    let model = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(5)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(model.epochs_taken(), 1);
    assert!(!model.converged());
}

/// A one-shot source with a budget of exactly one pass really did exhaust its
/// budget, so the strict policy raises instead of pretending the provider
/// clamped it.
#[test]
fn a_one_shot_source_with_a_budget_of_one_raises_under_strict() {
    let (features, raw) = separable_binary_design();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &raw, 4));
    let error = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .number_of_epochs_without_improvement(10)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations: 1 }));
}

/// A reopenable source that exhausts the whole requested budget without
/// converging raises under the strict policy.
#[test]
fn an_exhausted_reopenable_budget_raises_under_strict() {
    let (features, raw) = separable_binary_design();
    let batches = split_batches(&features, &raw, 4);
    let provider =
        SKReopenableSource::new(move || Ok(SKSupervisedFloatSource::<f64>::new(batches.clone())));
    let error = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(2)
        .number_of_epochs_without_improvement(10)
        .build()
        .unwrap()
        .fit_streaming(provider)
        .unwrap_err();
    assert!(matches!(error, SKError::NotConverged { iterations: 2 }));
}

#[test]
fn an_empty_source_is_rejected() {
    let mut source = SKSupervisedFloatSource::<f64>::new(Vec::new());
    let error = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(matches!(error, SKError::Conversion(_)));
}

#[test]
fn a_column_mismatch_is_rejected() {
    let batches = vec![
        (
            Array2::from_elem((2, 1), 1.0),
            Array1::from_vec(vec![0.0, 1.0]),
        ),
        (
            Array2::from_elem((2, 2), 1.0),
            Array1::from_vec(vec![0.0, 1.0]),
        ),
    ];
    let mut source = SKSupervisedFloatSource::new(batches);
    let error = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn early_stopping_over_a_stream_is_refused() {
    let (features, raw) = separable_binary_design();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &raw, 4));
    let error = SKSGDClassifier::<f64>::build()
        .early_stopping(true)
        .maximum_number_of_iterations(2)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(
        matches!(error, SKError::UnsupportedOperation { operation, .. } if operation == "early_stopping")
    );
}

#[test]
fn a_class_introduced_after_the_first_batch_is_rejected() {
    let batches = vec![
        (
            Array2::from_shape_vec((2, 1), vec![-1.0, 1.0]).unwrap(),
            Array1::from_vec(vec![0.0, 1.0]),
        ),
        (
            Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(),
            Array1::from_vec(vec![2.0]),
        ),
    ];
    let mut source = SKSupervisedFloatSource::new(batches);
    let error = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap_err();
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "targets"));
}
