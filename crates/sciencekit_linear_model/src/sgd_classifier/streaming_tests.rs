//! Companion tests for the single-pass out-of-core fit (spec
//! `stochastic-gradient-classifier`).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKReopenableSource, SKSupervisedFit};

use super::SKSGDClassifier;
use crate::sgd::SKConvergencePolicy;
use crate::test_datasets::{
    SKSupervisedFloatSource, assert_close, labels_i64, labels_scalar, separable_binary_design,
    to_f32,
};

fn split_batches<F: sciencekit_common::SKFloat>(
    features: &Array2<F>,
    labels: &[usize],
    split: usize,
) -> Vec<(Array2<F>, Array1<F>)> {
    let first = features.slice(ndarray::s![0..split, ..]).to_owned();
    let second = features.slice(ndarray::s![split.., ..]).to_owned();
    vec![
        (first, labels_scalar::<F>(&labels[0..split])),
        (second, labels_scalar::<F>(&labels[split..])),
    ]
}

#[test]
fn one_streaming_pass_equals_one_in_memory_epoch() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let in_memory = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &raw, 4));
    let streamed = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_close(
        in_memory.coefficients().as_slice().unwrap(),
        streamed.coefficients().as_slice().unwrap(),
        1e-9,
    );
    assert_close(
        in_memory.intercepts().as_slice().unwrap(),
        streamed.intercepts().as_slice().unwrap(),
        1e-9,
    );
    assert_eq!(streamed.epochs_taken(), 1);
    assert!(!streamed.converged());
}

/// A reopenable provider runs the same epoch budget as the in-memory fit: with
/// a fixed seed and one batch per pass the streamed trajectory is identical.
#[test]
fn a_reopenable_source_runs_the_epoch_budget_and_matches_the_in_memory_fit() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let estimator = || {
        SKSGDClassifier::<f64>::build()
            .shuffle(true)
            .random_seed(7)
            .maximum_number_of_iterations(3)
            .average(true)
            .convergence_policy(SKConvergencePolicy::Permissive)
            .build()
            .unwrap()
    };
    let in_memory = estimator().fit(features.view(), labels.view()).unwrap();
    let batches = vec![(features.clone(), labels_scalar::<f64>(&raw))];
    let provider =
        SKReopenableSource::new(move || Ok(SKSupervisedFloatSource::<f64>::new(batches.clone())));
    let streamed = estimator().fit_streaming(provider).unwrap();
    assert_eq!(streamed.epochs_taken(), 3);
    assert!(!streamed.converged());
    assert_close(
        in_memory.coefficients().as_slice().unwrap(),
        streamed.coefficients().as_slice().unwrap(),
        1e-12,
    );
    assert_close(
        in_memory.intercepts().as_slice().unwrap(),
        streamed.intercepts().as_slice().unwrap(),
        1e-12,
    );
}

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
fn an_f32_stream_fits() {
    let (features, raw) = separable_binary_design();
    let features = to_f32(&features);
    let mut source = SKSupervisedFloatSource::new(split_batches(&features, &raw, 4));
    let model = SKSGDClassifier::<f32>::build()
        .shuffle(false)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(model.number_of_heads(), 1);
    assert_eq!(model.epochs_taken(), 1);
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
