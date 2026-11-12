//! Determinism, parallel-prediction and concurrency tests (task 6.3).

use std::sync::Arc;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::classifier::SKDecisionTreeClassifier;
use crate::regressor::SKDecisionTreeRegressor;
use crate::splitter::SKSplitter;

fn classification_fixture() -> (Array2<f64>, Array1<i64>) {
    let rows = 400usize;
    let features = Array2::from_shape_fn((rows, 3), |(row, column)| {
        ((row * 13 + column * 7) % 29) as f64
    });
    let labels = Array1::from_shape_fn(rows, |row| (row % 3) as i64);
    (features, labels)
}

/// The same configuration and input produce the same node array twice.
#[test]
fn repeated_classifier_fits_agree() {
    let (features, labels) = classification_fixture();
    let first = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(5)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let second = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(5)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert_eq!(first.tree(), second.tree());
    assert_eq!(
        first.predict_labels(features.view()).expect("predict"),
        second.predict_labels(features.view()).expect("predict")
    );
}

/// The random splitter and feature subsampling are reproducible for a seed.
#[test]
fn seeded_random_fits_agree() {
    let (features, labels) = classification_fixture();
    let build = || {
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_depth(4)
            .maximum_features(crate::splitter::SKMaximumFeatures::Sqrt)
            .splitter(SKSplitter::Random)
            .random_seed(1234)
            .build()
            .expect("builds")
            .fit(&features, labels.view())
            .expect("fit succeeds")
    };
    assert_eq!(build().tree(), build().tree());
}

/// The parallel prediction path equals a sequential walk of the same tree.
#[test]
fn parallel_classifier_prediction_equals_sequential() {
    let (features, labels) = classification_fixture();
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(5)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let parallel = model
        .predict_probabilities(features.view())
        .expect("predict");
    let sequential = Array2::from_shape_fn(
        (features.nrows(), model.number_of_classes()),
        |(row, class)| {
            let leaf = model
                .tree()
                .leaf_index_for(|feature| features[[row, feature]]);
            model.tree().node(leaf).expect("leaf").value()[class]
        },
    );
    assert_eq!(parallel, sequential);
}

/// The parallel regression path equals a sequential walk of the same tree.
#[test]
fn parallel_regressor_prediction_equals_sequential() {
    let (features, _) = classification_fixture();
    let responses = Array1::from_shape_fn(features.nrows(), |row| (row % 11) as f64);
    let model = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(5)
        .build()
        .expect("builds")
        .fit(&features, responses.view())
        .expect("fit succeeds");
    let parallel = model.predict(features.view()).expect("predict");
    let sequential = Array1::from_shape_fn(features.nrows(), |row| {
        let leaf = model
            .tree()
            .leaf_index_for(|feature| features[[row, feature]]);
        model.tree().node(leaf).expect("leaf").value()[0]
    });
    assert_eq!(parallel, sequential);
}

/// A fitted model is shareable across threads without locking, and every
/// concurrent call returns the sequential result.
#[test]
fn one_model_serves_several_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKDecisionTreeClassifier<f64>>();
    assert_send_sync::<crate::classifier::SKDecisionTreeClassifierModel<f64>>();
    assert_send_sync::<crate::regressor::SKDecisionTreeRegressorModel<f64>>();

    let (features, labels) = classification_fixture();
    let model = Arc::new(
        SKDecisionTreeClassifier::<f64>::build()
            .maximum_depth(5)
            .build()
            .expect("builds")
            .fit(&features, labels.view())
            .expect("fit succeeds"),
    );
    let expected = model.predict_labels(features.view()).expect("predict");
    let shared_features = Arc::new(features);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let model = Arc::clone(&model);
            let features = Arc::clone(&shared_features);
            std::thread::spawn(move || model.predict_labels(features.view()).expect("predict"))
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().expect("thread joins"), expected);
    }
}

/// A depth-limited tree over the same data is independent of how many rows are
/// predicted at once.
#[test]
fn prediction_batches_agree_with_the_whole_batch() {
    let (features, labels) = classification_fixture();
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(4)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let whole = model.predict_labels(features.view()).expect("predict");
    for start in (0..features.nrows()).step_by(37) {
        let end = (start + 37).min(features.nrows());
        let batch = Array2::from_shape_fn((end - start, features.ncols()), |(row, column)| {
            features[[start + row, column]]
        });
        let predicted = model.predict_labels(batch.view()).expect("predict");
        for offset in 0..(end - start) {
            assert_eq!(predicted[offset], whole[start + offset]);
        }
    }
}
