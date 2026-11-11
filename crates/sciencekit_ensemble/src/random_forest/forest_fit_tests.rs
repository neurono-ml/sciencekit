//! Companion tests for parallel per-tree fitting and seed determinism (task 2.2).

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKExecutionMode, SKRegressorPredictor,
    SKSupervisedFit,
};
use sciencekit_tree::SKMaximumFeatures;

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

fn classification_fixture() -> (Array2<f64>, Array1<i64>) {
    let features = Array2::from_shape_fn((120, 4), |(row, column)| {
        ((row * 17 + column * 11) % 37) as f64 / 10.0
    });
    let labels = Array1::from_shape_fn(120, |row| {
        i64::from(features[[row, 0]] + features[[row, 1]] > 3.5)
    });
    (features, labels)
}

fn regression_fixture() -> (Array2<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((120, 4), |(row, column)| {
        ((row * 13 + column * 19) % 41) as f64 / 10.0
    });
    let responses = Array1::from_shape_fn(120, |row| {
        2.0 * features[[row, 0]] - features[[row, 2]] + 0.5
    });
    (features, responses)
}

/// Repeated fits with the same seed produce identical tree structures, leaf
/// values and predictions for both the classifier and the regressor.
#[test]
fn repeated_fits_produce_identical_trees() {
    let (features, labels) = classification_fixture();
    let classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(4)
        .random_seed(2026)
        .build()
        .expect("builds");
    let first = classifier.fit(&features, labels.view()).expect("fit");
    let second = classifier.fit(&features, labels.view()).expect("fit");
    assert_eq!((first.number_of_trees(), first.feature_count()), (16, 4));
    assert!((0..16).all(|index| first.tree(index) == second.tree(index)));
    assert_eq!(
        first.predict_probabilities(&features).expect("predict"),
        second.predict_probabilities(&features).expect("predict")
    );

    let (regression_features, responses) = regression_fixture();
    let regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(4)
        .random_seed(2026)
        .build()
        .expect("builds");
    let first_model = regressor
        .fit(&regression_features, responses.view())
        .expect("fit");
    let second_model = regressor
        .fit(&regression_features, responses.view())
        .expect("fit");
    assert!((0..16).all(|index| first_model.tree(index) == second_model.tree(index)));
    assert_eq!(
        first_model.predict(&regression_features).expect("predict"),
        second_model.predict(&regression_features).expect("predict")
    );
}

/// Fitting sequentially (`InProcessSynchronous`) and through the parallel path
/// (`Automatic`) produces the exact same forest.
#[test]
fn parallel_fit_equals_sequential_fit() {
    let (features, labels) = classification_fixture();
    let fit_classifier = |mode| {
        SKRandomForestClassifier::<f64>::build()
            .number_of_trees(20)
            .maximum_depth(5)
            .random_seed(77)
            .execution_mode(mode)
            .build()
            .expect("builds")
            .fit(&features, labels.view())
            .expect("fit")
    };
    let sequential = fit_classifier(SKExecutionMode::InProcessSynchronous);
    let parallel = fit_classifier(SKExecutionMode::Automatic);
    assert!((0..20).all(|index| sequential.tree(index) == parallel.tree(index)));

    let (regression_features, responses) = regression_fixture();
    let fit_regressor = |mode| {
        SKRandomForestRegressor::<f64>::build()
            .number_of_trees(20)
            .maximum_depth(5)
            .random_seed(77)
            .execution_mode(mode)
            .build()
            .expect("builds")
            .fit(&regression_features, responses.view())
            .expect("fit")
    };
    let sequential_model = fit_regressor(SKExecutionMode::InProcessSynchronous);
    let parallel_model = fit_regressor(SKExecutionMode::Automatic);
    assert!((0..20).all(|index| sequential_model.tree(index) == parallel_model.tree(index)));
}

/// Fitting on rayon pools of different thread counts produces identical trees
/// and predictions.
#[test]
fn thread_count_does_not_change_fitted_forest() {
    let (features, labels) = classification_fixture();
    let estimator = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(24)
        .maximum_depth(5)
        .random_seed(314)
        .build()
        .expect("builds");
    let fit_on_threads = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| estimator.fit(&features, labels.view()).expect("fit"))
    };
    let (one_thread, four_threads) = (fit_on_threads(1), fit_on_threads(4));
    assert!((0..24).all(|index| one_thread.tree(index) == four_threads.tree(index)));
    assert_eq!(
        one_thread
            .predict_probabilities(&features)
            .expect("predict"),
        four_threads
            .predict_probabilities(&features)
            .expect("predict")
    );
}

/// On nine features where feature 8 cleanly separates the classes, `All`
/// always picks feature 8 at root while `Sqrt` (3 features) diversifies.
#[test]
fn feature_subsampling_sqrt_and_all_control_candidate_splits() {
    assert_eq!(SKMaximumFeatures::<f64>::Sqrt.resolved_count(9), 3);
    assert_eq!(SKMaximumFeatures::<f64>::All.resolved_count(9), 9);

    let features = Array2::from_shape_fn((80, 9), |(row, column)| {
        if column == 8 {
            if row < 40 { 0.0 } else { 10.0 }
        } else {
            ((row + column * 3) % 11) as f64
        }
    });
    let labels = Array1::from_shape_fn(80, |row| i64::from(row >= 40));
    let root_features = |policy| -> HashSet<Option<usize>> {
        let model = SKRandomForestClassifier::<f64>::build()
            .number_of_trees(24)
            .maximum_depth(1)
            .bootstrap(false)
            .maximum_features(policy)
            .random_seed(5)
            .build()
            .expect("builds")
            .fit(&features, labels.view())
            .expect("fit");
        (0..24)
            .map(|index| {
                model
                    .tree(index)
                    .and_then(|tree| tree.node(0))
                    .and_then(|node| node.split_feature())
            })
            .collect()
    };
    assert_eq!(
        root_features(SKMaximumFeatures::All),
        HashSet::from([Some(8)])
    );
    assert!(root_features(SKMaximumFeatures::Sqrt).len() > 1);
}

/// Backend and execution intent transfer from the estimator into the model.
#[test]
fn backend_and_execution_intent_transfer_to_fitted_model() {
    let (features, labels) = classification_fixture();
    let model = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(4)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}
