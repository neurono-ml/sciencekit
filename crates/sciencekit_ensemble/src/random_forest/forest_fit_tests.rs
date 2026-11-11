//! Companion tests for parallel per-tree fitting and seed determinism (task 2.2).

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBackendKind, SKBuilder, SKExecutionMode, SKSupervisedFit};
use sciencekit_tree::SKMaximumFeatures;

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

fn classification_fixture() -> (Array2<f64>, Array1<i64>) {
    let rows = 120usize;
    let features = Array2::from_shape_fn((rows, 4), |(row, column)| {
        ((row * 17 + column * 11) % 37) as f64 / 10.0
    });
    let labels = Array1::from_shape_fn(rows, |row| {
        i64::from(features[[row, 0]] + features[[row, 1]] > 3.5)
    });
    (features, labels)
}

fn regression_fixture() -> (Array2<f64>, Array1<f64>) {
    let rows = 120usize;
    let features = Array2::from_shape_fn((rows, 4), |(row, column)| {
        ((row * 13 + column * 19) % 41) as f64 / 10.0
    });
    let responses = Array1::from_shape_fn(rows, |row| {
        2.0 * features[[row, 0]] - features[[row, 2]] + 0.5
    });
    (features, responses)
}

/// Repeated fits with the same seed produce identical tree structures and leaf
/// values for both the classifier and the regressor.
#[test]
fn repeated_fits_produce_identical_trees() {
    let (features, labels) = classification_fixture();
    let fit_once = || {
        SKRandomForestClassifier::<f64>::build()
            .number_of_trees(16)
            .maximum_depth(4)
            .random_seed(2026)
            .build()
            .expect("builds")
            .fit(&features, labels.view())
            .expect("fit succeeds")
    };
    let first = fit_once();
    let second = fit_once();
    assert_eq!(first.number_of_trees(), 16);
    assert_eq!(first.feature_count(), 4);
    for index in 0..first.number_of_trees() {
        assert_eq!(first.tree(index), second.tree(index));
    }

    let (reg_features, responses) = regression_fixture();
    let fit_regressor = || {
        SKRandomForestRegressor::<f64>::build()
            .number_of_trees(16)
            .maximum_depth(4)
            .random_seed(2026)
            .build()
            .expect("builds")
            .fit(&reg_features, responses.view())
            .expect("fit succeeds")
    };
    let reg_first = fit_regressor();
    let reg_second = fit_regressor();
    for index in 0..reg_first.number_of_trees() {
        assert_eq!(reg_first.tree(index), reg_second.tree(index));
    }
}

/// Fitting sequentially (`InProcessSynchronous`) and through the parallel path
/// (`Automatic`) produces the exact same forest.
#[test]
fn parallel_fit_equals_sequential_fit() {
    let (features, labels) = classification_fixture();
    let sequential = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(20)
        .maximum_depth(5)
        .random_seed(77)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("sequential fit");
    let parallel = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(20)
        .maximum_depth(5)
        .random_seed(77)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("parallel fit");
    for index in 0..sequential.number_of_trees() {
        assert_eq!(sequential.tree(index), parallel.tree(index));
    }

    let (reg_features, responses) = regression_fixture();
    let reg_seq = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(20)
        .maximum_depth(5)
        .random_seed(77)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("builds")
        .fit(&reg_features, responses.view())
        .expect("sequential regressor fit");
    let reg_par = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(20)
        .maximum_depth(5)
        .random_seed(77)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .expect("builds")
        .fit(&reg_features, responses.view())
        .expect("parallel regressor fit");
    for index in 0..reg_seq.number_of_trees() {
        assert_eq!(reg_seq.tree(index), reg_par.tree(index));
    }
}

/// Fitting on rayon pools of different thread counts produces identical trees.
#[test]
fn thread_count_does_not_change_fitted_forest() {
    let (features, labels) = classification_fixture();
    let estimator = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(24)
        .maximum_depth(5)
        .random_seed(314)
        .build()
        .expect("builds");

    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool")
        .install(|| estimator.fit(&features, labels.view()).expect("fit 1"));
    let four_threads = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool")
        .install(|| estimator.fit(&features, labels.view()).expect("fit 4"));

    for index in 0..one_thread.number_of_trees() {
        assert_eq!(one_thread.tree(index), four_threads.tree(index));
    }
}

/// On a nine-feature problem where feature 8 perfectly separates the classes
/// and features 0..8 carry weaker signals, `All` always splits on feature 8 at
/// the root while `Sqrt` (3 features per split) explores other root features.
#[test]
fn feature_subsampling_sqrt_and_all_control_candidate_splits() {
    assert_eq!(SKMaximumFeatures::<f64>::Sqrt.resolved_count(9), 3);
    assert_eq!(SKMaximumFeatures::<f64>::All.resolved_count(9), 9);

    let rows = 80usize;
    let features = Array2::from_shape_fn((rows, 9), |(row, column)| {
        if column == 8 {
            if row < 40 { 0.0 } else { 10.0 }
        } else {
            ((row + column * 3) % 11) as f64
        }
    });
    let labels = Array1::from_shape_fn(rows, |row| i64::from(row >= 40));

    let all_forest = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(24)
        .maximum_depth(1)
        .bootstrap(false)
        .maximum_features(SKMaximumFeatures::All)
        .random_seed(5)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let all_roots: HashSet<Option<usize>> = (0..24)
        .map(|index| {
            all_forest
                .tree(index)
                .expect("tree")
                .node(0)
                .expect("root")
                .split_feature()
        })
        .collect();
    assert_eq!(all_roots, HashSet::from([Some(8)]));

    let sqrt_forest = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(24)
        .maximum_depth(1)
        .bootstrap(false)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .random_seed(5)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let sqrt_roots: HashSet<Option<usize>> = (0..24)
        .map(|index| {
            sqrt_forest
                .tree(index)
                .expect("tree")
                .node(0)
                .expect("root")
                .split_feature()
        })
        .collect();
    assert!(
        sqrt_roots.len() > 1,
        "Sqrt over 9 features considers 3 per split, so root splits diversify"
    );
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
        .expect("fit succeeds");
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        model.execution_intent(),
        SKExecutionMode::InProcessSynchronous
    );
}
