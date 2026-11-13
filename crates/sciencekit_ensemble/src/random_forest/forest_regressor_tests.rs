//! Companion tests for regressor response averaging (task 2.4).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExecutionMode, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_tree::SKTreeCriterion;

use super::{SKMaximumSamples, SKRandomForestRegressor};

/// When three single-sample zero-depth trees predict `1.0`, `2.0` and `6.0`,
/// the forest predicts their arithmetic mean `3.0`.
#[test]
fn three_tree_forest_predicts_exact_arithmetic_mean() {
    let features = array![[0.0], [1.0], [2.0]];
    let responses = array![1.0, 2.0, 6.0];

    let model = (0_u64..100)
        .find_map(|seed| {
            let candidate = SKRandomForestRegressor::<f64>::build()
                .number_of_trees(3)
                .maximum_depth(0)
                .maximum_samples(SKMaximumSamples::Count(1))
                .random_seed(seed)
                .build()
                .expect("builds")
                .fit(&features, responses.view())
                .expect("fit succeeds");
            let mut leaves: Vec<f64> = (0..3)
                .map(|index| {
                    candidate
                        .tree(index)
                        .expect("tree")
                        .node(0)
                        .expect("root")
                        .value()[0]
                })
                .collect();
            leaves.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
            (leaves == vec![1.0, 2.0, 6.0]).then_some(candidate)
        })
        .expect("a seed in 0..100 draws 1.0, 2.0 and 6.0 across the three trees");

    let predictions = model
        .predict(array![[0.5], [1.5]].view())
        .expect("predict succeeds");
    assert!((predictions[0] - 3.0).abs() < 1e-12);
    assert!((predictions[1] - 3.0).abs() < 1e-12);
}

/// Squared-error and absolute-error regression forests fit piecewise functions
/// accurately and agree between sequential and parallel prediction.
#[test]
fn regression_forest_recovers_step_signal_under_both_criteria() {
    let features = Array2::from_shape_fn((80, 4), |(row, column)| match column {
        0 => row as f64 / 10.0,
        1 => row as f64 / 5.0,
        2 => ((row * 7) % 13) as f64,
        _ => ((row * 11) % 17) as f64,
    });
    let responses = Array1::from_shape_fn(80, |row| if row < 40 { 10.0 } else { 50.0 });

    for criterion in [
        SKTreeCriterion::SquaredError,
        SKTreeCriterion::AbsoluteError,
    ] {
        let parallel = SKRandomForestRegressor::<f64>::build()
            .number_of_trees(25)
            .criterion(criterion)
            .backend(SKBackendKind::MatrixMultiply)
            .random_seed(17)
            .build()
            .expect("builds")
            .fit(&features, responses.view())
            .expect("fit succeeds");
        let sequential = SKRandomForestRegressor::<f64>::build()
            .number_of_trees(25)
            .criterion(criterion)
            .execution_mode(SKExecutionMode::InProcessSynchronous)
            .random_seed(17)
            .build()
            .expect("builds")
            .fit(&features, responses.view())
            .expect("fit succeeds");

        assert_eq!(parallel.backend(), SKBackendKind::MatrixMultiply);
        let probe = array![[0.5, 1.0, 2.0, 3.0], [6.0, 12.0, 2.0, 3.0]];
        let parallel_predictions = parallel.predict(&probe).expect("parallel predict");
        let sequential_predictions = sequential.predict(&probe).expect("sequential predict");
        assert_eq!(parallel_predictions, sequential_predictions);
        assert!((parallel_predictions[0] - 10.0).abs() < 1.0);
        assert!((parallel_predictions[1] - 50.0).abs() < 1.0);
    }
}
