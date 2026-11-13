//! Companion tests for multiclass `SKGradientBoostingClassifier`: `C` trees per
//! stage, softmax probabilities, canonical labels and tie-breaking (task 4.3).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::SKGradientBoostingClassifier;

/// Scenarios: Multiclass fits `C` trees per stage, softmax probabilities sum to
/// one, and canonical class order is preserved.
#[test]
fn multiclass_fits_one_tree_per_class_per_stage_and_softmax_sums_to_one() {
    let features = array![
        [0.0, 0.1],
        [0.1, 0.0],
        [0.2, 0.1],
        [5.0, 5.1],
        [5.1, 5.0],
        [5.2, 5.1],
        [10.0, 0.1],
        [10.1, 0.0],
        [10.2, 0.1],
    ];
    let labels = [
        "red", "red", "red", "green", "green", "green", "blue", "blue", "blue",
    ];
    let model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(15)
        .learning_rate(0.25)
        .maximum_depth(2)
        .random_seed(11)
        .build()
        .expect("builds")
        .fit(&features, &labels[..])
        .expect("multiclass fit succeeds");

    assert_eq!(model.number_of_classes(), 3);
    assert_eq!(model.number_of_stages(), 15);
    assert_eq!(model.trees_per_stage(), 3);
    assert_eq!(model.initial_prediction(), None);
    assert_eq!(model.initial_predictions().len(), 3);
    (0..15).for_each(|stage| {
        assert_eq!(model.stage_trees(stage).expect("stage exists").len(), 3);
    });
    assert_eq!(model.classes().label_of(0), Some("red"));
    assert_eq!(model.classes().label_of(1), Some("green"));
    assert_eq!(model.classes().label_of(2), Some("blue"));

    let probabilities = model
        .predict_probabilities(&features)
        .expect("probabilities");
    assert_eq!(probabilities.shape(), &[9, 3]);
    probabilities.rows().into_iter().for_each(|row| {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(row.iter().all(|&p| p > 0.0 && p < 1.0));
    });
    assert_eq!(
        model.predict_labels(&features).expect("labels"),
        array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2]
    );
}

/// Uniform three-class depth-0 fit produces equal softmax probabilities `1/3`
/// and breaks ties toward canonical class `0`.
#[test]
fn multiclass_equal_scores_break_ties_toward_smallest_canonical_index() {
    let features = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let labels = ["gamma", "beta", "alpha", "gamma", "beta", "alpha"];
    let model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(2)
        .maximum_depth(0)
        .learning_rate(0.2)
        .build()
        .expect("builds")
        .fit(&features, &labels[..])
        .expect("fits");

    let probabilities = model
        .predict_probabilities(&features)
        .expect("probabilities");
    probabilities.rows().into_iter().for_each(|row| {
        assert!((row[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((row[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((row[2] - 1.0 / 3.0).abs() < 1e-12);
    });
    assert_eq!(
        model.predict_labels(&features).expect("labels"),
        array![0_i64, 0, 0, 0, 0, 0]
    );
}
