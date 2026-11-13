//! Multiclass histogram loop tests: refreshed gradients and per-stage row
//! subsampling.

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKSupervisedFit};

use super::SKHistGradientBoostingClassifier;

fn three_class_data() -> (Array2<f64>, ndarray::Array1<i64>) {
    let rows = 150usize;
    let features = Array2::from_shape_fn((rows, 2), |(row, column)| {
        (row % 3) as f64 + column as f64 * 0.1
    });
    let labels = ndarray::Array1::from_shape_fn(rows, |row| (row % 3) as i64);
    (features, labels)
}

#[test]
fn hist_multiclass_refreshes_gradients_between_stages() {
    let (features, labels) = three_class_data();
    let model = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(3)
        .learning_rate(0.3)
        .maximum_leaf_nodes(4)
        .minimum_samples_leaf(1)
        .maximum_bins(8)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    assert_eq!(model.number_of_stages(), 3);
    (0..model.number_of_classes()).for_each(|class_index| {
        assert_ne!(
            model.stage_tree(0, class_index).expect("first stage tree"),
            model.stage_tree(1, class_index).expect("second stage tree"),
            "class {class_index} refitted the first stage's gradients"
        );
    });
}

#[test]
fn hist_multiclass_applies_row_subsampling() {
    let (features, labels) = three_class_data();
    let model = SKHistGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(2)
        .subsample(0.5)
        .maximum_leaf_nodes(4)
        .minimum_samples_leaf(1)
        .maximum_bins(8)
        .without_early_stopping()
        .build()
        .expect("build")
        .fit(features.view(), labels.view())
        .expect("fit");
    let root = &model.stage_tree(0, 0).expect("tree").nodes()[0];
    assert!(
        root.sample_count() < features.nrows(),
        "subsample was ignored: root holds {} rows",
        root.sample_count()
    );
}
