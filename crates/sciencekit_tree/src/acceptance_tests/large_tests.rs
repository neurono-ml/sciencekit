//! Large-data acceptance: at least 20 000 rows, every criterion, dense and
//! sparse paths, constraints, feature subsampling and metrics.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_metrics::regression::{sk_mean_absolute_error, sk_mean_squared_error};
use sprs::CsMat;

use crate::classifier::SKDecisionTreeClassifier;
use crate::criterion::SKTreeCriterion;
use crate::regressor::SKDecisionTreeRegressor;
use crate::splitter::SKMaximumFeatures;

const ROWS: usize = 20_000;
const COLUMNS: usize = 5;

/// A deterministic continuous design with 5% label noise.
fn classification_data() -> (Array2<f64>, Array1<i64>) {
    let features = Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
        ((row * 17 + column * 31) % 101) as f64 / 50.0 - 1.0
    });
    let labels = Array1::from_shape_fn(ROWS, |row| {
        let score =
            features[[row, 0]] + features[[row, 1]] - features[[row, 2]] + 0.5 * features[[row, 3]];
        let signal = if score > 0.0 { 1 } else { 0 };
        if row % 20 == 0 { 1 - signal } else { signal }
    });
    (features, labels)
}

/// Canonical truth, translating raw labels into the fitted class indices.
fn canonical(
    labels: &Array1<i64>,
    model: &crate::classifier::SKDecisionTreeClassifierModel<f64>,
) -> Vec<i64> {
    labels
        .iter()
        .map(|label| {
            model
                .classes()
                .index_of(&label.to_string())
                .expect("fit saw this label") as i64
        })
        .collect()
}

/// A deterministic response surface with a small periodic disturbance.
fn regression_data() -> (Array2<f64>, Array1<f64>) {
    let features = Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
        ((row * 13 + column * 29) % 97) as f64 / 48.5 - 1.0
    });
    let responses = Array1::from_shape_fn(ROWS, |row| {
        3.0 * features[[row, 0]] - 2.0 * features[[row, 1]]
            + 0.5 * features[[row, 2]]
            + (row % 5) as f64 * 0.01
    });
    (features, responses)
}

/// Gini and entropy both fit 20 000 rows and produce accurate labels.
#[test]
fn large_classification_fits_every_criterion() {
    let (features, labels) = classification_data();
    for criterion in [SKTreeCriterion::Gini, SKTreeCriterion::Entropy] {
        let model = SKDecisionTreeClassifier::<f64>::build()
            .criterion(criterion)
            .maximum_depth(6)
            .build()
            .expect("builds")
            .fit(&features, labels.view())
            .expect("fit succeeds");
        let predicted = model.predict_labels(features.view()).expect("predict");
        let truth = canonical(&labels, &model);
        let accuracy: f64 =
            sk_accuracy_score(&truth, predicted.as_slice().expect("contiguous")).expect("accuracy");
        assert!(accuracy > 0.85, "{criterion:?} accuracy {accuracy}");
    }
}

/// Squared error and absolute error both fit 20 000 rows and produce small
/// metrics.
#[test]
fn large_regression_fits_every_criterion() {
    let (features, responses) = regression_data();
    for criterion in [
        SKTreeCriterion::SquaredError,
        SKTreeCriterion::AbsoluteError,
    ] {
        let model = SKDecisionTreeRegressor::<f64>::build()
            .criterion(criterion)
            .maximum_depth(6)
            .build()
            .expect("builds")
            .fit(&features, responses.view())
            .expect("fit succeeds");
        let predicted = model.predict(features.view()).expect("predict");
        let mean_squared: f64 =
            sk_mean_squared_error(responses.view(), predicted.view()).expect("mse");
        let mean_absolute: f64 =
            sk_mean_absolute_error(responses.view(), predicted.view()).expect("mae");
        assert!(mean_squared < 0.2, "{criterion:?} mse {mean_squared}");
        assert!(mean_absolute < 0.35, "{criterion:?} mae {mean_absolute}");
    }
}

/// A mostly-zero sparse design fits at the same scale.
#[test]
fn large_sparse_classification_fits() {
    let (dense, labels) = classification_data();
    let mut indptr = vec![0usize];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let value = dense[[row, column]].max(0.0);
            if value > 0.2 {
                indices.push(column);
                data.push(value);
            }
        }
        indptr.push(indices.len());
    }
    let sparse = CsMat::new((ROWS, COLUMNS), indptr, indices, data);
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(6)
        .build()
        .expect("builds")
        .fit(sparse.view(), labels.view())
        .expect("sparse fit succeeds");
    let predicted = model.predict_labels(sparse.view()).expect("predict");
    assert_eq!(predicted.len(), ROWS);
}

/// Depth, leaf and feature-subsampling constraints hold on large data.
#[test]
fn large_classification_honors_constraints() {
    let (features, labels) = classification_data();
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(3)
        .minimum_samples_leaf(100)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .random_seed(9)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert!(model.depth() <= 3);
    for node in model.tree().nodes().iter().filter(|node| node.is_leaf()) {
        assert!(node.sample_count() >= 100);
    }
    assert!(model.node_count() > 1);
}
