//! Acceptance over every algorithm/metric/weights combination.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKError, SKSupervisedFit};
use sprs::TriMat;

use crate::SKKNeighborsClassifier;
use crate::neighbors::{SKNeighborsAlgorithm, SKNeighborsMetric, SKNeighborsWeights};

const ROWS: usize = 90;

/// A design with strictly increasing, well-separated rows and queries that do
/// not sit exactly midway between two rows: the tree and the brute-force
/// kernels must select exactly the same neighbours, so their votes cannot
/// disagree on a floating-point tie.
fn design() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, 3), |(row, column)| {
        row as f64 * 0.1 + column as f64 * 0.037
    })
}

fn queries() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, 3), |(row, column)| {
        row as f64 * 0.1 + 0.031 + column as f64 * 0.037
    })
}

fn labels() -> Array1<i64> {
    Array1::from_shape_fn(ROWS, |row| (row % 2) as i64)
}

fn dense_model(
    algorithm: SKNeighborsAlgorithm,
    metric: SKNeighborsMetric,
    weights: SKNeighborsWeights,
) -> SKKNeighborsClassifier<f64> {
    SKKNeighborsClassifier::<f64>::build()
        .algorithm(algorithm)
        .metric(metric)
        .weights(weights)
        .build()
        .unwrap()
}

#[test]
fn acceptance_knn_every_dense_combination_is_consistent_with_brute_force() {
    let features = design();
    let labels = labels();
    let queries = queries();
    let weights_options = [SKNeighborsWeights::Uniform, SKNeighborsWeights::Distance];

    for metric in SKNeighborsMetric::ALL {
        let baseline = dense_model(
            SKNeighborsAlgorithm::BruteForce,
            metric,
            SKNeighborsWeights::Uniform,
        )
        .fit(features.view(), labels.view())
        .unwrap()
        .predict_labels(queries.view())
        .unwrap();

        for weights in weights_options {
            for algorithm in [
                SKNeighborsAlgorithm::Auto,
                SKNeighborsAlgorithm::KdTree,
                SKNeighborsAlgorithm::BallTree,
                SKNeighborsAlgorithm::BruteForce,
            ] {
                let estimator = dense_model(algorithm, metric, weights);
                let outcome = estimator.fit(features.view(), labels.view());
                if algorithm != SKNeighborsAlgorithm::BruteForce
                    && algorithm != SKNeighborsAlgorithm::Auto
                    && metric == SKNeighborsMetric::Cosine
                {
                    match outcome {
                        Err(SKError::UnsupportedRepresentation { representation, .. }) => {
                            assert_eq!(representation, "cosine distance")
                        }
                        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
                    }
                    continue;
                }
                let model = outcome.expect("the combination is supported");
                let predictions = model.predict_labels(queries.view()).unwrap();
                if weights == SKNeighborsWeights::Uniform {
                    assert_eq!(
                        predictions, baseline,
                        "metric={metric:?} algorithm={algorithm:?} weights={weights:?}"
                    );
                }
                assert_eq!(predictions.len(), ROWS);
            }
        }
    }
}

#[test]
fn acceptance_knn_every_sparse_combination_is_supported() {
    let dense = design();
    let mut triplet = TriMat::new((ROWS, 3));
    for ((row, column), value) in dense.indexed_iter() {
        if *value != 0.0 {
            triplet.add_triplet(row, column, *value);
        }
    }
    let sparse = triplet.to_csr();
    let labels = labels();

    for metric in SKNeighborsMetric::ALL {
        for weights in [SKNeighborsWeights::Uniform, SKNeighborsWeights::Distance] {
            let model = dense_model(SKNeighborsAlgorithm::BruteForce, metric, weights)
                .fit(sparse.view(), labels.view())
                .unwrap();
            let predictions = model.predict_labels(sparse.view()).unwrap();
            assert_eq!(predictions.len(), ROWS);
            assert_eq!(model.training_representation(), "csr");
        }
    }
}
