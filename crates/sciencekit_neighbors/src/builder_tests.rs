//! Companion tests for both spatial-index builders.

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKDataView, SKExecutionMode};

use crate::ball_tree::SKBallTreeBuilder;
use crate::kd_tree::SKKDTreeBuilder;
use crate::metric::{SKNeighborMetric, SKRequestedNeighborMetric};

fn data() -> ndarray::Array2<f64> {
    array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 4.0]]
}

#[test]
fn builder_kd_defaults_and_overrides_are_exposed_on_the_index() {
    let default = SKKDTreeBuilder::<f64>::default()
        .build(SKDataView::Dense(data().view()))
        .unwrap();
    assert_eq!(default.leaf_size(), 40);
    assert_eq!(default.metric(), SKNeighborMetric::Euclidean);
    assert_eq!(default.backend(), SKBackendKind::Faer);
    assert_eq!(default.execution_mode(), SKExecutionMode::Automatic);

    let custom = SKKDTreeBuilder::<f64>::new()
        .leaf_size(2)
        .metric(SKRequestedNeighborMetric::SquaredEuclidean)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build(SKDataView::Dense(data().view()))
        .unwrap();
    assert_eq!(custom.leaf_size(), 2);
    assert_eq!(custom.metric(), SKNeighborMetric::SquaredEuclidean);
    assert_eq!(custom.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(custom.execution_mode(), SKExecutionMode::InProcessSynchronous);
}

#[test]
fn builder_ball_defaults_and_overrides_are_exposed_on_the_index() {
    let default = SKBallTreeBuilder::<f64>::default()
        .build(SKDataView::Dense(data().view()))
        .unwrap();
    assert_eq!(default.leaf_size(), 40);
    assert_eq!(default.metric(), SKNeighborMetric::Euclidean);
    assert_eq!(default.backend(), SKBackendKind::Faer);
    assert_eq!(default.execution_mode(), SKExecutionMode::Automatic);

    let custom = SKBallTreeBuilder::<f64>::new()
        .leaf_size(3)
        .metric(SKNeighborMetric::Chebyshev)
        .backend(SKBackendKind::NdArrayLinalg)
        .execution_mode(SKExecutionMode::InProcessAsynchronous)
        .build(SKDataView::Dense(data().view()))
        .unwrap();
    assert_eq!(custom.leaf_size(), 3);
    assert_eq!(custom.metric(), SKNeighborMetric::Chebyshev);
    assert_eq!(custom.backend(), SKBackendKind::NdArrayLinalg);
    assert_eq!(custom.execution_mode(), SKExecutionMode::InProcessAsynchronous);
}

#[test]
fn builder_rejects_zero_leaf_size_on_the_ball_tree() {
    let error = SKBallTreeBuilder::<f64>::new()
        .leaf_size(0)
        .build(SKDataView::Dense(data().view()))
        .unwrap_err();

    match error {
        sciencekit_common::SKError::InvalidHyperparameter { name, .. } => {
            assert_eq!(name, "leaf_size");
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}
