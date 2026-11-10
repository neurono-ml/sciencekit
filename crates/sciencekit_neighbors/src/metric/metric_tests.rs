//! Companion tests for the metric catalogue point distances.

use ndarray::array;

use super::SKNeighborMetric;

fn point(a: [f64; 3], b: [f64; 3], metric: SKNeighborMetric) -> f64 {
    metric.point_distance(array![a].row(0), array![b].row(0))
}

#[test]
fn euclidean_metric_returns_the_geometric_distance() {
    assert!(
        (point(
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            SKNeighborMetric::Euclidean
        ) - 5.0)
            .abs()
            < 1e-12
    );
}

#[test]
fn squared_euclidean_metric_returns_the_squared_distance() {
    assert_eq!(
        point(
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            SKNeighborMetric::SquaredEuclidean
        ),
        25.0
    );
}

#[test]
fn manhattan_metric_sums_absolute_coordinate_gaps() {
    assert_eq!(
        point(
            [1.0, 2.0, 3.0],
            [4.0, 0.0, 5.0],
            SKNeighborMetric::Manhattan
        ),
        7.0
    );
}

#[test]
fn chebyshev_metric_returns_the_largest_coordinate_gap() {
    assert_eq!(
        point(
            [1.0, 2.0, 3.0],
            [4.0, 0.0, 5.0],
            SKNeighborMetric::Chebyshev
        ),
        3.0
    );
}

#[test]
fn identical_rows_have_zero_distance_in_every_metric() {
    for metric in SKNeighborMetric::ALL {
        assert_eq!(point([1.5, -2.0, 3.0], [1.5, -2.0, 3.0], metric), 0.0);
    }
}

#[test]
fn every_metric_is_symmetric_and_non_negative() {
    let a = [0.5, -1.0, 2.5];
    let b = [-3.0, 4.0, 0.0];
    for metric in SKNeighborMetric::ALL {
        let forward = point(a, b, metric);
        let backward = point(b, a, metric);
        assert_eq!(forward, backward, "metric {metric:?} is not symmetric");
        assert!(forward >= 0.0, "metric {metric:?} is negative");
    }
}
