//! Companion tests for the metric geometry lower bounds.

use ndarray::array;

use super::SKNeighborMetric;

fn query() -> ndarray::Array1<f64> {
    array![1.0, 1.0]
}

#[test]
fn kd_axis_aligned_bound_is_zero_inside_the_box() {
    for metric in SKNeighborMetric::ALL {
        let minimum = [0.0, 0.0];
        let maximum = [2.0, 2.0];
        assert_eq!(
            metric.axis_aligned_lower_bound(query().view(), &minimum, &maximum),
            0.0,
            "metric {metric:?}"
        );
    }
}

#[test]
fn kd_axis_aligned_bound_matches_each_metric_outside_the_box() {
    let minimum = [3.0, 0.0];
    let maximum = [4.0, 2.0];

    assert_eq!(
        SKNeighborMetric::Euclidean.axis_aligned_lower_bound(query().view(), &minimum, &maximum),
        2.0
    );
    assert_eq!(
        SKNeighborMetric::SquaredEuclidean.axis_aligned_lower_bound(
            query().view(),
            &minimum,
            &maximum
        ),
        4.0
    );
    assert_eq!(
        SKNeighborMetric::Manhattan.axis_aligned_lower_bound(query().view(), &minimum, &maximum),
        2.0
    );
    assert_eq!(
        SKNeighborMetric::Chebyshev.axis_aligned_lower_bound(query().view(), &minimum, &maximum),
        2.0
    );
}

#[test]
fn kd_axis_aligned_bound_handles_mixed_inside_and_outside_axes() {
    let minimum = [0.0, 4.0];
    let maximum = [0.5, 5.0];

    let expected_squared: f64 = 0.25 + 9.0;
    assert!(
        (SKNeighborMetric::Euclidean.axis_aligned_lower_bound(query().view(), &minimum, &maximum)
            - expected_squared.sqrt())
        .abs()
            < 1e-12
    );
    assert_eq!(
        SKNeighborMetric::SquaredEuclidean.axis_aligned_lower_bound(
            query().view(),
            &minimum,
            &maximum
        ),
        expected_squared
    );
}

#[test]
fn ball_bound_is_zero_when_the_query_lies_inside_the_ball() {
    let centroid = [1.0, 1.0];
    for metric in SKNeighborMetric::ALL {
        assert_eq!(metric.ball_lower_bound(query().view(), &centroid, 2.0), 0.0);
    }
}

#[test]
fn ball_bound_equals_the_distance_gap_outside_the_ball() {
    let centroid = [4.0, 5.0];
    let radius = 1.0;

    // The bound is conservative: it never exceeds, and may undercut by rounding.
    let assert_lower_bound = |actual: f64, expected: f64| {
        assert!(
            actual <= expected + 1e-12,
            "the bound {actual} must not exceed the true gap {expected}"
        );
        assert!(
            (actual - expected).abs() < 1e-9,
            "the bound {actual} must stay close to the true gap {expected}"
        );
    };

    assert_lower_bound(
        SKNeighborMetric::Euclidean.ball_lower_bound(query().view(), &centroid, radius),
        4.0,
    );
    assert_lower_bound(
        SKNeighborMetric::SquaredEuclidean.ball_lower_bound(query().view(), &centroid, radius),
        16.0,
    );
    assert_lower_bound(
        SKNeighborMetric::Manhattan.ball_lower_bound(query().view(), &centroid, radius),
        4.0,
    );
    assert_lower_bound(
        SKNeighborMetric::Chebyshev.ball_lower_bound(query().view(), &centroid, radius),
        4.0 / 2.0_f64.sqrt(),
    );
}
