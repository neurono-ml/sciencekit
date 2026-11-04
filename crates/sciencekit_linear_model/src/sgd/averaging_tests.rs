//! Companion tests for the running averaged iterate (spec
//! `stochastic-gradient-solver`, decision 5).

use ndarray::array;

use super::averaging::SKAveragedIterate;

#[test]
fn the_running_mean_equals_the_explicit_mean() {
    let mut average = SKAveragedIterate::new(array![0.0_f64, 0.0], 0.0);
    average.record(array![1.0, 2.0].view(), 1.0);
    average.record(array![3.0, 4.0].view(), 3.0);
    average.record(array![5.0, 6.0].view(), 5.0);
    assert_eq!(average.count(), 3);
    assert!((average.weights()[0] - 3.0).abs() < 1e-12);
    assert!((average.weights()[1] - 4.0).abs() < 1e-12);
    assert!((average.intercept() - 3.0).abs() < 1e-12);
}

#[test]
fn an_incremental_continuation_keeps_the_earlier_updates() {
    let mut first = SKAveragedIterate::new(array![0.0_f64], 0.0);
    first.record(array![1.0].view(), 0.0);
    first.record(array![3.0].view(), 0.0);
    let (weights, intercept, count) =
        (first.weights().to_owned(), first.intercept(), first.count());

    let mut continued = SKAveragedIterate::from_parts(weights, intercept, count);
    continued.record(array![5.0].view(), 0.0);
    assert_eq!(continued.count(), count + 1);
    assert!((continued.weights()[0] - 3.0).abs() < 1e-12);
}

#[test]
fn the_average_of_a_constant_trajectory_is_that_constant() {
    let mut average = SKAveragedIterate::new(array![1.0_f64, -1.0], 2.0);
    for _ in 0..10 {
        average.record(array![1.0, -1.0].view(), 2.0);
    }
    assert!((average.weights()[0] - 1.0).abs() < 1e-12);
    assert!((average.weights()[1] + 1.0).abs() < 1e-12);
    assert!((average.intercept() - 2.0).abs() < 1e-12);
}

#[test]
fn reset_restarts_the_average_from_the_chosen_iterate() {
    let mut average = SKAveragedIterate::new(array![0.0_f64], 0.0);
    average.record(array![4.0].view(), 1.0);
    average.record(array![6.0].view(), 2.0);
    average.reset(array![2.0].view(), 3.0);
    assert_eq!(average.count(), 0);
    assert!((average.weights()[0] - 2.0).abs() < 1e-12);
    assert!((average.intercept() - 3.0).abs() < 1e-12);
    average.record(array![2.0].view(), 3.0);
    assert_eq!(average.count(), 1);
    assert!((average.weights()[0] - 2.0).abs() < 1e-12);
}
