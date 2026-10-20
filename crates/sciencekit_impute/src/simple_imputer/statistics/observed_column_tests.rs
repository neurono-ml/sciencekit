//! Companion tests for the observed-set statistics (spec `simple-imputer`).

use super::observed_column::SKObservedColumn;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// The statistic of a set of explicit values and a zero multiplicity.
fn statistic(values: &[f64], implicit_zeros: usize, strategy: SKImputationStrategy) -> f64 {
    SKObservedColumn::new(values.to_vec(), implicit_zeros).statistic(strategy, None)
}

#[test]
fn implicit_zeros_count_in_the_mean_denominator() {
    assert_eq!(
        statistic(&[6.0], 2, SKImputationStrategy::Mean),
        2.0,
        "(6 + 0 + 0) / 3"
    );
}

#[test]
fn median_places_the_implicit_zeros_in_their_rank_block() {
    assert_eq!(
        statistic(&[10.0], 3, SKImputationStrategy::Median),
        0.0,
        "the sorted multiset 0, 0, 0, 10 has median 0"
    );
    assert_eq!(
        statistic(&[-4.0, 6.0], 2, SKImputationStrategy::Median),
        0.0,
        "the sorted multiset -4, 0, 0, 6 averages its central pair"
    );
    assert_eq!(
        statistic(&[1.0, 3.0, 5.0, 9.0], 0, SKImputationStrategy::Median),
        4.0,
        "an even explicit count averages the middle pair"
    );
}

#[test]
fn mode_merges_the_zero_run_and_breaks_ties_toward_the_smallest() {
    assert_eq!(
        statistic(&[5.0], 2, SKImputationStrategy::MostFrequent),
        0.0,
        "two zeros beat one five"
    );
    assert_eq!(
        statistic(&[2.0, 2.0, 5.0, 5.0], 0, SKImputationStrategy::MostFrequent),
        2.0,
        "a tie keeps the smallest value"
    );
    assert_eq!(
        statistic(&[2.0, 2.0, 5.0], 2, SKImputationStrategy::MostFrequent),
        0.0,
        "the zeros form a three-strong run"
    );
}

#[test]
fn an_empty_observed_set_yields_nan() {
    for strategy in [
        SKImputationStrategy::Mean,
        SKImputationStrategy::Median,
        SKImputationStrategy::MostFrequent,
    ] {
        assert!(statistic(&[], 0, strategy).is_nan());
    }
}

#[test]
fn constant_ignores_the_observed_set() {
    let column = SKObservedColumn::new(Vec::<f64>::new(), 0);
    assert_eq!(
        column.statistic(SKImputationStrategy::Constant, Some(-1.5)),
        -1.5
    );
    assert_eq!(column.statistic(SKImputationStrategy::Constant, None), 0.0);
}
