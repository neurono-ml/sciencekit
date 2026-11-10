//! Stratified train/test split tests (spec `model-selection-splits`,
//! requirement `Stratified split`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKTargetView};

use super::{SKTrainTestSplitter, sk_train_test_split};
use crate::SKTargetPartition;

fn count_test_labels(
    split: &crate::SKTrainTestSplit<f64>,
    _targets: &Array1<i64>,
) -> std::collections::BTreeMap<i64, usize> {
    let Some(SKTargetPartition::Integer(test_targets)) = split.test_targets() else {
        panic!("integer targets must stay integer");
    };
    let mut counts = std::collections::BTreeMap::new();
    for &value in test_targets {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
}

#[test]
fn stratified_split_preserves_binary_proportions() {
    let features = Array2::<f64>::zeros((100, 2));
    let mut labels = vec![0_i64; 50];
    labels.extend(std::iter::repeat_n(1, 50));
    let targets = Array1::from_vec(labels);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.2)
        .stratify(true)
        .random_seed(2)
        .build()
        .unwrap();

    let split = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
    )
    .unwrap();

    let counts = count_test_labels(&split, &targets);
    assert_eq!(counts.get(&0), Some(&10));
    assert_eq!(counts.get(&1), Some(&10));
}

#[test]
fn stratified_split_preserves_multiclass_proportions() {
    let features = Array2::<f64>::zeros((100, 2));
    let mut labels = vec![0_i64; 30];
    labels.extend(std::iter::repeat_n(1, 20));
    labels.extend(std::iter::repeat_n(2, 50));
    let targets = Array1::from_vec(labels);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.2)
        .stratify(true)
        .random_seed(8)
        .build()
        .unwrap();

    let split = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
    )
    .unwrap();

    let counts = count_test_labels(&split, &targets);
    assert_eq!(counts.get(&0), Some(&6));
    assert_eq!(counts.get(&1), Some(&4));
    assert_eq!(counts.get(&2), Some(&10));
}

#[test]
fn stratified_split_is_deterministic_for_a_seed() {
    let features = Array2::<f64>::zeros((40, 2));
    let mut labels = vec![0_i64; 20];
    labels.extend(std::iter::repeat_n(1, 20));
    let targets = Array1::from_vec(labels);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.25)
        .stratify(true)
        .random_seed(6)
        .build()
        .unwrap();

    let first = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
    )
    .unwrap();
    let second = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
    )
    .unwrap();

    assert_eq!(first.test_indices(), second.test_indices());
    assert_eq!(first.train_indices(), second.train_indices());
}

#[test]
fn stratified_split_rejects_continuous_targets() {
    let features = Array2::<f64>::zeros((20, 2));
    let targets = Array1::from_vec((0..20).map(f64::from).collect());
    let splitter = SKTrainTestSplitter::build()
        .stratify(true)
        .build()
        .unwrap();

    let error = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Continuous(targets.view())),
    )
    .expect_err("continuous targets cannot be stratified");

    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
}

#[test]
fn stratified_split_rejects_missing_targets() {
    let features = Array2::<f64>::zeros((20, 2));
    let splitter = SKTrainTestSplitter::build()
        .stratify(true)
        .build()
        .unwrap();

    let error = sk_train_test_split(&splitter, SKDataView::Dense(features.view()), None)
        .expect_err("stratification needs targets");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter { name: "targets", .. }
    ));
}

#[test]
fn stratified_split_supports_nominal_targets() {
    let features = Array2::<f64>::zeros((20, 2));
    let labels = [["red"; 10].as_slice(), ["blue"; 10].as_slice()].concat();
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.3)
        .stratify(true)
        .random_seed(1)
        .build()
        .unwrap();

    let split = sk_train_test_split(
        &splitter,
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Nominal(&labels)),
    )
    .unwrap();

    let Some(SKTargetPartition::Nominal(test_labels)) = split.test_targets() else {
        panic!("nominal targets must stay nominal");
    };
    let red = test_labels.iter().filter(|label| label.as_str() == "red").count();
    let blue = test_labels.iter().filter(|label| label.as_str() == "blue").count();
    assert_eq!(red, 3);
    assert_eq!(blue, 3);
}
