//! `SKIndexSplit` accessors and invariants.

use super::SKIndexSplit;

#[test]
fn index_split_exposes_the_train_and_test_indices() {
    let split = SKIndexSplit::new(vec![0, 2, 4], vec![1, 3, 5]);

    assert_eq!(split.train(), &[0, 2, 4]);
    assert_eq!(split.test(), &[1, 3, 5]);
    assert_eq!(split.number_of_train_rows(), 3);
    assert_eq!(split.number_of_test_rows(), 3);
}
