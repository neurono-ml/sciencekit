//! Companion tests for out-of-bag mask computation (task 1.2).

use std::collections::HashSet;

use super::{SKMaximumSamples, sk_out_of_bag_rows, sk_sample_indices};

/// The OOB rows are the exact set complement of the drawn indices.
#[test]
fn out_of_bag_rows_are_exact_complement_of_drawn_sample() {
    let drawn = sk_sample_indices(30, &SKMaximumSamples::<f64>::All, true, 123, 0);
    let oob = sk_out_of_bag_rows(30, &drawn);
    let drawn_set: HashSet<usize> = drawn.iter().copied().collect();
    let oob_set: HashSet<usize> = oob.iter().copied().collect();

    assert!(drawn_set.is_disjoint(&oob_set));
    assert_eq!(drawn_set.len() + oob_set.len(), 30);
    assert!(oob.windows(2).all(|pair| pair[0] < pair[1]));
}

/// Repeated draws of one row still leave it in-bag, and a full draw leaves no
/// OOB rows.
#[test]
fn repeated_in_bag_rows_and_no_bootstrap_produce_expected_masks() {
    assert_eq!(sk_out_of_bag_rows(5, &[1, 1, 3, 1]), vec![0, 2, 4]);
    let all_rows = sk_sample_indices(8, &SKMaximumSamples::<f64>::All, false, 0, 0);
    assert!(sk_out_of_bag_rows(8, &all_rows).is_empty());
}
