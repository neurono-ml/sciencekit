//! Deterministic KD-tree tie-ordering tests.

use ndarray::array;
use sciencekit_common::SKDataView;

use super::fixtures::build;
use crate::searcher::SKNearestNeighborsSearcher;

#[test]
fn kd_query_breaks_distance_ties_by_the_smallest_training_index() {
    let data = array![[1.0_f64, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
    let tree = build(&data, 2);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 2)
        .unwrap();

    assert_eq!(result.indices(), array![[0_usize, 1]].view());
    assert_eq!(result.distances(), array![[1.0_f64, 1.0]].view());
}

#[test]
fn kd_query_tie_order_is_independent_of_the_tree_partition() {
    let data = array![[0.0_f64, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]];
    let tree = build(&data, 1);
    let queries = array![[0.0_f64, 0.0]];

    let result = tree
        .k_nearest(SKDataView::Dense(queries.view()), 1)
        .unwrap();

    assert_eq!(result.indices()[[0, 0]], 0);
}

/// The same logical data partitioned by a different leaf size answers
/// identically, ties included.
#[test]
fn kd_partition_does_not_change_the_answer() {
    let training = ndarray::Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0,
        ],
    )
    .unwrap();
    let query = ndarray::Array2::from_shape_vec((2, 2), vec![0.5_f64, 0.5, 2.0, 2.0]).unwrap();
    let mut answers = Vec::new();
    for leaf_size in [1_usize, 2, 40] {
        let tree = crate::SKKDTreeBuilder::<f64>::new()
            .leaf_size(leaf_size)
            .build(sciencekit_common::SKDataView::Dense(training.view()))
            .unwrap();
        answers.push(
            crate::searcher::SKNearestNeighborsSearcher::k_nearest(
                &tree,
                sciencekit_common::SKDataView::Dense(query.view()),
                3,
            )
            .unwrap(),
        );
    }
    for answer in &answers[1..] {
        assert_eq!(answer.indices(), answers[0].indices());
        assert_eq!(answer.distances(), answers[0].distances());
    }
}
