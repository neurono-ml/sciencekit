//! Companion TDD tests for one-vs-one multiclass `SKSVC::fit`
//! (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKSupervisedFit};
use sciencekit_math::SKKernel;

use super::SKSVC;

#[test]
fn svc_multiclass_fits_deterministic_one_vs_one_pairs_using_only_each_pairs_rows() {
    let features = array![
        [-4.0_f64, 0.0],
        [-3.5, 0.5],
        [-3.5, -0.5],
        [0.0, 4.0],
        [0.5, 3.5],
        [-0.5, 3.5],
        [4.0, 0.0],
        [3.5, 0.5],
        [3.5, -0.5]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];

    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(1.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.number_of_classes(), 3);
    assert_eq!(model.pairwise_models().len(), 3);

    let expected_pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    for (pair_model, &(expected_neg, expected_pos)) in
        model.pairwise_models().iter().zip(expected_pairs.iter())
    {
        assert_eq!(pair_model.class_negative(), expected_neg);
        assert_eq!(pair_model.class_positive(), expected_pos);
        assert!(!pair_model.support_training_indices().is_empty());
        for &row_idx in pair_model.support_training_indices() {
            let row_class = labels[row_idx] as usize;
            assert!(
                row_class == expected_neg || row_class == expected_pos,
                "pair ({expected_neg}, {expected_pos}) used row {row_idx} of class {row_class}"
            );
        }
    }
}
