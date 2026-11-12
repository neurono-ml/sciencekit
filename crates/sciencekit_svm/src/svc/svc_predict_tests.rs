//! Companion TDD tests for `SKSVCModel::predict_labels` and OvO vote tie-breaking
//! (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sciencekit_math::SKKernel;

use super::SKSVC;
use super::predict::sk_resolve_one_vs_one_votes;

#[test]
fn svc_predict_classifies_three_class_fixture_by_majority_vote() {
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
        .kernel(SKKernel::RadialBasisFunction)
        .C(2.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted, labels);
}

#[test]
fn svc_predict_breaks_hand_built_vote_ties_by_smallest_canonical_index() {
    // 3-class Rock-Paper-Scissors cycle:
    // (0, 1): -1.0 -> class 0 wins
    // (0, 2): +1.0 -> class 2 wins
    // (1, 2): -1.0 -> class 1 wins
    // Votes = [1, 1, 1] => smallest canonical index 0 wins.
    let pairs_3 = [(0, 1), (0, 2), (1, 2)];
    assert_eq!(
        sk_resolve_one_vs_one_votes(&[-1.0_f64, 1.0, -1.0], &pairs_3, 3),
        0
    );

    // 4-class tie between classes 1 and 2 (votes [0, 2, 2, 2]):
    // Pairs: (0,1)+, (0,2)+, (0,3)+, (1,2)-, (1,3)-, (2,3)+
    // Wait: check votes:
    // (0,1)+ -> 1; (0,2)+ -> 2; (0,3)+ -> 3; (1,2)- -> 1; (1,3)+ -> 3; (2,3)- -> 2
    // Votes: class 0 = 0, class 1 = 2, class 2 = 2, class 3 = 2 => winner must be 1!
    let pairs_4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let decisions_4 = [1.0_f64, 1.0, 1.0, -1.0, 1.0, -1.0];
    assert_eq!(sk_resolve_one_vs_one_votes(&decisions_4, &pairs_4, 4), 1);

    // Also verify end-to-end tie-breaking through SKSVCModel::predict_labels
    // on exact binary symmetry around the origin (decision == 0.0 -> class 0).
    let binary_tie_features = array![[-1.0_f64, 0.0], [1.0, 0.0]];
    let binary_tie_labels = array![10_i64, 20];
    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(1.0)
        .build()
        .unwrap()
        .fit(binary_tie_features.view(), binary_tie_labels.view())
        .unwrap();
    let origin_prediction = model.predict_labels(array![[0.0_f64, 0.0]].view()).unwrap();
    assert_eq!(origin_prediction[0], 0);
}
