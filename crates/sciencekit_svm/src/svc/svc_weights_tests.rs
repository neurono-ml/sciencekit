//! Companion TDD tests for `SKSVC` class weights (`Balanced`, `Explicit` and
//! `LabelMap`) (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKError, SKSupervisedFit};
use sciencekit_math::SKKernel;

use super::{SKClassWeight, SKSVC};

#[test]
fn svc_weights_balanced_shifts_decision_boundary_on_imbalanced_fixture() {
    // 8 negative samples crowding near 0 and 2 positive samples near +1:
    // with small C = 0.2, unweighted C-SVC lets the 8 negatives dominate and
    // pushes the decision value at x = 0.4 negative; Balanced upweights the
    // minority class by (10 / (2*2)) / (10 / (2*8)) = 4x, flipping x = 0.4 positive.
    let features = array![
        [-1.2_f64],
        [-1.0],
        [-0.8],
        [-0.5],
        [-0.3],
        [-0.1],
        [0.1],
        [0.3],
        [0.7],
        [1.1]
    ];
    let labels = array![0_i64, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    let probe = array![[0.4_f64]];

    let unweighted = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(0.2)
        .class_weight(SKClassWeight::None)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let balanced = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(0.2)
        .class_weight(SKClassWeight::Balanced)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let unweighted_margin = unweighted.decision_function(probe.view()).unwrap()[[0, 0]];
    let balanced_margin = balanced.decision_function(probe.view()).unwrap()[[0, 0]];
    assert!(
        balanced_margin > unweighted_margin + 0.2,
        "expected balanced ({balanced_margin}) > unweighted ({unweighted_margin})"
    );
    assert!(unweighted_margin < 0.0);
    assert!(balanced_margin > 0.0);
}

#[test]
fn svc_weights_explicit_vector_and_label_map_produce_identical_weighted_models() {
    let features = array![[-1.0_f64], [-0.4], [-0.2], [0.2], [0.8], [1.2]];
    let labels = array![10_i64, 10, 10, 10, 20, 20];
    let probe = array![[0.3_f64]];

    let explicit_vec = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(0.25)
        .class_weight(SKClassWeight::Explicit(vec![0.5, 3.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let explicit_map = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(0.25)
        .class_weight(SKClassWeight::from_labels([(10_i64, 0.5), (20_i64, 3.0)]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let d_vec = explicit_vec.decision_function(probe.view()).unwrap()[[0, 0]];
    let d_map = explicit_map.decision_function(probe.view()).unwrap()[[0, 0]];
    assert!((d_vec - d_map).abs() < 1e-12);

    let wrong_len = SKSVC::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 2.0, 3.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(matches!(wrong_len, SKError::ShapeMismatch { .. }));
}
