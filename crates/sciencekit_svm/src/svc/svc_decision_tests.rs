//! Companion TDD tests for `SKSVCModel::decision_function`
//! (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};
use sciencekit_math::SKKernel;

use super::SKSVC;

#[test]
fn svc_decision_binary_returns_single_signed_column_monotonic_with_labels() {
    let features = array![[-3.0_f64], [-1.0], [1.0], [3.0]];
    let labels = array![0_i64, 0, 1, 1];

    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(10.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let decisions = model.decision_function(features.view()).unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();

    assert_eq!(decisions.dim(), (4, 1));
    assert!(decisions[[0, 0]] < decisions[[1, 0]]);
    assert!(decisions[[1, 0]] < 0.0);
    assert!(decisions[[2, 0]] > 0.0);
    assert!(decisions[[2, 0]] < decisions[[3, 0]]);
    for (row, &label) in predicted.iter().enumerate() {
        assert_eq!(label, i64::from(decisions[[row, 0]] > 0.0));
    }
}

#[test]
fn svc_decision_multiclass_returns_per_class_confidence_matrix_aligned_with_labels() {
    let features = array![
        [-4.0_f64, 0.0],
        [-3.5, 0.4],
        [0.0, 4.0],
        [0.4, 3.5],
        [4.0, 0.0],
        [3.5, -0.4]
    ];
    let labels = array![0_i64, 0, 1, 1, 2, 2];

    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(2.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let decisions = model.decision_function(features.view()).unwrap();
    let pairwise = model.pairwise_decision_function(features.view()).unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();

    assert_eq!(decisions.dim(), (6, 3));
    assert_eq!(pairwise.dim(), (6, 3));

    for row in 0..decisions.nrows() {
        let argmax = decisions
            .row(row)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(predicted[row], argmax as i64);
    }
}
