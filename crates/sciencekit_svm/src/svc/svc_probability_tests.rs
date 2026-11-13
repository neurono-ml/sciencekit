//! Companion TDD tests for `SKSVCModel::predict_probabilities`
//! (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKError, SKSupervisedFit};
use sciencekit_math::SKKernel;

use super::SKSVC;

#[test]
fn svc_probability_binary_and_multiclass_sum_to_one_and_are_monotonic() {
    let binary_x = array![
        [-3.0_f64, 0.0],
        [-2.2, 0.2],
        [-1.5, -0.1],
        [-1.1, 0.1],
        [-0.8, -0.2],
        [0.8, 0.2],
        [1.1, -0.1],
        [1.5, 0.1],
        [2.2, -0.2],
        [3.0, 0.0]
    ];
    let binary_y = array![0_i64, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    let binary_model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(1.0)
        .probability(true)
        .random_seed(11)
        .build()
        .unwrap()
        .fit(binary_x.view(), binary_y.view())
        .unwrap();

    let decisions = binary_model.decision_function(binary_x.view()).unwrap();
    let binary_probs = binary_model.predict_probabilities(binary_x.view()).unwrap();
    assert_eq!(binary_probs.dim(), (10, 2));

    let mut pairs: Vec<(f64, f64)> = (0..10)
        .map(|row| {
            let p0 = binary_probs[[row, 0]];
            let p1 = binary_probs[[row, 1]];
            assert!((0.0..=1.0).contains(&p0) && (0.0..=1.0).contains(&p1));
            assert!((p0 + p1 - 1.0).abs() < 1e-12);
            (decisions[[row, 0]], p1)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for window in pairs.windows(2) {
        assert!(window[1].1 >= window[0].1 - 1e-12);
    }
    assert!(pairs.last().unwrap().1 > pairs.first().unwrap().1 + 0.4);

    // Three-class coupled probabilities:
    let multi_x = array![
        [-4.0_f64, 0.0],
        [-3.6, 0.3],
        [-3.4, -0.3],
        [-3.2, 0.1],
        [0.0, 4.0],
        [0.3, 3.6],
        [-0.3, 3.4],
        [0.1, 3.2],
        [4.0, 0.0],
        [3.6, 0.3],
        [3.4, -0.3],
        [3.2, -0.1]
    ];
    let multi_y = array![0_i64, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    let multi_model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(2.0)
        .probability(true)
        .random_seed(19)
        .build()
        .unwrap()
        .fit(multi_x.view(), multi_y.view())
        .unwrap();

    let probs = multi_model.predict_probabilities(multi_x.view()).unwrap();
    assert_eq!(probs.dim(), (12, 3));
    for (row, &expected_class) in multi_y.iter().enumerate() {
        let row_view = probs.row(row);
        row_view
            .iter()
            .for_each(|&value| assert!((0.0..=1.0).contains(&value)));
        assert!((row_view.sum() - 1.0).abs() < 1e-10);
        let best_class = row_view
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best_class, expected_class as usize);
        // Coupled probability gives smooth non-zero mass to all 3 classes, not {0, 1/3, 2/3} vote fractions.
        assert!(row_view.iter().all(|&v| v > 0.0));
    }
}

#[test]
fn svc_probability_disabled_returns_unsupported_operation() {
    let features = array![[-1.0_f64, 0.0], [1.0, 0.0]];
    let labels = array![0_i64, 1];

    let model = SKSVC::<f64>::build()
        .probability(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    let err = model.predict_probabilities(features.view()).unwrap_err();
    assert!(matches!(
        err,
        SKError::UnsupportedOperation {
            operation: "predict_probabilities",
            ..
        }
    ));
}
