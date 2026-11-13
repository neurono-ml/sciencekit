//! Companion tests for deterministic fitting across repeated runs and thread
//! counts (spec `support-vector-machines`).

use ndarray::{Array1, Array2, array};
use rayon::ThreadPoolBuilder;
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_math::SKKernel;

use crate::{SKSVC, SKSVR};

#[test]
fn determinism_repeated_fits_and_thread_counts_produce_bit_identical_models() {
    let features = array![
        [-3.2_f64, 0.5],
        [-2.8, -0.3],
        [-2.5, 0.1],
        [-2.1, -0.4],
        [0.1, 3.1],
        [-0.3, 2.7],
        [0.4, 2.4],
        [0.0, 2.9],
        [3.1, -0.2],
        [2.7, 0.4],
        [2.5, -0.5],
        [2.9, 0.1]
    ];
    let labels = array![0_i64, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

    let pool_1 = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool_4 = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let fit_once = || {
        SKSVC::<f64>::build()
            .kernel(SKKernel::RadialBasisFunction)
            .C(1.5)
            .probability(true)
            .random_seed(123)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap()
    };

    let model_a = pool_1.install(fit_once);
    let model_b = pool_4.install(fit_once);

    assert_eq!(
        model_a.pairwise_models().len(),
        model_b.pairwise_models().len()
    );
    for (pair_a, pair_b) in model_a
        .pairwise_models()
        .iter()
        .zip(model_b.pairwise_models().iter())
    {
        assert_eq!(
            pair_a.support_training_indices(),
            pair_b.support_training_indices()
        );
        assert_eq!(pair_a.dual_coefficients(), pair_b.dual_coefficients());
        assert_eq!(pair_a.bias(), pair_b.bias());
        assert_eq!(pair_a.platt_parameters(), pair_b.platt_parameters());
        if let (SKDataView::Dense(sv_a), SKDataView::Dense(sv_b)) = (
            pair_a.support_vectors().view(),
            pair_b.support_vectors().view(),
        ) {
            assert_eq!(sv_a, sv_b);
        }
    }

    assert_eq!(
        model_a.predict_labels(features.view()).unwrap(),
        model_b.predict_labels(features.view()).unwrap()
    );
    assert_eq!(
        model_a.predict_probabilities(features.view()).unwrap(),
        model_b.predict_probabilities(features.view()).unwrap()
    );
}

#[test]
fn determinism_svr_repeated_fits_match_exactly() {
    let features =
        Array2::from_shape_fn((16, 2), |(r, c)| (r as f64 * 0.25 - 1.8) + (c as f64) * 0.3);
    let targets = Array1::from_shape_fn(16, |r| (features[[r, 0]] * 0.9).sin());

    let estimator = SKSVR::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .C(3.0)
        .epsilon(0.05)
        .build()
        .unwrap();

    let first = estimator.fit(features.view(), targets.view()).unwrap();
    let second = estimator.fit(features.view(), targets.view()).unwrap();

    assert_eq!(
        first.support_training_indices(),
        second.support_training_indices()
    );
    assert_eq!(first.dual_coefficients(), second.dual_coefficients());
    assert_eq!(first.bias(), second.bias());
    assert_eq!(
        first.predict(features.view()).unwrap(),
        second.predict(features.view()).unwrap()
    );
}
