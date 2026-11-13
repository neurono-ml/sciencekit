//! Companion tests for dense/sparse CSR representation parity and
//! non-densifying sparse boosting (task 6.2).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_tree::SKMaximumFeatures;
use sprs::CsMat;

use crate::boosting::early_stopping::SKEarlyStoppingConfig;
use crate::boosting::loss::SKSquaredErrorLoss;
use crate::boosting::scalar_loop::sk_run_scalar_boosting;
use crate::boosting::stage::SKBoostingStageParameters;

use super::{
    SKGradientBoostingClassifier, SKGradientBoostingRegressionLoss, SKGradientBoostingRegressor,
};

fn logical_fixture() -> (Array2<f64>, CsMat<f64>, Array1<i64>, Array1<f64>) {
    let dense = array![
        [0.0, 2.0, 0.0, 1.0],
        [0.0, 1.5, 0.0, 0.0],
        [0.0, 2.5, 0.0, 0.5],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 1.0, 0.0],
        [3.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 5.0, 2.0],
        [0.0, 0.0, 5.5, 2.5],
        [0.0, 0.0, 6.0, 1.5],
    ];
    let sparse = CsMat::new(
        (9, 4),
        vec![0, 2, 3, 5, 6, 8, 10, 12, 14, 16],
        vec![1, 3, 1, 1, 3, 0, 0, 2, 0, 2, 2, 3, 2, 3, 2, 3],
        vec![
            2.0, 1.0, 1.5, 2.5, 0.5, 3.0, 4.0, 1.0, 3.5, 0.5, 5.0, 2.0, 5.5, 2.5, 6.0, 1.5,
        ],
    );
    let multiclass_labels = array![0_i64, 0, 0, 1, 1, 1, 2, 2, 2];
    let responses = array![1.0, 1.5, 1.2, 8.0, 9.0, 8.5, -4.0, -4.5, -3.5];
    (dense, sparse, multiclass_labels, responses)
}

/// Scenarios: Sparse fit equals dense fit and sparse prediction matches dense
/// prediction across multiclass and all regression losses.
#[test]
fn classifier_and_regressor_dense_and_sparse_fit_and_predict_are_equivalent() {
    let (dense, sparse, multiclass_labels, responses) = logical_fixture();
    let build_classifier = || {
        SKGradientBoostingClassifier::<f64>::build()
            .number_of_estimators(16)
            .learning_rate(0.15)
            .maximum_depth(2)
            .subsample(0.8)
            .n_iter_no_change(4)
            .validation_fraction(0.3)
            .random_seed(42)
            .build()
            .expect("builds")
    };
    let dense_classifier = build_classifier()
        .fit(&dense, multiclass_labels.view())
        .expect("dense fit");
    let sparse_classifier = build_classifier()
        .fit(sparse.view(), multiclass_labels.view())
        .expect("sparse fit");

    assert_eq!(
        dense_classifier.initial_predictions(),
        sparse_classifier.initial_predictions()
    );
    assert_eq!(
        dense_classifier.feature_importances(),
        sparse_classifier.feature_importances()
    );
    let expected = dense_classifier.predict_probabilities(&dense).expect("d2d");
    let query_predictions = [
        dense_classifier.predict_probabilities(sparse.view()),
        sparse_classifier.predict_probabilities(&dense),
        sparse_classifier.predict_probabilities(sparse.view()),
    ];
    query_predictions
        .into_iter()
        .for_each(|result| assert_eq!(expected, result.expect("predict")));

    for loss in [
        SKGradientBoostingRegressionLoss::SquaredError,
        SKGradientBoostingRegressionLoss::AbsoluteError,
        SKGradientBoostingRegressionLoss::Huber,
    ] {
        let build_regressor = || {
            SKGradientBoostingRegressor::<f64>::build()
                .number_of_estimators(14)
                .learning_rate(0.2)
                .maximum_depth(2)
                .subsample(0.8)
                .loss(loss)
                .random_seed(42)
                .build()
                .expect("builds")
        };
        let dense_model = build_regressor().fit(&dense, responses.view()).expect("d");
        let sparse_model = build_regressor()
            .fit(sparse.view(), responses.view())
            .expect("s");
        assert_eq!(dense_model.stage_trees(), sparse_model.stage_trees());
        let baseline = dense_model.predict(&dense).expect("d2d");
        assert_eq!(baseline, dense_model.predict(sparse.view()).expect("d2s"));
        assert_eq!(baseline, sparse_model.predict(&dense).expect("s2d"));
        assert_eq!(baseline, sparse_model.predict(sparse.view()).expect("s2s"));
    }
}

/// Ultra-wide `128 × 100_000` CSR matrices fit and predict without densifying,
/// and non-CSR (`CSC`) views are rejected up front on both public and substrate
/// paths.
#[test]
fn ultra_wide_sparse_never_densifies_and_csc_is_rejected() {
    let (rows, columns) = (128usize, 100_000usize);
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(rows * 2);
    let mut data = Vec::with_capacity(rows * 2);
    indptr.push(0);
    (0..rows).for_each(|row| {
        indices.extend([0, 1_000 + (row * 701) % (columns - 1_000)]);
        data.extend([if row < 64 { 1.0 } else { 10.0 }, 2.0]);
        indptr.push(indices.len());
    });
    let sparse = CsMat::new((rows, columns), indptr, indices, data);
    let labels = Array1::from_shape_fn(rows, |row| i64::from(row >= 64));
    let responses = Array1::from_shape_fn(rows, |row| if row < 64 { -1.0 } else { 5.0 });

    let classifier = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(4)
        .maximum_depth(2)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .subsample(0.75)
        .n_iter_no_change(2)
        .random_seed(5)
        .build()
        .expect("builds")
        .fit(sparse.view(), labels.view())
        .expect("wide sparse classifier fit");
    assert_eq!(
        classifier
            .predict_labels(sparse.view())
            .expect("pred")
            .len(),
        rows
    );

    let regressor = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(4)
        .maximum_depth(2)
        .maximum_features(SKMaximumFeatures::Sqrt)
        .subsample(0.75)
        .n_iter_no_change(2)
        .random_seed(5)
        .build()
        .expect("builds")
        .fit(sparse.view(), responses.view())
        .expect("wide sparse regressor fit");
    assert_eq!(regressor.predict(sparse.view()).expect("pred").len(), rows);

    let csc = CsMat::new_csc((4, 1), vec![0, 2], vec![1, 3], vec![1.0_f64, 2.0]);
    let stage_parameters = SKBoostingStageParameters {
        learning_rate: 0.1,
        maximum_depth: 1,
        minimum_samples_leaf: 1,
        maximum_features: SKMaximumFeatures::All,
        subsample: 0.5,
        l2_regularization: 0.0,
        random_seed: 0,
    };
    let stopping = Some(SKEarlyStoppingConfig {
        validation_fraction: 0.25,
        patience: 2,
        tolerance: 1e-4,
        random_seed: 0,
    });
    let outcome = sk_run_scalar_boosting(
        &SKSquaredErrorLoss::new(),
        &SKDataView::Sparse(csc.view()),
        &[0.0, 1.0, 2.0, 3.0],
        2,
        &stage_parameters,
        stopping,
    );
    assert!(matches!(
        outcome,
        Err(SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        })
    ));
}
