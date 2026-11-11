//! Companion tests verifying feature-subset routing never leaks unchosen
//! columns during dense or sparse prediction.

use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, Axis};
use sciencekit_common::{
    SKBuilder, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit, SKTargetView,
};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_tree::SKMaximumFeatures;
use sprs::TriMat;

use crate::{SKBaggingRegressor, SKMaximumSamples};

use super::predict::sk_predict_regressor_chunk;
use super::sampling::sk_bagging_feature_indices;

#[derive(Clone, Debug)]
struct ColumnSpyRegressor {
    seen_prediction_matrices: Arc<Mutex<Vec<Array2<f64>>>>,
}

#[derive(Clone, Debug)]
struct ColumnSpyModel {
    seen_prediction_matrices: Arc<Mutex<Vec<Array2<f64>>>>,
}

impl SKSupervisedFit<f64> for ColumnSpyRegressor {
    type Model = ColumnSpyModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, _features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        Ok(ColumnSpyModel {
            seen_prediction_matrices: Arc::clone(&self.seen_prediction_matrices),
        })
    }
}

impl SKRegressorPredictor<f64> for ColumnSpyModel {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let dense = match features.try_into()? {
            SKDataView::Dense(view) => view.to_owned(),
            SKDataView::Sparse(sparse) => sparse.to_dense(),
            _ => Array2::zeros((0, 0)),
        };
        let sums = dense.sum_axis(Axis(1));
        self.seen_prediction_matrices.lock().unwrap().push(dense);
        Ok(sums)
    }
}

fn find_seed_producing_zero_and_two() -> u64 {
    (0..10_000_u64)
        .find(|&seed| {
            sk_bagging_feature_indices(3, &SKMaximumFeatures::Count(2), false, seed, 0)
                == vec![0, 2]
        })
        .expect("a seed draws subset [0, 2] from 3 columns")
}

#[test]
fn estimator_fitted_on_columns_zero_and_two_receives_only_those_columns() {
    let seed = find_seed_producing_zero_and_two();
    let train_features = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 100.0, 2.0, //
            2.0, -50.0, 1.0, //
            3.0, 200.0, 4.0, //
            4.0, -80.0, 3.0, //
            5.0, 300.0, 6.0, //
            6.0, -90.0, 5.0,
        ],
    )
    .unwrap();
    let train_targets = Array1::from_vec(vec![5.0, 5.0, 11.0, 11.0, 17.0, 17.0]);

    let seen = Arc::new(Mutex::new(Vec::new()));
    let spy = ColumnSpyRegressor {
        seen_prediction_matrices: Arc::clone(&seen),
    };
    let bagging_model = SKBaggingRegressor::build(spy)
        .number_of_estimators(1)
        .maximum_features(SKMaximumFeatures::Count(2))
        .random_seed(seed)
        .build()
        .unwrap()
        .fit(train_features.view(), train_targets.view())
        .unwrap();

    assert_eq!(bagging_model.estimator_features(0), Some(&[0usize, 2][..]));

    // Column 1 carries a huge sentinel (999_999.0) that must never reach the base model.
    let query = Array2::from_shape_vec(
        (2, 3),
        vec![
            10.0, 999_999.0, 30.0, //
            40.0, -888_888.0, 60.0,
        ],
    )
    .unwrap();
    let query_view = SKDataView::Dense(query.view());
    let routed_predictions = sk_predict_regressor_chunk(
        bagging_model.estimator(0).unwrap(),
        &query_view,
        0,
        2,
        bagging_model.estimator_features(0).unwrap(),
    )
    .unwrap();

    assert_eq!(routed_predictions, Array1::from_vec(vec![40.0, 100.0]));
    {
        let recorded = seen.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(
            recorded[0],
            Array2::from_shape_vec((2, 2), vec![10.0, 30.0, 40.0, 60.0]).unwrap()
        );
    }

    // Sparse CSR routing also passes only columns [0, 2].
    let mut triplets = TriMat::new((2, 3));
    triplets.add_triplet(0, 0, 10.0);
    triplets.add_triplet(0, 1, 999_999.0);
    triplets.add_triplet(0, 2, 30.0);
    triplets.add_triplet(1, 1, -888_888.0);
    triplets.add_triplet(1, 2, 60.0);
    let sparse_query = triplets.to_csr::<usize>();
    let sparse_routed = sk_predict_regressor_chunk(
        bagging_model.estimator(0).unwrap(),
        &SKDataView::Sparse(sparse_query.view()),
        0,
        2,
        bagging_model.estimator_features(0).unwrap(),
    )
    .unwrap();
    assert_eq!(sparse_routed, Array1::from_vec(vec![40.0, 60.0]));
}

#[test]
fn routed_single_estimator_matches_manually_subsetted_linear_regression() {
    let seed = find_seed_producing_zero_and_two();
    let train_features = Array2::from_shape_vec(
        (6, 3),
        vec![
            1.0, 500.0, 0.0, //
            0.0, -400.0, 1.0, //
            2.0, 300.0, 1.0, //
            1.0, -200.0, 2.0, //
            3.0, 100.0, 2.0, //
            2.0, -100.0, 3.0,
        ],
    )
    .unwrap();
    let train_targets = Array1::from_shape_fn(6, |row| {
        2.0 * train_features[[row, 0]] + 3.0 * train_features[[row, 2]] + 1.0
    });

    let linear = SKLinearRegression::build().build().unwrap();
    let bagging_model = SKBaggingRegressor::build(linear.clone())
        .number_of_estimators(1)
        .bootstrap(false)
        .maximum_samples(SKMaximumSamples::All)
        .maximum_features(SKMaximumFeatures::Count(2))
        .random_seed(seed)
        .build()
        .unwrap()
        .fit(train_features.view(), train_targets.view())
        .unwrap();

    let manual_train = train_features.select(Axis(1), &[0, 2]);
    let manual_model = linear
        .fit(manual_train.view(), train_targets.view())
        .unwrap();

    let query =
        Array2::<f64>::from_shape_vec((2, 3), vec![4.0, 77_777.0, 5.0, -1.0, -99_999.0, 2.0])
            .unwrap();
    let routed = sk_predict_regressor_chunk(
        bagging_model.estimator(0).unwrap(),
        &SKDataView::Dense(query.view()),
        0,
        2,
        bagging_model.estimator_features(0).unwrap(),
    )
    .unwrap();
    let manual = manual_model
        .predict(query.select(Axis(1), &[0, 2]).view())
        .unwrap();
    routed
        .iter()
        .zip(manual.iter())
        .for_each(|(left, right)| assert!((left - right).abs() < 1e-10));
}
