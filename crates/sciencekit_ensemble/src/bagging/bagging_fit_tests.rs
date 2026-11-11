//! Companion tests for parallel, deterministic per-estimator bagging fitting.

use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClassifierPredictor, SKDataView, SKError, SKExecutionMode,
    SKSupervisedFit, SKTargetView,
};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};

use crate::{SKBaggingClassifier, SKBaggingRegressor, SKMaximumSamples};

#[derive(Clone, Debug)]
struct RecordingBaseClassifier {
    recorded_shapes: Arc<Mutex<Vec<(usize, usize)>>>,
}

#[derive(Clone, Debug)]
struct RecordedModel;

impl SKClassifierPredictor<f64> for RecordedModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array1::zeros(rows))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array2::from_elem((rows, 2), 0.5))
    }
}

impl SKSupervisedFit<f64> for RecordingBaseClassifier {
    type Model = RecordedModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let shape = match features.try_into()? {
            SKDataView::Dense(dense) => (dense.nrows(), dense.ncols()),
            SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
            _ => (0, 0),
        };
        self.recorded_shapes.lock().unwrap().push(shape);
        Ok(RecordedModel)
    }
}

fn synthetic_classification_dataset() -> (Array2<f64>, Array1<i64>) {
    let features = Array2::from_shape_fn((36, 6), |(row, column)| {
        let sign = if row % 2 == 0 { -1.0 } else { 1.0 };
        sign * ((column + 1) as f64) + (row as f64) * 0.05
    });
    let labels = Array1::from_shape_fn(36, |row| (row % 2) as i64);
    (features, labels)
}

#[test]
fn each_base_estimator_fits_on_configured_row_and_feature_subset_shapes() {
    let (features, labels) = synthetic_classification_dataset();
    let recorded_shapes = Arc::new(Mutex::new(Vec::new()));
    let spy = RecordingBaseClassifier {
        recorded_shapes: Arc::clone(&recorded_shapes),
    };

    let model = SKBaggingClassifier::build(spy)
        .number_of_estimators(6)
        .maximum_samples(SKMaximumSamples::Count(20))
        .maximum_features(SKMaximumFeatures::Count(3))
        .bootstrap_features(true)
        .random_seed(17)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert_eq!(model.number_of_estimators(), 6);
    assert_eq!(model.feature_count(), 6);
    assert!(
        model
            .feature_subsets()
            .iter()
            .all(|subset| subset.len() == 3)
    );
    let shapes = recorded_shapes.lock().unwrap();
    assert_eq!(shapes.len(), 6);
    assert!(
        shapes
            .iter()
            .all(|&(rows, columns)| rows == 20 && columns == 3)
    );
}

#[test]
fn parallel_fit_matches_sequential_fit_and_repeated_fits_are_deterministic() {
    let (features, labels) = synthetic_classification_dataset();
    let fit_mode = |mode| {
        let base_tree = SKDecisionTreeClassifier::<f64>::build()
            .maximum_depth(4)
            .random_seed(5)
            .build()
            .unwrap();
        SKBaggingClassifier::build(base_tree)
            .number_of_estimators(12)
            .maximum_features(SKMaximumFeatures::Count(4))
            .bootstrap_features(true)
            .random_seed(77)
            .execution_mode(mode)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap()
    };

    let sequential = fit_mode(SKExecutionMode::InProcessSynchronous);
    let parallel_first = fit_mode(SKExecutionMode::Automatic);
    let parallel_second = fit_mode(SKExecutionMode::Automatic);

    assert_eq!(
        sequential.feature_subsets(),
        parallel_first.feature_subsets()
    );
    assert_eq!(
        parallel_first.feature_subsets(),
        parallel_second.feature_subsets()
    );
    (0..12).for_each(|index| {
        assert_eq!(
            sequential.estimator(index).map(|model| model.tree()),
            parallel_first.estimator(index).map(|model| model.tree())
        );
        assert_eq!(
            parallel_first.estimator(index).map(|model| model.tree()),
            parallel_second.estimator(index).map(|model| model.tree())
        );
    });
}

#[test]
fn regressor_fit_is_invariant_across_rayon_thread_counts_and_non_tree_bases() {
    let features = Array2::from_shape_fn((40, 4), |(row, column)| {
        ((row * 3 + column * 5) as f64) * 0.1 + ((column + 1) as f64)
    });
    let targets = Array1::from_shape_fn(40, |row| {
        2.0 * features[[row, 0]] - 0.5 * features[[row, 2]] + 1.0
    });

    let base_tree = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(4)
        .build()
        .unwrap();
    let estimator = SKBaggingRegressor::build(base_tree)
        .number_of_estimators(10)
        .maximum_features(SKMaximumFeatures::Count(3))
        .random_seed(123)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();

    let pool_one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pool_four = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let single_thread =
        pool_one.install(|| estimator.fit(features.view(), targets.view()).unwrap());
    let four_threads =
        pool_four.install(|| estimator.fit(features.view(), targets.view()).unwrap());

    assert_eq!(single_thread.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        single_thread.feature_subsets(),
        four_threads.feature_subsets()
    );
    (0..10).for_each(|index| {
        assert_eq!(
            single_thread.estimator(index).unwrap().tree(),
            four_threads.estimator(index).unwrap().tree()
        );
    });

    let linear = SKLinearRegression::build().build().unwrap();
    let linear_bagging = SKBaggingRegressor::build(linear)
        .number_of_estimators(5)
        .random_seed(9)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    assert_eq!(linear_bagging.number_of_estimators(), 5);
}
