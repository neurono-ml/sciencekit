//! Large-data acceptance scenarios (1,080 rows), factor variations (2, 3, 4),
//! geometric resource schedules and exact per-round fold subsample verification.

use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit, SKTargetView,
};

use super::super::fixtures::dense_design;
use crate::grid_search::mock_estimators::{SKMockConstantRegressor, SKMockThresholdClassifier};
use crate::halving::{SKHalvingGridSearchClassifier, SKHalvingGridSearchRegressor};
use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;

const LARGE_ROWS: usize = 1080;

#[test]
fn halving_acceptance_large_classifier_runs_multi_round_schedule_and_refits_full_dataset() {
    let features = Array2::from_shape_fn((LARGE_ROWS, 4), |(row, col)| (row + col) as f64);
    let labels = Array1::from_iter((0..LARGE_ROWS).map(|row| i64::from(row >= LARGE_ROWS / 2)));
    // 27 candidates across thresholds; index 13 sits at 540.0 (exact midpoint -> accuracy 1.0).
    let candidates: Vec<SKMockThresholdClassifier> = (0..27)
        .map(|index| SKMockThresholdClassifier::new(((index + 1) * 40 - 20) as f64))
        .collect();

    let estimator = SKHalvingGridSearchClassifier::build()
        .candidates(candidates)
        .splitter(
            SKStratifiedKFold::build()
                .number_of_splits(4)
                .shuffle(true)
                .random_seed(5)
                .build()
                .unwrap(),
        )
        .factor(3)
        .minimum_resource(40)
        .random_seed(17)
        .build()
        .unwrap();

    let model = estimator.fit(features.view(), labels.view()).unwrap();
    assert_eq!(model.resource_schedule(), vec![40, 120, 360]);
    assert_eq!(model.candidate_count_trajectory(), vec![27, 9, 3, 1]);
    assert_eq!(model.best_candidate_index(), 13);
    assert_eq!(model.best_model().unwrap().training_rows(), LARGE_ROWS);
    assert!(
        model
            .elimination_trace()
            .iter()
            .all(|round| round.fold_scores().iter().all(|score| score.is_finite()))
    );
}

#[test]
fn halving_acceptance_large_regressor_supports_factor_two_three_and_four_schedules() {
    let features = dense_design(LARGE_ROWS, 3);
    let targets = Array1::from_elem(LARGE_ROWS, 7.0);
    let candidates: Vec<SKMockConstantRegressor> = (0..16)
        .map(|index| SKMockConstantRegressor::new(index as f64))
        .collect();

    for (factor, min_resource, max_resource, expected_resources, expected_trajectory) in [
        (2, 30, 480, vec![30, 60, 120, 240], vec![16, 8, 4, 2, 1]),
        (3, 30, 810, vec![30, 90, 270], vec![16, 6, 2, 1]),
        (4, 30, 480, vec![30, 120], vec![16, 4, 1]),
    ] {
        let model = SKHalvingGridSearchRegressor::build()
            .candidates(candidates.clone())
            .splitter(SKKFold::build().number_of_splits(5).build().unwrap())
            .factor(factor)
            .minimum_resource(min_resource)
            .maximum_resource(max_resource)
            .random_seed(3)
            .build()
            .unwrap()
            .fit(features.view(), targets.view())
            .unwrap();

        assert_eq!(model.resource_schedule(), expected_resources);
        assert_eq!(model.candidate_count_trajectory(), expected_trajectory);
        assert_eq!(model.best_candidate_index(), 7);
        assert_eq!(model.best_score(), 0.0);
        assert_eq!(model.best_model().unwrap().training_rows(), LARGE_ROWS);
    }
}

#[derive(Clone)]
struct SKRowCountingRegressor {
    observed_fold_totals: Arc<Mutex<Vec<usize>>>,
}

struct SKRowCountingModel {
    training_rows: usize,
    observed_fold_totals: Arc<Mutex<Vec<usize>>>,
}

impl SKSupervisedFit<f64> for SKRowCountingRegressor {
    type Model = SKRowCountingModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let _ = features.try_into()?;
        let training_rows = targets.try_into()?.len();
        Ok(SKRowCountingModel {
            training_rows,
            observed_fold_totals: Arc::clone(&self.observed_fold_totals),
        })
    }
}

impl SKRegressorPredictor<f64> for SKRowCountingModel {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let validation_rows = match features.try_into()? {
            SKDataView::Dense(view) => view.nrows(),
            SKDataView::Sparse(view) => view.rows(),
            _ => 0,
        };
        self.observed_fold_totals
            .lock()
            .unwrap()
            .push(self.training_rows + validation_rows);
        Ok(Array1::zeros(validation_rows))
    }
}

#[test]
fn halving_acceptance_thirty_row_resource_evaluates_exact_thirty_rows_across_every_fold() {
    let features = dense_design(120, 2);
    let targets = Array1::<f64>::zeros(120);
    let observed = Arc::new(Mutex::new(Vec::new()));
    let candidates = vec![
        SKRowCountingRegressor {
            observed_fold_totals: Arc::clone(&observed),
        },
        SKRowCountingRegressor {
            observed_fold_totals: Arc::clone(&observed),
        },
        SKRowCountingRegressor {
            observed_fold_totals: Arc::clone(&observed),
        },
    ];

    let model = SKHalvingGridSearchRegressor::build()
        .candidates(candidates)
        .splitter(SKKFold::build().number_of_splits(3).build().unwrap())
        .factor(3)
        .minimum_resource(30)
        .maximum_resource(30)
        .refit(false)
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();

    assert_eq!(model.resource_schedule(), vec![30]);
    let totals = observed.lock().unwrap().clone();
    // 3 candidates * 3 folds = 9 fold evaluations, every single one over the 30-row subsample.
    assert_eq!(totals, vec![30; 9]);
}
