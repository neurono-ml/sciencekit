//! Hand-computed candidate/fold matrices (spec `grid-search`, requirement
//! `Candidate evaluation across folds`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKDataView, SKTargetView};

use super::sk_evaluate_candidates;
use crate::grid_search::error_score::SKErrorScore;
use crate::grid_search::mock_estimators::SKMockConstantRegressor;
use crate::grid_search::scoring::SKRegressorScoring;
use crate::split::kfold::SKKFold;

/// Targets `0..10` split into five contiguous folds of two rows each.
fn fixtures() -> (Array2<f64>, Array1<f64>) {
    let features = Array2::zeros((10, 1));
    let targets = Array1::from_iter((0..10).map(|value| value as f64));
    (features, targets)
}

#[test]
fn grid_evaluation_three_candidates_five_folds_matches_the_hand_computed_matrix() {
    let (features, targets) = fixtures();
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();
    let candidates = vec![
        SKMockConstantRegressor::new(0.0),
        SKMockConstantRegressor::new(1.0),
        SKMockConstantRegressor::new(2.0),
    ];

    let results = sk_evaluate_candidates(
        &candidates,
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        None,
        SKErrorScore::Raise,
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();

    assert_eq!(results.number_of_candidates(), 3);
    assert_eq!(results.number_of_folds(), 5);
    let expected = array![
        [-0.5, -6.5, -20.5, -42.5, -72.5],
        [-0.5, -2.5, -12.5, -30.5, -56.5],
        [-2.5, -0.5, -6.5, -20.5, -42.5],
    ];
    for ((candidate, fold), value) in results.fold_scores().indexed_iter() {
        assert!(
            (value - expected[[candidate, fold]]).abs() < 1e-12,
            "candidate {candidate} fold {fold}: {value} != {}",
            expected[[candidate, fold]]
        );
    }
}

#[test]
fn grid_evaluation_fold_count_matches_the_splitter() {
    let (features, targets) = fixtures();
    let splitter = SKKFold::build().number_of_splits(5).build().unwrap();
    let results = sk_evaluate_candidates(
        &[SKMockConstantRegressor::new(0.0)],
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        None,
        SKErrorScore::Raise,
        &SKRegressorScoring::NegativeMeanSquaredError,
    )
    .unwrap();
    assert_eq!(results.fold_scores().nrows(), 1);
    assert_eq!(results.fold_scores().ncols(), 5);
}

/// The splitter runs exactly once, no matter how many candidates are scored:
/// every candidate sees the same fold partition.
#[test]
fn grid_evaluation_splits_once_for_every_candidate() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ndarray::ArrayView1;

    use crate::SKSplitter;
    use crate::index_split::SKIndexSplit;
    use sciencekit_common::{SKError, SKFloat};

    struct CountingSplitter {
        inner: SKKFold,
        calls: Arc<AtomicUsize>,
    }

    impl<F: SKFloat> SKSplitter<F> for CountingSplitter {
        fn split(
            &self,
            features: SKDataView<'_, F>,
            targets: Option<SKTargetView<'_, F>>,
            groups: Option<ArrayView1<'_, i64>>,
        ) -> Result<Vec<SKIndexSplit>, SKError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.split(features, targets, groups)
        }
    }

    let (features, targets) = fixtures();
    let calls = Arc::new(AtomicUsize::new(0));
    let splitter = CountingSplitter {
        inner: SKKFold::build().number_of_splits(5).build().unwrap(),
        calls: Arc::clone(&calls),
    };
    let candidates = vec![
        SKMockConstantRegressor::new(0.0),
        SKMockConstantRegressor::new(4.0),
        SKMockConstantRegressor::new(8.0),
    ];
    sk_evaluate_candidates(
        &candidates,
        &splitter,
        SKDataView::Dense(features.view()),
        SKTargetView::Continuous(targets.view()),
        None,
        SKErrorScore::Raise,
        &SKRegressorScoring::R2,
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
