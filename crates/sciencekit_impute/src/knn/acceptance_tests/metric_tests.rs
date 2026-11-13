//! Imputation-error acceptance through `sciencekit_metrics`.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sciencekit_metrics::regression::sk_mean_absolute_error;

use super::fixtures::{dense, linear_design};
use crate::knn::core_implementation::SKKNNImputer;

const ROWS: usize = 300;
const COLUMNS: usize = 3;

#[test]
fn acceptance_knn_imputer_produces_a_low_imputation_error() {
    let truth = linear_design(ROWS, COLUMNS);
    let mut masked = truth.clone();
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            if (row + column) % 5 == 0 {
                masked[[row, column]] = f64::NAN;
            }
        }
    }
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(5)
        .weights(crate::knn::SKImputationWeights::Distance)
        .build()
        .unwrap()
        .fit(masked.view())
        .unwrap();

    let imputed = dense(model.transform(&masked).unwrap());
    let (targets, predictions) = masked_cells(&truth, &masked, &imputed);
    assert!(
        targets.len() > 150,
        "the fixture must mask a meaningful share of cells"
    );
    let error = sk_mean_absolute_error(
        Array1::from(targets).view(),
        Array1::from(predictions).view(),
    )
    .unwrap();
    assert!(error < 2.0, "mean absolute imputation error was {error}");
}

/// The ground-truth and imputed values at the cells the fixture masked.
fn masked_cells(
    truth: &Array2<f64>,
    masked: &Array2<f64>,
    imputed: &Array2<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut targets = Vec::new();
    let mut predictions = Vec::new();
    for ((row, column), value) in masked.indexed_iter() {
        if value.is_nan() {
            targets.push(truth[[row, column]]);
            predictions.push(imputed[[row, column]]);
        }
    }
    (targets, predictions)
}
