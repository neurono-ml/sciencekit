//! Large-data and concurrency acceptance tests for `SKFeatureUnion` inside
//! `SKPipeline` (spec `feature-union`, Task 3.2).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKFeatureBlock};
use sciencekit_impute::missing_indicator::{SKMissingFeatureScope, SKMissingIndicator};
use sciencekit_impute::simple_imputer::SKSimpleImputer;
use sciencekit_linear_model::logistic_regression::SKLogisticRegression;
use sciencekit_metrics::classification::sk_accuracy_score;
use sciencekit_preprocessing::minmax_scaler::core_implementation::SKMinMaxScaler;
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;
use sprs::{CsMat, TriMat};

use crate::{SKConfiguredStages, SKFeatureUnion, SKPipeline};

fn separable_large_dataset(rows: usize, columns: usize) -> (Array2<f64>, Array1<i64>) {
    let values: Vec<f64> = (0..rows)
        .flat_map(|row| {
            let class_offset = if row % 2 == 0 { -2.0 } else { 2.0 };
            (0..columns).map(move |column| {
                let variation = ((row * 13 + column * 7) % 29) as f64 / 29.0;
                class_offset + variation
            })
        })
        .collect();
    let features = Array2::from_shape_vec((rows, columns), values).unwrap();
    let labels = Array1::from_vec((0..rows).map(|row| (row % 2) as i64).collect());
    (features, labels)
}

fn separable_large_sparse(rows: usize, columns: usize) -> (CsMat<f64>, Array1<i64>) {
    let mut triplets = TriMat::new((rows, columns));
    (0..rows).for_each(|row| {
        let sign = if row % 2 == 0 { -1.5 } else { 1.5 };
        let primary_column = row % (columns / 2);
        let secondary_column = (columns / 2) + (row % (columns / 2));
        triplets.add_triplet(row, primary_column, sign);
        triplets.add_triplet(row, secondary_column, sign * 2.0);
    });
    let labels = Array1::from_vec((0..rows).map(|row| (row % 2) as i64).collect());
    (triplets.to_csr(), labels)
}

#[test]
fn large_feature_union_pipeline_fits_predicts_and_produces_metrics() {
    let (features, labels) = separable_large_dataset(500, 6);
    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();

    let fitted = SKPipeline::start::<f64>()
        .then(union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();

    assert_eq!(fitted.number_of_features(), 12);
    let predictions = fitted.predict_labels(&features).unwrap();
    let accuracy: f64 =
        sk_accuracy_score(labels.as_slice().unwrap(), predictions.as_slice().unwrap()).unwrap();
    assert!(accuracy > 0.95, "expected accuracy > 0.95, got {accuracy}");
}

#[test]
fn large_sparse_feature_union_pipeline_fits_without_densification() {
    let (sparse_features, labels) = separable_large_sparse(600, 16);
    let union = SKFeatureUnion::build()
        .stages((
            SKSimpleImputer::<f64>::build().build().unwrap(),
            SKMissingIndicator::<f64>::build()
                .features(SKMissingFeatureScope::All)
                .sparse_output(true)
                .error_on_new(false)
                .build()
                .unwrap(),
        ))
        .build()
        .unwrap();

    let fitted = SKPipeline::start::<f64>()
        .then(union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(sparse_features.view(), labels.view())
        .unwrap();

    assert_eq!(fitted.number_of_features(), 32);
    let block = fitted
        .stages()
        .transform_stages(
            fitted.fitted_stages(),
            SKFeatureBlock::Sparse(sparse_features.clone()),
        )
        .unwrap();
    assert!(block.is_sparse());

    let predictions = fitted.predict_labels(sparse_features.view()).unwrap();
    let accuracy: f64 =
        sk_accuracy_score(labels.as_slice().unwrap(), predictions.as_slice().unwrap()).unwrap();
    assert!(accuracy > 0.90, "expected accuracy > 0.90, got {accuracy}");
}

#[test]
fn concurrent_predictions_over_fitted_feature_union_pipeline_agree() {
    let (features, labels) = separable_large_dataset(200, 4);
    let union = SKFeatureUnion::build()
        .stages((
            SKStandardScaler::build().build().unwrap(),
            SKMinMaxScaler::<f64>::build().build().unwrap(),
        ))
        .build()
        .unwrap();
    let fitted = SKPipeline::start::<f64>()
        .then(union)
        .finalize_classifier(SKLogisticRegression::build().build().unwrap())
        .unwrap()
        .fit(&features, labels.view())
        .unwrap();

    let expected_labels = fitted.predict_labels(&features).unwrap();
    let expected_probabilities = fitted.predict_probabilities(&features).unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let model = &fitted;
                let query = &features;
                scope.spawn(move || {
                    (
                        model.predict_labels(query).unwrap(),
                        model.predict_probabilities(query).unwrap(),
                    )
                })
            })
            .collect();
        handles.into_iter().for_each(|handle| {
            let (found_labels, found_probabilities) = handle.join().unwrap();
            assert_eq!(found_labels, expected_labels);
            assert_eq!(found_probabilities, expected_probabilities);
        });
    });
}
