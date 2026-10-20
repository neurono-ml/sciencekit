//! Companion tests for the incremental fit (spec `simple-imputer`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::core_implementation::SKSimpleImputer;
use super::model::SKSimpleImputerModel;
use super::strategy::SKImputationStrategy;

/// The documented relative tolerance of the online median sketch.
const SKETCH_RELATIVE_TOLERANCE: f64 = 0.05;

/// Build a CSR matrix from triplets for the sparse cases.
fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

/// The statistics as a plain vector.
fn statistics(model: &SKSimpleImputerModel<f64>) -> Vec<f64> {
    model.statistics().to_vec()
}

/// Assert two statistic vectors agree within a relative tolerance.
fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        let allowed = tolerance * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= allowed,
            "expected {expected}, found {actual} (tolerance {allowed})"
        );
    }
}

/// A deterministic column of values with every seventh cell missing.
fn median_features(rows_count: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows_count, 1), |(row, _)| {
        if row % 7 == 0 {
            f64::NAN
        } else {
            ((row * 37) % rows_count) as f64 + 1.0
        }
    })
}

#[test]
fn split_batches_equal_the_whole_for_mean() {
    let reference = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![
            [1.0, 10.0],
            [3.0, f64::NAN],
            [5.0, 14.0],
            [7.0, 18.0]
        ])
        .unwrap();
    let mut incremental = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, 10.0]])
        .unwrap();
    incremental
        .partial_fit(&array![[3.0, f64::NAN], [5.0, 14.0]])
        .unwrap();
    incremental.partial_fit(&array![[7.0, 18.0]]).unwrap();
    assert_eq!(statistics(&incremental), statistics(&reference));
    let probe = array![[f64::NAN, f64::NAN], [2.0, 12.0]];
    let reference_output = reference.transform(&probe).unwrap();
    let incremental_output = incremental.transform(&probe).unwrap();
    assert_eq!(incremental_output, reference_output);
}

#[test]
fn split_batches_equal_the_whole_for_most_frequent() {
    let reference = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::MostFrequent)
        .build()
        .unwrap()
        .fit(&array![[2.0], [5.0], [2.0], [5.0], [2.0]])
        .unwrap();
    let mut incremental = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::MostFrequent)
        .build()
        .unwrap()
        .fit(&array![[2.0], [5.0]])
        .unwrap();
    incremental
        .partial_fit(&array![[2.0], [5.0], [2.0]])
        .unwrap();
    assert_eq!(statistics(&incremental), statistics(&reference));
}

#[test]
fn constant_statistics_ignore_the_batches() {
    let mut model = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Constant)
        .fill_value(Some(-1.0))
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN]])
        .unwrap();
    model.partial_fit(&array![[2.0, 3.0]]).unwrap();
    assert_eq!(statistics(&model), vec![-1.0, -1.0]);
}

#[test]
fn median_stays_within_the_sketch_tolerance_of_the_dense_fit() {
    let features = median_features(600);
    let reference = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let mut incremental = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap()
        .fit(features.slice(ndarray::s![..300, ..]))
        .unwrap();
    incremental
        .partial_fit(features.slice(ndarray::s![300.., ..]))
        .unwrap();
    assert_close(
        &statistics(&incremental),
        &statistics(&reference),
        SKETCH_RELATIVE_TOLERANCE,
    );
}

#[test]
fn empty_batch_is_a_no_op() {
    let mut model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0], [3.0]])
        .unwrap();
    let before = statistics(&model);
    model
        .partial_fit(&Array2::<f64>::zeros((0, 1)))
        .expect("an empty batch succeeds");
    assert_eq!(statistics(&model), before);
}

#[test]
fn column_count_mismatch_is_rejected() {
    let mut model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0], [3.0]])
        .unwrap();
    let error = model
        .partial_fit(&array![[1.0, 2.0]])
        .err()
        .expect("a two-column batch cannot extend a one-column model");
    assert!(
        error.to_string().contains("shape mismatch"),
        "the error reports the shape contract: {error}"
    );
}

#[test]
fn partial_fit_chains_on_the_model() {
    let mut model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[2.0]])
        .unwrap();
    model
        .partial_fit(&array![[4.0]])
        .unwrap()
        .partial_fit(&array![[6.0]])
        .unwrap()
        .partial_fit(&array![[8.0]])
        .unwrap();
    assert_eq!(statistics(&model), vec![5.0]);
}

#[test]
fn sparse_batch_accumulates_like_the_dense_concatenation() {
    let reference = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[4.0], [2.0], [0.0]])
        .unwrap();
    let mut model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(csr((1, 1), &[(0, 0, 4.0)]).view())
        .unwrap();
    model
        .partial_fit(csr((2, 1), &[(0, 0, 2.0)]).view())
        .unwrap();
    assert_eq!(statistics(&model), statistics(&reference));
}
