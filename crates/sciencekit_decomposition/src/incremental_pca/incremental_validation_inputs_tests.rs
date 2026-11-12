//! Companion validation tests for feature-count mismatches, non-finite values
//! and unsupported CSC storage on `SKIncrementalPCA` (spec `incremental-pca`).

use std::collections::VecDeque;

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKDataBatch, SKError, SKUnsupervisedFit};
use sprs::TriMat;

use super::SKIncrementalPCA;
use super::incremental_validation_tests::SliceBatchSource;

#[test]
fn incremental_validation_rejects_feature_mismatch_non_finite_and_csc() {
    let valid = Array2::from_shape_fn((8, 4), |(row, column)| {
        ((row + 1) * (column + 2)) as f64 + (row % 3) as f64 * 0.5
    });
    let mut model = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .build()
        .unwrap()
        .fit(&valid)
        .unwrap();
    let prior_samples = model.samples_seen();

    let wrong_columns = Array2::<f64>::ones((4, 5));
    assert!(matches!(
        model.partial_fit(&wrong_columns).unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![4, 4] && found == vec![4, 5]
    ));
    assert_eq!(model.samples_seen(), prior_samples);

    let mut mismatched_stream = SliceBatchSource {
        batches: VecDeque::from(vec![
            SKDataBatch::new_dense(valid.clone(), 0, false),
            SKDataBatch::new_dense(wrong_columns, 1, true),
        ]),
    };
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(2)
            .build()
            .unwrap()
            .fit_streaming(&mut mismatched_stream)
            .unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));

    let mut dense_nan = valid.clone();
    dense_nan[(1, 2)] = f64::NAN;
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(2)
            .build()
            .unwrap()
            .fit(&dense_nan)
            .unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));
    assert!(matches!(
        model.partial_fit(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));
    assert!(matches!(
        model.transform(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));

    let mut sparse_triplet = TriMat::new((4, 4));
    sparse_triplet.add_triplet(0, 1, 2.0_f64);
    sparse_triplet.add_triplet(2, 3, f64::INFINITY);
    let sparse_inf = sparse_triplet.to_csr();
    assert!(matches!(
        model.partial_fit(sparse_inf.view()).unwrap_err(),
        SKError::NonFiniteValue { position: 11 }
    ));

    let csc = {
        let mut triplet = TriMat::new((4, 4));
        triplet.add_triplet(0, 0, 1.0_f64);
        triplet.to_csc()
    };
    assert!(matches!(
        model.partial_fit(csc.view()).unwrap_err(),
        SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));

    assert!(matches!(
        model.transform(&Array2::<f64>::zeros((2, 5))).unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![2, 4] && found == vec![2, 5]
    ));
    assert!(matches!(
        model
            .inverse_transform(&Array2::<f64>::zeros((2, 2)))
            .unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![2, 3] && found == vec![2, 2]
    ));
}
