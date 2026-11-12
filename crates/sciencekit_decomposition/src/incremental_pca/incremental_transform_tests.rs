//! Companion tests for `transform`, `fit_transform`, `inverse_transform`,
//! whitening and dense/CSR representation parity on `SKIncrementalPCA`.

use ndarray::{Array2, Axis};
use sciencekit_common::{SKBackendKind, SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKIncrementalPCA;
use crate::pca::test_tracing::PcaCaptureGuard;

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

#[test]
fn incremental_transform_shape_roundtrip_and_sparse_dense_parity() {
    let train = Array2::from_shape_fn((48, 6), |(row, column)| {
        if (row * 5 + column * 3) % 4 == 0 {
            let angle = (row as f64 + 1.0) * 0.21 + (column as f64) * 0.53;
            angle.sin() * (7.0 - column as f64) + ((row * 3 + column * 7) % 9) as f64 * 0.45
        } else {
            0.0
        }
    });
    let query = Array2::from_shape_fn((19, 6), |(row, column)| {
        if (row + column * 2) % 3 == 0 {
            (row as f64 + 0.5) * 0.4 - (column as f64) * 0.75
        } else {
            0.0
        }
    });
    let query_csr = csr_from_dense(&query);

    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .batch_size(16)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();

    let model = estimator.fit(&train).unwrap();
    let capture = PcaCaptureGuard::begin();
    let projected_dense = model.transform(&query).unwrap();
    let projected_sparse = SKFeatureTransformer::transform(&model, query_csr.view()).unwrap();
    let spans = capture.spans();

    assert_eq!(projected_dense.dim(), (19, 3));
    assert_eq!(projected_sparse.dim(), (19, 3));
    assert!(spans.iter().any(|(name, fields)| {
        name == "sciencekit.operation"
            && fields.get("operation").map(String::as_str) == Some("transform")
            && fields.get("backend").map(String::as_str) == Some("matrixmultiply")
    }));

    for (left, right) in projected_dense.iter().zip(projected_sparse.iter()) {
        assert!((left - right).abs() < 1e-11);
    }

    let fit_transformed = estimator.fit_transform(&train).unwrap();
    let direct_transformed = model.transform(&train).unwrap();
    for (left, right) in fit_transformed.iter().zip(direct_transformed.iter()) {
        assert!((left - right).abs() < 1e-12);
    }

    // Full-rank k == d round-trip reconstructs original inputs exactly.
    let full_rank_model = SKIncrementalPCA::<f64>::build()
        .number_of_components(6)
        .batch_size(16)
        .build()
        .unwrap()
        .fit(&train)
        .unwrap();
    let coordinates = full_rank_model.transform(&query).unwrap();
    let coordinates_csr = csr_from_dense(&coordinates);
    let reconstructed_dense = full_rank_model.inverse_transform(&coordinates).unwrap();
    let reconstructed_sparse = full_rank_model
        .inverse_transform(coordinates_csr.view())
        .unwrap();
    for (left, right) in reconstructed_dense.iter().zip(query.iter()) {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
    for (left, right) in reconstructed_sparse.iter().zip(query.iter()) {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
}

#[test]
fn incremental_transform_whitening_normalizes_unit_variance_and_inverts() {
    let data = Array2::from_shape_fn((80, 5), |(row, column)| {
        let angle = (row as f64 + 1.0) * 0.17;
        match column {
            0 => angle.sin() * 12.0 + 3.0,
            1 => angle.cos() * 5.0 - 2.0,
            2 => (angle * 2.3).sin() * 2.2 + 1.0,
            _ => ((row * 7 + column * 13) % 17) as f64 * 0.35,
        }
    });

    let whitened_model = SKIncrementalPCA::<f64>::build()
        .number_of_components(4)
        .batch_size(20)
        .whiten(true)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let unwhitened_model = SKIncrementalPCA::<f64>::build()
        .number_of_components(4)
        .batch_size(20)
        .whiten(false)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    let whitened_scores = whitened_model.transform(&data).unwrap();
    assert_eq!(whitened_scores.dim(), (80, 4));

    let column_means = whitened_scores.mean_axis(Axis(0)).unwrap();
    for &mean in column_means.iter() {
        assert!(mean.abs() < 1e-11, "mean={mean}");
    }

    let degrees_of_freedom = (data.nrows() - 1) as f64;
    for column in whitened_scores.columns() {
        let variance = column.iter().map(|&value| value * value).sum::<f64>() / degrees_of_freedom;
        assert!((variance - 1.0).abs() < 1e-10, "variance={variance}");
    }

    let reconstructed_whitened = whitened_model.inverse_transform(&whitened_scores).unwrap();
    let unwhitened_scores = unwhitened_model.transform(&data).unwrap();
    let reconstructed_unwhitened = unwhitened_model
        .inverse_transform(&unwhitened_scores)
        .unwrap();
    for (left, right) in reconstructed_whitened
        .iter()
        .zip(reconstructed_unwhitened.iter())
    {
        assert!((left - right).abs() < 1e-10);
    }
}
