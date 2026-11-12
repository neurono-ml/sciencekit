//! Companion tests for `SKTruncatedSVDModel::transform` and
//! `SKTruncatedSVDModel::inverse_transform`.

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExecutionMode, SKFeatureTransformer, SKUnsupervisedFit,
};
use sprs::{CsMat, TriMat};

use super::product::{sk_peak_materialized_elements, sk_reset_peak_materialized_elements};
use super::{SK_TRUNCATED_SVD_PARITY_TOLERANCE, SKTruncatedSVD};
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
fn truncated_transform_and_inverse_transform_shapes_and_rank_k_reconstruction() {
    // Rank-2 matrix (6 x 5) with non-zero feature means, factorized with k = 3 >= 2.
    let mode_one = array![1.0_f64, 3.0, 5.0, 7.0, 9.0, 11.0];
    let mode_two = array![2.0_f64, -1.0, 4.0, -3.0, 6.0, -2.0];
    let direction_one = array![0.5_f64, 0.5, 0.5, 0.5, 0.0];
    let direction_two = array![0.5_f64, -0.5, 0.5, -0.5, 0.0];
    let train = Array2::from_shape_fn((6, 5), |(row, column)| {
        mode_one[row] * direction_one[column] + mode_two[row] * direction_two[column]
    });
    let query = Array2::from_shape_fn((4, 5), |(row, column)| {
        (row as f64 + 1.5) * direction_one[column] - (row as f64 * 0.5) * direction_two[column]
    });

    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(3)
        .random_seed(7)
        .build()
        .unwrap()
        .fit(&train)
        .unwrap();

    // Transform shape: m x k (4 x 3).
    let projected_query = model.transform(&query).unwrap();
    assert_eq!(projected_query.dim(), (4, 3));

    // Inverse transform shape: m x d (4 x 5), reconstructing the rank-2 query within tolerance.
    let reconstructed_query = model.inverse_transform(&projected_query).unwrap();
    assert_eq!(reconstructed_query.dim(), (4, 5));
    for (actual, expected) in reconstructed_query.iter().zip(query.iter()) {
        assert!((actual - expected).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }

    // Trait dispatch via SKFeatureTransformer agrees identically.
    let via_trait = SKFeatureTransformer::transform(&model, &train).unwrap();
    let reconstructed_train = model.inverse_transform(&via_trait).unwrap();
    for (actual, expected) in reconstructed_train.iter().zip(train.iter()) {
        assert!((actual - expected).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }
}

#[test]
fn truncated_transform_accepts_dense_and_sparse_views_across_all_combinations() {
    let dense_train = Array2::from_shape_fn((24, 10), |(row, column)| {
        if column == row % 10 {
            (row as f64 + 2.0) * 1.3
        } else if column == (row * 3 + 4) % 10 {
            -((row % 5) as f64 + 1.0)
        } else {
            0.0
        }
    });
    let sparse_train = csr_from_dense(&dense_train);
    let dense_query = Array2::from_shape_fn((9, 10), |(row, column)| {
        if (row + column) % 4 == 0 {
            (row as f64 + 1.0) * 0.8
        } else {
            0.0
        }
    });
    let sparse_query = csr_from_dense(&dense_query);

    let estimator = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .random_seed(31)
        .build()
        .unwrap();
    let dense_model = estimator.fit(&dense_train).unwrap();
    let sparse_model = estimator.fit(sparse_train.view()).unwrap();

    // All four dense/sparse fit x transform combinations agree.
    let dense_on_dense = dense_model.transform(&dense_query).unwrap();
    sk_reset_peak_materialized_elements();
    let dense_on_sparse = dense_model.transform(sparse_query.view()).unwrap();
    assert_eq!(sk_peak_materialized_elements(), 9 * 2);
    let sparse_on_dense = sparse_model.transform(&dense_query).unwrap();
    let sparse_on_sparse = sparse_model.transform(sparse_query.view()).unwrap();

    for candidate in [&dense_on_sparse, &sparse_on_dense, &sparse_on_sparse] {
        for (left, right) in dense_on_dense.iter().zip(candidate.iter()) {
            assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
        }
    }

    // Sparse CSR input to inverse_transform matches dense inverse_transform.
    let sparse_projected = csr_from_dense(&dense_on_dense);
    let rebuilt_dense = dense_model.inverse_transform(&dense_on_dense).unwrap();
    let rebuilt_sparse = dense_model
        .inverse_transform(sparse_projected.view())
        .unwrap();
    for (left, right) in rebuilt_dense.iter().zip(rebuilt_sparse.iter()) {
        assert!((left - right).abs() < 1e-12);
    }
}

#[test]
fn truncated_transform_and_inverse_transform_record_transferred_backend_in_spans() {
    let data = array![[1.0_f64, 2.0, 0.5], [3.0, 5.0, 1.5], [5.0, 8.0, 2.5]];
    let model = SKTruncatedSVD::<f64>::build()
        .number_of_components(2)
        .random_seed(3)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();

    let capture = PcaCaptureGuard::begin();
    let projected = model.transform(&data).unwrap();
    let _reconstructed = model.inverse_transform(&projected).unwrap();
    let spans = capture.spans();

    let matching: Vec<_> = spans
        .iter()
        .filter(|(name, fields)| {
            name == "sciencekit.operation"
                && fields.get("operation").map(String::as_str) == Some("transform")
                && fields.get("backend").map(String::as_str) == Some("matrixmultiply")
                && fields.get("mode").map(String::as_str) == Some("InProcessSynchronous")
        })
        .collect();
    assert_eq!(matching.len(), 2);
}
