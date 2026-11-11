//! Companion tests for `transform`, `inverse_transform`, `fit_transform` and
//! whitening on `SKPCA` and `SKPCAModel`.

use ndarray::{Array2, Axis, array};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKExecutionMode, SKFeatureTransformer, SKUnsupervisedFit,
};
use sprs::{CsMat, TriMat};

use super::SKPCA;
use super::test_tracing::PcaCaptureGuard;

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
fn pca_transform_projects_rows_with_expected_shape_and_matches_fit_transform() {
    let training = array![
        [1.0_f64, 2.0, 0.5, -1.0],
        [2.5, 4.1, 1.2, -0.5],
        [4.0, 6.2, 2.1, 0.2],
        [5.5, 8.0, 2.9, 1.0],
        [7.0, 10.3, 3.8, 1.6],
        [8.5, 12.1, 4.5, 2.3]
    ];
    let query = array![
        [0.0_f64, 1.5, 0.0, -2.0],
        [3.0, 0.0, 2.0, 0.0],
        [6.0, 9.0, 0.0, 1.5]
    ];
    let estimator = SKPCA::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap();
    let model = estimator.fit(&training).unwrap();

    let projected = model.transform(&query).unwrap();
    assert_eq!(projected.dim(), (3, 2));

    let via_trait = SKFeatureTransformer::transform(&model, &query).unwrap();
    assert_eq!(projected, via_trait);

    let sparse_query = csr_from_dense(&query);
    let projected_sparse = model.transform(sparse_query.view()).unwrap();
    for (left, right) in projected.iter().zip(projected_sparse.iter()) {
        assert!((left - right).abs() < 1e-12);
    }

    let combined = estimator.fit_transform(&training).unwrap();
    let separate = model.transform(&training).unwrap();
    for (left, right) in combined.iter().zip(separate.iter()) {
        assert!((left - right).abs() < 1e-12);
    }
}

#[test]
fn pca_transform_full_components_inverse_transform_reconstructs_training_data() {
    let data = array![
        [2.0_f64, -1.0, 4.0],
        [5.0, 2.5, 1.0],
        [-3.0, 0.5, 7.0],
        [4.0, -3.5, -2.0],
        [1.0, 6.0, 3.0]
    ];
    let model = SKPCA::<f64>::build().build().unwrap().fit(&data).unwrap();
    let projected = model.transform(&data).unwrap();
    let reconstructed = model.inverse_transform(&projected).unwrap();

    for (left, right) in reconstructed.iter().zip(data.iter()) {
        assert!((left - right).abs() < 1e-11);
    }

    let mean_row = model.mean().insert_axis(Axis(0));
    let centered_original = &data - &mean_row;
    let centered_reconstructed = &reconstructed - &mean_row;
    for (left, right) in centered_reconstructed.iter().zip(centered_original.iter()) {
        assert!((left - right).abs() < 1e-11);
    }
}

#[test]
fn pca_transform_whitening_normalizes_component_variance_and_handles_near_zero_variance() {
    let data = Array2::from_shape_fn((40, 3), |(row, column)| {
        let row_scalar = row as f64;
        match column {
            0 => 4.0 * (0.21 * row_scalar).sin() + 10.0,
            1 => 1.5 * (0.37 * row_scalar + 0.4).cos() - 3.0,
            _ => 0.35 * ((row % 7) as f64 - 3.0) + 2.0,
        }
    });
    let model = SKPCA::<f64>::build()
        .whiten(true)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let whitened = model.transform(&data).unwrap();
    let degrees_of_freedom = (data.nrows() - 1) as f64;

    for column in whitened.columns() {
        let mean = column.mean().unwrap();
        assert!(mean.abs() < 1e-11);
        let sample_variance =
            column.mapv(|value| (value - mean) * (value - mean)).sum() / degrees_of_freedom;
        assert!((sample_variance - 1.0).abs() < 1e-10);
    }

    let round_trip = model.inverse_transform(&whitened).unwrap();
    for (left, right) in round_trip.iter().zip(data.iter()) {
        assert!((left - right).abs() < 1e-10);
    }

    // Rank-deficient / near-zero variance input stays finite under whitening.
    let collinear = array![
        [1.0_f64, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0]
    ];
    let rank_deficient = SKPCA::<f64>::build()
        .whiten(true)
        .build()
        .unwrap()
        .fit(&collinear)
        .unwrap();
    let whitened_collinear = rank_deficient.transform(&collinear).unwrap();
    assert!(whitened_collinear.iter().all(|value| value.is_finite()));
    let rebuilt_collinear = rank_deficient
        .inverse_transform(&whitened_collinear)
        .unwrap();
    assert!(rebuilt_collinear.iter().all(|value| value.is_finite()));
}

#[test]
fn pca_transform_records_model_backend_in_transform_and_fit_transform_spans() {
    let data = array![
        [1.0_f64, 3.0, 2.0],
        [2.0, 5.0, 4.0],
        [3.0, 7.0, 5.0],
        [4.0, 9.0, 8.0]
    ];
    let estimator = SKPCA::<f64>::build()
        .number_of_components(2)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    let model = estimator.fit(&data).unwrap();

    let capture = PcaCaptureGuard::begin();
    let projected = model.transform(&data).unwrap();
    let _ = model.inverse_transform(&projected).unwrap();
    let _ = estimator.fit_transform(&data).unwrap();
    let spans = capture.spans();

    let transform_spans = spans
        .iter()
        .filter(|(name, fields)| {
            name == "sciencekit.operation"
                && fields.get("operation").map(String::as_str) == Some("transform")
                && fields.get("backend").map(String::as_str) == Some("matrixmultiply")
        })
        .count();
    assert_eq!(transform_spans, 2);
    assert!(spans.iter().any(|(name, fields)| {
        name == "sciencekit.operation"
            && fields.get("operation").map(String::as_str) == Some("fit_transform")
            && fields.get("backend").map(String::as_str) == Some("matrixmultiply")
    }));
}
