//! Companion tests for dense/sparse CSR representation parity and the
//! no-densification guarantee (`spec` `truncated-svd` and `data-view-boundary`).

use ndarray::Array2;
use sciencekit_common::{SKBackendKind, SKBuilderState, SKDataView, SKError};
use sprs::{CsMat, TriMat};

use super::SK_TRUNCATED_SVD_PARITY_TOLERANCE;
use super::config::{SKTruncatedSVDAlgorithm, SKTruncatedSVDConfiguration};
use super::fit::sk_fit_with_projection;
use super::product::{sk_peak_materialized_elements, sk_reset_peak_materialized_elements};
use crate::pca::test_tracing::PcaCaptureGuard;

fn csr_from_dense_with_explicit_zero(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 || (row == 0 && column == 0) {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

#[test]
fn truncated_representation_sparse_csr_equals_dense_within_documented_tolerance() {
    assert_eq!(SK_TRUNCATED_SVD_PARITY_TOLERANCE, 1e-6);

    let dense = Array2::from_shape_fn((40, 12), |(row, column)| {
        if column == row % 12 {
            (row as f64 + 1.0) * 1.4
        } else if column == (row * 5 + 3) % 12 {
            -((row % 6) as f64 + 0.8) * 1.9
        } else {
            0.0
        }
    });
    let sparse = csr_from_dense_with_explicit_zero(&dense);

    let configuration = SKTruncatedSVDConfiguration {
        number_of_components: 3,
        algorithm: SKTruncatedSVDAlgorithm::Randomized,
        number_of_iterations: 5,
        tolerance: 0.0,
        random_seed: 23,
        backend: SKBackendKind::Faer,
        builder_state: SKBuilderState::new(),
    };

    let (dense_model, dense_projected) =
        sk_fit_with_projection(&configuration, SKDataView::Dense(dense.view()), "fit").unwrap();
    let (sparse_model, sparse_projected) =
        sk_fit_with_projection(&configuration, SKDataView::Sparse(sparse.view()), "fit").unwrap();

    for (left, right) in dense_model
        .components()
        .iter()
        .zip(sparse_model.components().iter())
    {
        assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }
    for (left, right) in dense_model
        .singular_values()
        .iter()
        .zip(sparse_model.singular_values().iter())
    {
        assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }
    for (left, right) in dense_model
        .explained_variance()
        .iter()
        .zip(sparse_model.explained_variance().iter())
    {
        assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }
    for (left, right) in dense_model
        .explained_variance_ratio()
        .iter()
        .zip(sparse_model.explained_variance_ratio().iter())
    {
        assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }
    for (left, right) in dense_projected.iter().zip(sparse_projected.iter()) {
        assert!((left - right).abs() < SK_TRUNCATED_SVD_PARITY_TOLERANCE);
    }
}

#[test]
fn truncated_representation_wide_sparse_matrix_fits_without_densification() {
    // 1_500 x 25_000 logical matrix (300 MB if densified as f64) with only
    // 6_000 non-zeros (~96 KB CSR). Fitting and projecting touch only stored
    // entries and thin k+oversampling factors, emitting zero warnings.
    let (rows, columns) = (1_500usize, 25_000usize);
    let mut triplet = TriMat::with_capacity((rows, columns), rows * 4);
    for row in 0..rows {
        let topic_one = ((row % 13) as f64 + 1.0) * 1.5;
        let topic_two = ((row % 7) as f64 + 0.5) * 0.9;
        let tail_column = 4 + (row * 97) % (columns - 4);
        triplet.add_triplet(row, 0, topic_one * 3.0);
        triplet.add_triplet(row, 1, topic_one * 1.5);
        triplet.add_triplet(row, 2 + (row % 2), topic_two);
        triplet.add_triplet(row, tail_column, 0.05 * ((row % 5) as f64 + 1.0));
    }
    let sparse = triplet.to_csr();

    let configuration = SKTruncatedSVDConfiguration {
        number_of_components: 3,
        algorithm: SKTruncatedSVDAlgorithm::Randomized,
        number_of_iterations: 4,
        tolerance: 0.0,
        random_seed: 19,
        backend: SKBackendKind::Faer,
        builder_state: SKBuilderState::new(),
    };

    let capture = PcaCaptureGuard::begin();
    sk_reset_peak_materialized_elements();
    let (model, projected) =
        sk_fit_with_projection(&configuration, SKDataView::Sparse(sparse.view()), "fit").unwrap();
    let peak_elements = sk_peak_materialized_elements();
    assert!(capture.warning_events().is_empty());
    // Full dense materialization would allocate rows * columns = 37_500_000 elements;
    // the sparse path only materializes rows * k = 4_500 projected elements.
    assert_eq!(peak_elements, rows * 3);
    assert!(peak_elements < (rows * columns) / 1_000);
    assert_eq!(model.components().dim(), (3, columns));
    assert_eq!(projected.dim(), (rows, 3));
    assert!(projected.iter().all(|value| value.is_finite()));
    assert!(model.explained_variance_ratio().sum() <= 1.0);
}

#[test]
fn truncated_representation_rejects_csc_up_front_before_scanning_elements() {
    let mut triplet = TriMat::new((4, 4));
    triplet.add_triplet(0, 1, 2.0_f64);
    triplet.add_triplet(2, 3, f64::NAN);
    let csc = triplet.to_csc();
    let view = SKDataView::<f64>::try_from(csc.view());

    match view {
        Err(SKError::UnsupportedRepresentation { representation, .. }) => {
            assert_eq!(representation, "csc");
        }
        Ok(converted) => {
            let configuration = SKTruncatedSVDConfiguration {
                number_of_components: 2,
                algorithm: SKTruncatedSVDAlgorithm::Randomized,
                number_of_iterations: 5,
                tolerance: 0.0,
                random_seed: 0,
                backend: SKBackendKind::Faer,
                builder_state: SKBuilderState::new(),
            };
            let error = sk_fit_with_projection(&configuration, converted, "fit").unwrap_err();
            assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
        }
        Err(other) => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}
