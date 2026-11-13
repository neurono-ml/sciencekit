//! Companion TDD tests for owned support-vector blocks (spec
//! `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_math::{SKKernel, SKKernelParameters};
use sprs::{CsMat, TriMat};

use super::{SKOwnedFeatureBlock, sk_evaluate_kernel_diagonal};
use crate::{SKSVC, SKSVR};

#[test]
fn owned_feature_block_preserves_csr_representation_without_densifying() {
    let csr = CsMat::new(
        (4, 50_000_000),
        vec![0, 1, 3, 3, 4],
        vec![7, 12, 49_999_999, 100],
        vec![2.0_f64, -3.0, 5.0, 1.5],
    );
    let block = SKOwnedFeatureBlock::select_rows(SKDataView::Sparse(csr.view()), &[1, 3]).unwrap();
    assert!(block.is_sparse());
    assert_eq!(block.rows(), 2);
    assert_eq!(block.columns(), 50_000_000);

    let dense = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let dense_block =
        SKOwnedFeatureBlock::select_rows(SKDataView::Dense(dense.view()), &[0, 2]).unwrap();
    assert!(!dense_block.is_sparse());
    assert_eq!(dense_block.rows(), 2);
}

#[test]
fn precomputed_kernel_rejects_sparse_before_scanning_elements_and_on_empty_support_vectors() {
    let mut nan_triplets = TriMat::new((2, 2));
    nan_triplets.add_triplet(0, 0, f64::NAN);
    let nan_csr = nan_triplets.to_csr();

    let fit_error = SKSVC::<f64>::build()
        .kernel(SKKernel::Precomputed)
        .build()
        .unwrap()
        .fit(nan_csr.view(), array![0_i64, 1].view())
        .unwrap_err();
    assert!(matches!(
        fit_error,
        SKError::UnsupportedRepresentation { .. }
    ));

    // Wide epsilon yields 0 support vectors on a valid precomputed Gram matrix;
    // predicting on a sparse query view must still fail with UnsupportedRepresentation.
    let gram = array![[1.0_f64, 0.2], [0.2, 1.0]];
    let svr_model = SKSVR::<f64>::build()
        .kernel(SKKernel::Precomputed)
        .epsilon(10.0)
        .build()
        .unwrap()
        .fit(gram.view(), array![0.1_f64, -0.1].view())
        .unwrap();
    assert_eq!(svr_model.support_vectors().rows(), 0);

    let mut query_triplets = TriMat::new((1, 2));
    query_triplets.add_triplet(0, 0, 1.0_f64);
    let query_csr = query_triplets.to_csr();
    let predict_error = svr_model.predict(query_csr.view()).unwrap_err();
    assert!(matches!(
        predict_error,
        SKError::UnsupportedRepresentation { .. }
    ));

    // RadialBasisFunction diagonal evaluation also rejects CSC views up front.
    let csc = query_triplets.to_csc();
    let rbf_parameters = SKKernelParameters::new(SKKernel::RadialBasisFunction, 0.5_f64, 1, 0.0);
    let diagonal_error =
        sk_evaluate_kernel_diagonal(SKDataView::Sparse(csc.view()), &rbf_parameters).unwrap_err();
    assert!(matches!(
        diagonal_error,
        SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));
}
