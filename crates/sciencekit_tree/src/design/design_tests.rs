//! Representation tests for the fit design (spec `data-view-boundary`).

use ndarray::array;
use sprs::CsMat;

use super::SKDesign;

/// A CSR design exposes the same logical values as its dense twin: stored
/// entries are read directly and absent entries read as the observed zero.
#[test]
fn sparse_design_reads_stored_entries_and_implicit_zeros() {
    // Dense twin of the CSR matrix below.
    let dense = array![[1.0_f64, 0.0, 2.0], [0.0, 3.0, 0.0]];
    let sparse = CsMat::new(
        (2, 3),
        vec![0, 2, 3],
        vec![0, 2, 1],
        vec![1.0_f64, 2.0, 3.0],
    );
    let csc = super::sk_to_csc(&sparse.view());
    let design = SKDesign::Sparse {
        csr: sparse.view(),
        csc: csc.view(),
    };
    assert!(!design.is_dense());
    for row in 0..2 {
        for column in 0..3 {
            assert_eq!(design.feature_value(row, column), dense[[row, column]]);
        }
    }
    let values = design.sorted_feature_values(&[0, 1], 2);
    assert_eq!(values, vec![(0.0, 1), (2.0, 0)]);
}

/// Non-finite stored entries are rejected at their flattened position.
#[test]
fn sparse_design_rejects_non_finite_stored_values() {
    let sparse = CsMat::new((1, 3), vec![0, 1], vec![2], vec![f64::NAN]);
    let csc = super::sk_to_csc(&sparse.view());
    let design = SKDesign::Sparse {
        csr: sparse.view(),
        csc: csc.view(),
    };
    let error = design.validate_finite().expect_err("NaN must be rejected");
    assert!(matches!(
        error,
        sciencekit_common::SKError::NonFiniteValue { position: 2 }
    ));
}
