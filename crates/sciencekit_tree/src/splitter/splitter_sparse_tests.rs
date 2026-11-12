//! Sparse split-scan tests (spec `data-view-boundary`, `decision-trees`).

use ndarray::{Array2, array};
use sprs::CsMat;

use super::{SKScanContext, SKThresholdChoice, sk_scan_feature};
use crate::criterion::SKTreeCriterion;
use crate::design::SKDesign;

/// The logical fixture: feature 0 separates zeros from `3, 4`, feature 1
/// separates `1, 2` from the implicit zeros, feature 2 is constant zero.
fn fixtures() -> (Array2<f64>, CsMat<f64>, Vec<usize>) {
    let dense = array![
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ];
    let sparse = CsMat::new(
        (4, 3),
        vec![0, 1, 2, 3, 4],
        vec![1, 1, 0, 0],
        vec![1.0, 2.0, 3.0, 4.0],
    );
    let classes = vec![0, 0, 1, 1];
    (dense, sparse, classes)
}

fn context<'a>(classes: &'a [usize], weights: &'a [f64]) -> SKScanContext<'a, f64> {
    SKScanContext::Classification {
        class_indices: classes,
        sample_weights: weights,
        number_of_classes: 2,
        criterion: SKTreeCriterion::Gini,
    }
}

/// Dense and sparse scans agree on every feature, including the features whose
/// zeros are implicit in the CSR matrix.
#[test]
fn sparse_scan_matches_the_dense_scan() {
    let (dense, sparse, classes) = fixtures();
    let csc = crate::design::sk_to_csc(&sparse.view());
    let dense_design = SKDesign::Dense(dense.view());
    let sparse_design = SKDesign::Sparse {
        csr: sparse.view(),
        csc: csc.view(),
    };
    let weights = [1.0; 4];
    let indices = [0, 1, 2, 3];
    for feature in 0..3 {
        let dense_candidate = sk_scan_feature(
            &dense_design,
            &indices,
            feature,
            1,
            context(&classes, &weights),
            SKThresholdChoice::Best,
        );
        let sparse_candidate = sk_scan_feature(
            &sparse_design,
            &indices,
            feature,
            1,
            context(&classes, &weights),
            SKThresholdChoice::Best,
        );
        assert_eq!(dense_candidate, sparse_candidate, "feature {feature}");
    }
}

/// A sparse design enters the scan as a sparse representation, never as a
/// densified array.
#[test]
fn sparse_design_stays_sparse() {
    let (_dense, sparse, _classes) = fixtures();
    let csc = crate::design::sk_to_csc(&sparse.view());
    let design = SKDesign::Sparse {
        csr: sparse.view(),
        csc: csc.view(),
    };
    assert!(!design.is_dense());
    assert_eq!(design.rows(), 4);
    assert_eq!(design.columns(), 3);
}

/// The sparse scan recovers the threshold that separates implicit zeros from
/// stored positive values.
#[test]
fn sparse_scan_recovers_the_implicit_zero_split() {
    let (_dense, sparse, classes) = fixtures();
    let csc = crate::design::sk_to_csc(&sparse.view());
    let design = SKDesign::Sparse {
        csr: sparse.view(),
        csc: csc.view(),
    };
    let weights = [1.0; 4];
    let candidate = sk_scan_feature(
        &design,
        &[0, 1, 2, 3],
        0,
        1,
        context(&classes, &weights),
        SKThresholdChoice::Best,
    )
    .expect("a split exists");
    assert_eq!(candidate.threshold(), 1.5);
    assert!((candidate.impurity_decrease() - 0.5).abs() < 1e-12);
}
