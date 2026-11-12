//! Validation of non-finite feature and target values.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKError, SKTargetView};
use sprs::TriMat;

use crate::SKSplitter;
use crate::split::kfold::SKKFold;

#[test]
fn validation_rejects_non_finite_dense_features() {
    let mut features = Array2::<f64>::zeros((3, 2));
    features[[1, 0]] = f64::NAN;
    let kfold = SKKFold::build().number_of_splits(2).build().unwrap();

    let error = kfold
        .split(SKDataView::Dense(features.view()), None, None)
        .expect_err("a NaN feature value must be rejected");

    assert!(matches!(error, SKError::NonFiniteValue { position: 2 }));
}

#[test]
fn validation_rejects_non_finite_sparse_features() {
    let mut triplets = TriMat::new((3, 3));
    triplets.add_triplet(2, 1, f64::INFINITY);
    let sparse = triplets.to_csr();
    let kfold = SKKFold::build().number_of_splits(2).build().unwrap();

    let error = kfold
        .split(SKDataView::Sparse(sparse.view()), None, None)
        .expect_err("an infinite sparse value must be rejected");

    assert!(matches!(error, SKError::NonFiniteValue { position: 7 }));
}

#[test]
fn validation_rejects_non_finite_continuous_targets() {
    let features = Array2::<f64>::zeros((4, 2));
    let targets = Array1::from_vec(vec![0.0_f64, f64::NAN, 2.0, 3.0]);
    let kfold = SKKFold::build().number_of_splits(2).build().unwrap();

    let error = kfold
        .split(
            SKDataView::Dense(features.view()),
            Some(SKTargetView::Continuous(targets.view())),
            None,
        )
        .expect_err("a NaN target must be rejected");

    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}
