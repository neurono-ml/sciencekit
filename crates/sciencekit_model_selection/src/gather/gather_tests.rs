//! Dense and sparse row gathering (spec `model-selection-splits`, requirement
//! `Row gathering`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKDataView, SKError, SKFeatureBlock, SKTargetView};
use sprs::{CsMat, TriMat};

use super::{SKTargetPartition, sk_gather_rows};

fn sk_sparse_fixture() -> CsMat<f64> {
    let mut triplets = TriMat::new((4, 3));
    triplets.add_triplet(0, 1, 1.0);
    triplets.add_triplet(2, 0, 2.0);
    triplets.add_triplet(2, 2, 3.0);
    triplets.add_triplet(3, 1, 4.0);
    triplets.to_csr()
}

#[test]
fn gather_dense_features_returns_selected_rows_in_index_order() {
    let features = array![[0.0_f64, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];

    let (gathered, targets) =
        sk_gather_rows(SKDataView::Dense(features.view()), None, &[3, 0, 2])
            .expect("the dense gather succeeds");

    let SKFeatureBlock::Dense(matrix) = gathered else {
        panic!("a dense input must yield a dense block");
    };
    assert_eq!(matrix, array![[6.0, 7.0], [0.0, 1.0], [4.0, 5.0]]);
    assert!(targets.is_none());
}

#[test]
fn gather_sparse_features_keeps_csr_without_densifying() {
    let sparse = sk_sparse_fixture();
    let stored_entries = sparse.data().len();

    let (gathered, _) = sk_gather_rows(SKDataView::Sparse(sparse.view()), None, &[3, 2])
        .expect("the sparse gather succeeds");

    let SKFeatureBlock::Sparse(matrix) = gathered else {
        panic!("a sparse input must yield a CSR block, never a dense copy");
    };
    assert_eq!((matrix.rows(), matrix.cols()), (2, 3));
    assert_eq!(matrix.data().len(), 3);
    assert_eq!(matrix.data().len(), stored_entries - 1);
    assert_eq!(matrix.outer_view(0).unwrap().data(), &[4.0]);
    assert_eq!(matrix.outer_view(1).unwrap().data(), &[2.0, 3.0]);
    assert_eq!(matrix.outer_view(1).unwrap().indices(), &[0, 2]);
}

#[test]
fn gather_continuous_targets_preserves_values_and_order() {
    let features = Array2::<f64>::zeros((4, 1));
    let targets = Array1::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0]);

    let (_, gathered) = sk_gather_rows(
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Continuous(targets.view())),
        &[2, 0],
    )
    .expect("the target gather succeeds");

    let Some(SKTargetPartition::Continuous(values)) = gathered else {
        panic!("continuous targets must stay continuous");
    };
    assert_eq!(values, Array1::from_vec(vec![30.0, 10.0]));
}

#[test]
fn gather_integer_targets_preserves_values_and_order() {
    let features = Array2::<f64>::zeros((3, 1));
    let targets = Array1::from_vec(vec![4_i64, 5, 6]);

    let (_, gathered) = sk_gather_rows(
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
        &[1, 2, 0],
    )
    .expect("the target gather succeeds");

    let Some(SKTargetPartition::Integer(values)) = gathered else {
        panic!("integer targets must stay integer");
    };
    assert_eq!(values, Array1::from_vec(vec![5, 6, 4]));
}

#[test]
fn gather_nominal_targets_borrows_the_gathered_text() {
    let features = Array2::<f64>::zeros((3, 1));
    let labels = ["red", "green", "blue"];

    let (_, gathered) = sk_gather_rows(
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Nominal(&labels)),
        &[2, 0],
    )
    .expect("the nominal gather succeeds");

    let Some(SKTargetPartition::Nominal(ref values)) = gathered else {
        panic!("nominal targets must stay nominal");
    };
    assert_eq!(values.as_slice(), ["blue".to_owned(), "red".to_owned()]);
    gathered
        .as_ref()
        .expect("the partition is present")
        .with_view(|view| match view {
            SKTargetView::Nominal(text) => assert_eq!(text, &["blue", "red"]),
            _ => panic!("the gathered partition must stay nominal"),
        });
}

#[test]
fn gather_rejects_a_feature_index_outside_the_rows() {
    let features = Array2::<f64>::zeros((3, 2));

    let error = sk_gather_rows(SKDataView::Dense(features.view()), None, &[0, 3])
        .expect_err("an out-of-range row must be rejected");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "row_indices",
            ..
        }
    ));
}

#[test]
fn gather_rejects_a_target_index_outside_the_targets() {
    let features = Array2::<f64>::zeros((3, 2));
    let targets = Array1::from_vec(vec![0_i64, 1, 2]);

    let error = sk_gather_rows(
        SKDataView::Dense(features.view()),
        Some(SKTargetView::Integer(targets.view())),
        &[4],
    )
    .expect_err("an out-of-range target index must be rejected");

    assert!(matches!(
        error,
        SKError::InvalidHyperparameter {
            name: "row_indices",
            ..
        }
    ));
}
