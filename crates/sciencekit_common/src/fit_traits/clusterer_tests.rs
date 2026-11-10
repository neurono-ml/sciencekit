//! Tests for the typed cluster assignment contract (spec `clustering-contracts`).

use ndarray::{Array1, array};

use super::{
    SKClusterIdentifier, SKClusterLabel, SKClustererPredictor, sk_validate_cluster_assignments,
};
use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;

#[test]
fn cluster_zero_is_a_cluster_not_noise() {
    let label = SKClusterLabel::cluster(0);
    assert!(!label.is_noise());
    assert_eq!(label.cluster_index(), Some(0));
}

#[test]
fn a_non_zero_cluster_keeps_its_index() {
    let label = SKClusterLabel::cluster(7);
    assert_eq!(label.cluster_index(), Some(7));
    assert!(!label.is_noise());
}

#[test]
fn noise_is_explicit_and_has_no_cluster_index() {
    let label = SKClusterLabel::NOISE;
    assert!(label.is_noise());
    assert_eq!(label.cluster_index(), None);
}

#[test]
fn the_label_fits_in_four_bytes() {
    assert_eq!(core::mem::size_of::<SKClusterLabel>(), 4);
    assert_eq!(core::mem::size_of::<SKClusterIdentifier>(), 4);
}

#[test]
fn the_i64_conversion_round_trips() {
    assert_eq!(SKClusterLabel::cluster(0).to_i64(), 0);
    assert_eq!(SKClusterLabel::cluster(42).to_i64(), 42);
    assert_eq!(SKClusterLabel::NOISE.to_i64(), -1);
    for label in [
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(9),
        SKClusterLabel::NOISE,
    ] {
        assert_eq!(SKClusterLabel::from_i64(label.to_i64()).unwrap(), label);
    }
}

#[test]
fn an_external_value_below_minus_one_is_rejected() {
    let error = SKClusterLabel::from_i64(-2).unwrap_err();
    assert!(matches!(error, SKError::Conversion(_)));
}

#[test]
fn fast_and_sparse_views_are_the_same_boundary() {
    // A constant clusterer returning one cluster per row: dense and sparse
    // inputs go through the same method and produce one assignment per row.
    let model = ConstantClusterer;
    let dense = Array1::from_vec(vec![0.0_f64; 6])
        .to_shape((3, 2))
        .unwrap()
        .to_owned();
    let sparse = sprs::CsMat::new((3, 2), vec![0, 1, 2, 3], vec![0, 1, 0], vec![1.0; 3]);

    let dense_labels = model.predict_clusters(&dense).unwrap();
    let sparse_labels = model.predict_clusters(sparse.view()).unwrap();
    assert_eq!(dense_labels.len(), 3);
    assert_eq!(sparse_labels.len(), 3);
    assert!(
        dense_labels
            .iter()
            .all(|label| label.cluster_index() == Some(0))
    );
}

#[test]
fn a_feature_count_mismatch_is_rejected() {
    let model = ConstantClusterer;
    let wrong = Array1::from_vec(vec![0.0_f64; 3])
        .to_shape((1, 3))
        .unwrap()
        .to_owned();
    let error = model.predict_clusters(&wrong).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn the_assignment_shape_helper_accepts_and_rejects() {
    let assignments = array![SKClusterLabel::cluster(0), SKClusterLabel::NOISE];
    assert!(sk_validate_cluster_assignments(2, assignments.view()).is_ok());

    let error = sk_validate_cluster_assignments(3, assignments.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn a_shared_model_serves_several_threads() {
    let model = ConstantClusterer;
    let dense = Array1::from_vec(vec![0.0_f64; 8])
        .to_shape((4, 2))
        .unwrap()
        .to_owned();
    let sequential = model.predict_clusters(&dense).unwrap();
    let observed = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| model.predict_clusters(&dense).unwrap()))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for result in observed {
        assert_eq!(result, sequential);
    }
}

/// A minimal fitted clusterer: one cluster for every row, two features.
struct ConstantClusterer;

impl<F: SKFloat> SKClustererPredictor<F> for ConstantClusterer {
    type Error = SKError;

    fn predict_clusters<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let number_of_rows = match view {
            SKDataView::Dense(dense) => {
                if dense.ncols() != 2 {
                    return Err(SKError::shape_mismatch_2d(dense.ncols(), 1, 2, 1));
                }
                dense.nrows()
            }
            SKDataView::Sparse(sparse) => {
                if sparse.cols() != 2 {
                    return Err(SKError::shape_mismatch_2d(sparse.cols(), 1, 2, 1));
                }
                sparse.rows()
            }
        };
        Ok(Array1::from_elem(
            number_of_rows,
            SKClusterLabel::cluster(0),
        ))
    }
}
