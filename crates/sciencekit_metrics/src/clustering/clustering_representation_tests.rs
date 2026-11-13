//! Dense/sparse parity for the clustering scores (spec `clustering-metrics`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKClusterLabel, SKDataView};
use sprs::CsMat;

use super::{sk_calinski_harabasz_score, sk_davies_bouldin_score};

fn compact_separated_clusters() -> Array2<f64> {
    array![
        [0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [10.0, 11.0, 3.0],
        [11.0, 10.0, 0.0],
    ]
}

fn two_cluster_assignments() -> Array1<SKClusterLabel> {
    Array1::from_vec(vec![
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(0),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
        SKClusterLabel::cluster(1),
    ])
}

#[test]
fn clustering_representation_davies_bouldin_sparse_equals_dense() {
    let features = compact_separated_clusters();
    let assignments = two_cluster_assignments();
    let dense =
        sk_davies_bouldin_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    let sparse = CsMat::csr_from_dense(features.view(), 0.0);
    let sparse_view: SKDataView<'_, f64> = sparse.view().try_into().unwrap();
    let from_sparse = sk_davies_bouldin_score(sparse_view, assignments.view()).unwrap();
    assert!(
        (dense - from_sparse).abs() < 1e-12,
        "dense {dense} vs sparse {from_sparse}"
    );
}

#[test]
fn clustering_representation_calinski_harabasz_sparse_equals_dense() {
    let features = compact_separated_clusters();
    let assignments = two_cluster_assignments();
    let dense =
        sk_calinski_harabasz_score(SKDataView::Dense(features.view()), assignments.view()).unwrap();
    let sparse = CsMat::csr_from_dense(features.view(), 0.0);
    let sparse_view: SKDataView<'_, f64> = sparse.view().try_into().unwrap();
    let from_sparse = sk_calinski_harabasz_score(sparse_view, assignments.view()).unwrap();
    assert!(
        (dense - from_sparse).abs() < 1e-9 * dense.abs().max(1.0),
        "dense {dense} vs sparse {from_sparse}"
    );
}

#[test]
fn clustering_representation_scores_a_wide_sparse_input_without_densifying() {
    // 500_000 rows × 1_024 columns with one non-zero each: a dense copy would
    // need 4 GiB, while the sparse input is a few megabytes. Completing proves
    // the centroid path reads the CSR representation directly.
    let number_of_rows = 500_000_usize;
    let number_of_features = 1_024_usize;
    let mut indptr = Vec::with_capacity(number_of_rows + 1);
    indptr.push(0_usize);
    let mut indices = Vec::with_capacity(number_of_rows);
    let mut data = Vec::with_capacity(number_of_rows);
    for row in 0..number_of_rows {
        indices.push(row % number_of_features);
        data.push(1.0_f64);
        indptr.push(row + 1);
    }
    let sparse = CsMat::new((number_of_rows, number_of_features), indptr, indices, data);
    let assignments = Array1::from_vec(
        (0..number_of_rows)
            .map(|row| SKClusterLabel::cluster((row % 2) as u32))
            .collect(),
    );
    let sparse_view: SKDataView<'_, f64> = sparse.view().try_into().unwrap();

    let davies_bouldin = sk_davies_bouldin_score(sparse_view, assignments.view()).unwrap();
    let calinski_harabasz = sk_calinski_harabasz_score(sparse_view, assignments.view()).unwrap();
    assert!(davies_bouldin.is_finite());
    assert!(davies_bouldin >= 0.0);
    assert!(calinski_harabasz.is_finite());
    assert!(calinski_harabasz > 0.0);
}
