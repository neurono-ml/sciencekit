//! Column-major binned store, category mapping and sparse-input tests
//! (task 1.3).

use ndarray::Array2;
use sciencekit_common::SKDataView;
use sprs::CsMat;

use super::dataset::SKBinnedDataset;

#[test]
fn hist_dataset_stores_columns_with_u8_bins() {
    let dense = Array2::from_shape_vec(
        (3, 2),
        vec![0.0, 10.0, 1.0, 20.0, 2.0, 30.0],
    )
    .expect("fixture");
    let dataset = SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 8).expect("build");
    assert_eq!(dataset.number_of_rows(), 3);
    assert_eq!(dataset.number_of_features(), 2);
    assert_eq!(dataset.bins_for_feature(0).len(), 3);
    assert_eq!(dataset.bins_for_feature(1).len(), 3);
    assert_eq!(dataset.bin(0, 0), dataset.bins_for_feature(0)[0]);
    assert!(dataset.bin(1, 2) > dataset.bin(1, 0));
}

#[test]
fn hist_dataset_maps_categories_and_unknown_bin() {
    let dense = Array2::from_shape_vec(
        (4, 1),
        vec![10.0, 20.0, 10.0, 99.0],
    )
    .expect("fixture");
    let dataset = SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[0], 8).expect("build");
    let binning = dataset.feature_binning(0);
    assert!(binning.is_categorical());
    assert_eq!(binning.categories(), &[10, 20, 99]);
    assert_eq!(dataset.bin(0, 0), dataset.bin(0, 2));
    assert_ne!(dataset.bin(0, 0), dataset.bin(0, 1));
    assert_eq!(dataset.bin(0, 3), 2);
    assert_eq!(binning.unknown_bin(), 3);
    assert_eq!(binning.bin(77.0), 3);
}

#[test]
fn hist_dataset_does_not_densify_sparse_input() {
    let dense = Array2::from_shape_vec(
        (3, 2),
        vec![0.0, 5.0, 0.0, 0.0, 2.0, 7.0],
    )
    .expect("fixture");
    let sparse = CsMat::new(
        (3, 2),
        vec![0, 1, 1, 3],
        vec![1, 0, 1],
        vec![5.0, 2.0, 7.0],
    );
    let dense_dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 8).expect("dense build");
    let sparse_dataset =
        SKBinnedDataset::build(&SKDataView::Sparse(sparse.view()), &[], 8).expect("sparse build");
    assert_eq!(sparse_dataset.bins(), dense_dataset.bins());
    assert_eq!(
        sparse_dataset.feature_binning(0).edges(),
        dense_dataset.feature_binning(0).edges()
    );
}
