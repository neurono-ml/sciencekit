//! Tiny-data acceptance for the nearest-neighbor imputer.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::dense;
use crate::knn::core_implementation::SKKNNImputer;

#[test]
fn acceptance_knn_imputer_one_receiver_and_one_donor() {
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap()
        .fit(&array![[1.0_f64, 2.0]])
        .unwrap();
    let output = dense(model.transform(&array![[f64::NAN, f64::NAN]]).unwrap());
    assert_eq!(output, array![[1.0_f64, 2.0]]);
}

#[test]
fn acceptance_knn_imputer_a_single_missing_cell_in_a_single_row() {
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(1)
        .build()
        .unwrap()
        .fit(&array![[1.0_f64, 2.0], [4.0, 8.0]])
        .unwrap();
    let output = dense(model.transform(&array![[3.0_f64, f64::NAN]]).unwrap());
    assert_eq!(output, array![[3.0_f64, 8.0]]);
}
