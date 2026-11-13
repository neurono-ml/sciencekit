//! Weighted versus uniform donor acceptance for the nearest-neighbor imputer.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::{assert_close, dense};
use crate::knn::core_implementation::SKKNNImputer;
use crate::knn::weights::SKImputationWeights;

fn imputed(weights: SKImputationWeights, probe: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let training = array![[0.0_f64, 2.0], [1.0, 4.0], [2.0, 6.0]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .weights(weights)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    dense(model.transform(probe).unwrap())
}

#[test]
fn acceptance_knn_imputer_uniform_and_distance_donors_differ_as_hand_computed() {
    let probe = array![[0.2_f64, f64::NAN]];
    let uniform = imputed(SKImputationWeights::Uniform, &probe);
    let distance = imputed(SKImputationWeights::Distance, &probe);
    assert_close(uniform[[0, 1]], 3.0);
    assert_close(distance[[0, 1]], 2.4);
}

#[test]
fn acceptance_knn_imputer_zero_distance_duplicates_stay_finite() {
    let training = array![[1.0_f64, 2.0], [1.0, 4.0], [5.0, 6.0]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .weights(SKImputationWeights::Distance)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let output = dense(model.transform(&array![[1.0_f64, f64::NAN]]).unwrap());
    assert_close(output[[0, 1]], 3.0);
}
