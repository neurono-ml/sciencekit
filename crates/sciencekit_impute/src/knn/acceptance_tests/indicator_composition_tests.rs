//! Indicator-composition acceptance for the nearest-neighbor imputer.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::dense;
use crate::knn::core_implementation::SKKNNImputer;

fn training() -> Array2<f64> {
    Array2::from_shape_fn((80, 3), |(row, column)| {
        if column == 2 && row % 5 == 0 {
            f64::NAN
        } else {
            row as f64 * 0.25 + column as f64
        }
    })
}

#[test]
fn acceptance_knn_imputer_indicator_follows_the_imputed_features() {
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(4)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(training().view())
        .unwrap();
    assert_eq!(model.indicator().unwrap().scope(), &[2]);

    let probe = array![[10.0_f64, 11.0, f64::NAN], [20.0, 21.0, 22.0]];
    let output = dense(model.transform(&probe).unwrap());
    assert_eq!(output.dim(), (2, 4));
    assert!(output[[0, 2]].is_finite());
    assert_eq!(output[[0, 3]], 1.0, "the missing feature is masked");
    assert_eq!(output[[1, 3]], 0.0, "a complete row is not masked");
}

#[test]
fn acceptance_knn_imputer_indicator_tolerates_new_missingness() {
    let complete = Array2::from_shape_fn((40, 3), |(row, column)| row as f64 * 0.5 + column as f64);
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(complete.view())
        .unwrap();
    assert!(model.indicator().unwrap().scope().is_empty());

    let probe = array![[f64::NAN, 6.0, 7.0]];
    let output = dense(model.transform(&probe).unwrap());
    assert_eq!(output.dim(), (1, 3));
    assert!(output[[0, 0]].is_finite());
}
