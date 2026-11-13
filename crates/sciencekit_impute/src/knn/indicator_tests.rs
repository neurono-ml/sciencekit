//! Companion tests for the missingness-indicator composition
//! (spec `nearest-neighbor-imputation`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use crate::knn::core_implementation::SKKNNImputer;
use crate::simple_imputer::SKImputedOutput;

fn dense(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => panic!("expected a dense result, found {matrix:?}"),
    }
}

#[test]
fn knn_indicator_stacks_the_mask_after_the_imputed_features() {
    let training = array![[1.0_f64, f64::NAN], [2.0, 3.0], [3.0, 5.0]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    assert_eq!(model.indicator().unwrap().scope(), &[1]);

    let output = dense(
        model
            .transform(&array![[4.0_f64, f64::NAN], [5.0, 6.0]])
            .unwrap(),
    );
    assert_eq!(output, array![[4.0_f64, 4.0, 1.0], [5.0, 6.0, 0.0]]);
}

#[test]
fn knn_indicator_does_not_add_columns_when_disabled() {
    let training = array![[1.0_f64, f64::NAN], [2.0, 3.0], [3.0, 5.0]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    assert!(model.indicator().is_none());
    let output = dense(model.transform(&array![[4.0_f64, f64::NAN]]).unwrap());
    assert_eq!(output.dim(), (1, 2));
}

#[test]
fn knn_indicator_tolerates_missingness_unseen_at_fit() {
    let training = array![[1.0_f64, 2.0], [2.0, 4.0], [3.0, 6.0]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    assert!(model.indicator().unwrap().scope().is_empty());

    let output = dense(model.transform(&array![[2.0_f64, f64::NAN]]).unwrap());
    assert_eq!(output.dim(), (1, 2));
    assert_eq!(output[[0, 0]], 2.0);
    assert_eq!(output[[0, 1]], 3.0);
}

#[test]
fn knn_indicator_masks_a_kept_empty_column() {
    let training = array![[1.0_f64, f64::NAN], [2.0, f64::NAN]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .keep_empty_features(true)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    assert_eq!(model.indicator().unwrap().scope(), &[1]);

    let output = dense(model.transform(&array![[f64::NAN, f64::NAN]]).unwrap());
    assert_eq!(output, array![[1.5_f64, 0.0, 1.0]]);
}
