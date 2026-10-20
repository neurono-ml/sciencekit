//! Companion tests for stacking on sparse inputs (spec `simple-imputer`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::csr;
use crate::simple_imputer::core_implementation::SKSimpleImputer;
use crate::simple_imputer::output::SKImputedOutput;

#[test]
fn sparse_stacking_appends_the_mask_columns() {
    let fitted = csr(
        (2, 2),
        &[(0, 0, 1.0), (0, 1, f64::NAN), (1, 0, 3.0), (1, 1, 5.0)],
    );
    let model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(fitted.view())
        .unwrap();
    assert_eq!(model.indicator().unwrap().scope(), &[1]);
    let features = csr(
        (2, 2),
        &[(0, 0, 1.0), (0, 1, f64::NAN), (1, 0, 3.0), (1, 1, 5.0)],
    );
    match model.transform(features.view()).unwrap() {
        SKImputedOutput::Sparse(matrix) => {
            assert_eq!(matrix.shape(), (2, 3), "two features plus one mask");
            assert_eq!(matrix.get(0, 1), Some(&5.0), "the missing cell is filled");
            assert_eq!(
                matrix.get(0, 2),
                Some(&1.0),
                "the mask follows the features"
            );
            assert_eq!(matrix.get(1, 2), None, "an observed cell is not masked");
        }
        SKImputedOutput::Dense(matrix) => panic!("expected a csr result, found {matrix:?}"),
    }
}

#[test]
fn a_sparse_incremental_batch_extends_the_indicator_scope() {
    let mut model = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0], [3.0, 4.0]])
        .unwrap();
    let batch = csr(
        (2, 2),
        &[(0, 0, 5.0), (0, 1, f64::NAN), (1, 0, 7.0), (1, 1, 8.0)],
    );
    model.partial_fit(batch.view()).unwrap();

    assert_eq!(
        model.indicator().unwrap().scope(),
        &[1],
        "the sparse batch contributes the missing column"
    );
    let output = match model.transform(&array![[f64::NAN, f64::NAN]]).unwrap() {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => panic!("expected a dense result, found {matrix:?}"),
    };
    assert_eq!(output.dim(), (1, 3), "two features plus one mask");
    assert_eq!(output[[0, 2]], 1.0, "the missing column is masked");
}
