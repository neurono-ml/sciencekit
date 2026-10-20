//! Companion tests for the sparse transform (spec `simple-imputer`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::{csr, sparse};
use crate::simple_imputer::core_implementation::SKSimpleImputer;

#[test]
fn csr_input_produces_csr_output() {
    let features = csr((3, 2), &[(0, 0, 2.0), (1, 1, 4.0)]);
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    let output = model.transform(features.view()).unwrap();
    assert_eq!(output.representation(), "csr");
    assert_eq!((output.nrows(), output.ncols()), (3, 2));
}

#[test]
fn sparse_fill_stores_only_the_filled_cells() {
    // Column 0 observes 2.0 explicitly, NaN explicitly (missing) and an
    // implicit zero (observed under a NaN marker): mean = 1.0.
    let features = csr((3, 1), &[(0, 0, 2.0), (1, 0, f64::NAN)]);
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    let imputed = sparse(model.transform(features.view()).unwrap());
    assert_eq!(
        imputed.nnz(),
        2,
        "only the filled cell joins the observed one"
    );
    assert_eq!(imputed.get(0, 0), Some(&2.0));
    assert_eq!(imputed.get(1, 0), Some(&1.0));
    assert_eq!(
        imputed.get(2, 0),
        None,
        "the observed implicit zero stays implicit"
    );
    assert_eq!(imputed.to_dense(), array![[2.0], [1.0], [0.0]]);
}

#[test]
fn zero_marker_fills_the_implicit_zeros() {
    let features = csr((3, 1), &[(0, 0, 4.0)]);
    let model = SKSimpleImputer::<f64>::build()
        .missing_values(0.0)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    let imputed = sparse(model.transform(features.view()).unwrap());
    assert_eq!(
        imputed.nnz(),
        3,
        "every structural zero was missing under a zero marker"
    );
    assert_eq!(imputed.to_dense(), array![[4.0], [4.0], [4.0]]);
}

#[test]
fn sparse_all_missing_column_is_dropped_by_default() {
    let features = csr(
        (2, 2),
        &[(0, 0, 1.0), (1, 0, 2.0), (0, 1, f64::NAN), (1, 1, f64::NAN)],
    );
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    let imputed = sparse(model.transform(features.view()).unwrap());
    assert_eq!((imputed.rows(), imputed.cols()), (2, 1));
    assert_eq!(imputed.to_dense(), array![[1.0], [2.0]]);
}
