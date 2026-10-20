//! Companion tests for the imputation transform (spec `simple-imputer`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::core_implementation::SKSimpleImputer;
use super::output::SKImputedOutput;
use super::strategy::SKImputationStrategy;

/// Build a CSR matrix from triplets for the sparse cases.
fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

/// The dense matrix behind a dense imputation output.
fn dense(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        other => panic!("expected a dense output, found {other:?}"),
    }
}

/// The sparse matrix behind a sparse imputation output.
fn sparse(output: SKImputedOutput<f64>) -> CsMat<f64> {
    match output {
        SKImputedOutput::Sparse(matrix) => matrix,
        other => panic!("expected a sparse output, found {other:?}"),
    }
}

#[test]
fn dense_fills_only_the_missing_cells() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [3.0, 8.0], [5.0, 10.0]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, 8.0], [3.0, f64::NAN], [7.0, 12.0]])
            .unwrap(),
    );
    assert_eq!(
        imputed,
        array![[3.0, 8.0], [3.0, 9.0], [7.0, 12.0]],
        "the statistic fills the missing cells and leaves the observed ones"
    );
}

#[test]
fn observed_cells_stay_bit_identical() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[0.1, 1.0], [0.1, 1.0 / 3.0]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[0.1, 1.0 / 3.0], [f64::NAN, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(
        imputed[[0, 0]].to_bits(),
        0.1f64.to_bits(),
        "an observed cell is copied, never recomputed"
    );
    assert_eq!(imputed[[0, 1]].to_bits(), (1.0f64 / 3.0).to_bits());
}

#[test]
fn column_count_mismatch_is_rejected() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, 2.0]])
        .unwrap();
    let error = model
        .transform(&array![[1.0, 2.0, 3.0]])
        .expect_err("three columns cannot be transformed by a two-column model");
    assert!(
        error.to_string().contains("shape mismatch"),
        "the error reports the shape contract: {error}"
    );
}

#[test]
fn dense_input_produces_dense_output() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0], [f64::NAN]])
        .unwrap();
    let output = model.transform(&array![[f64::NAN], [2.0]]).unwrap();
    assert_eq!(output.representation(), "dense");
    assert_eq!((output.nrows(), output.ncols()), (2, 1));
}

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
fn all_missing_column_is_dropped_by_default() {
    let model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [2.0, f64::NAN]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, f64::NAN], [2.0, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(imputed, array![[1.5], [2.0]]);
}

#[test]
fn keep_empty_features_retains_the_column_as_zero() {
    let model = SKSimpleImputer::<f64>::build()
        .keep_empty_features(true)
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [2.0, f64::NAN]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, f64::NAN], [2.0, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(imputed, array![[1.5, 0.0], [2.0, 0.0]]);
}

#[test]
fn constant_strategy_fills_every_missing_cell_and_keeps_the_columns() {
    let model = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Constant)
        .fill_value(Some(-1.0))
        .build()
        .unwrap()
        .fit(&array![[1.0, f64::NAN], [f64::NAN, f64::NAN]])
        .unwrap();
    let imputed = dense(
        model
            .transform(&array![[f64::NAN, f64::NAN], [2.0, f64::NAN]])
            .unwrap(),
    );
    assert_eq!(imputed, array![[-1.0, -1.0], [2.0, -1.0]]);
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
