//! Acceptance across representations: dense and sparse inputs must produce
//! provably equal statistics, imputations and masks.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sprs::TriMat;

use super::fixtures::{assert_statistics_agree, dense_imputation, dense_mask};
use crate::missing_indicator::{SKIndicatorOutput, SKMissingFeatureScope, SKMissingIndicator};
use crate::simple_imputer::{SKImputedOutput, SKSimpleImputer};

#[test]
fn dense_and_sparse_inputs_produce_equal_outputs() {
    let dense = array![[1.0, 0.0, f64::NAN], [0.0, 5.0, 6.0], [7.0, f64::NAN, 0.0]];
    // The sparse twin stores the explicit values and the missing cells; the
    // structural zeros are observed under the NaN marker.
    let mut triplets = TriMat::new((3, 3));
    for ((row, column), &value) in dense.indexed_iter() {
        if value.is_nan() {
            triplets.add_triplet(row, column, f64::NAN);
        } else if value != 0.0 {
            triplets.add_triplet(row, column, value);
        }
    }
    let sparse = triplets.to_csr();

    let imputer = SKSimpleImputer::<f64>::build()
        .add_indicator(true)
        .build()
        .unwrap();
    let dense_model = imputer.fit(&dense).unwrap();
    let sparse_model = imputer.fit(sparse.view()).unwrap();
    assert_statistics_agree(sparse_model.statistics(), dense_model.statistics());
    assert_eq!(
        sparse_model.dropped_columns(),
        dense_model.dropped_columns()
    );

    let dense_features = dense_imputation(dense_model.transform(&dense).unwrap());
    let sparse_features = match sparse_model.transform(sparse.view()).unwrap() {
        SKImputedOutput::Sparse(matrix) => matrix.to_dense(),
        SKImputedOutput::Dense(matrix) => panic!("expected a csr result, found {matrix:?}"),
    };
    assert_eq!(dense_features, sparse_features, "the stacked results agree");

    let indicator = SKMissingIndicator::<f64>::build()
        .features(SKMissingFeatureScope::All)
        .build()
        .unwrap();
    let dense_mask = dense_mask(indicator.fit(&dense).unwrap().transform(&dense).unwrap());
    let sparse_mask = match indicator
        .fit(sparse.view())
        .unwrap()
        .transform(sparse.view())
        .unwrap()
    {
        SKIndicatorOutput::Sparse(matrix) => matrix.to_dense(),
        SKIndicatorOutput::Dense(matrix) => panic!("expected a csr mask, found {matrix:?}"),
    };
    assert_eq!(dense_mask, sparse_mask, "the masks agree");
}
