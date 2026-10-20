//! Companion tests for the sparse incremental fit (spec `simple-imputer`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use super::fixtures::{csr, statistics};
use crate::simple_imputer::core_implementation::SKSimpleImputer;

#[test]
fn sparse_batch_accumulates_like_the_dense_concatenation() {
    let reference = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&array![[4.0], [2.0], [0.0]])
        .unwrap();
    let mut model = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(csr((1, 1), &[(0, 0, 4.0)]).view())
        .unwrap();
    model
        .partial_fit(csr((2, 1), &[(0, 0, 2.0)]).view())
        .unwrap();
    assert_eq!(statistics(&model), statistics(&reference));
}
