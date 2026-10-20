//! Companion tests for the owned imputation output (spec `simple-imputer`).

use ndarray::Array2;
use sprs::CsMat;

use super::output::SKImputedOutput;

#[test]
fn dense_variant_exposes_its_shape() {
    let output = SKImputedOutput::<f64>::Dense(Array2::zeros((3, 2)));
    assert_eq!(output.nrows(), 3);
    assert_eq!(output.ncols(), 2);
    assert_eq!(output.representation(), "dense");
}

#[test]
fn sparse_variant_exposes_its_shape() {
    let matrix = CsMat::new((2, 4), vec![0, 1, 2], vec![0, 3], vec![1.0, 1.0]);
    let output = SKImputedOutput::<f64>::Sparse(matrix);
    assert_eq!(output.nrows(), 2);
    assert_eq!(output.ncols(), 4);
    assert_eq!(output.representation(), "csr");
}

#[test]
fn variants_compare_by_value() {
    let dense = SKImputedOutput::<f32>::Dense(Array2::zeros((1, 1)));
    let same = SKImputedOutput::<f32>::Dense(Array2::zeros((1, 1)));
    let other = SKImputedOutput::<f32>::Dense(Array2::from_elem((1, 1), 1.0));
    assert_eq!(dense, same);
    assert_ne!(dense, other);
}
