//! Companion tests for the randomized SVD result container.

use ndarray::array;

use super::result::SKRandomizedSvdResult;

#[test]
fn randomized_result_exposes_the_truncated_factors() {
    let result = SKRandomizedSvdResult {
        u: array![[1.0, 0.0], [0.0, 1.0]],
        singular_values: vec![2.0, 1.0],
        v: array![[1.0, 0.0], [0.0, 1.0]],
    };
    assert_eq!(result.number_of_components(), 2);
    assert_eq!(result.u.dim(), (2, 2));
    assert_eq!(result.v.dim(), (2, 2));
}

#[test]
fn randomized_result_reconstructs_the_truncated_matrix() {
    let result = SKRandomizedSvdResult {
        u: array![[1.0, 0.0], [0.0, 1.0]],
        singular_values: vec![3.0, 1.0],
        v: array![[1.0, 0.0], [0.0, 1.0]],
    };
    let reconstructed = result.reconstruct();
    assert_eq!(reconstructed, array![[3.0, 0.0], [0.0, 1.0]]);
}
