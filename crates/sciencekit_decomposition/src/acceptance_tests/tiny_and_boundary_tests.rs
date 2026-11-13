//! Acceptance tests for tiny data, `f32`/`f64` scalars, component-count
//! boundaries and whitening.

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use crate::{SKPCA, SKPCASolver};

#[test]
fn acceptance_pca_tiny_dense_and_single_precision_f32() {
    // 1 x 3 degenerate single-observation boundary.
    let single_row = array![[3.0_f64, -1.0, 5.0]];
    let single_model = SKPCA::<f64>::build()
        .build()
        .unwrap()
        .fit(&single_row)
        .unwrap();
    assert_eq!(single_model.number_of_components(), 1);
    assert_eq!(single_model.explained_variance()[0], 0.0);
    assert_eq!(single_model.explained_variance_ratio()[0], 0.0);

    // 2 x 2 exact boundary.
    let two_by_two = array![[1.0_f64, 2.0], [5.0, 6.0]];
    let two_model = SKPCA::<f64>::build()
        .build()
        .unwrap()
        .fit(&two_by_two)
        .unwrap();
    let rebuilt_two = two_model
        .inverse_transform(&two_model.transform(&two_by_two).unwrap())
        .unwrap();
    for (left, right) in rebuilt_two.iter().zip(two_by_two.iter()) {
        assert!((left - right).abs() < 1e-12);
    }

    // Single-precision f32 end-to-end across Full, Randomized and CovarianceEigh.
    let data_f32 = array![
        [1.0_f32, 2.0, 0.5],
        [2.0, 4.1, 1.0],
        [3.0, 5.9, 1.6],
        [4.0, 8.2, 2.1],
        [5.0, 10.0, 2.5]
    ];
    for solver in [
        SKPCASolver::Full,
        SKPCASolver::Randomized,
        SKPCASolver::CovarianceEigh,
    ] {
        let model = SKPCA::<f32>::build()
            .number_of_components(2)
            .svd_solver(solver)
            .random_seed(5)
            .build()
            .unwrap()
            .fit(&data_f32)
            .unwrap();
        let projected = model.transform(&data_f32).unwrap();
        assert_eq!(projected.dim(), (5, 2));
        assert!(projected.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn acceptance_pca_component_count_boundaries_and_whitening() {
    // Tall (12 x 4): min(n, d) = 4; test k = 1 and k = 4 (None).
    let tall = Array2::from_shape_fn((12, 4), |(row, column)| {
        (row as f64) * (column as f64 + 1.0) + ((row + column) % 3) as f64 * 0.4
    });
    let one_component = SKPCA::<f64>::build()
        .number_of_components(1)
        .build()
        .unwrap()
        .fit(&tall)
        .unwrap();
    assert_eq!(one_component.transform(&tall).unwrap().dim(), (12, 1));

    let all_tall = SKPCA::<f64>::build()
        .whiten(true)
        .build()
        .unwrap()
        .fit(&tall)
        .unwrap();
    assert_eq!(all_tall.number_of_components(), 4);
    let whitened = all_tall.transform(&tall).unwrap();
    let round_trip = all_tall.inverse_transform(&whitened).unwrap();
    for (left, right) in round_trip.iter().zip(tall.iter()) {
        assert!((left - right).abs() < 1e-10);
    }

    // Wide (4 x 9): min(n, d) = 4; full reconstruction still exact on training rows.
    let wide = Array2::from_shape_fn((4, 9), |(row, column)| {
        ((row + 1) as f64 * 0.7 * (column as f64 + 0.3)).sin() * 5.0 + column as f64
    });
    let wide_model = SKPCA::<f64>::build().build().unwrap().fit(&wide).unwrap();
    assert_eq!(wide_model.number_of_components(), 4);
    let rebuilt_wide = wide_model
        .inverse_transform(&wide_model.transform(&wide).unwrap())
        .unwrap();
    for (left, right) in rebuilt_wide.iter().zip(wide.iter()) {
        assert!((left - right).abs() < 1e-10);
    }
}
