//! Companion TDD tests for the linear SVC design abstraction
//! (spec `linear-support-vector-classifier`).

use ndarray::array;

use super::design::{SKDenseLinearSvcDesign, SKLinearSvcDesignAccess};

#[test]
fn dense_design_row_dot_matches_the_hand_computed_augmented_row() {
    let features = array![[1.0_f64, 2.0], [3.0, -1.0]];
    let design = SKDenseLinearSvcDesign::new(features.view(), true);

    assert_eq!(design.number_of_rows(), 2);
    assert_eq!(design.number_of_columns(), 2);
    assert!(design.has_intercept());

    // coefficients = [w_0, w_1, bias]; row 0 = [1, 2, 1], row 1 = [3, -1, 1].
    let coefficients = [0.5_f64, -1.5, 2.0];
    assert!((design.row_dot(0, &coefficients) - (0.5 - 3.0 + 2.0)).abs() < 1e-12);
    assert!((design.row_dot(1, &coefficients) - (1.5 + 1.5 + 2.0)).abs() < 1e-12);

    assert!((design.row_norm_squared(0) - (1.0 + 4.0 + 1.0)).abs() < 1e-12);
    assert!((design.row_norm_squared(1) - (9.0 + 1.0 + 1.0)).abs() < 1e-12);
}

#[test]
fn dense_design_adds_scaled_augmented_rows_into_the_target() {
    let features = array![[1.0_f64, 2.0], [3.0, -1.0]];
    let design = SKDenseLinearSvcDesign::new(features.view(), true);

    let mut target = [0.0_f64; 3];
    design.add_scaled_row(0, 2.0, &mut target);
    design.add_scaled_row(1, -1.0, &mut target);

    assert!((target[0] - (2.0 - 3.0)).abs() < 1e-12);
    assert!((target[1] - (4.0 + 1.0)).abs() < 1e-12);
    assert!((target[2] - (2.0 - 1.0)).abs() < 1e-12);
}

#[test]
fn dense_design_without_intercept_uses_the_stored_columns_only() {
    let features = array![[1.0_f64, 2.0]];
    let design = SKDenseLinearSvcDesign::new(features.view(), false);
    let coefficients = [2.0_f64, 3.0];
    assert!((design.row_dot(0, &coefficients) - 8.0).abs() < 1e-12);
    assert!((design.row_norm_squared(0) - 5.0).abs() < 1e-12);
}

#[test]
fn sparse_design_scans_stored_entries_only_and_rejects_csc() {
    use sciencekit_common::SKDataView;
    use sprs::{CsMat, TriMat};

    use super::design::SKLinearSvcDesign;

    let mut triplets = TriMat::new((2, 3));
    triplets.add_triplet(0, 0, 1.0_f64);
    triplets.add_triplet(0, 2, 2.0);
    triplets.add_triplet(1, 1, -1.0);
    let csr = triplets.to_csr();

    let design = SKLinearSvcDesign::dispatch(SKDataView::Sparse(csr.view()), true).unwrap();
    let access = super::design::sk_linear_svc_design_access(&design);
    let coefficients = [0.5_f64, 1.5, -2.0, 0.25];
    // row 0 stored: 1·0.5 + 2·(-2) + bias 0.25
    assert!((access.row_dot(0, &coefficients) - (0.5 - 4.0 + 0.25)).abs() < 1e-12);
    assert!((access.row_norm_squared(0) - (1.0 + 4.0 + 1.0)).abs() < 1e-12);

    let mut target = [0.0_f64; 4];
    access.add_scaled_row(1, 3.0, &mut target);
    assert!((target[1] + 3.0).abs() < 1e-12);
    assert!((target[3] - 3.0).abs() < 1e-12);
    assert!((target[0] - 0.0).abs() < 1e-12);
    assert!((target[2] - 0.0).abs() < 1e-12);

    let csc = CsMat::new_csc(
        (2, 3),
        vec![0, 1, 2, 3],
        vec![0, 1, 0],
        vec![1.0_f64, 2.0, 3.0],
    );
    assert!(matches!(
        SKLinearSvcDesign::dispatch(SKDataView::Sparse(csc.view()), true).unwrap_err(),
        sciencekit_common::SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));
}

#[test]
fn sparse_and_dense_designs_agree_on_the_same_logical_matrix() {
    use ndarray::Array2;
    use sciencekit_common::SKDataView;
    use sprs::TriMat;

    use super::design::SKLinearSvcDesign;

    let dense = Array2::from_shape_vec((2, 3), vec![1.0_f64, 0.0, 2.0, 0.0, -1.0, 0.0]).unwrap();
    let mut triplets = TriMat::new((2, 3));
    triplets.add_triplet(0, 0, 1.0_f64);
    triplets.add_triplet(0, 2, 2.0);
    triplets.add_triplet(1, 1, -1.0);
    let sparse = triplets.to_csr();

    let dense_design = SKLinearSvcDesign::dispatch(SKDataView::Dense(dense.view()), true).unwrap();
    let sparse_design =
        SKLinearSvcDesign::dispatch(SKDataView::Sparse(sparse.view()), true).unwrap();
    let dense_access = super::design::sk_linear_svc_design_access(&dense_design);
    let sparse_access = super::design::sk_linear_svc_design_access(&sparse_design);

    let coefficients = [0.75_f64, -0.5, 1.25, 0.5];
    (0..2).for_each(|row| {
        assert!(
            (dense_access.row_dot(row, &coefficients) - sparse_access.row_dot(row, &coefficients))
                .abs()
                < 1e-12
        );
        assert!(
            (dense_access.row_norm_squared(row) - sparse_access.row_norm_squared(row)).abs()
                < 1e-12
        );
    });
}
