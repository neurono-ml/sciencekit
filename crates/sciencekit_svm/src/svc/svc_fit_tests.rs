//! Companion TDD tests for binary `SKSVC::fit`
//! (spec `support-vector-machines`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKDataView, SKSupervisedFit};
use sciencekit_math::{SKKernel, SKKernelGamma, sk_kernel_matrix};

use super::SKSVC;

#[test]
fn svc_fit_binary_separable_retains_support_vectors_and_separates_training_rows() {
    let features = array![
        [-2.5_f64, -2.0],
        [-1.5, -1.0],
        [-1.0, -1.5],
        [1.0, 1.5],
        [1.5, 1.0],
        [2.5, 2.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];

    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::Linear)
        .C(1.0)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert!(model.converged());
    assert!(model.iterations() > 0);
    assert_eq!(model.number_of_classes(), 2);
    assert_eq!(model.pairwise_models().len(), 1);

    let binary = &model.pairwise_models()[0];
    assert_eq!(binary.class_negative(), 0);
    assert_eq!(binary.class_positive(), 1);
    assert!(!binary.dual_coefficients().is_empty());
    assert_eq!(
        binary.support_vectors().rows(),
        binary.dual_coefficients().len()
    );
    // Support vectors are a strict subset of the 6 separable rows (the extreme points are inactive).
    assert!(binary.support_vectors().rows() < features.nrows());

    // Every support vector matches its training row.
    if let SKDataView::Dense(sv_dense) = binary.support_vectors().view() {
        for (sv_row, &train_idx) in binary.support_training_indices().iter().enumerate() {
            assert_eq!(sv_dense.row(sv_row), features.row(train_idx));
        }
    }

    // Check decision values separate all 6 rows cleanly.
    let gram = sk_kernel_matrix(
        &model.kernel_parameters(),
        SKDataView::Dense(features.view()),
        binary.support_vectors().view(),
    )
    .unwrap();
    for row in 0..features.nrows() {
        let margin = gram.row(row).dot(binary.dual_coefficients()) + binary.bias();
        if labels[row] == 1 {
            assert!(
                margin > 0.0,
                "expected positive margin on row {row}, got {margin}"
            );
        } else {
            assert!(
                margin < 0.0,
                "expected negative margin on row {row}, got {margin}"
            );
        }
    }
}

#[test]
fn svc_fit_binary_overlapping_converges_with_bounded_dual_coefficients() {
    let features = array![
        [-2.0_f64, 0.0],
        [-1.0, 0.2],
        [0.3, -0.1], // overlapping negative sample on positive side
        [-0.2, 0.1], // overlapping positive sample on negative side
        [1.0, -0.2],
        [2.0, 0.0]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];

    let model = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .gamma(SKKernelGamma::Scale)
        .C(0.5)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();

    assert!(model.converged());
    assert!(model.kernel_parameters().gamma > 0.0);
    let binary = &model.pairwise_models()[0];
    binary
        .dual_coefficients()
        .iter()
        .for_each(|&coef| assert!(coef.abs() <= 0.5 + 1e-10));
}
