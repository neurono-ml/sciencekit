//! Companion TDD tests for kernel configuration and gamma resolution
//! (spec `kernel-functions`).

use ndarray::array;
use sciencekit_common::{SKDataView, SKError};
use sprs::TriMat;

use super::{
    SKKernel, SKKernelGamma, sk_feature_population_variance, sk_resolve_kernel_gamma,
    sk_resolve_kernel_parameters, sk_validate_kernel_hyperparameters,
};

#[test]
fn kernel_defaults_select_rbf_and_scale_gamma() {
    assert_eq!(SKKernel::default(), SKKernel::RadialBasisFunction);
    assert_eq!(SKKernelGamma::<f64>::default(), SKKernelGamma::Scale);
}

#[test]
fn kernel_gamma_resolves_scale_auto_and_explicit_value() {
    // Elements: [0, 2, 2, 4] -> mean = 2, population variance = ((4 + 0 + 0 + 4) / 4) = 2.0.
    // n_features = 2 => Scale = 1 / (2 * 2.0) = 0.25, Auto = 1 / 2 = 0.5.
    let dense = array![[0.0_f64, 2.0], [2.0, 4.0]];
    let view = SKDataView::Dense(dense.view());

    let scale = sk_resolve_kernel_gamma(SKKernelGamma::Scale, view).unwrap();
    let auto = sk_resolve_kernel_gamma(SKKernelGamma::Auto, view).unwrap();
    let explicit = sk_resolve_kernel_gamma(SKKernelGamma::Value(1.75), view).unwrap();

    assert!((scale - 0.25).abs() < 1e-12);
    assert!((auto - 0.5).abs() < 1e-12);
    assert!((explicit - 1.75).abs() < 1e-12);
}

#[test]
fn kernel_gamma_scale_matches_between_dense_and_csr() {
    let dense = array![[0.0_f64, 3.0, 0.0], [0.0, 0.0, 6.0]];
    let mut triplets = TriMat::new((2, 3));
    triplets.add_triplet(0, 1, 3.0_f64);
    triplets.add_triplet(1, 2, 6.0_f64);
    let csr = triplets.to_csr();

    let dense_var = sk_feature_population_variance(SKDataView::Dense(dense.view())).unwrap();
    let sparse_var = sk_feature_population_variance(SKDataView::Sparse(csr.view())).unwrap();
    assert!((dense_var - sparse_var).abs() < 1e-12);

    let dense_gamma =
        sk_resolve_kernel_gamma(SKKernelGamma::Scale, SKDataView::Dense(dense.view())).unwrap();
    let sparse_gamma =
        sk_resolve_kernel_gamma(SKKernelGamma::Scale, SKDataView::Sparse(csr.view())).unwrap();
    assert!((dense_gamma - sparse_gamma).abs() < 1e-12);
}

#[test]
fn kernel_invalid_parameters_are_rejected_with_named_errors() {
    let dense = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let view = SKDataView::Dense(dense.view());

    let negative_gamma = sk_resolve_kernel_gamma(SKKernelGamma::Value(-0.1), view).unwrap_err();
    assert!(matches!(
        negative_gamma,
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));

    let zero_degree = sk_resolve_kernel_parameters(
        SKKernel::Polynomial,
        SKKernelGamma::Auto,
        0,
        0.0,
        view,
    )
    .unwrap_err();
    assert!(matches!(
        zero_degree,
        SKError::InvalidHyperparameter { name: "degree", .. }
    ));

    let nan_coef0 =
        sk_validate_kernel_hyperparameters(SKKernel::Sigmoid, SKKernelGamma::Auto, 3, f64::NAN)
            .unwrap_err();
    assert!(matches!(
        nan_coef0,
        SKError::InvalidHyperparameter { name: "coef0", .. }
    ));

    let constant = array![[2.0_f64, 2.0], [2.0, 2.0]];
    let zero_variance =
        sk_resolve_kernel_gamma(SKKernelGamma::Scale, SKDataView::Dense(constant.view()))
            .unwrap_err();
    assert!(matches!(
        zero_variance,
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));
}
