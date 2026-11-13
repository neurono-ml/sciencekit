//! Companion TDD tests for [`SKSVRBuilder`] (spec `support-vector-machines`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};
use sciencekit_math::{SKKernel, SKKernelGamma};

use super::SKSVR;

#[test]
fn svr_builder_applies_documented_defaults() {
    let svr = SKSVR::<f64>::build().build().unwrap();
    assert!((svr.C() - 1.0).abs() < 1e-12);
    assert!((svr.epsilon() - 0.1).abs() < 1e-12);
    assert_eq!(svr.kernel(), SKKernel::RadialBasisFunction);
    assert_eq!(svr.degree(), 3);
    assert_eq!(svr.gamma(), SKKernelGamma::Scale);
    assert!((svr.coef0() - 0.0).abs() < 1e-12);
    assert!((svr.tolerance() - 1e-3).abs() < 1e-12);
    assert_eq!(svr.maximum_number_of_iterations(), 10_000);
    assert_eq!(svr.cache_size(), 200);
    assert!(svr.shrinking());
    assert_eq!(svr.backend(), SKBackendKind::Faer);
    assert_eq!(svr.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn svr_builder_honors_overrides_and_rejects_invalid_hyperparameters() {
    let svr = SKSVR::<f64>::build()
        .C(5.0)
        .epsilon(0.0)
        .kernel(SKKernel::Sigmoid)
        .degree(2)
        .gamma(SKKernelGamma::Auto)
        .coef0(-0.5)
        .tolerance(1e-5)
        .maximum_number_of_iterations(2_000)
        .cache_size(32)
        .shrinking(false)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert!((svr.C() - 5.0).abs() < 1e-12);
    assert!((svr.epsilon() - 0.0).abs() < 1e-12);
    assert_eq!(svr.kernel(), SKKernel::Sigmoid);
    assert_eq!(svr.degree(), 2);
    assert_eq!(svr.gamma(), SKKernelGamma::Auto);
    assert!((svr.coef0() - (-0.5)).abs() < 1e-12);
    assert!((svr.tolerance() - 1e-5).abs() < 1e-12);
    assert_eq!(svr.maximum_number_of_iterations(), 2_000);
    assert_eq!(svr.cache_size(), 32);
    assert!(!svr.shrinking());
    assert_eq!(svr.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(svr.execution_mode(), SKExecutionMode::InProcessSynchronous);

    assert!(matches!(
        SKSVR::<f64>::build().C(0.0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKSVR::<f64>::build().epsilon(-0.01).build().unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "epsilon",
            ..
        }
    ));
    assert!(matches!(
        SKSVR::<f64>::build()
            .gamma(SKKernelGamma::Value(-1.0))
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));
    assert!(matches!(
        SKSVR::<f64>::build().degree(0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "degree", .. }
    ));
}
