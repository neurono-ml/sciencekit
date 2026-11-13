//! Companion TDD tests for [`SKSVCBuilder`] (spec `support-vector-machines`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};
use sciencekit_math::{SKKernel, SKKernelGamma};

use super::{SKClassWeight, SKSVC};

#[test]
fn svc_builder_applies_documented_defaults() {
    let svc = SKSVC::<f64>::build().build().unwrap();
    assert!((svc.C() - 1.0).abs() < 1e-12);
    assert_eq!(svc.kernel(), SKKernel::RadialBasisFunction);
    assert_eq!(svc.degree(), 3);
    assert_eq!(svc.gamma(), SKKernelGamma::Scale);
    assert!((svc.coef0() - 0.0).abs() < 1e-12);
    assert!((svc.tolerance() - 1e-3).abs() < 1e-12);
    assert_eq!(svc.maximum_number_of_iterations(), 10_000);
    assert_eq!(svc.cache_size(), 200);
    assert!(svc.shrinking());
    assert_eq!(svc.class_weight(), &SKClassWeight::None);
    assert!(!svc.probability());
    assert_eq!(svc.random_seed(), 0);
    assert_eq!(svc.backend(), SKBackendKind::Faer);
    assert_eq!(svc.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn svc_builder_honors_every_setter_and_rejects_invalid_hyperparameters() {
    let svc = SKSVC::<f64>::build()
        .C(2.5)
        .kernel(SKKernel::Polynomial)
        .degree(4)
        .gamma(SKKernelGamma::Value(0.25))
        .coef0(1.5)
        .tolerance(1e-4)
        .maximum_number_of_iterations(500)
        .cache_size(64)
        .shrinking(false)
        .class_weight(SKClassWeight::Balanced)
        .probability(true)
        .random_seed(42)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert!((svc.C() - 2.5).abs() < 1e-12);
    assert_eq!(svc.kernel(), SKKernel::Polynomial);
    assert_eq!(svc.degree(), 4);
    assert_eq!(svc.gamma(), SKKernelGamma::Value(0.25));
    assert!((svc.coef0() - 1.5).abs() < 1e-12);
    assert!((svc.tolerance() - 1e-4).abs() < 1e-12);
    assert_eq!(svc.maximum_number_of_iterations(), 500);
    assert_eq!(svc.cache_size(), 64);
    assert!(!svc.shrinking());
    assert_eq!(svc.class_weight(), &SKClassWeight::Balanced);
    assert!(svc.probability());
    assert_eq!(svc.random_seed(), 42);
    assert_eq!(svc.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(svc.execution_mode(), SKExecutionMode::InProcessSynchronous);

    assert!(matches!(
        SKSVC::<f64>::build().C(0.0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKSVC::<f64>::build().C(-1.0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKSVC::<f64>::build().degree(0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "degree", .. }
    ));
    assert!(matches!(
        SKSVC::<f64>::build()
            .gamma(SKKernelGamma::Value(-0.2))
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter { name: "gamma", .. }
    ));
    assert!(matches!(
        SKSVC::<f64>::build().tolerance(0.0).build().unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "tolerance",
            ..
        }
    ));
    assert!(matches!(
        SKSVC::<f64>::build()
            .maximum_number_of_iterations(0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "maximum_number_of_iterations",
            ..
        }
    ));
    assert!(matches!(
        SKSVC::<f64>::build()
            .class_weight(SKClassWeight::Explicit(vec![1.0, -1.0]))
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "class_weight",
            ..
        }
    ));
}
