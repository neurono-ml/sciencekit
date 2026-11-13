//! Companion TDD tests for [`SKLinearSVCBuilder`]
//! (spec `linear-support-vector-classifier`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode};

use super::{SKLinearSVC, SKLinearSVCLoss};
use crate::svc::SKClassWeight;

#[test]
fn linear_svc_builder_applies_documented_defaults() {
    let estimator = SKLinearSVC::<f64>::build().build().unwrap();
    assert!((estimator.C() - 1.0).abs() < 1e-12);
    assert_eq!(estimator.loss(), SKLinearSVCLoss::SquaredHinge);
    assert_eq!(estimator.class_weight(), &SKClassWeight::None);
    assert!(estimator.include_intercept());
    assert!((estimator.tolerance() - 1e-4).abs() < 1e-15);
    assert_eq!(estimator.maximum_number_of_iterations(), 1_000);
    assert_eq!(estimator.backend(), SKBackendKind::Faer);
    assert_eq!(estimator.execution_mode(), SKExecutionMode::Automatic);
}

#[test]
fn linear_svc_builder_honors_every_setter() {
    let estimator = SKLinearSVC::<f64>::build()
        .C(2.5)
        .loss(SKLinearSVCLoss::Hinge)
        .class_weight(SKClassWeight::Balanced)
        .include_intercept(false)
        .tolerance(1e-3)
        .maximum_number_of_iterations(250)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();

    assert!((estimator.C() - 2.5).abs() < 1e-12);
    assert_eq!(estimator.loss(), SKLinearSVCLoss::Hinge);
    assert_eq!(estimator.class_weight(), &SKClassWeight::Balanced);
    assert!(!estimator.include_intercept());
    assert!((estimator.tolerance() - 1e-3).abs() < 1e-15);
    assert_eq!(estimator.maximum_number_of_iterations(), 250);
    assert_eq!(estimator.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}

#[test]
fn linear_svc_builder_rejects_invalid_hyperparameters() {
    assert!(matches!(
        SKLinearSVC::<f64>::build().C(0.0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build().C(-1.0).build().unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .C(f64::INFINITY)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter { name: "C", .. }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .tolerance(0.0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "tolerance",
            ..
        }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .tolerance(f64::NAN)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "tolerance",
            ..
        }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .maximum_number_of_iterations(0)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "maximum_number_of_iterations",
            ..
        }
    ));
    assert!(matches!(
        SKLinearSVC::<f64>::build()
            .class_weight(SKClassWeight::Explicit(vec![1.0, -1.0]))
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "class_weight",
            ..
        }
    ));
}
