//! Companion TDD tests for loss selection and class-weight configuration of
//! [`SKLinearSVC`] (spec `linear-support-vector-classifier`).

use sciencekit_common::{SKBuilder, SKError};

use super::{SKLinearSVC, SKLinearSvcLoss, sk_linear_svc_loss_from_code};
use crate::svc::SKClassWeight;

#[test]
fn linear_svc_config_defaults_to_squared_hinge() {
    let estimator = SKLinearSVC::<f64>::build().build().unwrap();
    assert_eq!(estimator.loss(), SKLinearSvcLoss::SquaredHinge);
    assert_eq!(estimator.loss().name(), "squared_hinge");
}

#[test]
fn linear_svc_config_honors_explicit_hinge_loss() {
    let estimator = SKLinearSVC::<f64>::build()
        .loss(SKLinearSvcLoss::Hinge)
        .build()
        .unwrap();
    assert_eq!(estimator.loss(), SKLinearSvcLoss::Hinge);
    assert_eq!(estimator.loss().name(), "hinge");
}

#[test]
fn linear_svc_config_translates_binding_loss_codes() {
    assert_eq!(
        sk_linear_svc_loss_from_code(0).unwrap(),
        SKLinearSvcLoss::Hinge
    );
    assert_eq!(
        sk_linear_svc_loss_from_code(1).unwrap(),
        SKLinearSvcLoss::SquaredHinge
    );
    assert!(matches!(
        sk_linear_svc_loss_from_code(7).unwrap_err(),
        SKError::InvalidHyperparameter { name: "loss", .. }
    ));
}

#[test]
fn linear_svc_config_accepts_none_balanced_and_explicit_weights() {
    let none = SKLinearSVC::<f64>::build().build().unwrap();
    assert_eq!(none.class_weight(), &SKClassWeight::None);

    let balanced = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Balanced)
        .build()
        .unwrap();
    assert_eq!(balanced.class_weight(), &SKClassWeight::Balanced);

    let explicit = SKLinearSVC::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 3.0]))
        .build()
        .unwrap();
    assert_eq!(
        explicit.class_weight(),
        &SKClassWeight::Explicit(vec![1.0, 3.0])
    );
}

#[test]
fn linear_svc_config_carries_the_backend_and_execution_mode() {
    use sciencekit_common::{SKBackendKind, SKExecutionMode};

    let estimator = SKLinearSVC::<f64>::build()
        .backend(SKBackendKind::NdArrayLinalg)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(estimator.backend(), SKBackendKind::NdArrayLinalg);
    assert_eq!(
        estimator.execution_mode(),
        SKExecutionMode::InProcessSynchronous
    );
}
