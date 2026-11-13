//! Companion TDD tests for loss selection and class-weight configuration of
//! [`SKLinearSVC`] (spec `linear-support-vector-classifier`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKError, SKSupervisedFit};

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

fn primal_objective(
    loss: SKLinearSvcLoss,
    features: &ndarray::Array2<f64>,
    labels: &ndarray::Array1<i64>,
    coefficients: &[f64],
    intercept: f64,
    regularization_c: f64,
) -> f64 {
    let regularization = 0.5 * coefficients.iter().map(|value| value * value).sum::<f64>();
    let empirical = (0..features.nrows()).fold(0.0, |accumulator, row| {
        let signed = if labels[row] == 1 { 1.0 } else { -1.0 };
        let margin: f64 = coefficients
            .iter()
            .zip(features.row(row).iter())
            .map(|(weight, value)| weight * value)
            .sum::<f64>()
            + intercept;
        let residual = (1.0 - signed * margin).max(0.0);
        let loss_term = match loss {
            SKLinearSvcLoss::Hinge => residual,
            SKLinearSvcLoss::SquaredHinge => residual * residual,
        };
        accumulator + regularization_c * loss_term
    });
    regularization + empirical
}

#[test]
fn each_loss_minimizes_its_own_objective_better_than_the_other() {
    let features = array![
        [-2.0_f64, 0.5],
        [-1.0, -1.0],
        [-0.5, 0.2],
        [0.5, -0.4],
        [1.0, 0.6],
        [2.0, -0.2]
    ];
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let fit = |loss: SKLinearSvcLoss| {
        SKLinearSVC::<f64>::build()
            .C(1.0)
            .loss(loss)
            .tolerance(1e-10)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap()
    };
    let squared = fit(SKLinearSvcLoss::SquaredHinge);
    let hinge = fit(SKLinearSvcLoss::Hinge);

    let squared_head = &squared.heads()[0];
    let hinge_head = &hinge.heads()[0];
    let squared_weights = squared_head.coefficients().to_vec();
    let hinge_weights = hinge_head.coefficients().to_vec();

    assert_ne!(
        squared_weights, hinge_weights,
        "the loss must change the fitted coefficients"
    );

    let squared_objective_of = |weights: &[f64], intercept: f64| {
        primal_objective(
            SKLinearSvcLoss::SquaredHinge,
            &features,
            &labels,
            weights,
            intercept,
            1.0,
        )
    };
    let hinge_objective_of = |weights: &[f64], intercept: f64| {
        primal_objective(
            SKLinearSvcLoss::Hinge,
            &features,
            &labels,
            weights,
            intercept,
            1.0,
        )
    };

    assert!(
        squared_objective_of(&squared_weights, squared_head.intercept())
            < squared_objective_of(&hinge_weights, hinge_head.intercept()),
        "the squared-hinge fit must minimize the squared-hinge objective"
    );
    assert!(
        hinge_objective_of(&hinge_weights, hinge_head.intercept())
            < hinge_objective_of(&squared_weights, squared_head.intercept()),
        "the hinge fit must minimize the hinge objective"
    );
}
