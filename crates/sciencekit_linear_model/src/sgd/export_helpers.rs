//! Shared state encoding for the stochastic-gradient models.

use sciencekit_common::{
    SKError, SKFloat, SKModelMetadata, SKParameterValue, sk_boolean_parameter, sk_float_parameter,
    sk_integer_parameter, sk_string_parameter,
};

use super::config::SKSGDConfig;
use super::convergence::SKConvergencePolicy;
use super::learning_rate::SKSGDLearningRate;
use super::penalty::SKSGDPenalty;

/// Attach every validated configuration field to the metadata.
pub(crate) fn with_sgd_config<F: SKFloat>(
    mut metadata: SKModelMetadata,
    config: &SKSGDConfig<F>,
) -> Result<SKModelMetadata, SKError> {
    metadata = metadata
        .with_parameter(
            "penalty",
            SKParameterValue::from(penalty_name(config.penalty)),
        )?
        .with_parameter(
            "alpha",
            SKParameterValue::from(config.alpha.to_f64().unwrap_or(1e-4)),
        )?
        .with_parameter(
            "l1_ratio",
            SKParameterValue::from(config.l1_ratio.to_f64().unwrap_or(0.15)),
        )?
        .with_parameter(
            "include_intercept",
            SKParameterValue::from(config.include_intercept),
        )?
        .with_parameter(
            "learning_rate",
            SKParameterValue::from(learning_rate_name(config.learning_rate)),
        )?
        .with_parameter(
            "initial_learning_rate",
            SKParameterValue::from(config.initial_learning_rate.to_f64().unwrap_or(0.01)),
        )?
        .with_parameter(
            "inverse_scaling_power",
            SKParameterValue::from(config.inverse_scaling_power.to_f64().unwrap_or(0.25)),
        )?
        .with_parameter(
            "tolerance",
            SKParameterValue::from(config.tolerance.to_f64().unwrap_or(1e-3)),
        )?
        .with_parameter(
            "maximum_number_of_iterations",
            SKParameterValue::from(config.maximum_number_of_iterations as i64),
        )?
        .with_parameter(
            "number_of_epochs_without_improvement",
            SKParameterValue::from(config.number_of_epochs_without_improvement as i64),
        )?
        .with_parameter(
            "early_stopping",
            SKParameterValue::from(config.early_stopping),
        )?
        .with_parameter(
            "validation_fraction",
            SKParameterValue::from(config.validation_fraction.to_f64().unwrap_or(0.1)),
        )?
        .with_parameter("shuffle", SKParameterValue::from(config.shuffle))?
        .with_parameter(
            "random_seed",
            SKParameterValue::from(config.random_seed as i64),
        )?
        .with_parameter("average", SKParameterValue::from(config.average))?
        .with_parameter(
            "convergence_policy",
            SKParameterValue::from(convergence_name(config.convergence_policy)),
        )?;
    Ok(metadata)
}

/// Decode the validated configuration from metadata.
pub(crate) fn read_sgd_config<F: SKFloat>(
    metadata: &SKModelMetadata,
) -> Result<SKSGDConfig<F>, SKError> {
    Ok(SKSGDConfig {
        penalty: penalty_from_name(sk_string_parameter(metadata, "penalty", "l2")),
        alpha: F::from(sk_float_parameter(metadata, "alpha", 1e-4))
            .ok_or(SKError::Conversion("alpha is not finite".to_string()))?,
        l1_ratio: F::from(sk_float_parameter(metadata, "l1_ratio", 0.15))
            .ok_or(SKError::Conversion("l1_ratio is not finite".to_string()))?,
        include_intercept: sk_boolean_parameter(metadata, "include_intercept", true),
        learning_rate: learning_rate_from_name(sk_string_parameter(
            metadata,
            "learning_rate",
            "optimal",
        )),
        initial_learning_rate: F::from(sk_float_parameter(metadata, "initial_learning_rate", 0.01))
            .ok_or(SKError::Conversion(
                "initial_learning_rate is not finite".to_string(),
            ))?,
        inverse_scaling_power: F::from(sk_float_parameter(metadata, "inverse_scaling_power", 0.25))
            .ok_or(SKError::Conversion(
                "inverse_scaling_power is not finite".to_string(),
            ))?,
        tolerance: F::from(sk_float_parameter(metadata, "tolerance", 1e-3))
            .ok_or(SKError::Conversion("tolerance is not finite".to_string()))?,
        maximum_number_of_iterations: sk_integer_parameter(
            metadata,
            "maximum_number_of_iterations",
            1000,
        )
        .max(1) as usize,
        number_of_epochs_without_improvement: sk_integer_parameter(
            metadata,
            "number_of_epochs_without_improvement",
            5,
        )
        .max(1) as usize,
        early_stopping: sk_boolean_parameter(metadata, "early_stopping", false),
        validation_fraction: F::from(sk_float_parameter(metadata, "validation_fraction", 0.1))
            .ok_or(SKError::Conversion(
                "validation_fraction is not finite".to_string(),
            ))?,
        shuffle: sk_boolean_parameter(metadata, "shuffle", true),
        random_seed: sk_integer_parameter(metadata, "random_seed", 0).max(0) as u64,
        average: sk_boolean_parameter(metadata, "average", false),
        convergence_policy: convergence_from_name(sk_string_parameter(
            metadata,
            "convergence_policy",
            "strict",
        )),
    })
}

fn penalty_name(penalty: SKSGDPenalty) -> &'static str {
    match penalty {
        SKSGDPenalty::L2 => "l2",
        SKSGDPenalty::L1 => "l1",
        SKSGDPenalty::ElasticNet => "elastic_net",
        SKSGDPenalty::None => "none",
    }
}

fn penalty_from_name(name: &str) -> SKSGDPenalty {
    match name {
        "l1" => SKSGDPenalty::L1,
        "elastic_net" => SKSGDPenalty::ElasticNet,
        "none" => SKSGDPenalty::None,
        _ => SKSGDPenalty::L2,
    }
}

fn learning_rate_name(policy: SKSGDLearningRate) -> &'static str {
    match policy {
        SKSGDLearningRate::Constant => "constant",
        SKSGDLearningRate::Optimal => "optimal",
        SKSGDLearningRate::InverseScaling => "inverse_scaling",
        SKSGDLearningRate::Adaptive => "adaptive",
    }
}

fn learning_rate_from_name(name: &str) -> SKSGDLearningRate {
    match name {
        "constant" => SKSGDLearningRate::Constant,
        "inverse_scaling" => SKSGDLearningRate::InverseScaling,
        "adaptive" => SKSGDLearningRate::Adaptive,
        _ => SKSGDLearningRate::Optimal,
    }
}

fn convergence_name(policy: SKConvergencePolicy) -> &'static str {
    match policy {
        SKConvergencePolicy::Strict => "strict",
        SKConvergencePolicy::Permissive => "permissive",
    }
}

fn convergence_from_name(name: &str) -> SKConvergencePolicy {
    match name {
        "permissive" => SKConvergencePolicy::Permissive,
        _ => SKConvergencePolicy::Strict,
    }
}
