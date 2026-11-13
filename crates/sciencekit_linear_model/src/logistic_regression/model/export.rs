//! Format-agnostic state export/import for
//! [`SKLogisticRegressionModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKParameterValue, SKTensor, sk_backend_parameter, sk_boolean_parameter,
    sk_canonicalize_labels, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::SKLogisticRegressionModel;
use crate::class_weight::SKClassWeight;
use crate::logistic::SKLogisticState;
use crate::logistic::config::{
    SKLogisticConfig, SKLogisticPenalty, SKLogisticSolver, SKMultiClass,
};
use crate::logistic::resolution::SKResolvedMultiClass;

/// Stable estimator identifier for exported logistic-regression states.
pub const SK_LOGISTIC_REGRESSION_ESTIMATOR_NAME: &str = "SKLogisticRegression";
/// The tensor carrying the per-head coefficients.
pub const SK_LOGISTIC_COEFFICIENTS_TENSOR: &str = "logistic.coefficients";
/// The tensor carrying the per-head intercepts.
pub const SK_LOGISTIC_INTERCEPTS_TENSOR: &str = "logistic.intercepts";

impl<F: SKFloat> SKExportable<F> for SKLogisticRegressionModel<F> {
    fn export_state(&self) -> Result<sciencekit_common::SKModelState<F>, SKError> {
        let feature_count = self.coefficients.ncols();
        let mut metadata = SKModelMetadata::new(
            SK_LOGISTIC_REGRESSION_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            feature_count,
        )?
        .with_class_labels(self.classes.labels().map(str::to_string).collect())?
        .with_parameter(
            "number_of_classes",
            SKParameterValue::from(self.number_of_classes as i64),
        )?
        .with_parameter(
            "strategy",
            SKParameterValue::from(strategy_name(self.strategy)),
        )?
        .with_parameter(
            "alpha",
            SKParameterValue::from(self.config.alpha.to_f64().unwrap_or(1.0)),
        )?
        .with_parameter(
            "tolerance",
            SKParameterValue::from(self.config.tolerance.to_f64().unwrap_or(1e-4)),
        )?
        .with_parameter(
            "maximum_number_of_iterations",
            SKParameterValue::from(self.config.maximum_number_of_iterations as i64),
        )?
        .with_parameter(
            "penalty",
            SKParameterValue::from(penalty_name(self.config.penalty)),
        )?
        .with_parameter(
            "l1_ratio",
            SKParameterValue::from(self.config.l1_ratio.to_f64().unwrap_or(0.5)),
        )?
        .with_parameter(
            "solver",
            SKParameterValue::from(solver_name(self.config.solver)),
        )?
        .with_parameter(
            "multi_class",
            SKParameterValue::from(multi_class_name(self.config.multi_class)),
        )?
        .with_parameter(
            "include_intercept",
            SKParameterValue::from(self.config.include_intercept),
        )?
        .with_parameter("warm_start", SKParameterValue::from(self.config.warm_start))?
        .with_parameter(
            "random_seed",
            SKParameterValue::from(self.config.random_seed as i64),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        match &self.config.class_weight {
            SKClassWeight::None => {
                metadata = metadata
                    .with_parameter("class_weight_policy", SKParameterValue::from("none"))?;
            }
            SKClassWeight::Balanced => {
                metadata = metadata
                    .with_parameter("class_weight_policy", SKParameterValue::from("balanced"))?;
            }
            SKClassWeight::Explicit(weights) => {
                metadata = metadata
                    .with_parameter("class_weight_policy", SKParameterValue::from("explicit"))?
                    .with_parameter(
                        "class_weight",
                        SKParameterValue::List(
                            weights
                                .iter()
                                .map(|weight| {
                                    SKParameterValue::from(weight.to_f64().unwrap_or(1.0))
                                })
                                .collect(),
                        ),
                    )?;
            }
        }
        let coefficients = SKTensor::floating(
            SK_LOGISTIC_COEFFICIENTS_TENSOR,
            vec![self.coefficients.nrows(), feature_count],
            self.coefficients.iter().copied().collect(),
        )?;
        let intercepts = SKTensor::floating(
            SK_LOGISTIC_INTERCEPTS_TENSOR,
            vec![self.intercepts.len()],
            self.intercepts.to_vec(),
        )?;
        sciencekit_common::SKModelState::new(metadata, vec![coefficients, intercepts])
    }
}

impl<F: SKFloat> SKImportable<F> for SKLogisticRegressionModel<F> {
    fn import_state(state: &sciencekit_common::SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_LOGISTIC_REGRESSION_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let (coefficient_shape, coefficient_values) =
            sk_read_floating_tensor(state, SK_LOGISTIC_COEFFICIENTS_TENSOR)?;
        let &[heads, columns] = coefficient_shape.as_slice() else {
            return Err(SKError::ShapeMismatch {
                expected: vec![1, feature_count],
                found: coefficient_shape,
            });
        };
        if columns != feature_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![heads, feature_count],
                found: vec![heads, columns],
            });
        }
        let coefficients = Array2::from_shape_vec((heads, columns), coefficient_values)
            .map_err(|error| SKError::Conversion(error.to_string()))?;
        let (_, intercepts) = sk_read_floating_tensor(state, SK_LOGISTIC_INTERCEPTS_TENSOR)?;
        if intercepts.len() != heads {
            return Err(SKError::ShapeMismatch {
                expected: vec![heads],
                found: vec![intercepts.len()],
            });
        }
        let metadata = state.metadata();
        let labels: Vec<String> = metadata
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        let class_weight = class_weight_from_state(state)?;
        let config = SKLogisticConfig {
            alpha: F::from(sk_float_parameter(metadata, "alpha", 1.0))
                .ok_or(SKError::Conversion("alpha is not finite".to_string()))?,
            tolerance: F::from(sk_float_parameter(metadata, "tolerance", 1e-4))
                .ok_or(SKError::Conversion("tolerance is not finite".to_string()))?,
            maximum_number_of_iterations: sk_integer_parameter(
                metadata,
                "maximum_number_of_iterations",
                100,
            )
            .max(1) as usize,
            penalty: penalty_from_name(sk_string_parameter(metadata, "penalty", "l2")),
            l1_ratio: F::from(sk_float_parameter(metadata, "l1_ratio", 0.5))
                .ok_or(SKError::Conversion("l1_ratio is not finite".to_string()))?,
            solver: solver_from_name(sk_string_parameter(metadata, "solver", "lbfgs")),
            multi_class: multi_class_from_name(sk_string_parameter(
                metadata,
                "multi_class",
                "auto",
            )),
            class_weight,
            include_intercept: sk_boolean_parameter(metadata, "include_intercept", true),
            warm_start: sk_boolean_parameter(metadata, "warm_start", false),
            random_seed: sk_integer_parameter(metadata, "random_seed", 0).max(0) as u64,
        };
        Ok(SKLogisticRegressionModel {
            coefficients,
            intercepts: Array1::from_vec(intercepts),
            classes,
            number_of_classes: sk_integer_parameter(metadata, "number_of_classes", heads as i64)
                .max(1) as usize,
            strategy: strategy_from_name(sk_string_parameter(metadata, "strategy", "binary")),
            config,
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
            state: SKLogisticState::new_dense(),
        })
    }
}

/// Rebuild the class-weighting policy from metadata.
fn class_weight_from_state<F: SKFloat>(
    state: &sciencekit_common::SKModelState<F>,
) -> Result<SKClassWeight<F>, SKError> {
    match sk_string_parameter(state.metadata(), "class_weight_policy", "none") {
        "balanced" => Ok(SKClassWeight::Balanced),
        "explicit" => {
            let values = state
                .metadata()
                .parameter("class_weight")
                .and_then(SKParameterValue::as_list)
                .ok_or(SKError::InvalidHyperparameter {
                    name: "class_weight",
                    reason: "an explicit policy must carry its weights".to_string(),
                })?;
            let weights = values
                .iter()
                .filter_map(SKParameterValue::as_float)
                .map(|weight| F::from(weight).unwrap_or_else(F::one))
                .collect();
            Ok(SKClassWeight::Explicit(weights))
        }
        _ => Ok(SKClassWeight::None),
    }
}

/// Stable string for a resolved multi-class strategy.
fn strategy_name(strategy: SKResolvedMultiClass) -> &'static str {
    match strategy {
        SKResolvedMultiClass::Binary => "binary",
        SKResolvedMultiClass::OneVersusRest => "one_versus_rest",
        SKResolvedMultiClass::Multinomial => "multinomial",
    }
}

/// The resolved multi-class strategy for its stable string.
fn strategy_from_name(name: &str) -> SKResolvedMultiClass {
    match name {
        "one_versus_rest" => SKResolvedMultiClass::OneVersusRest,
        "multinomial" => SKResolvedMultiClass::Multinomial,
        _ => SKResolvedMultiClass::Binary,
    }
}

/// Stable string for a penalty.
fn penalty_name(penalty: SKLogisticPenalty) -> &'static str {
    match penalty {
        SKLogisticPenalty::L2 => "l2",
        SKLogisticPenalty::L1 => "l1",
        SKLogisticPenalty::ElasticNet => "elastic_net",
        SKLogisticPenalty::None => "none",
    }
}

/// The penalty for its stable string.
fn penalty_from_name(name: &str) -> SKLogisticPenalty {
    match name {
        "l1" => SKLogisticPenalty::L1,
        "elastic_net" => SKLogisticPenalty::ElasticNet,
        "none" => SKLogisticPenalty::None,
        _ => SKLogisticPenalty::L2,
    }
}

/// Stable string for a solver.
fn solver_name(solver: SKLogisticSolver) -> &'static str {
    match solver {
        SKLogisticSolver::Lbfgs => "lbfgs",
        SKLogisticSolver::Liblinear => "liblinear",
        SKLogisticSolver::Saga => "saga",
    }
}

/// The solver for its stable string.
fn solver_from_name(name: &str) -> SKLogisticSolver {
    match name {
        "liblinear" => SKLogisticSolver::Liblinear,
        "saga" => SKLogisticSolver::Saga,
        _ => SKLogisticSolver::Lbfgs,
    }
}

/// Stable string for a configured multi-class option.
fn multi_class_name(multi_class: SKMultiClass) -> &'static str {
    match multi_class {
        SKMultiClass::Auto => "auto",
        SKMultiClass::OneVersusRest => "one_versus_rest",
        SKMultiClass::Multinomial => "multinomial",
    }
}

/// The configured multi-class option for its stable string.
fn multi_class_from_name(name: &str) -> SKMultiClass {
    match name {
        "one_versus_rest" => SKMultiClass::OneVersusRest,
        "multinomial" => SKMultiClass::Multinomial,
        _ => SKMultiClass::Auto,
    }
}
