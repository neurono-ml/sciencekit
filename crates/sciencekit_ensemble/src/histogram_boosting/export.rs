//! Format-agnostic export/import of the histogram-gradient-boosting models.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_canonicalize_labels, sk_float_parameter, sk_integer_parameter, sk_model_tensor_data_type,
    sk_parse_backend_parameter, sk_read_floating_tensor, sk_string_parameter,
    sk_validate_import_metadata,
};

use super::classifier_model::SKHistGradientBoostingClassifierModel;
use super::export_helpers::{
    export_binnings, export_histogram_tree, import_binnings, import_histogram_tree,
};
use super::loss_kind::SKHistGradientBoostingRegressionLoss;
use super::regressor_model::SKHistGradientBoostingRegressorModel;

/// Stable estimator identifier for exported histogram-boosting regressor states.
pub const SK_HIST_GRADIENT_BOOSTING_REGRESSOR_ESTIMATOR_NAME: &str =
    "SKHistGradientBoostingRegressor";
/// Stable estimator identifier for exported histogram-boosting classifier states.
pub const SK_HIST_GRADIENT_BOOSTING_CLASSIFIER_ESTIMATOR_NAME: &str =
    "SKHistGradientBoostingClassifier";

impl<F: SKFloat> SKExportable<F> for SKHistGradientBoostingRegressorModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_HIST_GRADIENT_BOOSTING_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.feature_count,
        )?
        .with_parameter(
            "number_of_stages",
            SKParameterValue::from(self.stages.len() as i64),
        )?
        .with_parameter(
            "initial_prediction",
            SKParameterValue::from(self.initial_prediction.to_f64().unwrap_or(0.0)),
        )?
        .with_parameter(
            "learning_rate",
            SKParameterValue::from(self.learning_rate.to_f64().unwrap_or(0.1)),
        )?
        .with_parameter("loss", SKParameterValue::from(loss_name(self.loss)))?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut tensors = vec![
            SKTensor::floating(
                "hist_boosting.feature_importances",
                vec![self.feature_importances.len()],
                self.feature_importances.clone(),
            )?,
            SKTensor::floating(
                "hist_boosting.stage_importances",
                vec![self.stage_importances.iter().map(Vec::len).sum()],
                self.stage_importances.iter().flatten().copied().collect(),
            )?,
        ];
        tensors.extend(export_binnings(&self.feature_binnings)?);
        for (index, tree) in self.stages.iter().enumerate() {
            tensors.extend(export_histogram_tree(
                tree,
                &format!("hist_boosting.stage.{index}"),
            )?);
        }
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKHistGradientBoostingRegressorModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_HIST_GRADIENT_BOOSTING_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let number_of_stages =
            sk_integer_parameter(metadata, "number_of_stages", 0).max(0) as usize;
        let mut stages = Vec::with_capacity(number_of_stages);
        for index in 0..number_of_stages {
            stages.push(import_histogram_tree(
                state,
                &format!("hist_boosting.stage.{index}"),
                feature_count,
            )?);
        }
        let (_, importance_values) =
            sk_read_floating_tensor(state, "hist_boosting.stage_importances")?;
        let expected: usize = stages.iter().map(|tree| tree.feature_count()).sum();
        if importance_values.len() != expected {
            return Err(SKError::ShapeMismatch {
                expected: vec![expected],
                found: vec![importance_values.len()],
            });
        }
        let mut remaining = importance_values.as_slice();
        let mut stage_importances = Vec::with_capacity(stages.len());
        for tree in &stages {
            let (head, tail) = remaining.split_at(tree.feature_count());
            stage_importances.push(head.to_vec());
            remaining = tail;
        }
        Ok(SKHistGradientBoostingRegressorModel {
            stages,
            stage_importances,
            initial_prediction: F::from(sk_float_parameter(metadata, "initial_prediction", 0.0))
                .ok_or(SKError::Conversion(
                    "initial_prediction is not finite".to_string(),
                ))?,
            learning_rate: F::from(sk_float_parameter(metadata, "learning_rate", 0.1)).ok_or(
                SKError::Conversion("learning_rate is not finite".to_string()),
            )?,
            feature_count,
            feature_importances: read_importances(state, feature_count)?,
            feature_binnings: import_binnings(state, feature_count)?,
            loss: loss_from_name(sk_string_parameter(metadata, "loss", "squared_error")),
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}

impl<F: SKFloat> SKExportable<F> for SKHistGradientBoostingClassifierModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = SKModelMetadata::new(
            SK_HIST_GRADIENT_BOOSTING_CLASSIFIER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.feature_count,
        )?
        .with_class_labels(self.classes.labels().map(str::to_string).collect())?
        .with_parameter(
            "number_of_classes",
            SKParameterValue::from(self.number_of_classes as i64),
        )?
        .with_parameter(
            "number_of_stages",
            SKParameterValue::from(self.stages.len() as i64),
        )?
        .with_parameter(
            "learning_rate",
            SKParameterValue::from(self.learning_rate.to_f64().unwrap_or(0.1)),
        )?
        .with_parameter(
            "initial_predictions",
            SKParameterValue::List(
                self.initial_predictions
                    .iter()
                    .map(|value| SKParameterValue::from(value.to_f64().unwrap_or(0.0)))
                    .collect(),
            ),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        let mut tensors = vec![SKTensor::floating(
            "hist_boosting.feature_importances",
            vec![self.feature_importances.len()],
            self.feature_importances.clone(),
        )?];
        tensors.extend(export_binnings(&self.feature_binnings)?);
        let mut trees_per_stage = Vec::new();
        for (stage_index, trees) in self.stages.iter().enumerate() {
            trees_per_stage.push(SKParameterValue::from(trees.len() as i64));
            for (tree_index, tree) in trees.iter().enumerate() {
                tensors.extend(export_histogram_tree(
                    tree,
                    &format!("hist_boosting.stage.{stage_index}.tree.{tree_index}"),
                )?);
            }
        }
        let metadata = metadata
            .with_parameter("trees_per_stage", SKParameterValue::List(trees_per_stage))?
            .with_parameter(
                "stage_importances",
                SKParameterValue::List(
                    self.stage_importances
                        .iter()
                        .map(|stage| {
                            SKParameterValue::List(
                                stage
                                    .iter()
                                    .map(|tree| {
                                        SKParameterValue::List(
                                            tree.iter()
                                                .map(|value| {
                                                    SKParameterValue::from(
                                                        value.to_f64().unwrap_or(0.0),
                                                    )
                                                })
                                                .collect(),
                                        )
                                    })
                                    .collect(),
                            )
                        })
                        .collect(),
                ),
            )?;
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKHistGradientBoostingClassifierModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_HIST_GRADIENT_BOOSTING_CLASSIFIER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let number_of_stages =
            sk_integer_parameter(metadata, "number_of_stages", 0).max(0) as usize;
        let trees_per_stage = metadata
            .parameter("trees_per_stage")
            .and_then(SKParameterValue::as_list)
            .map(<[SKParameterValue]>::to_vec)
            .unwrap_or_default();
        let mut stages = Vec::with_capacity(number_of_stages);
        for stage_index in 0..number_of_stages {
            let count = trees_per_stage
                .get(stage_index)
                .and_then(SKParameterValue::as_integer)
                .unwrap_or(1)
                .max(0) as usize;
            let mut trees = Vec::with_capacity(count);
            for tree_index in 0..count {
                trees.push(import_histogram_tree(
                    state,
                    &format!("hist_boosting.stage.{stage_index}.tree.{tree_index}"),
                    feature_count,
                )?);
            }
            stages.push(trees);
        }
        let initial_predictions: Vec<F> = metadata
            .parameter("initial_predictions")
            .and_then(SKParameterValue::as_list)
            .map(|values| {
                values
                    .iter()
                    .filter_map(SKParameterValue::as_float)
                    .map(|value| F::from(value).unwrap_or_else(F::zero))
                    .collect()
            })
            .unwrap_or_default();
        let stage_importances: Vec<Vec<Vec<F>>> = metadata
            .parameter("stage_importances")
            .and_then(SKParameterValue::as_list)
            .map(|stages| {
                stages
                    .iter()
                    .map(|stage| {
                        stage
                            .as_list()
                            .unwrap_or_default()
                            .iter()
                            .map(|tree| {
                                tree.as_list()
                                    .unwrap_or_default()
                                    .iter()
                                    .filter_map(SKParameterValue::as_float)
                                    .map(|value| F::from(value).unwrap_or_else(F::zero))
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let labels: Vec<String> = metadata
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        Ok(SKHistGradientBoostingClassifierModel {
            stages,
            stage_importances,
            initial_predictions,
            learning_rate: F::from(sk_float_parameter(metadata, "learning_rate", 0.1)).ok_or(
                SKError::Conversion("learning_rate is not finite".to_string()),
            )?,
            classes,
            number_of_classes: sk_integer_parameter(metadata, "number_of_classes", 1).max(1)
                as usize,
            feature_count,
            feature_importances: read_importances(state, feature_count)?,
            feature_binnings: import_binnings(state, feature_count)?,
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}

fn read_importances<F: SKFloat>(
    state: &SKModelState<F>,
    feature_count: usize,
) -> Result<Vec<F>, SKError> {
    let (_, values) = sk_read_floating_tensor(state, "hist_boosting.feature_importances")?;
    if values.len() != feature_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![feature_count],
            found: vec![values.len()],
        });
    }
    Ok(values)
}

fn loss_name(loss: SKHistGradientBoostingRegressionLoss) -> &'static str {
    match loss {
        SKHistGradientBoostingRegressionLoss::SquaredError => "squared_error",
        SKHistGradientBoostingRegressionLoss::AbsoluteError => "absolute_error",
    }
}

fn loss_from_name(name: &str) -> SKHistGradientBoostingRegressionLoss {
    match name {
        "absolute_error" => SKHistGradientBoostingRegressionLoss::AbsoluteError,
        _ => SKHistGradientBoostingRegressionLoss::SquaredError,
    }
}
