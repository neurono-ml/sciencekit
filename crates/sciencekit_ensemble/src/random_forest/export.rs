//! Format-agnostic export/import of the random-forest models.
//!
//! Each fitted tree is itself a contract model, so its whole exported state is
//! embedded under a `forest.tree.<index>` namespace. A pure-leaf classifier
//! member is stored as its constant class index instead of a tree.

use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable,
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, sk_backend_parameter,
    sk_boolean_parameter, sk_canonicalize_labels, sk_float_parameter, sk_integer_parameter,
    sk_model_tensor_data_type, sk_parse_backend_parameter, sk_string_parameter,
    sk_validate_import_metadata,
};
use sciencekit_tree::classifier::SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME;
use sciencekit_tree::regressor::SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME;
use sciencekit_tree::{SKDecisionTreeClassifierModel, SKDecisionTreeRegressorModel};

use super::classifier_model::{
    SKClassifierTreeMember, SKForestOobState, SKRandomForestClassifierModel,
};
use super::regressor_model::SKRandomForestRegressorModel;
use crate::export_support::{criterion_from_name, criterion_name};

/// Stable estimator identifier for exported random-forest classifier states.
pub const SK_RANDOM_FOREST_CLASSIFIER_ESTIMATOR_NAME: &str = "SKRandomForestClassifier";
/// Stable estimator identifier for exported random-forest regressor states.
pub const SK_RANDOM_FOREST_REGRESSOR_ESTIMATOR_NAME: &str = "SKRandomForestRegressor";

impl<F: SKFloat> SKExportable<F> for SKRandomForestClassifierModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = SKModelMetadata::new(
            SK_RANDOM_FOREST_CLASSIFIER_ESTIMATOR_NAME,
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
            "number_of_trees",
            SKParameterValue::from(self.members.len() as i64),
        )?
        .with_parameter(
            "criterion",
            SKParameterValue::from(criterion_name(self.criterion)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        metadata = with_oob_metadata(metadata, &self.oob_state)?;
        let mut tensors = vec![SKTensor::floating(
            "forest.feature_importances",
            vec![self.feature_importances.len()],
            self.feature_importances.clone(),
        )?];
        for (index, member) in self.members.iter().enumerate() {
            match member {
                SKClassifierTreeMember::Fitted {
                    model,
                    local_to_forest,
                } => {
                    metadata = metadata.with_parameter(
                        format!("forest.tree.{index}.kind"),
                        SKParameterValue::from("fitted"),
                    )?;
                    metadata = metadata.with_parameter(
                        format!("forest.tree.{index}.local_to_forest"),
                        SKParameterValue::List(
                            local_to_forest
                                .iter()
                                .map(|&class| SKParameterValue::from(class as i64))
                                .collect(),
                        ),
                    )?;
                    let state = SKExportable::<F>::export_state(model)?;
                    metadata = embed_parameters(
                        metadata,
                        &format!("forest.tree.{index}"),
                        state.metadata(),
                    )?;
                    tensors.extend(embed_tensors(&format!("forest.tree.{index}"), &state)?);
                }
                SKClassifierTreeMember::PureLeaf { class_index } => {
                    metadata = metadata
                        .with_parameter(
                            format!("forest.tree.{index}.kind"),
                            SKParameterValue::from("pure_leaf"),
                        )?
                        .with_parameter(
                            format!("forest.tree.{index}.class_index"),
                            SKParameterValue::from(*class_index as i64),
                        )?;
                }
            }
        }
        metadata = with_out_of_bag_rows(metadata, self.out_of_bag_rows.as_deref())?;
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKRandomForestClassifierModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_RANDOM_FOREST_CLASSIFIER_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let number_of_trees = sk_integer_parameter(metadata, "number_of_trees", 0).max(0) as usize;
        let mut members = Vec::with_capacity(number_of_trees);
        for index in 0..number_of_trees {
            let prefix = format!("forest.tree.{index}");
            let kind = sk_string_parameter(metadata, &format!("{prefix}.kind"), "fitted");
            if kind == "pure_leaf" {
                members.push(SKClassifierTreeMember::PureLeaf {
                    class_index: sk_integer_parameter(metadata, &format!("{prefix}.class_index"), 0)
                        .max(0) as usize,
                });
            } else {
                let tree_state = embedded_state(
                    state,
                    &prefix,
                    SK_DECISION_TREE_CLASSIFIER_ESTIMATOR_NAME,
                    feature_count,
                )?;
                let model = SKDecisionTreeClassifierModel::import_state(&tree_state)?;
                let local_to_forest: Vec<usize> = metadata
                    .parameter(&format!("{prefix}.local_to_forest"))
                    .and_then(SKParameterValue::as_list)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(SKParameterValue::as_integer)
                            .map(|class| class.max(0) as usize)
                            .collect()
                    })
                    .unwrap_or_default();
                members.push(SKClassifierTreeMember::Fitted {
                    model,
                    local_to_forest,
                });
            }
        }
        let labels: Vec<String> = metadata
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        Ok(SKRandomForestClassifierModel {
            members,
            out_of_bag_rows: read_out_of_bag_rows(state, number_of_trees)?,
            oob_state: read_oob_state(state)?,
            classes,
            number_of_classes: sk_integer_parameter(metadata, "number_of_classes", 1).max(1)
                as usize,
            feature_count,
            feature_importances: read_importances(state)?,
            criterion: criterion_from_name(sk_string_parameter(metadata, "criterion", "gini")),
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}

impl<F: SKFloat> SKExportable<F> for SKRandomForestRegressorModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = SKModelMetadata::new(
            SK_RANDOM_FOREST_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            sk_model_tensor_data_type::<F>(),
            self.feature_count,
        )?
        .with_parameter(
            "number_of_trees",
            SKParameterValue::from(self.trees.len() as i64),
        )?
        .with_parameter(
            "criterion",
            SKParameterValue::from(criterion_name(self.criterion)),
        )?
        .with_parameter("backend", sk_backend_parameter(self.backend))?;
        metadata = with_oob_metadata(metadata, &self.oob_state)?;
        let mut tensors = vec![SKTensor::floating(
            "forest.feature_importances",
            vec![self.feature_importances.len()],
            self.feature_importances.clone(),
        )?];
        for (index, tree) in self.trees.iter().enumerate() {
            let prefix = format!("forest.tree.{index}");
            let tree_state = SKExportable::<F>::export_state(tree)?;
            metadata = embed_parameters(metadata, &prefix, tree_state.metadata())?;
            tensors.extend(embed_tensors(&prefix, &tree_state)?);
        }
        metadata = with_out_of_bag_rows(metadata, self.out_of_bag_rows.as_deref())?;
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKRandomForestRegressorModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_RANDOM_FOREST_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let metadata = state.metadata();
        let number_of_trees = sk_integer_parameter(metadata, "number_of_trees", 0).max(0) as usize;
        let mut trees = Vec::with_capacity(number_of_trees);
        for index in 0..number_of_trees {
            let prefix = format!("forest.tree.{index}");
            let tree_state = embedded_state(
                state,
                &prefix,
                SK_DECISION_TREE_REGRESSOR_ESTIMATOR_NAME,
                feature_count,
            )?;
            trees.push(SKDecisionTreeRegressorModel::import_state(&tree_state)?);
        }
        Ok(SKRandomForestRegressorModel {
            trees,
            out_of_bag_rows: read_out_of_bag_rows(state, number_of_trees)?,
            oob_state: read_oob_state(state)?,
            feature_count,
            feature_importances: read_importances(state)?,
            criterion: criterion_from_name(sk_string_parameter(
                metadata,
                "criterion",
                "squared_error",
            )),
            backend: sk_parse_backend_parameter(metadata),
            execution_intent: SKExecutionMode::Automatic,
        })
    }
}

fn embed_parameters(
    mut metadata: SKModelMetadata,
    prefix: &str,
    inner: &SKModelMetadata,
) -> Result<SKModelMetadata, SKError> {
    for (name, value) in inner.parameters() {
        metadata = metadata.with_parameter(format!("{prefix}.p.{name}"), value.clone())?;
    }
    Ok(metadata)
}

fn embed_tensors<F: SKFloat>(
    prefix: &str,
    state: &SKModelState<F>,
) -> Result<Vec<SKTensor<F>>, SKError> {
    state
        .tensors()
        .iter()
        .map(|(name, tensor)| {
            SKTensor::new(
                format!("{prefix}.t.{name}"),
                tensor.data_type(),
                tensor.shape().to_vec(),
                tensor.values().clone(),
            )
        })
        .collect()
}

fn embedded_state<F: SKFloat>(
    state: &SKModelState<F>,
    prefix: &str,
    estimator_name: &str,
    feature_count: usize,
) -> Result<SKModelState<F>, SKError> {
    let parameter_prefix = format!("{prefix}.p.");
    let tensor_prefix = format!("{prefix}.t.");
    let mut metadata = SKModelMetadata::new(
        estimator_name,
        SK_MODEL_FORMAT_VERSION,
        sk_model_tensor_data_type::<F>(),
        feature_count,
    )?;
    for (name, value) in state.metadata().parameters() {
        if let Some(inner) = name.strip_prefix(&parameter_prefix) {
            metadata = metadata.with_parameter(inner, value.clone())?;
        }
    }
    if let Some(labels) = state.metadata().class_labels() {
        metadata = metadata.with_class_labels(labels.to_vec())?;
    }
    let mut tensors = Vec::new();
    for (name, tensor) in state.tensors() {
        if let Some(inner) = name.strip_prefix(&tensor_prefix) {
            tensors.push(SKTensor::new(
                inner,
                tensor.data_type(),
                tensor.shape().to_vec(),
                tensor.values().clone(),
            )?);
        }
    }
    SKModelState::new(metadata, tensors)
}

fn with_oob_metadata<F: SKFloat>(
    mut metadata: SKModelMetadata,
    oob_state: &SKForestOobState<F>,
) -> Result<SKModelMetadata, SKError> {
    match oob_state {
        SKForestOobState::Disabled => {
            metadata = metadata.with_parameter("oob_kind", SKParameterValue::from("disabled"))?;
        }
        SKForestOobState::NoScoredRows => {
            metadata =
                metadata.with_parameter("oob_kind", SKParameterValue::from("no_scored_rows"))?;
        }
        SKForestOobState::Scored { score, scored_rows } => {
            metadata = metadata
                .with_parameter("oob_kind", SKParameterValue::from("scored"))?
                .with_parameter(
                    "oob_score",
                    SKParameterValue::from(score.to_f64().unwrap_or(0.0)),
                )?
                .with_parameter(
                    "oob_scored_rows",
                    SKParameterValue::from(*scored_rows as i64),
                )?;
        }
    }
    Ok(metadata)
}

fn read_oob_state<F: SKFloat>(state: &SKModelState<F>) -> Result<SKForestOobState<F>, SKError> {
    match sk_string_parameter(state.metadata(), "oob_kind", "disabled") {
        "scored" => Ok(SKForestOobState::Scored {
            score: F::from(sk_float_parameter(state.metadata(), "oob_score", 0.0))
                .ok_or(SKError::Conversion("oob_score is not finite".to_string()))?,
            scored_rows: sk_integer_parameter(state.metadata(), "oob_scored_rows", 0).max(0)
                as usize,
        }),
        "no_scored_rows" => Ok(SKForestOobState::NoScoredRows),
        _ => Ok(SKForestOobState::Disabled),
    }
}

fn with_out_of_bag_rows(
    mut metadata: SKModelMetadata,
    rows: Option<&[Vec<usize>]>,
) -> Result<SKModelMetadata, SKError> {
    if let Some(rows) = rows {
        let lists = rows
            .iter()
            .map(|members| {
                SKParameterValue::List(
                    members
                        .iter()
                        .map(|&row| SKParameterValue::from(row as i64))
                        .collect(),
                )
            })
            .collect();
        metadata = metadata
            .with_parameter("out_of_bag_present", SKParameterValue::from(true))?
            .with_parameter("out_of_bag_rows", SKParameterValue::List(lists))?;
    } else {
        metadata = metadata.with_parameter("out_of_bag_present", SKParameterValue::from(false))?;
    }
    Ok(metadata)
}

fn read_out_of_bag_rows<F: SKFloat>(
    state: &SKModelState<F>,
    _number_of_trees: usize,
) -> Result<Option<Vec<Vec<usize>>>, SKError> {
    if !sk_boolean_parameter(state.metadata(), "out_of_bag_present", false) {
        return Ok(None);
    }
    let rows = state
        .metadata()
        .parameter("out_of_bag_rows")
        .and_then(SKParameterValue::as_list)
        .map(<[SKParameterValue]>::to_vec)
        .ok_or(SKError::InvalidHyperparameter {
            name: "out_of_bag_rows",
            reason: "the exported state does not carry the out-of-bag rows".to_string(),
        })?;
    let decoded: Vec<Vec<usize>> = rows
        .iter()
        .map(|members| {
            members
                .as_list()
                .unwrap_or_default()
                .iter()
                .filter_map(SKParameterValue::as_integer)
                .map(|row| row.max(0) as usize)
                .collect()
        })
        .collect();
    Ok(Some(decoded))
}

fn read_importances<F: SKFloat>(state: &SKModelState<F>) -> Result<Vec<F>, SKError> {
    let (_, values) =
        sciencekit_common::sk_read_floating_tensor(state, "forest.feature_importances")?;
    Ok(values)
}
