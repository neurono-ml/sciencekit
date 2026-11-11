//! Format-agnostic `SKExportable` implementations for fitted bagging models
//! whose base models implement `SKExportable`.

use sciencekit_common::{
    SKError, SKExportable, SKFloat, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    SKTensorDataType,
};

use super::model::{
    SKBaggingClassifierMember, SKBaggingClassifierModel, SKBaggingOobState, SKBaggingRegressorModel,
};

/// Stable estimator name carried by exported bagging classifier states.
pub const SK_BAGGING_CLASSIFIER_ESTIMATOR_NAME: &str = "SKBaggingClassifier";
/// Stable estimator name carried by exported bagging regressor states.
pub const SK_BAGGING_REGRESSOR_ESTIMATOR_NAME: &str = "SKBaggingRegressor";
/// Format version for exported bagging states.
pub const SK_BAGGING_FORMAT_VERSION: u32 = 1;

impl<F: SKFloat, M: SKExportable<F>> SKExportable<F> for SKBaggingClassifierModel<F, M> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let labels: Vec<String> = self.classes.labels().map(ToString::to_string).collect();
        let metadata = sk_bagging_metadata::<F>(
            SK_BAGGING_CLASSIFIER_ESTIMATOR_NAME,
            self.feature_count,
            self.members.len(),
            self.bootstrap,
            self.bootstrap_features,
            self.oob_scored_rows(),
            format!("{:?}", self.backend),
        )?
        .with_class_labels(labels)?;

        let mut tensors =
            sk_feature_subset_and_oob_tensors(&self.feature_subsets, &self.oob_state)?;
        self.members
            .iter()
            .enumerate()
            .try_for_each(|(index, member)| match member {
                SKBaggingClassifierMember::Fitted {
                    model,
                    local_to_ensemble,
                } => {
                    let mapping: Vec<i64> = local_to_ensemble
                        .iter()
                        .map(|&class| class as i64)
                        .collect();
                    tensors.push(SKTensor::integer(
                        format!("estimator.{index}.local_to_ensemble"),
                        vec![mapping.len()],
                        mapping,
                    )?);
                    sk_append_prefixed_base_tensors(&mut tensors, index, model)
                }
                SKBaggingClassifierMember::ConstantClass { class_index } => {
                    tensors.push(SKTensor::integer(
                        format!("estimator.{index}.constant_class"),
                        Vec::new(),
                        vec![*class_index as i64],
                    )?);
                    Ok(())
                }
            })?;

        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat, M: SKExportable<F>> SKExportable<F> for SKBaggingRegressorModel<F, M> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let metadata = sk_bagging_metadata::<F>(
            SK_BAGGING_REGRESSOR_ESTIMATOR_NAME,
            self.feature_count,
            self.estimators.len(),
            self.bootstrap,
            self.bootstrap_features,
            self.oob_scored_rows(),
            format!("{:?}", self.backend),
        )?;

        let mut tensors =
            sk_feature_subset_and_oob_tensors(&self.feature_subsets, &self.oob_state)?;
        self.estimators
            .iter()
            .enumerate()
            .try_for_each(|(index, estimator)| {
                sk_append_prefixed_base_tensors(&mut tensors, index, estimator)
            })?;

        SKModelState::new(metadata, tensors)
    }
}

fn sk_bagging_metadata<F: SKFloat>(
    estimator_name: &str,
    feature_count: usize,
    number_of_estimators: usize,
    bootstrap: bool,
    bootstrap_features: bool,
    oob_scored_rows: usize,
    backend_label: String,
) -> Result<SKModelMetadata, SKError> {
    let scalar_data_type = if size_of::<F>() == size_of::<f32>() {
        SKTensorDataType::F32
    } else {
        SKTensorDataType::F64
    };
    SKModelMetadata::new(
        estimator_name,
        SK_BAGGING_FORMAT_VERSION,
        scalar_data_type,
        feature_count,
    )?
    .with_parameter(
        "number_of_estimators",
        SKParameterValue::from(number_of_estimators as i64),
    )?
    .with_parameter("bootstrap", SKParameterValue::from(bootstrap))?
    .with_parameter(
        "bootstrap_features",
        SKParameterValue::from(bootstrap_features),
    )?
    .with_parameter(
        "oob_scored_rows",
        SKParameterValue::from(oob_scored_rows as i64),
    )?
    .with_parameter("backend", SKParameterValue::from(backend_label))
}

fn sk_feature_subset_and_oob_tensors<F: SKFloat>(
    feature_subsets: &[Vec<usize>],
    oob_state: &SKBaggingOobState<F>,
) -> Result<Vec<SKTensor<F>>, SKError> {
    let subset_width = feature_subsets.first().map_or(0, Vec::len);
    let flat_subsets: Vec<i64> = feature_subsets
        .iter()
        .flat_map(|subset| subset.iter().map(|&column| column as i64))
        .collect();
    let mut tensors = vec![SKTensor::integer(
        "bagging.feature_subsets",
        vec![feature_subsets.len(), subset_width],
        flat_subsets,
    )?];
    if let SKBaggingOobState::Scored { score, .. } = oob_state {
        tensors.push(SKTensor::floating(
            "bagging.oob_score",
            Vec::new(),
            vec![*score],
        )?);
    }
    Ok(tensors)
}

fn sk_append_prefixed_base_tensors<F: SKFloat, M: SKExportable<F>>(
    tensors: &mut Vec<SKTensor<F>>,
    index: usize,
    model: &M,
) -> Result<(), SKError> {
    let base_state = model.export_state()?;
    base_state.tensors().values().try_for_each(|tensor| {
        let renamed = SKTensor::new(
            format!("estimator.{index}.{}", tensor.name()),
            tensor.data_type(),
            tensor.shape().to_vec(),
            tensor.values().clone(),
        )?;
        tensors.push(renamed);
        Ok(())
    })
}
