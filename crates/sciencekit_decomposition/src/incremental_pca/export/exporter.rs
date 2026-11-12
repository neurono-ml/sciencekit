//! `SKExportable` implementation for [`crate::incremental_pca::SKIncrementalPCAModel`].

use std::mem::size_of;

use sciencekit_common::{
    SKError, SKExportable, SKFloat, SKModelMetadata, SKModelState, SKParameterValue, SKTensor,
    SKTensorDataType,
};

use crate::incremental_pca::SKIncrementalPCAModel;

pub(super) const ESTIMATOR_NAME: &str = "SKIncrementalPCA";
pub(super) const FORMAT_VERSION: u32 = 1;
pub(super) const COMPONENTS_TENSOR: &str = "incremental_pca.components";
pub(super) const MEAN_TENSOR: &str = "incremental_pca.mean";
pub(super) const SCATTER_TENSOR: &str = "incremental_pca.scatter";
pub(super) const EXPLAINED_VARIANCE_TENSOR: &str = "incremental_pca.explained_variance";
pub(super) const EXPLAINED_VARIANCE_RATIO_TENSOR: &str = "incremental_pca.explained_variance_ratio";
pub(super) const SINGULAR_VALUES_TENSOR: &str = "incremental_pca.singular_values";

impl<F: SKFloat> SKExportable<F> for SKIncrementalPCAModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let components_count = self.number_of_components();
        let features_count = self.number_of_features();
        let metadata = SKModelMetadata::new(
            ESTIMATOR_NAME,
            FORMAT_VERSION,
            sk_tensor_data_type::<F>(),
            features_count,
        )?
        .with_parameter(
            "number_of_components",
            SKParameterValue::from(components_count as i64),
        )?
        .with_parameter(
            "batch_size",
            SKParameterValue::from(self.batch_size() as i64),
        )?
        .with_parameter("whiten", SKParameterValue::from(self.whiten()))?
        .with_parameter(
            "samples_seen",
            SKParameterValue::from(self.samples_seen() as i64),
        )?
        .with_parameter(
            "epochs_consumed",
            SKParameterValue::from(self.epochs_consumed() as i64),
        )?
        .with_parameter("backend", SKParameterValue::from(self.backend().name()))?;

        let tensors = vec![
            SKTensor::floating(
                COMPONENTS_TENSOR,
                vec![components_count, features_count],
                self.components().iter().copied().collect(),
            )?,
            SKTensor::floating(
                MEAN_TENSOR,
                vec![features_count],
                self.mean().iter().copied().collect(),
            )?,
            SKTensor::floating(
                SCATTER_TENSOR,
                vec![features_count, features_count],
                self.scatter_matrix().iter().copied().collect(),
            )?,
            SKTensor::floating(
                EXPLAINED_VARIANCE_TENSOR,
                vec![components_count],
                self.explained_variance().to_vec(),
            )?,
            SKTensor::floating(
                EXPLAINED_VARIANCE_RATIO_TENSOR,
                vec![components_count],
                self.explained_variance_ratio().to_vec(),
            )?,
            SKTensor::floating(
                SINGULAR_VALUES_TENSOR,
                vec![components_count],
                self.singular_values().to_vec(),
            )?,
        ];
        SKModelState::new(metadata, tensors)
    }
}

fn sk_tensor_data_type<F: SKFloat>() -> SKTensorDataType {
    if size_of::<F>() == size_of::<f32>() {
        SKTensorDataType::F32
    } else {
        SKTensorDataType::F64
    }
}
