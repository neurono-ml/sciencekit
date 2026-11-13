//! Format-agnostic state export/import for [`SKKNeighborsRegressorModel`].

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExportable, SKFloat, SKImportable, SKModelMetadata,
    SKModelState, SKParameterValue, SKTensor, sk_model_tensor_data_type, sk_read_floating_tensor,
    sk_string_parameter, sk_validate_import_metadata,
};

use super::SKKNeighborsRegressorModel;
use crate::neighbors::export_helpers::{
    SK_TRAINING_REPRESENTATION_PARAMETER, export_training_data, import_training_data,
    read_neighbors_configuration, with_neighbors_configuration,
};
use crate::neighbors::sk_prepare_search;

/// Stable estimator identifier for exported KNN-regressor states.
pub const SK_KNN_REGRESSOR_ESTIMATOR_NAME: &str = "SKKNeighborsRegressor";
/// The tensor carrying each training row's response.
pub const SK_KNN_RESPONSES_TENSOR: &str = "knn.responses";

impl<F: SKFloat> SKExportable<F> for SKKNeighborsRegressorModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = with_neighbors_configuration(
            SKModelMetadata::new(
                SK_KNN_REGRESSOR_ESTIMATOR_NAME,
                SK_MODEL_FORMAT_VERSION,
                sk_model_tensor_data_type::<F>(),
                self.search.number_of_features(),
            )?,
            &self.configuration,
        )?;
        let (mut tensors, representation) = export_training_data(&self.search.training, "knn")?;
        metadata = metadata.with_parameter(
            SK_TRAINING_REPRESENTATION_PARAMETER,
            SKParameterValue::from(representation),
        )?;
        tensors.push(SKTensor::floating(
            SK_KNN_RESPONSES_TENSOR,
            vec![self.responses.len()],
            self.responses.to_vec(),
        )?);
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKKNeighborsRegressorModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_KNN_REGRESSOR_ESTIMATOR_NAME,
            SK_MODEL_FORMAT_VERSION,
            feature_count,
        )?;
        let configuration = read_neighbors_configuration(state.metadata())?;
        let representation = sk_string_parameter(
            state.metadata(),
            SK_TRAINING_REPRESENTATION_PARAMETER,
            "dense",
        );
        let training = import_training_data(state, "knn", representation)?;
        let search = sk_prepare_search(training.view(), &configuration)?;
        let (_, responses) = sk_read_floating_tensor(state, SK_KNN_RESPONSES_TENSOR)?;
        if responses.len() != search.training_rows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![search.training_rows()],
                found: vec![responses.len()],
            });
        }
        Ok(SKKNeighborsRegressorModel {
            search,
            configuration,
            responses: Array1::from_vec(responses),
        })
    }
}
