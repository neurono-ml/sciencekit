//! Format-agnostic state export/import for [`SKKNeighborsClassifierModel`].

use ndarray::Array1;
use sciencekit_common::{
    SK_MODEL_FORMAT_VERSION, SKError, SKExportable, SKFloat, SKImportable, SKModelMetadata,
    SKModelState, SKParameterValue, SKTensor, sk_canonicalize_labels, sk_model_tensor_data_type,
    sk_read_integer_tensor, sk_string_parameter, sk_validate_import_metadata,
};

use super::SKKNeighborsClassifierModel;
use crate::neighbors::export_helpers::{
    SK_TRAINING_REPRESENTATION_PARAMETER, export_training_data, import_training_data,
    read_neighbors_configuration, with_neighbors_configuration,
};
use crate::neighbors::sk_prepare_search;

/// Stable estimator identifier for exported KNN-classifier states.
pub const SK_KNN_CLASSIFIER_ESTIMATOR_NAME: &str = "SKKNeighborsClassifier";
/// The tensor carrying each training row's canonical class index.
pub const SK_KNN_CLASS_INDICES_TENSOR: &str = "knn.class_indices";

impl<F: SKFloat> SKExportable<F> for SKKNeighborsClassifierModel<F> {
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let mut metadata = with_neighbors_configuration(
            SKModelMetadata::new(
                SK_KNN_CLASSIFIER_ESTIMATOR_NAME,
                SK_MODEL_FORMAT_VERSION,
                sk_model_tensor_data_type::<F>(),
                self.search.number_of_features(),
            )?
            .with_class_labels(self.classes.labels().map(str::to_string).collect())?,
            &self.configuration,
        )?;
        let (mut tensors, representation) = export_training_data(&self.search.training, "knn")?;
        metadata = metadata.with_parameter(
            SK_TRAINING_REPRESENTATION_PARAMETER,
            SKParameterValue::from(representation),
        )?;
        tensors.push(SKTensor::integer(
            SK_KNN_CLASS_INDICES_TENSOR,
            vec![self.class_indices.len()],
            self.class_indices
                .iter()
                .map(|&index| index as i64)
                .collect(),
        )?);
        SKModelState::new(metadata, tensors)
    }
}

impl<F: SKFloat> SKImportable<F> for SKKNeighborsClassifierModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let feature_count = state.metadata().feature_count();
        sk_validate_import_metadata(
            state,
            SK_KNN_CLASSIFIER_ESTIMATOR_NAME,
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
        let class_indices: Vec<usize> = sk_read_integer_tensor(state, SK_KNN_CLASS_INDICES_TENSOR)?
            .into_iter()
            .map(|index| index.max(0) as usize)
            .collect();
        if class_indices.len() != search.training_rows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![search.training_rows()],
                found: vec![class_indices.len()],
            });
        }
        let labels: Vec<String> = state
            .metadata()
            .class_labels()
            .ok_or(SKError::InvalidHyperparameter {
                name: "class_labels",
                reason: "a classifier state must carry its class table".to_string(),
            })?
            .to_vec();
        let references: Vec<&str> = labels.iter().map(String::as_str).collect();
        let (_, classes) = sk_canonicalize_labels(&references);
        Ok(SKKNeighborsClassifierModel {
            search,
            configuration,
            classes,
            class_indices: Array1::from_vec(class_indices),
        })
    }
}
