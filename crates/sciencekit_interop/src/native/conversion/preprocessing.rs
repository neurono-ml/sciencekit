//! The transformer and clusterer arms of the native conversion table
//! (spec `native-model-import`, requirement "Safetensors native import").

use sciencekit_cluster::kmeans::{SK_KMEANS_ESTIMATOR_NAME, SKKMeansModel};
use sciencekit_common::{SKFloat, SKModelState};
use sciencekit_preprocessing::minmax_scaler::model::{
    SK_MINMAX_SCALER_ESTIMATOR_NAME, SKMinMaxScalerModel,
};
use sciencekit_preprocessing::robust_scaler::model::{
    SK_ROBUST_SCALER_ESTIMATOR_NAME, SKRobustScalerModel,
};
use sciencekit_preprocessing::standard_scaler::model::{
    SK_STANDARD_SCALER_ESTIMATOR_NAME, SKStandardScalerModel,
};

use super::dispatch::{descriptor, import_model, not_retrainable};
use super::tensor_names;
use super::validation::validate_tensor_names;
use crate::native::conversion::result::SKNativeImport;
use crate::native::error::SKNativeImportError;
use crate::native::model::{SKNativeClusterer, SKNativeTransformer};

/// Convert a preprocessing state, or return `None` when it is not a transformer.
pub(crate) fn convert_transformer<F: SKFloat>(
    state: &SKModelState<F>,
    estimator_name: &str,
) -> Result<Option<SKNativeImport<F>>, SKNativeImportError> {
    let model = match estimator_name {
        SK_STANDARD_SCALER_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::STANDARD_SCALER)?;
            let model: SKStandardScalerModel<F> = import_model(state)?;
            SKNativeTransformer::StandardScaler(model)
        }
        SK_MINMAX_SCALER_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::MINMAX_SCALER)?;
            let model: SKMinMaxScalerModel<F> = import_model(state)?;
            SKNativeTransformer::MinMaxScaler(model)
        }
        SK_ROBUST_SCALER_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::ROBUST_SCALER)?;
            let model: SKRobustScalerModel<F> = import_model(state)?;
            SKNativeTransformer::RobustScaler(model)
        }
        _ => return Ok(None),
    };
    Ok(Some(SKNativeImport::Transformer {
        model,
        descriptor: descriptor(
            state,
            not_retrainable("a fitted preprocessing state is inference-only"),
        ),
    }))
}

/// Convert a clustering state, or return `None` when it is not a clusterer.
pub(crate) fn convert_clusterer<F: SKFloat>(
    state: &SKModelState<F>,
    estimator_name: &str,
) -> Result<Option<SKNativeImport<F>>, SKNativeImportError> {
    let model = match estimator_name {
        SK_KMEANS_ESTIMATOR_NAME => {
            validate_tensor_names(state, tensor_names::KMEANS)?;
            let model: SKKMeansModel<F> = import_model(state)?;
            SKNativeClusterer::KMeans(model)
        }
        _ => return Ok(None),
    };
    Ok(Some(SKNativeImport::Clusterer {
        model,
        descriptor: descriptor(
            state,
            not_retrainable("a fitted centroid table is inference-only"),
        ),
    }))
}
