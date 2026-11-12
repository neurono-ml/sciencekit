//! `SKImportable` implementation for [`crate::incremental_pca::SKIncrementalPCAModel`].

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKFloat, SKImportable, SKModelState, SKParameterValue,
    SKTensorValues, sk_validate_import_metadata,
};

use super::exporter::{
    COMPONENTS_TENSOR, ESTIMATOR_NAME, EXPLAINED_VARIANCE_RATIO_TENSOR, EXPLAINED_VARIANCE_TENSOR,
    FORMAT_VERSION, MEAN_TENSOR, SCATTER_TENSOR, SINGULAR_VALUES_TENSOR,
};
use crate::incremental_pca::SKIncrementalPCAModel;
use crate::incremental_pca::accumulator::SKCovarianceAccumulator;
use crate::incremental_pca::config::{
    DEFAULT_BATCH_SIZE, DEFAULT_MAXIMUM_NUMBER_OF_EPOCHS, SKIncrementalPCAConfiguration,
};
use crate::pca::SKPCADecomposition;

impl<F: SKFloat> SKImportable<F> for SKIncrementalPCAModel<F> {
    fn import_state(state: &SKModelState<F>) -> Result<Self, SKError> {
        let features_count = state.metadata().feature_count();
        sk_validate_import_metadata(state, ESTIMATOR_NAME, FORMAT_VERSION, features_count)?;

        let components_count = state
            .metadata()
            .parameter("number_of_components")
            .and_then(SKParameterValue::as_integer)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| SKError::InvalidHyperparameter {
                name: "number_of_components",
                reason: "missing or negative component count in exported metadata".to_string(),
            })?;
        let batch_size = state
            .metadata()
            .parameter("batch_size")
            .and_then(SKParameterValue::as_integer)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(DEFAULT_BATCH_SIZE);
        let whiten = state
            .metadata()
            .parameter("whiten")
            .and_then(SKParameterValue::as_bool)
            .unwrap_or(false);
        let samples_seen = state
            .metadata()
            .parameter("samples_seen")
            .and_then(SKParameterValue::as_integer)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        let epochs_consumed = state
            .metadata()
            .parameter("epochs_consumed")
            .and_then(SKParameterValue::as_integer)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(1);
        let backend = state
            .metadata()
            .parameter("backend")
            .and_then(SKParameterValue::as_string)
            .and_then(|name| name.parse::<SKBackendKind>().ok())
            .unwrap_or(SKBackendKind::Faer);

        let components = sk_read_2d(state, COMPONENTS_TENSOR, components_count, features_count)?;
        let mean = sk_read_1d(state, MEAN_TENSOR, features_count)?;
        let scatter = sk_read_2d(state, SCATTER_TENSOR, features_count, features_count)?;
        let explained_variance = sk_read_1d(state, EXPLAINED_VARIANCE_TENSOR, components_count)?;
        let explained_variance_ratio =
            sk_read_1d(state, EXPLAINED_VARIANCE_RATIO_TENSOR, components_count)?;
        let singular_values = sk_read_1d(state, SINGULAR_VALUES_TENSOR, components_count)?;

        let accumulator = SKCovarianceAccumulator {
            samples_seen,
            mean,
            scatter,
        };
        let decomposition = SKPCADecomposition {
            components,
            explained_variance,
            explained_variance_ratio,
            singular_values,
        };
        let configuration = SKIncrementalPCAConfiguration {
            number_of_components: Some(components_count),
            batch_size,
            whiten,
            maximum_number_of_epochs: DEFAULT_MAXIMUM_NUMBER_OF_EPOCHS,
            backend,
            builder_state: SKBuilderState::new(),
        };
        Ok(Self::from_accumulator_and_decomposition(
            accumulator,
            decomposition,
            epochs_consumed,
            configuration,
        ))
    }
}

fn sk_read_1d<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
    expected_length: usize,
) -> Result<Array1<F>, SKError> {
    let values = sk_floating_values(state, name)?;
    if values.len() != expected_length {
        return Err(SKError::ShapeMismatch {
            expected: vec![expected_length],
            found: vec![values.len()],
        });
    }
    Ok(Array1::from_vec(values))
}

fn sk_read_2d<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
    rows: usize,
    columns: usize,
) -> Result<Array2<F>, SKError> {
    let values = sk_floating_values(state, name)?;
    Array2::from_shape_vec((rows, columns), values).map_err(|_| SKError::ShapeMismatch {
        expected: vec![rows, columns],
        found: vec![0],
    })
}

fn sk_floating_values<F: SKFloat>(
    state: &SKModelState<F>,
    name: &'static str,
) -> Result<Vec<F>, SKError> {
    let tensor = state
        .tensor(name)
        .ok_or_else(|| SKError::InvalidHyperparameter {
            name,
            reason: "exported state does not carry this tensor".to_string(),
        })?;
    match tensor.values() {
        SKTensorValues::Floating(values) => Ok(values.clone()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "non-floating tensor",
            suggestion: "export this tensor with floating values",
        }),
    }
}
