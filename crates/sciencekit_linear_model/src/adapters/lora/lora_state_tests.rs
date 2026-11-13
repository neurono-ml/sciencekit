//! Companion tests for adapter export/import (task 3.2).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKError, SKExportable, SKImportable, SKModelState, SKRegressorPredictor, SKSupervisedFit,
    SKTensor, SKTensorValues,
};

use crate::adapters::lora::{
    SKLoraAdapterArtifact, SKLoraAdapterBuilder, SKLoraBaseModel, SKLoraTargetModule,
    sk_lora_a_tensor_name, sk_lora_b_tensor_name,
};
use crate::linear_regression::core_implementation::SKLinearRegression;

fn linear_dataset() -> (Array2<f64>, Array1<f64>) {
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0]
    ];
    let targets = features.dot(&array![2.0, -3.0]) + 1.0;
    (features, targets)
}

fn linear_base() -> SKLoraBaseModel<f64> {
    let (features, targets) = linear_dataset();
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    SKLoraBaseModel::linear_regression(model)
}

fn fitted_adapter() -> crate::adapters::lora::SKLoraAttachedModel<f64> {
    let features = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let targets = features.dot(&array![2.7, -2.6]) + 1.0;
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .identifier("ticket-classifier")
        .rank(2)
        .alpha(4.0)
        .maximum_number_of_iterations(25)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    attached.fit(features.view(), targets.view()).unwrap();
    attached
}

#[test]
fn a_round_trip_preserves_predictions_exactly() {
    let attached = fitted_adapter();
    let features = array![[0.25, 0.5], [1.5, -0.5], [-1.0, 2.0]];

    let state = attached.export_state().unwrap();
    let artifact = SKLoraAdapterArtifact::<f64>::import_state(&state).unwrap();
    let restored = artifact.attach(linear_base()).unwrap();

    assert_eq!(
        restored.predict(features.view()).unwrap().to_vec(),
        attached.predict(features.view()).unwrap().to_vec()
    );
    assert!(restored.is_fitted());
}

#[test]
fn the_exported_metadata_names_rank_alpha_and_target() {
    let state = fitted_adapter().export_state().unwrap();
    let metadata = state.metadata();
    assert_eq!(metadata.estimator_name(), "SKLoraAdapter");
    assert_eq!(
        metadata
            .parameter("rank")
            .and_then(|value| value.as_integer()),
        Some(2)
    );
    assert_eq!(
        metadata
            .parameter("alpha")
            .and_then(|value| value.as_float()),
        Some(4.0)
    );
    assert_eq!(
        metadata
            .parameter("target")
            .and_then(|value| value.as_string()),
        Some(SKLoraTargetModule::LinearCoefficients.parameter_name())
    );
    assert_eq!(
        metadata
            .parameter("identifier")
            .and_then(|value| value.as_string()),
        Some("ticket-classifier")
    );
}

#[test]
fn the_exported_tensors_use_the_documented_names_and_shapes() {
    let state = fitted_adapter().export_state().unwrap();
    let target = SKLoraTargetModule::LinearCoefficients;
    let a = state.tensor(&sk_lora_a_tensor_name(target)).unwrap();
    let b = state.tensor(&sk_lora_b_tensor_name(target)).unwrap();
    assert_eq!(a.shape(), &[2, 2]);
    assert_eq!(b.shape(), &[1, 2]);
}

#[test]
fn an_adapter_can_be_imported_without_a_fit_flag() {
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let state = attached.export_state().unwrap();
    let artifact = SKLoraAdapterArtifact::<f64>::import_state(&state).unwrap();
    assert!(!artifact.is_fitted());
    assert!(!artifact.attach(linear_base()).unwrap().is_fitted());
}

#[test]
fn a_tampered_factor_shape_is_rejected_without_a_panic() {
    let attached = fitted_adapter();
    let state = attached.export_state().unwrap();
    let target = SKLoraTargetModule::LinearCoefficients;
    let mut tensors = state.tensors().values().cloned().collect::<Vec<_>>();
    tensors.retain(|tensor| tensor.name() != sk_lora_b_tensor_name(target));
    tensors
        .push(SKTensor::floating(sk_lora_b_tensor_name(target), vec![3, 2], vec![0.0; 6]).unwrap());
    let tampered = SKModelState::new(state.metadata().clone(), tensors).unwrap();

    let error = SKLoraAdapterArtifact::<f64>::import_state(&tampered)
        .expect_err("three heads cannot fit a one-head base");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn a_missing_factor_tensor_is_rejected() {
    let attached = fitted_adapter();
    let state = attached.export_state().unwrap();
    let target = SKLoraTargetModule::LinearCoefficients;
    let tensors = state
        .tensors()
        .iter()
        .filter(|(name, _)| *name != &sk_lora_a_tensor_name(target))
        .map(|(_, tensor)| tensor.clone())
        .collect::<Vec<_>>();
    let incomplete = SKModelState::new(state.metadata().clone(), tensors).unwrap();

    let error = SKLoraAdapterArtifact::<f64>::import_state(&incomplete)
        .expect_err("a state without A cannot rebuild the adapter");
    assert!(matches!(error, SKError::InvalidHyperparameter { .. }));
}

fn state_with_replaced_factor(
    attached: &crate::adapters::lora::SKLoraAttachedModel<f64>,
    tensor_name: &str,
    replacement: f64,
) -> SKModelState<f64> {
    let state = attached.export_state().unwrap();
    let tensors = state
        .tensors()
        .values()
        .map(|tensor| {
            if tensor.name() != tensor_name {
                return tensor.clone();
            }
            let shape = tensor.shape().to_vec();
            let values = match tensor.values() {
                SKTensorValues::Floating(values) => values.clone(),
                _ => unreachable!("adapter factors are floating tensors"),
            };
            let mut values = values;
            values[0] = replacement;
            SKTensor::floating(tensor_name, shape, values).unwrap()
        })
        .collect::<Vec<_>>();
    SKModelState::new(state.metadata().clone(), tensors).unwrap()
}

#[test]
fn a_non_finite_stored_factor_is_rejected_on_import() {
    let attached = fitted_adapter();
    let target = SKLoraTargetModule::LinearCoefficients;
    for replacement in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a_state =
            state_with_replaced_factor(&attached, &sk_lora_a_tensor_name(target), replacement);
        let a_error = SKLoraAdapterArtifact::<f64>::import_state(&a_state)
            .expect_err("a non-finite A factor cannot be imported");
        assert!(matches!(a_error, SKError::NonFiniteValue { .. }));

        let b_state =
            state_with_replaced_factor(&attached, &sk_lora_b_tensor_name(target), replacement);
        let b_error = SKLoraAdapterArtifact::<f64>::import_state(&b_state)
            .expect_err("a non-finite B factor cannot be imported");
        assert!(matches!(b_error, SKError::NonFiniteValue { .. }));
    }
}

#[test]
fn an_artifact_cannot_attach_to_a_different_base_shape() {
    let attached = fitted_adapter();
    let state = attached.export_state().unwrap();
    let artifact = SKLoraAdapterArtifact::<f64>::import_state(&state).unwrap();

    let narrow_features = array![[0.0], [1.0], [2.0]];
    let narrow_targets = array![1.0, 2.0, 3.0];
    let narrow_model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(narrow_features.view(), narrow_targets.view())
        .unwrap();
    let error = artifact
        .attach(SKLoraBaseModel::linear_regression(narrow_model))
        .expect_err("the stored A has two feature columns, the base has one");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
