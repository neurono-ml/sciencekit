//! Tests for the named tensor representation (spec `model-export-contracts`).

use crate::SKError;

use super::{SKTensor, SKTensorDataType, SKTensorValues};

/// A floating tensor keeps its shape, data_type and values.
#[test]
fn interop_tensor_floating_keeps_shape_values_and_dtype() {
    let tensor = SKTensor::floating(
        "linear.coefficients",
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    assert_eq!(tensor.name(), "linear.coefficients");
    assert_eq!(tensor.data_type(), SKTensorDataType::F64);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.number_of_elements(), 6);
    assert_eq!(
        tensor.values(),
        &SKTensorValues::Floating(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    );
}

/// A shape whose product differs from the value count is a shape mismatch.
#[test]
fn interop_tensor_shape_mismatch_is_rejected() {
    let outcome = SKTensor::floating("linear.coefficients", vec![2, 3], vec![1.0_f64; 5]);

    assert!(matches!(outcome, Err(SKError::ShapeMismatch { .. })));
}

/// The floating data_type follows the model scalar, never a hardcoded one.
#[test]
fn interop_tensor_data_type_follows_the_model_scalar() {
    let single = SKTensor::floating("intercept", vec![], vec![0.5_f32]).unwrap();
    let double = SKTensor::floating("intercept", vec![], vec![0.5_f64]).unwrap();

    assert_eq!(single.data_type(), SKTensorDataType::F32);
    assert_eq!(double.data_type(), SKTensorDataType::F64);
}

/// A scalar tensor (empty shape) holds exactly one value.
#[test]
fn interop_tensor_scalar_shape_holds_one_element() {
    let tensor = SKTensor::floating("linear.intercept", vec![], vec![2.5_f64]).unwrap();

    assert_eq!(tensor.number_of_elements(), 1);
    assert!(tensor.shape().is_empty());
}

/// Integer and boolean tensors carry their own storage and data_type.
#[test]
fn interop_tensor_integer_and_boolean_payloads() {
    let integer =
        SKTensor::<f64>::integer("tree.nodes.feature", vec![3], vec![0_i64, 2, 1]).unwrap();
    let boolean =
        SKTensor::<f64>::boolean("tree.nodes.is_leaf", vec![3], vec![true, false, true]).unwrap();

    assert_eq!(integer.data_type(), SKTensorDataType::I64);
    assert_eq!(integer.values(), &SKTensorValues::Integer(vec![0, 2, 1]));
    assert_eq!(boolean.data_type(), SKTensorDataType::Bool);
    assert_eq!(
        boolean.values(),
        &SKTensorValues::Boolean(vec![true, false, true])
    );
}

/// An explicit name must be present.
#[test]
fn interop_tensor_empty_name_is_rejected() {
    let outcome = SKTensor::floating("", vec![1], vec![1.0_f64]);

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}

/// A data_type that disagrees with the stored enum variant is rejected.
#[test]
fn interop_tensor_data_type_disagreeing_with_values_is_rejected() {
    let outcome = SKTensor::new(
        "tree.nodes.feature",
        SKTensorDataType::F32,
        vec![2],
        SKTensorValues::<f64>::Integer(vec![0, 1]),
    );

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}

/// A floating tensor whose data_type disagrees with the model scalar is rejected.
#[test]
fn interop_tensor_float_data_type_disagreeing_with_scalar_is_rejected() {
    let outcome = SKTensor::new(
        "linear.coefficients",
        SKTensorDataType::F32,
        vec![2],
        SKTensorValues::Floating(vec![1.0_f64, 2.0]),
    );

    assert!(matches!(
        outcome,
        Err(SKError::InvalidHyperparameter { .. })
    ));
}
