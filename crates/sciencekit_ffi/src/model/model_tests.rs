//! End-to-end tests for fitting and the model operations (spec `ffi-abi`).

use std::ffi::c_void;
use std::ptr;

use crate::array::{SKFfiArray, SKFfiDataType};
use crate::boundary::{
    sk_ffi_array_free, sk_ffi_builder_create, sk_ffi_builder_free, sk_ffi_fit, sk_ffi_model_free,
    sk_ffi_predict, sk_ffi_score, sk_ffi_transform,
};
use crate::status::SKFfiStatus;

/// A row-major f64 matrix descriptor over a live slice.
fn matrix_descriptor(values: &[f64], rows: i64, columns: i64, shape: &[i64]) -> SKFfiArray {
    SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 2,
        shape: shape.as_ptr(),
        strides: ptr::null(),
        data: values.as_ptr() as *const c_void,
        len: rows * columns,
    }
}

/// A row-major f64 vector descriptor over a live slice.
fn vector_descriptor(values: &[f64], length: i64, shape: &[i64]) -> SKFfiArray {
    SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 1,
        shape: shape.as_ptr(),
        strides: ptr::null(),
        data: values.as_ptr() as *const c_void,
        len: length,
    }
}

/// Copy an owned f64 output into a vector, then release it.
fn copy_and_release(output: &mut SKFfiArray) -> Vec<f64> {
    assert_eq!(output.dtype, SKFfiDataType::F64 as i32);
    // SAFETY: the library published a buffer of exactly `len` f64 values.
    let values =
        unsafe { std::slice::from_raw_parts(output.data as *const f64, output.len as usize) }
            .to_vec();
    assert_eq!(sk_ffi_array_free(output), SKFfiStatus::Ok);
    values
}

/// Fit an exactly linear dataset, predict it back and score it.
#[test]
fn fit_linear_regression_predicts_and_scores() {
    let features = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
    let targets = [1.0_f64, 3.0, 4.0, 14.0];
    let feature_shape = [4_i64, 2];
    let target_shape = [4_i64];
    let x = matrix_descriptor(&features, 4, 2, &feature_shape);
    let y = vector_descriptor(&targets, 4, &target_shape);
    let builder = sk_ffi_builder_create(0, SKFfiDataType::F64 as i32);
    assert!(!builder.is_null());
    let model = sk_ffi_fit(builder, &x, &y);
    assert!(!model.is_null(), "fit must return a model handle");

    let mut output = SKFfiArray {
        dtype: 0,
        ndim: 0,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    };
    assert_eq!(sk_ffi_predict(model, &x, &mut output), SKFfiStatus::Ok);
    let predictions = copy_and_release(&mut output);
    for (prediction, target) in predictions.iter().zip(targets.iter()) {
        assert!(
            (prediction - target).abs() < 1.0e-9,
            "prediction {prediction} must match {target}"
        );
    }

    let mut score = 0.0_f64;
    assert_eq!(sk_ffi_score(model, &x, &y, &mut score), SKFfiStatus::Ok);
    assert!((score - 1.0).abs() < 1.0e-9, "R² must be 1, found {score}");

    assert_eq!(sk_ffi_model_free(model), SKFfiStatus::Ok);
    assert_eq!(sk_ffi_builder_free(builder), SKFfiStatus::Ok);
}

/// A standard scaler transforms what it fitted and rejects predict.
#[test]
fn fit_standard_scaler_transforms() {
    let features = [1.0_f64, 10.0, 2.0, 20.0, 3.0, 30.0];
    let feature_shape = [3_i64, 2];
    let x = matrix_descriptor(&features, 3, 2, &feature_shape);
    let builder = sk_ffi_builder_create(1, SKFfiDataType::F64 as i32);
    assert!(!builder.is_null());
    let model = sk_ffi_fit(builder, &x, ptr::null());
    assert!(!model.is_null(), "scaler fit must return a model handle");

    let mut transformed = SKFfiArray {
        dtype: 0,
        ndim: 0,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    };
    assert_eq!(
        sk_ffi_transform(model, &x, &mut transformed),
        SKFfiStatus::Ok
    );
    assert_eq!(transformed.ndim, 2);
    assert_eq!(transformed.len, 6);
    let values = copy_and_release(&mut transformed);
    let column_mean = (values[0] + values[2] + values[4]) / 3.0;
    assert!(column_mean.abs() < 1.0e-9);

    let mut output = SKFfiArray {
        dtype: 0,
        ndim: 0,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    };
    assert_eq!(
        sk_ffi_predict(model, &x, &mut output),
        SKFfiStatus::Unsupported
    );
    assert_eq!(sk_ffi_model_free(model), SKFfiStatus::Ok);
    assert_eq!(sk_ffi_builder_free(builder), SKFfiStatus::Ok);
}

/// A freed handle is rejected without being dereferenced.
#[test]
fn predict_after_free_returns_invalid_handle() {
    let features = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let feature_shape = [3_i64, 2];
    let x = matrix_descriptor(&features, 3, 2, &feature_shape);
    let builder = sk_ffi_builder_create(0, SKFfiDataType::F64 as i32);
    let model = sk_ffi_fit(builder, &x, ptr::null());
    assert!(model.is_null(), "linear regression requires targets");
    let targets = [3.0_f64, 7.0, 11.0];
    let target_shape = [3_i64];
    let y = vector_descriptor(&targets, 3, &target_shape);
    let model = sk_ffi_fit(builder, &x, &y);
    assert_eq!(sk_ffi_model_free(model), SKFfiStatus::Ok);
    let mut output = SKFfiArray {
        dtype: 0,
        ndim: 0,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    };
    assert_eq!(
        sk_ffi_predict(model, &x, &mut output),
        SKFfiStatus::InvalidHandle
    );
    assert_eq!(sk_ffi_builder_free(builder), SKFfiStatus::Ok);
}

/// A dtype the builder did not configure is rejected.
#[test]
fn fit_rejects_a_mismatched_dtype() {
    let features = [1.0_f32, 2.0, 3.0, 4.0];
    let feature_shape = [2_i64, 2];
    let descriptor = SKFfiArray {
        dtype: SKFfiDataType::F32 as i32,
        ndim: 2,
        shape: feature_shape.as_ptr(),
        strides: ptr::null(),
        data: features.as_ptr() as *const c_void,
        len: 4,
    };
    let builder = sk_ffi_builder_create(1, SKFfiDataType::F64 as i32);
    let model = sk_ffi_fit(builder, &descriptor, ptr::null());
    assert!(model.is_null());
    assert!(crate::error::sk_ffi_last_error_message().contains("dtype mismatch"));
    assert_eq!(sk_ffi_builder_free(builder), SKFfiStatus::Ok);
}
