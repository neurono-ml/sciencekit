//! Tests for model persistence through the interop state (spec `ffi-abi`).

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use sciencekit_common::{
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};

use crate::array::{SKFfiArray, SKFfiDataType};
use crate::boundary::{
    sk_ffi_array_free, sk_ffi_builder_create, sk_ffi_builder_free, sk_ffi_fit, sk_ffi_model_free,
    sk_ffi_model_load, sk_ffi_model_save, sk_ffi_predict,
};
use crate::persistence::{
    decoding::sk_ffi_decode_model_state, encoding::sk_ffi_encode_model_state,
    sk_ffi_read_model_state, sk_ffi_write_model_state,
};
use crate::status::SKFfiStatus;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique path under the system temporary directory.
fn temporary_path(name: &str) -> PathBuf {
    let identifier = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "sciencekit_ffi_{name}_{}_{}.bin",
        std::process::id(),
        identifier
    ))
}

/// Fit a linear model, save it, load it and compare predictions.
#[test]
fn persistence_save_and_load_keep_predictions() {
    let features = [0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0];
    let targets = [1.0_f64, 3.0, 4.0, 14.0];
    let feature_shape = [4_i64, 2];
    let target_shape = [4_i64];
    let x = SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 2,
        shape: feature_shape.as_ptr(),
        strides: ptr::null(),
        data: features.as_ptr() as *const c_void,
        len: 8,
    };
    let y = SKFfiArray {
        dtype: SKFfiDataType::F64 as i32,
        ndim: 1,
        shape: target_shape.as_ptr(),
        strides: ptr::null(),
        data: targets.as_ptr() as *const c_void,
        len: 4,
    };
    let builder = sk_ffi_builder_create(0, SKFfiDataType::F64 as i32);
    let model = sk_ffi_fit(builder, &x, &y);
    assert!(!model.is_null());

    let path = temporary_path("round_trip");
    let path_text = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    assert_eq!(
        sk_ffi_model_save(model, path_text.as_ptr()),
        SKFfiStatus::Ok
    );
    let mut loaded = std::ptr::null_mut();
    assert_eq!(
        sk_ffi_model_load(path_text.as_ptr(), &mut loaded),
        SKFfiStatus::Ok
    );
    assert!(!loaded.is_null());

    let original = predict(model, &x);
    let restored = predict(loaded, &x);
    assert_eq!(original.len(), restored.len());
    for (left, right) in original.iter().zip(restored.iter()) {
        assert!((left - right).abs() < 1.0e-12);
    }
    std::fs::remove_file(&path).ok();
    assert_eq!(sk_ffi_model_free(model), SKFfiStatus::Ok);
    assert_eq!(sk_ffi_model_free(loaded), SKFfiStatus::Ok);
    assert_eq!(sk_ffi_builder_free(builder), SKFfiStatus::Ok);
}

fn predict(model: *mut crate::model::SKFfiModel, features: &SKFfiArray) -> Vec<f64> {
    let mut output = SKFfiArray {
        dtype: 0,
        ndim: 0,
        shape: ptr::null(),
        strides: ptr::null(),
        data: ptr::null(),
        len: 0,
    };
    assert_eq!(
        sk_ffi_predict(model, features, &mut output),
        SKFfiStatus::Ok
    );
    // SAFETY: the library published a buffer of exactly `len` f64 values.
    let values = unsafe {
        std::slice::from_raw_parts(output.data as *const f64, output.len as usize).to_vec()
    };
    assert_eq!(sk_ffi_array_free(&mut output), SKFfiStatus::Ok);
    values
}

/// The state encoder and decoder round-trip metadata and tensors.
#[test]
fn persistence_state_round_trips() {
    let metadata = SKModelMetadata::new("SKExample", 3, SKTensorDataType::F64, 2)
        .unwrap()
        .with_parameter("flag", SKParameterValue::Bool(true))
        .unwrap()
        .with_parameter("count", SKParameterValue::Integer(7))
        .unwrap();
    let tensor = SKTensor::floating("example.values", vec![2], vec![1.5_f64, -2.5]).unwrap();
    let state = SKModelState::new(metadata, vec![tensor]).unwrap();
    let decoded = sk_ffi_decode_model_state(&sk_ffi_encode_model_state(&state)).unwrap();
    assert_eq!(decoded, state);
}

/// A truncated or foreign file is rejected with an invalid-argument status.
#[test]
fn persistence_rejects_a_corrupt_file() {
    let path = temporary_path("corrupt");
    std::fs::write(&path, b"not a sciencekit model").unwrap();
    let error = sk_ffi_read_model_state(&path).expect_err("a foreign file must be rejected");
    assert_eq!(error.status(), SKFfiStatus::InvalidArgument);
    std::fs::remove_file(&path).ok();
}

/// Writing a state to disk and reading it back preserves it.
#[test]
fn persistence_file_round_trips() {
    let metadata = SKModelMetadata::new("SKExample", 1, SKTensorDataType::F64, 1).unwrap();
    let tensor = SKTensor::floating("example.values", vec![1], vec![4.0_f64]).unwrap();
    let state = SKModelState::new(metadata, vec![tensor]).unwrap();
    let path = temporary_path("file");
    sk_ffi_write_model_state(&path, &state).unwrap();
    let restored = sk_ffi_read_model_state(&path).unwrap();
    assert_eq!(restored, state);
    std::fs::remove_file(&path).ok();
}
