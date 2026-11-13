//! Tests for concurrent predictions on one model handle (spec `ffi-abi`).

use std::ffi::c_void;
use std::ptr;

use crate::array::{SKFfiArray, SKFfiDataType};
use crate::boundary::{
    sk_ffi_array_free, sk_ffi_builder_create, sk_ffi_builder_free, sk_ffi_fit, sk_ffi_model_free,
    sk_ffi_predict,
};
use crate::status::SKFfiStatus;

/// Several threads predict from the same handle; every result is correct and
/// the handle stays usable.
#[test]
fn concurrency_predict_is_safe_and_correct() {
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
    let address = model as usize;

    let mut workers = Vec::new();
    for _ in 0..8 {
        workers.push(std::thread::spawn(move || {
            let model = address as *mut crate::model::SKFfiModel;
            let mut output = SKFfiArray {
                dtype: 0,
                ndim: 0,
                shape: ptr::null(),
                strides: ptr::null(),
                data: ptr::null(),
                len: 0,
            };
            assert_eq!(
                sk_ffi_predict(model, &descriptor(), &mut output),
                SKFfiStatus::Ok
            );
            // SAFETY: the library published a buffer of exactly `len` f64 values.
            let values = unsafe {
                std::slice::from_raw_parts(output.data as *const f64, output.len as usize).to_vec()
            };
            assert_eq!(sk_ffi_array_free(&mut output), SKFfiStatus::Ok);
            values
        }));
    }
    for worker in workers {
        let values = worker.join().expect("no worker may panic");
        for (prediction, target) in values.iter().zip(targets.iter()) {
            assert!((prediction - target).abs() < 1.0e-9);
        }
    }
    // The handle is still alive after all readers finished.
    assert_eq!(sk_ffi_model_free(model), SKFfiStatus::Ok);
    assert_eq!(sk_ffi_builder_free(builder), SKFfiStatus::Ok);
}

/// The shared dataset descriptor, rebuilt per thread so no pointer crosses
/// thread boundaries.
fn descriptor() -> SKFfiArray {
    thread_local! {
        static DATA: [f64; 8] = const { [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0] };
        static SHAPE: [i64; 2] = const { [4, 2] };
    }
    DATA.with(|data| {
        SHAPE.with(|shape| SKFfiArray {
            dtype: SKFfiDataType::F64 as i32,
            ndim: 2,
            shape: shape.as_ptr(),
            strides: ptr::null(),
            data: data.as_ptr() as *const c_void,
            len: 8,
        })
    })
}
