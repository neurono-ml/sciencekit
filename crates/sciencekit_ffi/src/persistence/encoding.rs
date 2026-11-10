//! The little-endian writer for the model-state file.

use sciencekit_common::{SKModelState, SKParameterValue, SKTensorDataType, SKTensorValues};

use crate::persistence::decoding::{SK_MODEL_STATE_MAGIC, SK_MODEL_STATE_VERSION};

/// Serialize a double-precision model state into the documented file layout.
pub fn sk_ffi_encode_model_state(state: &SKModelState<f64>) -> Vec<u8> {
    let mut writer = SKModelStateWriter::new();
    writer.bytes.extend_from_slice(SK_MODEL_STATE_MAGIC);
    writer.u16(SK_MODEL_STATE_VERSION);
    writer.string(state.metadata().estimator_name());
    writer.u32(state.metadata().format_version());
    writer.u8(dtype_code(state.metadata().scalar_data_type()));
    writer.u64(state.metadata().feature_count() as u64);
    match state.metadata().class_labels() {
        Some(labels) => {
            writer.u8(1);
            writer.u32(labels.len() as u32);
            for label in labels {
                writer.string(label);
            }
        }
        None => writer.u8(0),
    }
    writer.u32(state.metadata().parameters().len() as u32);
    for (name, value) in state.metadata().parameters() {
        writer.string(name);
        writer.parameter(value);
    }
    writer.u32(state.number_of_tensors() as u32);
    for (name, tensor) in state.tensors() {
        writer.string(name);
        writer.u8(dtype_code(tensor.data_type()));
        writer.u32(tensor.shape().len() as u32);
        for dimension in tensor.shape() {
            writer.u64(*dimension as u64);
        }
        writer.u64(tensor.number_of_elements() as u64);
        writer.tensor_values(tensor.values());
    }
    writer.bytes
}

/// Map a tensor dtype onto its stable byte.
///
/// The writer supports the same dtypes the reader accepts: `F64`, `I64` and
/// `Bool`. `F32` is rejected by the caller before reaching this function, and an
/// unknown dtype would be a writer bug, not a caller error.
fn dtype_code(dtype: SKTensorDataType) -> u8 {
    match dtype {
        SKTensorDataType::F64 => 1,
        SKTensorDataType::I64 => 2,
        SKTensorDataType::Bool => 3,
        _ => unreachable!("the caller rejects unsupported tensor dtypes"),
    }
}

/// A growing little-endian byte buffer.
struct SKModelStateWriter {
    bytes: Vec<u8>,
}

impl SKModelStateWriter {
    fn new() -> SKModelStateWriter {
        SKModelStateWriter { bytes: Vec::new() }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn string(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn parameter(&mut self, value: &SKParameterValue) {
        match value {
            SKParameterValue::Bool(flag) => {
                self.u8(0);
                self.u8(u8::from(*flag));
            }
            SKParameterValue::Integer(integer) => {
                self.u8(1);
                self.i64(*integer);
            }
            SKParameterValue::Float(float) => {
                self.u8(2);
                self.f64(*float);
            }
            SKParameterValue::String(text) => {
                self.u8(3);
                self.string(text);
            }
            SKParameterValue::List(values) => {
                self.u8(4);
                self.u32(values.len() as u32);
                for value in values {
                    self.parameter(value);
                }
            }
            _ => self.u8(0),
        }
    }

    fn tensor_values(&mut self, values: &SKTensorValues<f64>) {
        match values {
            SKTensorValues::Floating(values) => {
                for value in values {
                    self.f64(*value);
                }
            }
            SKTensorValues::Integer(values) => {
                for value in values {
                    self.i64(*value);
                }
            }
            SKTensorValues::Boolean(values) => {
                for value in values {
                    self.u8(u8::from(*value));
                }
            }
            _ => {}
        }
    }
}
