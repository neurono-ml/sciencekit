//! Reading a model state from the documented little-endian file layout.

use sciencekit_common::{SKModelMetadata, SKModelState};

use crate::error_kind::SKFfiError;
use crate::persistence::reader::{SKModelStateReader, dtype_from_code, malformed};

/// The file magic, including its terminator, that identifies a model state.
pub const SK_MODEL_STATE_MAGIC: &[u8] = b"SKFFIMODEL\0";
/// The on-disk format version written by this build.
pub const SK_MODEL_STATE_VERSION: u16 = 1;

/// Parse a model state from the documented file layout.
pub fn sk_ffi_decode_model_state(bytes: &[u8]) -> Result<SKModelState<f64>, SKFfiError> {
    let mut reader = SKModelStateReader::new(bytes);
    if reader.take(SK_MODEL_STATE_MAGIC.len())? != SK_MODEL_STATE_MAGIC {
        return Err(malformed(
            "the file does not start with the model-state magic",
        ));
    }
    let version = reader.u16()?;
    if version != SK_MODEL_STATE_VERSION {
        return Err(malformed(format!(
            "unsupported model-state format version {version}; expected {SK_MODEL_STATE_VERSION}"
        )));
    }
    let estimator_name = reader.string()?;
    let format_version = reader.u32()?;
    let scalar_dtype = dtype_from_code(reader.u8()?)?;
    let feature_count = reader.u64()? as usize;
    let class_labels = read_class_labels(&mut reader)?;
    let parameters = read_parameters(&mut reader)?;
    let number_of_tensors = reader.u32()? as usize;

    let mut metadata =
        SKModelMetadata::new(estimator_name, format_version, scalar_dtype, feature_count)?;
    if let Some(labels) = class_labels {
        metadata = metadata.with_class_labels(labels)?;
    }
    for (name, value) in parameters {
        metadata = metadata.with_parameter(name, value)?;
    }
    let mut tensors = Vec::with_capacity(number_of_tensors);
    for _ in 0..number_of_tensors {
        tensors.push(reader.tensor()?);
    }
    SKModelState::new(metadata, tensors).map_err(SKFfiError::from)
}

fn read_class_labels(
    reader: &mut SKModelStateReader<'_>,
) -> Result<Option<Vec<String>>, SKFfiError> {
    if reader.u8()? == 0 {
        return Ok(None);
    }
    let count = reader.u32()? as usize;
    let mut labels = Vec::with_capacity(count);
    for _ in 0..count {
        labels.push(reader.string()?);
    }
    Ok(Some(labels))
}

fn read_parameters(
    reader: &mut SKModelStateReader<'_>,
) -> Result<Vec<(String, sciencekit_common::SKParameterValue)>, SKFfiError> {
    let count = reader.u32()? as usize;
    let mut parameters = Vec::with_capacity(count);
    for _ in 0..count {
        let name = reader.string()?;
        parameters.push((name, reader.parameter()?));
    }
    Ok(parameters)
}
