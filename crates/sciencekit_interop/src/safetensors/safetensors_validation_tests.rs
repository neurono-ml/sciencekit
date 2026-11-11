//! Corrupt-file, offset, shape, dtype and duplicate-name validation tests
//! (spec `safetensors-format`).

use sciencekit_common::{SKError, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType};

use super::{
    SK_SAFETENSORS_FORMAT_VERSION, sk_read_model_state_safetensors,
    sk_write_model_state_safetensors,
};

fn build_raw_safetensors_bytes(json_header: &str, payload: &[u8]) -> Vec<u8> {
    let mut header_bytes = json_header.as_bytes().to_vec();
    let aligned = header_bytes.len().next_multiple_of(8);
    header_bytes.resize(aligned, b' ');
    let mut file_bytes = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.extend_from_slice(payload);
    file_bytes
}

fn valid_metadata_fragment() -> String {
    format!(
        r#""__metadata__":{{"sciencekit.format_version":"{}","sciencekit.estimator_name":"SKLinearRegression","sciencekit.scalar_data_type":"F64","sciencekit.feature_count":"2","sciencekit.parameters":"{{}}"}}"#,
        SK_SAFETENSORS_FORMAT_VERSION
    )
}

/// Truncated files (prefix < 8 bytes, truncated header and truncated payload) return structured errors.
#[test]
fn safetensors_validation_rejects_truncated_files() {
    let metadata = SKModelMetadata::new("SKLinearRegression", 1, SKTensorDataType::F64, 4).unwrap();
    let weights =
        SKTensor::floating("linear.weights", vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let state = SKModelState::new(metadata, vec![weights]).unwrap();
    let valid_bytes = sk_write_model_state_safetensors(&state).unwrap();

    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&valid_bytes[..4]),
        Err(SKError::Conversion(_))
    ));
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&valid_bytes[..12]),
        Err(SKError::Conversion(_))
    ));
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&valid_bytes[..valid_bytes.len() - 5]),
        Err(SKError::Conversion(_))
    ));
}

/// Inconsistent `data_offsets` (inverted, gapped or overlapping) are rejected before constructing state.
#[test]
fn safetensors_validation_rejects_inconsistent_offsets() {
    let meta = valid_metadata_fragment();
    let inverted_header =
        format!(r#"{{{meta},"w":{{"dtype":"F64","shape":[1],"data_offsets":[8,0]}}}}"#);
    let inverted_bytes = build_raw_safetensors_bytes(&inverted_header, &[0_u8; 8]);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&inverted_bytes),
        Err(SKError::Conversion(_))
    ));

    let gapped_header = format!(
        r#"{{{meta},"a":{{"dtype":"F64","shape":[1],"data_offsets":[0,8]}},"b":{{"dtype":"F64","shape":[1],"data_offsets":[16,24]}}}}"#
    );
    let gapped_bytes = build_raw_safetensors_bytes(&gapped_header, &[0_u8; 24]);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&gapped_bytes),
        Err(SKError::Conversion(_))
    ));

    let overlapping_header = format!(
        r#"{{{meta},"a":{{"dtype":"F64","shape":[2],"data_offsets":[0,16]}},"b":{{"dtype":"F64","shape":[2],"data_offsets":[8,24]}}}}"#
    );
    let overlapping_bytes = build_raw_safetensors_bytes(&overlapping_header, &[0_u8; 24]);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&overlapping_bytes),
        Err(SKError::Conversion(_))
    ));
}

/// A tensor whose `shape` product disagrees with its byte slice length returns `SKError::ShapeMismatch`.
#[test]
fn safetensors_validation_rejects_shape_and_offset_length_mismatch() {
    let meta = valid_metadata_fragment();
    let mismatched_header =
        format!(r#"{{{meta},"w":{{"dtype":"F64","shape":[2,3],"data_offsets":[0,16]}}}}"#);
    let mismatched_bytes = build_raw_safetensors_bytes(&mismatched_header, &[0_u8; 16]);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&mismatched_bytes),
        Err(SKError::ShapeMismatch { .. })
    ));

    let overflow_header = format!(
        r#"{{{meta},"w":{{"dtype":"F64","shape":[18446744073709551615,2],"data_offsets":[0,16]}}}}"#
    );
    let overflow_bytes = build_raw_safetensors_bytes(&overflow_header, &[0_u8; 16]);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&overflow_bytes),
        Err(SKError::ShapeMismatch { .. })
    ));
}

/// Duplicate tensor names in the raw JSON header are deterministically rejected with `InvalidHyperparameter`.
#[test]
fn safetensors_validation_rejects_duplicate_tensor_names_in_header() {
    let meta = valid_metadata_fragment();
    let duplicate_header = format!(
        r#"{{{meta},"w":{{"dtype":"F64","shape":[1],"data_offsets":[0,8]}},"w":{{"dtype":"F64","shape":[1],"data_offsets":[8,16]}}}}"#
    );
    let duplicate_bytes = build_raw_safetensors_bytes(&duplicate_header, &[0_u8; 16]);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&duplicate_bytes),
        Err(SKError::InvalidHyperparameter {
            name: "tensor_names",
            ..
        })
    ));
}

/// Unsupported Safetensors dtypes (`F16`) and mismatched model scalars (`F32` read as `f64`) are rejected.
#[test]
fn safetensors_validation_rejects_unsupported_and_mismatched_dtypes() {
    let meta = valid_metadata_fragment();
    let f16_header =
        format!(r#"{{{meta},"w":{{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}}}"#);
    let f16_bytes = build_raw_safetensors_bytes(&f16_header, &[0_u8; 4]);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&f16_bytes),
        Err(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            ..
        })
    ));

    let single_meta =
        SKModelMetadata::new("SKLinearRegression", 1, SKTensorDataType::F32, 2).unwrap();
    let single_weights = SKTensor::floating("w", vec![2], vec![1.0_f32, 2.0_f32]).unwrap();
    let single_state = SKModelState::new(single_meta, vec![single_weights]).unwrap();
    let single_bytes = sk_write_model_state_safetensors(&single_state).unwrap();

    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&single_bytes),
        Err(SKError::InvalidHyperparameter {
            name: "scalar_data_type",
            ..
        })
    ));
}
