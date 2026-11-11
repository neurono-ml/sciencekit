//! Header padding, in-place tensor append and shard-overflow tests
//! (spec `safetensors-format`).

use sciencekit_common::{
    SKError, SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};

use super::{
    SK_SAFETENSORS_PADDING_TENSOR_NAME, SKAppendOutcome, SKSafetensorsWriter,
    sk_append_safetensors_tensor, sk_read_model_state_safetensors_file,
    sk_read_safetensors_shard_index, sk_safetensors_index_path,
};

fn extract_header_end_offset(file_bytes: &[u8]) -> usize {
    let header_len = u64::from_le_bytes(file_bytes[..8].try_into().unwrap()) as usize;
    8 + header_len
}

/// Small appends fit inside reserved header and payload padding without moving existing payloads.
#[test]
fn safetensors_padding_reserves_space_excludes_padding_tensor_and_appends_in_place() {
    let metadata = SKModelMetadata::new("SKLinearRegression", 1, SKTensorDataType::F64, 4)
        .unwrap()
        .with_parameter("include_intercept", SKParameterValue::from(true))
        .unwrap();
    let base_weights = SKTensor::floating(
        "linear.coefficients",
        vec![4],
        vec![1.5_f64, -2.5, 3.5, -4.5],
    )
    .unwrap();
    let initial_state = SKModelState::new(metadata.clone(), vec![base_weights.clone()]).unwrap();

    let temporary_directory = tempfile::tempdir().unwrap();
    let file_path = temporary_directory.path().join("padded.safetensors");

    let writer = SKSafetensorsWriter::builder()
        .padding_bytes(128)
        .build()
        .unwrap();
    assert_eq!(writer.payload_padding_bytes(), 128);
    writer
        .write_state_to_path(&initial_state, &file_path)
        .unwrap();

    let before_bytes = std::fs::read(&file_path).unwrap();
    assert!(safetensors::SafeTensors::deserialize(&before_bytes).is_ok());
    let before_header_end = extract_header_end_offset(&before_bytes);
    let original_payload_slice = before_bytes[before_header_end..before_header_end + 32].to_vec();

    let read_before = sk_read_model_state_safetensors_file::<f64>(&file_path).unwrap();
    assert_eq!(read_before, initial_state);
    assert!(
        read_before
            .tensor(SK_SAFETENSORS_PADDING_TENSOR_NAME)
            .is_none()
    );

    let adapter_a =
        SKTensor::floating("lora.adapter_a", vec![2, 2], vec![0.1_f64, 0.2, 0.3, 0.4]).unwrap();
    let adapter_b =
        SKTensor::floating("lora.adapter_b", vec![2, 2], vec![0.5_f64, 0.6, 0.7, 0.8]).unwrap();

    let first_outcome = writer
        .append_tensor_to_path(&file_path, &adapter_a)
        .unwrap();
    let second_outcome = sk_append_safetensors_tensor(&file_path, &adapter_b).unwrap();
    assert_eq!(first_outcome, SKAppendOutcome::InPlace);
    assert_eq!(second_outcome, SKAppendOutcome::InPlace);
    assert!(!sk_safetensors_index_path(&file_path).exists());

    let after_bytes = std::fs::read(&file_path).unwrap();
    assert_eq!(after_bytes.len(), before_bytes.len());
    assert_eq!(extract_header_end_offset(&after_bytes), before_header_end);
    assert_eq!(
        &after_bytes[before_header_end..before_header_end + 32],
        original_payload_slice.as_slice()
    );
    assert!(safetensors::SafeTensors::deserialize(&after_bytes).is_ok());

    let expected_state =
        SKModelState::new(metadata, vec![base_weights, adapter_a, adapter_b]).unwrap();
    let read_after = sk_read_model_state_safetensors_file::<f64>(&file_path).unwrap();
    assert_eq!(read_after, expected_state);
}

/// When an append exceeds remaining padding, a new shard is written and the index is updated.
#[test]
fn safetensors_padding_overflow_creates_new_shard_and_updates_index() {
    let metadata = SKModelMetadata::new("SKLogisticRegression", 1, SKTensorDataType::F64, 2)
        .unwrap()
        .with_class_labels(vec!["no".to_string(), "yes".to_string()])
        .unwrap();
    let base_weights =
        SKTensor::floating("logistic.weights", vec![2], vec![1.0_f64, -1.0]).unwrap();
    let initial_state = SKModelState::new(metadata.clone(), vec![base_weights.clone()]).unwrap();

    let temporary_directory = tempfile::tempdir().unwrap();
    let file_path = temporary_directory.path().join("overflow.safetensors");

    let writer = SKSafetensorsWriter::builder()
        .padding_bytes(32)
        .build()
        .unwrap();
    writer
        .write_state_to_path(&initial_state, &file_path)
        .unwrap();

    let small_bias = SKTensor::floating("logistic.bias", vec![2], vec![0.25_f64, -0.25]).unwrap();
    assert_eq!(
        writer
            .append_tensor_to_path(&file_path, &small_bias)
            .unwrap(),
        SKAppendOutcome::InPlace
    );

    let duplicate_attempt = writer.append_tensor_to_path(&file_path, &small_bias);
    assert!(matches!(
        duplicate_attempt,
        Err(SKError::InvalidHyperparameter {
            name: "tensor_names",
            ..
        })
    ));

    let base_snapshot_before_overflow = std::fs::read(&file_path).unwrap();

    let large_delta = SKTensor::floating(
        "lora.weights_delta",
        vec![4, 2],
        vec![0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    .unwrap();
    let overflow_outcome = writer
        .append_tensor_to_path(&file_path, &large_delta)
        .unwrap();
    assert!(matches!(overflow_outcome, SKAppendOutcome::NewShard { .. }));

    assert_eq!(
        std::fs::read(&file_path).unwrap(),
        base_snapshot_before_overflow
    );
    let index_path = sk_safetensors_index_path(&file_path);
    assert!(index_path.exists());

    let extra_shard_tensor = SKTensor::<f64>::integer("lora.step", vec![1], vec![42_i64]).unwrap();
    let second_shard_outcome = writer
        .append_tensor_to_path(&file_path, &extra_shard_tensor)
        .unwrap();
    assert!(matches!(
        second_shard_outcome,
        SKAppendOutcome::NewShard { .. }
    ));

    let index = sk_read_safetensors_shard_index(&index_path).unwrap();
    assert_eq!(index.shards().len(), 3);

    let expected_state = SKModelState::new(
        metadata,
        vec![base_weights, small_bias, large_delta, extra_shard_tensor],
    )
    .unwrap();
    let merged = sk_read_model_state_safetensors_file::<f64>(&file_path).unwrap();
    assert_eq!(merged, expected_state);
}
