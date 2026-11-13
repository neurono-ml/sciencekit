//! Acceptance scenarios for multi-threaded concurrent reads and corrupt inputs
//! (PRD §8.7, spec `safetensors-format`).

use std::path::Path;
use std::sync::Arc;
use std::thread;

use sciencekit_common::{
    SKError, SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};

use crate::safetensors::{
    SKSafetensorsReader, SKSafetensorsWriter, sk_read_model_state_safetensors,
    sk_read_model_state_safetensors_file, sk_write_model_state_safetensors,
};

/// Eight concurrent threads reading both a padded single file and a sharded set agree with single-threaded reads.
#[test]
fn safetensors_acceptance_concurrent_reads_agree_for_single_and_sharded_files() {
    let metadata = SKModelMetadata::new("SKMLPRegressor", 1, SKTensorDataType::F64, 8)
        .unwrap()
        .with_parameter("hidden_width", SKParameterValue::from(64_i64))
        .unwrap();
    let layer_one: Vec<f64> = (0..512).map(|index| (index as f64) * 0.0625).collect();
    let layer_two: Vec<f64> = (0..512).map(|index| -(index as f64) * 0.03125).collect();
    let state = SKModelState::new(
        metadata,
        vec![
            SKTensor::floating("mlp.layer_one", vec![8, 64], layer_one).unwrap(),
            SKTensor::floating("mlp.layer_two", vec![64, 8], layer_two).unwrap(),
        ],
    )
    .unwrap();

    let temporary_directory = tempfile::tempdir().unwrap();
    let single_path = Arc::new(temporary_directory.path().join("single.safetensors"));
    let sharded_path = Arc::new(temporary_directory.path().join("sharded.safetensors"));

    SKSafetensorsWriter::builder()
        .padding_bytes(128)
        .build()
        .unwrap()
        .write_state_to_path(&state, &single_path)
        .unwrap();
    SKSafetensorsWriter::builder()
        .maximum_shard_bytes(4_096)
        .build()
        .unwrap()
        .write_state_to_path(&state, &sharded_path)
        .unwrap();

    let reference = sk_read_model_state_safetensors_file::<f64>(&single_path).unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let single_clone = Arc::clone(&single_path);
            let sharded_clone = Arc::clone(&sharded_path);
            thread::spawn(move || {
                let reader = SKSafetensorsReader::builder().build().unwrap();
                let single_read = reader.read_state_from_path::<f64>(&single_clone).unwrap();
                let sharded_read = reader.read_state_from_path::<f64>(&sharded_clone).unwrap();
                (single_read, sharded_read)
            })
        })
        .collect();

    for handle in handles {
        let (single_read, sharded_read) = handle.join().unwrap();
        assert_eq!(single_read, reference);
        assert_eq!(sharded_read, reference);
    }
}

/// Every truncated byte prefix, corrupt payload byte and missing path returns a structured error without panicking.
#[test]
fn safetensors_acceptance_corrupt_and_truncated_inputs_never_panic() {
    let metadata = SKModelMetadata::new("SKLinearRegression", 1, SKTensorDataType::F64, 2).unwrap();
    let weights = SKTensor::floating("linear.weights", vec![2], vec![1.0_f64, 2.0]).unwrap();
    let flags = SKTensor::<f64>::boolean("linear.mask", vec![2], vec![true, false]).unwrap();
    let state = SKModelState::new(metadata, vec![weights, flags]).unwrap();
    let valid_bytes = sk_write_model_state_safetensors(&state).unwrap();

    for prefix_length in 0..valid_bytes.len() {
        let outcome = sk_read_model_state_safetensors::<f64>(&valid_bytes[..prefix_length]);
        assert!(matches!(outcome, Err(SKError::Conversion(_))));
    }

    let header_end = 8 + (u64::from_le_bytes(valid_bytes[..8].try_into().unwrap()) as usize);
    let mut nan_corrupted = valid_bytes.clone();
    let nan_bytes = f64::NAN.to_le_bytes();
    nan_corrupted[header_end..header_end + 8].copy_from_slice(&nan_bytes);
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&nan_corrupted),
        Err(SKError::NonFiniteValue { position: 0 })
    ));

    let mut invalid_bool = valid_bytes.clone();
    let last_index = invalid_bool.len() - 1;
    invalid_bool[last_index] = 2;
    assert!(matches!(
        sk_read_model_state_safetensors::<f64>(&invalid_bool),
        Err(SKError::InvalidHyperparameter {
            name: "tensor_data_type",
            ..
        })
    ));

    assert!(matches!(
        sk_read_model_state_safetensors_file::<f64>(Path::new(
            "/nonexistent/safetensors/model.safetensors"
        )),
        Err(SKError::Io(_))
    ));
}
