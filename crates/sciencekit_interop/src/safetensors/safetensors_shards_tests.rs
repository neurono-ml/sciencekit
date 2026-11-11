//! Sharding plan, shard writer and merged index reader tests (spec `safetensors-format`).

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKModelMetadata, SKModelState,
    SKParameterValue, SKTensor, SKTensorDataType,
};

use super::{
    SKSafetensorsReader, SKSafetensorsWriter, SKShardPlan, sk_read_model_state_safetensors,
    sk_read_model_state_safetensors_file, sk_read_safetensors_shard_index,
    sk_safetensors_index_path, sk_write_model_state_safetensors,
};

/// `SKShardPlan` validates `maximum_shard_bytes > 0` and records backend and execution mode.
#[test]
fn safetensors_shards_plan_validates_threshold_and_preserves_builder_options() {
    let invalid = SKShardPlan::builder().maximum_shard_bytes(0).build();
    assert!(matches!(
        invalid,
        Err(SKError::InvalidHyperparameter {
            name: "maximum_shard_bytes",
            ..
        })
    ));

    let plan = SKShardPlan::builder()
        .maximum_shard_bytes(32_768)
        .backend(SKBackendKind::MatrixMultiply)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    assert_eq!(plan.maximum_shard_bytes(), 32_768);
    assert_eq!(plan.backend(), SKBackendKind::MatrixMultiply);
    assert_eq!(plan.execution_mode(), SKExecutionMode::InProcessSynchronous);
}

/// A large multi-tensor state exceeding the shard threshold splits into shards and merges identically.
#[test]
fn safetensors_shards_large_state_splits_into_immutable_shards_and_merges_identically() {
    let metadata = SKModelMetadata::new("SKRandomForestClassifier", 1, SKTensorDataType::F64, 16)
        .unwrap()
        .with_class_labels(vec!["negative".to_string(), "positive".to_string()])
        .unwrap()
        .with_parameter("number_of_trees", SKParameterValue::from(128_i64))
        .unwrap();

    let weights_a: Vec<f64> = (0..6_000).map(|index| (index as f64) * 0.125).collect();
    let weights_b: Vec<f64> = (0..6_000).map(|index| -(index as f64) * 0.25).collect();
    let split_features: Vec<i64> = (0..6_000).map(|index| index % 16).collect();
    let leaf_flags: Vec<bool> = (0..6_000).map(|index| index % 2 == 0).collect();

    let original = SKModelState::new(
        metadata,
        vec![
            SKTensor::floating("forest.thresholds_a", vec![100, 60], weights_a).unwrap(),
            SKTensor::floating("forest.thresholds_b", vec![100, 60], weights_b).unwrap(),
            SKTensor::<f64>::integer("forest.features", vec![6_000], split_features).unwrap(),
            SKTensor::<f64>::boolean("forest.is_leaf", vec![6_000], leaf_flags).unwrap(),
        ],
    )
    .unwrap();

    let single_bytes = sk_write_model_state_safetensors(&original).unwrap();
    let single_state = sk_read_model_state_safetensors::<f64>(&single_bytes).unwrap();

    let temporary_directory = tempfile::tempdir().unwrap();
    let base_path = temporary_directory.path().join("forest.safetensors");
    let writer = SKSafetensorsWriter::builder()
        .maximum_shard_bytes(50_000)
        .build()
        .unwrap();
    writer.write_state_to_path(&original, &base_path).unwrap();

    let index_path = sk_safetensors_index_path(&base_path);
    assert!(index_path.exists());
    assert!(!base_path.exists());

    let shard_index = sk_read_safetensors_shard_index(&index_path).unwrap();
    assert!(shard_index.shards().len() >= 3);
    assert_eq!(shard_index.tensor_to_shard().len(), 4);

    for shard_file in shard_index.shards() {
        let shard_bytes = std::fs::read(temporary_directory.path().join(shard_file)).unwrap();
        assert!(safetensors::SafeTensors::deserialize(&shard_bytes).is_ok());
    }

    let merged_via_base = sk_read_model_state_safetensors_file::<f64>(&base_path).unwrap();
    let merged_via_index = SKSafetensorsReader::builder()
        .build()
        .unwrap()
        .read_state_from_path::<f64>(&index_path)
        .unwrap();

    assert_eq!(merged_via_base, single_state);
    assert_eq!(merged_via_index, original);
}

/// A state whose total payload stays within `maximum_shard_bytes` writes a single file without an index.
#[test]
fn safetensors_shards_state_below_threshold_stays_single_file() {
    let metadata = SKModelMetadata::new("SKRidge", 1, SKTensorDataType::F64, 3).unwrap();
    let weights = SKTensor::floating("ridge.weights", vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let state = SKModelState::new(metadata, vec![weights]).unwrap();

    let temporary_directory = tempfile::tempdir().unwrap();
    let file_path = temporary_directory.path().join("ridge.safetensors");
    let writer = SKSafetensorsWriter::builder()
        .maximum_shard_bytes(65_536)
        .build()
        .unwrap();
    writer.write_state_to_path(&state, &file_path).unwrap();

    assert!(file_path.exists());
    assert!(!sk_safetensors_index_path(&file_path).exists());
    assert_eq!(
        sk_read_model_state_safetensors_file::<f64>(&file_path).unwrap(),
        state
    );
}
