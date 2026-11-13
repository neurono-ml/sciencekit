//! Missing-shard, duplicate-shard, cross-shard mismatch and future-version tests
//! (spec `safetensors-format`).

use std::collections::HashMap;
use std::io::ErrorKind;

use sciencekit_common::{SKError, SKModelMetadata, SKModelState, SKTensor, SKTensorDataType};

use super::{
    SKSafetensorsWriter, sk_read_model_state_safetensors, sk_read_model_state_safetensors_file,
    sk_safetensors_index_path,
};

/// Deleting a referenced shard returns a structured I/O NotFound error, and an empty shard list returns `EmptyInput`.
#[test]
fn safetensors_shards_rejects_missing_shard_and_empty_shard_manifest() {
    let metadata = SKModelMetadata::new("SKRidge", 1, SKTensorDataType::F64, 4).unwrap();
    let weights_a =
        SKTensor::floating("ridge.weights_a", vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let weights_b =
        SKTensor::floating("ridge.weights_b", vec![4], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
    let state = SKModelState::new(metadata, vec![weights_a, weights_b]).unwrap();

    let temporary_directory = tempfile::tempdir().unwrap();
    let base_path = temporary_directory.path().join("ridge.safetensors");
    let writer = SKSafetensorsWriter::builder()
        .maximum_shard_bytes(32)
        .build()
        .unwrap();
    writer.write_state_to_path(&state, &base_path).unwrap();

    let shard_two = temporary_directory
        .path()
        .join("ridge.shard-00002.safetensors");
    std::fs::remove_file(&shard_two).unwrap();

    let missing_outcome = sk_read_model_state_safetensors_file::<f64>(&base_path);
    match missing_outcome {
        Err(SKError::Io(io_error)) => assert_eq!(io_error.kind(), ErrorKind::NotFound),
        other => panic!("expected SKError::Io(NotFound), got {other:?}"),
    }

    let empty_index_path = temporary_directory
        .path()
        .join("empty.safetensors.index.json");
    std::fs::write(
        &empty_index_path,
        r#"{"format_version":1,"metadata":{},"shards":[],"weight_map":{}}"#,
    )
    .unwrap();
    assert!(matches!(
        sk_read_model_state_safetensors_file::<f64>(&empty_index_path),
        Err(SKError::EmptyInput { .. })
    ));
}

/// Duplicate shard entries, path-traversal shard names, duplicate tensors across shards and mismatched shard metadata fail.
#[test]
fn safetensors_shards_rejects_duplicate_shards_duplicate_tensors_and_mismatched_metadata() {
    let temporary_directory = tempfile::tempdir().unwrap();
    let duplicate_shard_index = temporary_directory
        .path()
        .join("duplicate.safetensors.index.json");
    std::fs::write(
        &duplicate_shard_index,
        r#"{"format_version":1,"metadata":{},"shards":["a.safetensors","a.safetensors"],"weight_map":{}}"#,
    )
    .unwrap();
    assert!(matches!(
        sk_read_model_state_safetensors_file::<f64>(&duplicate_shard_index),
        Err(SKError::InvalidHyperparameter { name: "shards", .. })
    ));

    let traversal_index = temporary_directory
        .path()
        .join("escape.safetensors.index.json");
    std::fs::write(
        &traversal_index,
        r#"{"format_version":1,"metadata":{},"shards":["../escape.safetensors"],"weight_map":{}}"#,
    )
    .unwrap();
    assert!(matches!(
        sk_read_model_state_safetensors_file::<f64>(&traversal_index),
        Err(SKError::InvalidHyperparameter { name: "shards", .. })
    ));

    let metadata_one = SKModelMetadata::new("SKLasso", 1, SKTensorDataType::F64, 2).unwrap();
    let state_one = SKModelState::new(
        metadata_one,
        vec![SKTensor::floating("lasso.weights", vec![2], vec![1.0_f64, 2.0]).unwrap()],
    )
    .unwrap();
    let writer = SKSafetensorsWriter::builder().build().unwrap();
    let shard_a_path = temporary_directory.path().join("s1.safetensors");
    let shard_b_path = temporary_directory.path().join("s2.safetensors");
    writer
        .write_state_to_path(&state_one, &shard_a_path)
        .unwrap();
    writer
        .write_state_to_path(&state_one, &shard_b_path)
        .unwrap();

    let duplicate_tensor_index = temporary_directory
        .path()
        .join("duplicate_tensor.safetensors.index.json");
    std::fs::write(
        &duplicate_tensor_index,
        r#"{
            "format_version": 1,
            "metadata": {
                "sciencekit.format_version": "1",
                "sciencekit.estimator_name": "SKLasso",
                "sciencekit.scalar_data_type": "F64",
                "sciencekit.feature_count": "2",
                "sciencekit.parameters": "{}"
            },
            "shards": ["s1.safetensors", "s2.safetensors"],
            "weight_map": { "lasso.weights": "s1.safetensors" }
        }"#,
    )
    .unwrap();
    assert!(matches!(
        sk_read_model_state_safetensors_file::<f64>(&duplicate_tensor_index),
        Err(SKError::InvalidHyperparameter {
            name: "tensor_names",
            ..
        })
    ));

    let foreign_metadata = SKModelMetadata::new("SKRidge", 1, SKTensorDataType::F64, 2).unwrap();
    let foreign_state = SKModelState::new(
        foreign_metadata,
        vec![SKTensor::floating("ridge.intercept", vec![], vec![0.5_f64]).unwrap()],
    )
    .unwrap();
    let shard_c_path = temporary_directory.path().join("s3.safetensors");
    writer
        .write_state_to_path(&foreign_state, &shard_c_path)
        .unwrap();

    let mismatched_metadata_index = temporary_directory
        .path()
        .join("mismatched.safetensors.index.json");
    std::fs::write(
        &mismatched_metadata_index,
        r#"{"format_version":1,"metadata":{"sciencekit.format_version":"1","sciencekit.estimator_name":"SKLasso","sciencekit.scalar_data_type":"F64","sciencekit.feature_count":"2","sciencekit.parameters":"{}"},"shards":["s1.safetensors","s3.safetensors"],"weight_map":{"lasso.weights":"s1.safetensors","ridge.intercept":"s3.safetensors"}}"#,
    )
    .unwrap();
    assert!(matches!(
        sk_read_model_state_safetensors_file::<f64>(&mismatched_metadata_index),
        Err(SKError::InvalidHyperparameter { name: "shards", .. })
    ));
}

/// Future major container versions (`2` or `2.1`) in single files or shard indexes return `UnsupportedRepresentation`.
#[test]
fn safetensors_shards_and_single_files_reject_future_major_versions() {
    let payload: Vec<u8> = [1.0_f64].into_iter().flat_map(f64::to_le_bytes).collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F64, vec![1], &payload).unwrap();
    let mut future_metadata = HashMap::new();
    future_metadata.insert("sciencekit.format_version".to_string(), "2".to_string());
    future_metadata.insert(
        "sciencekit.estimator_name".to_string(),
        "SKLinearRegression".to_string(),
    );
    future_metadata.insert(
        "sciencekit.future_tensor_layout".to_string(),
        "block_sparse".to_string(),
    );
    let future_bytes =
        safetensors::tensor::serialize([("w", view)], Some(future_metadata)).unwrap();

    let single_outcome = sk_read_model_state_safetensors::<f64>(&future_bytes);
    match single_outcome {
        Err(SKError::UnsupportedRepresentation { suggestion, .. }) => {
            assert!(!suggestion.is_empty());
        }
        other => panic!("expected UnsupportedRepresentation for v2 file, got {other:?}"),
    }

    let temporary_directory = tempfile::tempdir().unwrap();
    let base_path = temporary_directory.path().join("future.safetensors");
    let future_index = sk_safetensors_index_path(&base_path);
    std::fs::write(&future_index, r#"{"format_version":2,"future_field":true}"#).unwrap();
    assert!(matches!(
        sk_read_model_state_safetensors_file::<f64>(&base_path),
        Err(SKError::UnsupportedRepresentation { .. })
    ));
}
