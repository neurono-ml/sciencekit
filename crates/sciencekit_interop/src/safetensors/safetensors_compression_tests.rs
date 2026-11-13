//! Compressed Safetensors round-trip tests behind optional codec features
//! (spec `safetensors-format`).

#[cfg(any(
    feature = "compression-gzip",
    feature = "compression-brotli",
    feature = "compression-snappy"
))]
use sciencekit_common::{
    SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};

use super::SKSafetensorsCompression;
#[cfg(any(
    feature = "compression-gzip",
    feature = "compression-brotli",
    feature = "compression-snappy"
))]
use super::{
    SKSafetensorsWriter, sk_detect_safetensors_compression, sk_read_model_state_safetensors,
    sk_read_model_state_safetensors_file, sk_write_model_state_safetensors,
};

#[cfg(any(
    feature = "compression-gzip",
    feature = "compression-brotli",
    feature = "compression-snappy"
))]
fn sample_compressible_state() -> SKModelState<f64> {
    let metadata =
        SKModelMetadata::new("SKGradientBoostingClassifier", 1, SKTensorDataType::F64, 8)
            .unwrap()
            .with_class_labels(vec!["stay".to_string(), "churn".to_string()])
            .unwrap()
            .with_parameter("learning_rate", SKParameterValue::from(0.05_f64))
            .unwrap();
    let repeated_weights: Vec<f64> = (0..2_048).map(|index| ((index % 8) as f64) * 0.5).collect();
    let leaf_mask: Vec<bool> = (0..1_024).map(|index| index % 4 == 0).collect();
    SKModelState::new(
        metadata,
        vec![
            SKTensor::floating("boosting.weights", vec![256, 8], repeated_weights).unwrap(),
            SKTensor::<f64>::boolean("boosting.is_leaf", vec![1_024], leaf_mask).unwrap(),
        ],
    )
    .unwrap()
}

/// Extension metadata for every compression variant matches PRD §8.4 naming.
#[test]
fn safetensors_compression_variants_report_expected_extensions() {
    assert_eq!(
        SKSafetensorsCompression::None.file_extension(),
        "safetensors"
    );
    assert_eq!(
        SKSafetensorsCompression::Gzip.file_extension(),
        "safetensors.gz"
    );
    assert_eq!(
        SKSafetensorsCompression::Brotli.file_extension(),
        "safetensors.brotli"
    );
    assert_eq!(
        SKSafetensorsCompression::Snappy.file_extension(),
        "snappy.safetensors"
    );
}

/// Gzip-compressed bytes and `.safetensors.gz` files round-trip transparently.
#[cfg(feature = "compression-gzip")]
#[test]
fn safetensors_compression_gzip_round_trips_bytes_files_and_shards() {
    let state = sample_compressible_state();
    let uncompressed = sk_write_model_state_safetensors(&state).unwrap();

    let writer = SKSafetensorsWriter::builder()
        .compression(SKSafetensorsCompression::Gzip)
        .build()
        .unwrap();
    let compressed = writer.write_state_to_bytes(&state).unwrap();
    assert!(compressed.len() < uncompressed.len());
    assert_eq!(
        sk_detect_safetensors_compression(&compressed),
        Some(SKSafetensorsCompression::Gzip)
    );
    assert_eq!(
        sk_read_model_state_safetensors::<f64>(&compressed).unwrap(),
        sk_read_model_state_safetensors::<f64>(&uncompressed).unwrap()
    );

    let temporary_directory = tempfile::tempdir().unwrap();
    let gzip_path = temporary_directory.path().join("model.safetensors.gz");
    writer.write_state_to_path(&state, &gzip_path).unwrap();
    assert_eq!(
        sk_read_model_state_safetensors_file::<f64>(&gzip_path).unwrap(),
        state
    );

    let sharded_gzip_path = temporary_directory.path().join("sharded.safetensors.gz");
    SKSafetensorsWriter::builder()
        .compression(SKSafetensorsCompression::Gzip)
        .maximum_shard_bytes(4_096)
        .build()
        .unwrap()
        .write_state_to_path(&state, &sharded_gzip_path)
        .unwrap();
    assert_eq!(
        sk_read_model_state_safetensors_file::<f64>(&sharded_gzip_path).unwrap(),
        state
    );
}

/// Brotli-compressed bytes and `.safetensors.brotli` files round-trip transparently.
#[cfg(feature = "compression-brotli")]
#[test]
fn safetensors_compression_brotli_round_trips_bytes_and_files() {
    let state = sample_compressible_state();
    let uncompressed = sk_write_model_state_safetensors(&state).unwrap();

    let writer = SKSafetensorsWriter::builder()
        .compression(SKSafetensorsCompression::Brotli)
        .build()
        .unwrap();
    let compressed = writer.write_state_to_bytes(&state).unwrap();
    assert!(compressed.len() < uncompressed.len());
    assert_eq!(
        sk_detect_safetensors_compression(&compressed),
        Some(SKSafetensorsCompression::Brotli)
    );
    assert_eq!(
        sk_read_model_state_safetensors::<f64>(&compressed).unwrap(),
        state
    );

    let temporary_directory = tempfile::tempdir().unwrap();
    let brotli_path = temporary_directory.path().join("model.safetensors.brotli");
    writer.write_state_to_path(&state, &brotli_path).unwrap();
    assert_eq!(
        sk_read_model_state_safetensors_file::<f64>(&brotli_path).unwrap(),
        state
    );
}

/// Snappy-compressed bytes and `.snappy.safetensors` files round-trip transparently.
#[cfg(feature = "compression-snappy")]
#[test]
fn safetensors_compression_snappy_round_trips_bytes_and_files() {
    let state = sample_compressible_state();
    let uncompressed = sk_write_model_state_safetensors(&state).unwrap();

    let writer = SKSafetensorsWriter::builder()
        .compression(SKSafetensorsCompression::Snappy)
        .build()
        .unwrap();
    let compressed = writer.write_state_to_bytes(&state).unwrap();
    assert!(compressed.len() < uncompressed.len());
    assert_eq!(
        sk_detect_safetensors_compression(&compressed),
        Some(SKSafetensorsCompression::Snappy)
    );
    assert_eq!(
        sk_read_model_state_safetensors::<f64>(&compressed).unwrap(),
        state
    );

    let temporary_directory = tempfile::tempdir().unwrap();
    let snappy_path = temporary_directory.path().join("model.snappy.safetensors");
    writer.write_state_to_path(&state, &snappy_path).unwrap();
    assert_eq!(
        sk_read_model_state_safetensors_file::<f64>(&snappy_path).unwrap(),
        state
    );
}
