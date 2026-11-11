//! Acceptance scenarios for tiny/large states, sharding, padding, compression,
//! observability spans and dense/sparse model metrics (PRD §8.7).

use ndarray::array;
use sciencekit_common::{
    SKModelMetadata, SKModelState, SKParameterValue, SKRegressorPredictor, SKTensor,
    SKTensorDataType, sk_canonicalize_labels,
};
use sprs::CsMat;
use tracing_subscriber::layer::SubscriberExt;

use super::fixtures::{AcceptanceLinearRegressor, SpanRecorder};
use crate::safetensors::{
    SKAppendOutcome, SKSafetensorsCompression, SKSafetensorsReader, SKSafetensorsWriter,
    SKShardPlan, sk_export_model_safetensors_file, sk_import_model_safetensors_file,
    sk_read_model_state_safetensors, sk_read_model_state_safetensors_file,
    sk_write_model_state_safetensors,
};

/// Tiny zero-tensor and single-element states round-trip for both `f32` and `f64` via inherent `.build()`.
#[test]
fn safetensors_acceptance_tiny_states_round_trip_for_both_scalars() {
    let f32_meta = SKModelMetadata::new("SKDummyTransformer", 1, SKTensorDataType::F32, 1).unwrap();
    let f32_empty = SKModelState::<f32>::new(f32_meta, Vec::new()).unwrap();
    let writer = SKSafetensorsWriter::builder().build().unwrap();
    let reader = SKSafetensorsReader::builder().build().unwrap();
    let f32_bytes = writer.write_state_to_bytes(&f32_empty).unwrap();
    assert_eq!(
        reader.read_state_from_bytes::<f32>(&f32_bytes).unwrap(),
        f32_empty
    );

    let f64_meta = SKModelMetadata::new("SKDummyTransformer", 1, SKTensorDataType::F64, 0).unwrap();
    let f64_empty = SKModelState::<f64>::new(f64_meta, Vec::new()).unwrap();
    let f64_bytes = sk_write_model_state_safetensors(&f64_empty).unwrap();
    assert_eq!(
        sk_read_model_state_safetensors::<f64>(&f64_bytes).unwrap(),
        f64_empty
    );
}

/// Large state exercises sharding, padding, spans and enabled compression codecs end to end.
#[test]
fn safetensors_acceptance_large_state_sharded_padded_compressed_and_traced() {
    let raw_labels = ["setosa", "versicolor", "virginica", "setosa"];
    let (_encoded, label_table) = sk_canonicalize_labels(&raw_labels);
    let canonical_labels: Vec<String> = label_table.labels().map(str::to_string).collect();

    let metadata = SKModelMetadata::new("SKExtraTreesClassifier", 1, SKTensorDataType::F64, 16)
        .unwrap()
        .with_class_labels(canonical_labels.clone())
        .unwrap()
        .with_parameter("number_of_trees", SKParameterValue::from(100_i64))
        .unwrap()
        .with_parameter("bootstrap", SKParameterValue::from(true))
        .unwrap()
        .with_parameter("subsample_ratio", SKParameterValue::from(0.8_f64))
        .unwrap();

    let thresholds: Vec<f64> = (0..40_000).map(|index| (index as f64) * 0.03125).collect();
    let indices: Vec<i64> = (0..20_000).map(|index| (index % 16) - 1).collect();
    let leaves: Vec<bool> = (0..10_000).map(|index| index % 3 == 0).collect();

    let state = SKModelState::new(
        metadata,
        vec![
            SKTensor::floating("ensemble.thresholds", vec![40, 1_000], thresholds).unwrap(),
            SKTensor::<f64>::integer("ensemble.features", vec![20_000], indices).unwrap(),
            SKTensor::<f64>::boolean("ensemble.is_leaf", vec![10_000], leaves).unwrap(),
        ],
    )
    .unwrap();

    let temporary_directory = tempfile::tempdir().unwrap();
    let sharded_path = temporary_directory.path().join("large_sharded.safetensors");
    let shard_plan = SKShardPlan::builder()
        .maximum_shard_bytes(128 * 1024)
        .build()
        .unwrap();

    let recorder = SpanRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let restored_sharded = tracing::subscriber::with_default(subscriber, || {
        tracing::callsite::rebuild_interest_cache();
        let writer = SKSafetensorsWriter::builder()
            .shard_plan(shard_plan)
            .build()
            .unwrap();
        writer.write_state_to_path(&state, &sharded_path).unwrap();
        sk_read_model_state_safetensors_file::<f64>(&sharded_path).unwrap()
    });
    assert_eq!(restored_sharded, state);
    assert_eq!(
        restored_sharded.metadata().class_labels(),
        Some(canonical_labels.as_slice())
    );
    let recorded = recorder.names.lock().unwrap().clone();
    assert!(
        recorded
            .iter()
            .any(|name| name == "sciencekit.interop.safetensors.write")
    );
    assert!(
        recorded
            .iter()
            .any(|name| name == "sciencekit.interop.safetensors.read")
    );

    let padded_path = temporary_directory.path().join("large_padded.safetensors");
    let padded_writer = SKSafetensorsWriter::builder()
        .padding_bytes(256)
        .build()
        .unwrap();
    padded_writer
        .write_state_to_path(&state, &padded_path)
        .unwrap();
    let oob_score = SKTensor::floating("ensemble.oob_score", vec![], vec![0.9375_f64]).unwrap();
    assert_eq!(
        padded_writer
            .append_tensor_to_path(&padded_path, &oob_score)
            .unwrap(),
        SKAppendOutcome::InPlace
    );
    let restored_padded = sk_read_model_state_safetensors_file::<f64>(&padded_path).unwrap();
    assert_eq!(restored_padded.number_of_tensors(), 4);

    let codecs: &[SKSafetensorsCompression] = &[
        SKSafetensorsCompression::None,
        #[cfg(feature = "compression-gzip")]
        SKSafetensorsCompression::Gzip,
        #[cfg(feature = "compression-brotli")]
        SKSafetensorsCompression::Brotli,
        #[cfg(feature = "compression-snappy")]
        SKSafetensorsCompression::Snappy,
    ];
    for &codec in codecs {
        let compressed = SKSafetensorsWriter::builder()
            .compression(codec)
            .build()
            .unwrap()
            .write_state_to_bytes(&state)
            .unwrap();
        assert_eq!(
            sk_read_model_state_safetensors::<f64>(&compressed).unwrap(),
            state
        );
    }
}

/// Exported and imported regressor reproduces identical dense and CSR predictions with zero MSE and R² = 1.
#[test]
fn safetensors_acceptance_exported_regressor_reproduces_dense_and_sparse_metrics() {
    let fitted = AcceptanceLinearRegressor {
        coefficients: vec![2.5, -1.25, 0.75],
        intercept: 0.5,
    };
    let dense_features = array![[1.0_f64, 0.0, 4.0], [0.0, -4.0, 2.0], [3.0, 0.0, -2.0]];
    let sparse_features = CsMat::new(
        (3, 3),
        vec![0, 2, 4, 6],
        vec![0, 2, 1, 2, 0, 2],
        vec![1.0_f64, 4.0, -4.0, 2.0, 3.0, -2.0],
    );
    let targets = array![6.0_f64, 7.0, 6.5];

    let temporary_directory = tempfile::tempdir().unwrap();
    let model_path = temporary_directory.path().join("linear.safetensors");
    sk_export_model_safetensors_file(&model_path, &fitted).unwrap();
    let imported: AcceptanceLinearRegressor =
        sk_import_model_safetensors_file(&model_path).unwrap();

    let dense_predictions = imported.predict(dense_features.view()).unwrap();
    let sparse_predictions = imported.predict(sparse_features.view()).unwrap();
    assert_eq!(
        dense_predictions,
        fitted.predict(dense_features.view()).unwrap()
    );
    assert_eq!(sparse_predictions, dense_predictions);

    let mean_squared_error = dense_predictions
        .iter()
        .zip(targets.iter())
        .map(|(predicted, actual)| (predicted - actual).powi(2))
        .sum::<f64>()
        / (targets.len() as f64);
    assert_eq!(mean_squared_error, 0.0);
}
