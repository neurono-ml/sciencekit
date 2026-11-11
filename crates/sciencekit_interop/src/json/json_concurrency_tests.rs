//! Companion equivalence, large-tensor and concurrency tests for JSON debug
//! (spec `json-debug-format`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use sciencekit_common::{
    SKBuilder, SKError, SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};

use super::{
    SKJsonDebugReader, SKJsonDebugReaderBuilder, SKJsonDebugWriter, SKJsonDebugWriterBuilder,
    SKJsonFormatMode, sk_read_model_state_json, sk_read_model_state_json_file,
    sk_write_model_state_json_compact, sk_write_model_state_json_file,
    sk_write_model_state_json_pretty,
};

static NEXT_CONCURRENCY_FILE_INDEX: AtomicUsize = AtomicUsize::new(0);

fn assert_send_and_sync<T: Send + Sync>() {}

fn build_rich_state() -> SKModelState<f64> {
    let metadata = SKModelMetadata::new("SKRandomForestClassifier", 1, SKTensorDataType::F64, 4)
        .unwrap()
        .with_class_labels(vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
        ])
        .unwrap()
        .with_parameter("number_of_trees", SKParameterValue::from(16_i64))
        .unwrap()
        .with_parameter("bootstrap", SKParameterValue::from(true))
        .unwrap()
        .with_parameter("minimum_impurity_decrease", SKParameterValue::from(1.0e-7))
        .unwrap();
    let importances = SKTensor::floating(
        "forest.feature_importances",
        vec![4],
        vec![0.4_f64, 0.3, 0.2, 0.1],
    )
    .unwrap();
    let thresholds = SKTensor::floating(
        "forest.thresholds",
        vec![2, 3],
        vec![-1.25_f64, 0.0, 1.25, 2.5, 3.75, 5.0],
    )
    .unwrap();
    let children =
        SKTensor::<f64>::integer("forest.children_left", vec![6], vec![1, 2, -1, -1, 5, -1])
            .unwrap();
    let leaves = SKTensor::<f64>::boolean(
        "forest.is_leaf",
        vec![6],
        vec![false, false, true, true, false, true],
    )
    .unwrap();
    SKModelState::new(metadata, vec![importances, thresholds, children, leaves]).unwrap()
}

/// Pretty and compact serializations of the same state parse into identical `SKModelState`s.
#[test]
fn json_concurrency_pretty_and_compact_modes_parse_to_identical_state() {
    let state = build_rich_state();

    let pretty = sk_write_model_state_json_pretty(&state).unwrap();
    let compact = sk_write_model_state_json_compact(&state).unwrap();
    assert_ne!(pretty, compact);

    let from_pretty = sk_read_model_state_json::<f64>(&pretty).unwrap();
    let from_compact = sk_read_model_state_json::<f64>(&compact).unwrap();

    assert_eq!(from_pretty, state);
    assert_eq!(from_compact, state);
    assert_eq!(from_pretty, from_compact);
}

/// Large tensors round-trip without precision loss and reject mismatched shapes.
#[test]
fn json_concurrency_large_tensor_round_trips_and_validates_shape() {
    let rows = 256_usize;
    let columns = 256_usize;
    let element_count = rows * columns;
    let values: Vec<f64> = (0..element_count)
        .map(|index| (index as f64) * 0.000_976_562_5 - 32.0)
        .collect();

    let metadata = SKModelMetadata::new(
        "SKPrincipalComponentAnalysis",
        1,
        SKTensorDataType::F64,
        columns,
    )
    .unwrap();
    let components = SKTensor::floating("pca.components", vec![rows, columns], values).unwrap();
    let state = SKModelState::new(metadata, vec![components]).unwrap();

    let compact = sk_write_model_state_json_compact(&state).unwrap();
    // JSON decimal text expands past the raw 8-byte-per-element binary footprint.
    assert!(compact.len() > element_count * std::mem::size_of::<f64>());

    let restored = sk_read_model_state_json::<f64>(&compact).unwrap();
    assert_eq!(restored, state);

    let mismatched = compact.replacen("\"shape\":[256,256]", "\"shape\":[256,255]", 1);
    assert!(matches!(
        sk_read_model_state_json::<f64>(&mismatched),
        Err(SKError::ShapeMismatch { .. })
    ));
}

/// Multiple OS threads reading the same JSON file concurrently agree with single-threaded read.
#[test]
fn json_concurrency_concurrent_reads_from_same_file_agree() {
    assert_send_and_sync::<SKJsonDebugWriter>();
    assert_send_and_sync::<SKJsonDebugWriterBuilder>();
    assert_send_and_sync::<SKJsonDebugReader>();
    assert_send_and_sync::<SKJsonDebugReaderBuilder>();

    let expected = Arc::new(build_rich_state());
    let index = NEXT_CONCURRENCY_FILE_INDEX.fetch_add(1, Ordering::Relaxed);
    let shared_path = Arc::new(std::env::temp_dir().join(format!(
        "sciencekit_json_concurrent_read_{}_{index}.json",
        std::process::id()
    )));
    sk_write_model_state_json_file(&shared_path, &expected, SKJsonFormatMode::Pretty).unwrap();

    let single_threaded = Arc::new(sk_read_model_state_json_file::<f64>(&shared_path).unwrap());
    assert_eq!(*single_threaded, *expected);

    let reader = SKJsonDebugReader::builder().build().unwrap();
    let writer = SKJsonDebugWriter::builder().build().unwrap();
    let baseline_json = Arc::new(writer.write_state_to_string(&expected).unwrap());

    let mut handles = Vec::new();
    for _worker in 0..8 {
        let path = Arc::clone(&shared_path);
        let expected_state = Arc::clone(&single_threaded);
        let expected_json = Arc::clone(&baseline_json);
        handles.push(thread::spawn(move || {
            let parsed = reader.read_state_from_path::<f64>(&path).unwrap();
            assert_eq!(parsed, *expected_state);
            let reencoded = writer.write_state_to_string(&parsed).unwrap();
            assert_eq!(reencoded, *expected_json);
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }

    let _ = std::fs::remove_file(shared_path.as_ref());
}
