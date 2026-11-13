//! TDD suite for the Arrow IPC memory-mapped source (tasks 3.1–3.4, 4.3).

use sciencekit_common::{SKDataView, SKError, SKLazySource, SKMappableSource};

use super::arrow_ipc_source::SKArrowIpcMappableSource;
use super::test_fixtures::{sk_write_bytes, sk_write_ipc_f32, sk_write_ipc_f64};

#[test]
fn opening_indexes_batches_without_decoding_them() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrow");
    sk_write_ipc_f64(&path, 6, 2);
    let source = SKArrowIpcMappableSource::<f64>::open(&path).expect("an open IPC source");
    assert_eq!(source.number_of_batches(), 3);
    assert_eq!(source.number_of_rows(), 6);
    assert_eq!(source.number_of_columns(), 2);
    assert_eq!(source.decoded_batch_count(), 0);
}

#[test]
fn row_lookup_decodes_only_the_containing_batch() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrow");
    sk_write_ipc_f64(&path, 6, 2);
    let source = SKArrowIpcMappableSource::<f64>::open(&path).expect("an open IPC source");
    assert_eq!(source.row(0).unwrap().to_vec(), vec![0.0, 0.5]);
    assert_eq!(source.decoded_batch_count(), 1);
    assert_eq!(source.row(1).unwrap().to_vec(), vec![1.0, 1.5]);
    assert_eq!(source.decoded_batch_count(), 1);
    assert_eq!(source.row(2).unwrap().to_vec(), vec![2.0, 2.5]);
    assert_eq!(source.decoded_batch_count(), 2);
    assert_eq!(source.row(4).unwrap().to_vec(), vec![4.0, 4.5]);
    assert_eq!(source.row(5).unwrap().to_vec(), vec![5.0, 5.5]);
    assert_eq!(source.decoded_batch_count(), 3);
}

#[test]
fn first_middle_and_last_rows_are_correct() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrow");
    sk_write_ipc_f64(&path, 100, 10);
    let source = SKArrowIpcMappableSource::<f64>::open(&path).expect("an open IPC source");
    assert_eq!(source.row(0).unwrap()[0], 0.0);
    assert_eq!(source.row(55).unwrap()[0], 55.0);
    assert_eq!(source.row(99).unwrap()[0], 99.0);
    assert!(source.decoded_batch_count() <= 3);
}

#[test]
fn float32_columns_are_converted() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrow");
    sk_write_ipc_f32(&path, 4, 2);
    let source = SKArrowIpcMappableSource::<f32>::open(&path).expect("an open f32 source");
    assert_eq!(source.row(3).unwrap().to_vec(), vec![3.0, 3.5]);
}

#[test]
fn truncated_file_is_rejected_on_open() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("truncated.arrow");
    sk_write_ipc_f64(&path, 6, 2);
    let bytes = std::fs::read(&path).unwrap();
    sk_write_bytes(&path, &bytes[..bytes.len() / 2]);
    let error = SKArrowIpcMappableSource::<f64>::open(&path).expect_err("a truncated file");
    assert!(matches!(error, SKError::MalformedMappedHeader { .. }));
}

#[test]
fn non_ipc_file_is_rejected() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("not.arrow");
    sk_write_bytes(&path, b"this is not an arrow file, not at all");
    let error = SKArrowIpcMappableSource::<f64>::open(&path).expect_err("not an IPC file");
    assert!(matches!(error, SKError::MalformedMappedHeader { .. }));
}

#[test]
fn sequential_streaming_agrees_with_random_access() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("table.arrow");
    sk_write_ipc_f64(&path, 10, 2);
    let mut source = SKArrowIpcMappableSource::<f64>::open(&path)
        .expect("an open IPC source")
        .with_block_size(4)
        .expect("a non-zero block size");
    let mut seen = Vec::new();
    let mut finals = 0;
    while let Some(batch) = source.next_batch().unwrap() {
        assert!(batch.number_of_rows() <= 4);
        if batch.is_final() {
            finals += 1;
        }
        match batch.view() {
            SKDataView::Dense(view) => seen.extend(view.iter().copied()),
            _ => panic!("expected a dense block"),
        }
    }
    assert_eq!(finals, 1);
    assert_eq!(seen.len(), 20);
    let random = SKArrowIpcMappableSource::<f64>::open(&path).expect("a second open");
    assert_eq!(seen[0], random.row(0).unwrap()[0]);
    assert_eq!(seen[19], random.row(9).unwrap()[1]);
}
