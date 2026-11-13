//! Fixture writers for the memory-mapped source tests.

use std::path::Path;

use arrow_array::{ArrayRef, Float32Array, Float64Array, RecordBatch};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

/// Write arbitrary bytes to `path`.
pub(crate) fn sk_write_bytes(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).expect("a written fixture");
}

/// Encode `f64` values in little-endian order.
pub(crate) fn sk_f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Encode `f32` values in little-endian order.
pub(crate) fn sk_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Encode `i64` values in little-endian order.
pub(crate) fn sk_i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn sk_npy_header(
    descr: &str,
    number_of_rows: usize,
    number_of_columns: usize,
    fortran_order: bool,
    padded: bool,
) -> Vec<u8> {
    let fortran_order = if fortran_order { "True" } else { "False" };
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': {fortran_order}, 'shape': ({number_of_rows}, {number_of_columns}), }}"
    );
    let mut text = dict.into_bytes();
    if padded {
        let framing = 10 + text.len() + 1;
        let padding = (64 - framing % 64) % 64;
        text.extend(std::iter::repeat_n(b' ', padding));
    } else {
        // Nudge an unpadded header onto an offset that is deliberately not a
        // multiple of eight bytes.
        while (10 + text.len() + 1) % 8 == 0 {
            text.push(b' ');
        }
    }
    text.push(b'\n');
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.push(1);
    bytes.push(0);
    bytes.extend((text.len() as u16).to_le_bytes());
    bytes.extend(text);
    bytes
}

/// Write a C-order (or Fortran-order) `.npy` fixture.
pub(crate) fn sk_write_npy(
    path: &Path,
    descr: &str,
    number_of_rows: usize,
    number_of_columns: usize,
    fortran_order: bool,
    padded: bool,
    payload: &[u8],
) {
    let mut bytes = sk_npy_header(
        descr,
        number_of_rows,
        number_of_columns,
        fortran_order,
        padded,
    );
    bytes.extend_from_slice(payload);
    sk_write_bytes(path, &bytes);
}

/// Write a two-column `f64` Arrow IPC file split into batches of
/// `rows_per_batch`.
pub(crate) fn sk_write_ipc_f64(path: &Path, number_of_rows: usize, rows_per_batch: usize) {
    let schema = Schema::new(vec![
        Field::new("first", DataType::Float64, false),
        Field::new("second", DataType::Float64, false),
    ]);
    let file = std::fs::File::create(path).expect("an IPC fixture");
    let mut writer = FileWriter::try_new(file, &schema).expect("a file writer");
    let mut start = 0;
    while start < number_of_rows {
        let end = (start + rows_per_batch).min(number_of_rows);
        let first: Vec<f64> = (start..end).map(|row| row as f64).collect();
        let second: Vec<f64> = (start..end).map(|row| row as f64 + 0.5).collect();
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema.clone()),
            vec![
                std::sync::Arc::new(Float64Array::from(first)) as ArrayRef,
                std::sync::Arc::new(Float64Array::from(second)) as ArrayRef,
            ],
        )
        .expect("a record batch");
        writer.write(&batch).expect("a written batch");
        start = end;
    }
    writer.finish().expect("a finished IPC file");
}

/// Write a two-column `f32` Arrow IPC file split into batches.
pub(crate) fn sk_write_ipc_f32(path: &Path, number_of_rows: usize, rows_per_batch: usize) {
    let schema = Schema::new(vec![
        Field::new("first", DataType::Float32, false),
        Field::new("second", DataType::Float32, false),
    ]);
    let file = std::fs::File::create(path).expect("an IPC fixture");
    let mut writer = FileWriter::try_new(file, &schema).expect("a file writer");
    let mut start = 0;
    while start < number_of_rows {
        let end = (start + rows_per_batch).min(number_of_rows);
        let first: Vec<f32> = (start..end).map(|row| row as f32).collect();
        let second: Vec<f32> = (start..end).map(|row| row as f32 + 0.5).collect();
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema.clone()),
            vec![
                std::sync::Arc::new(Float32Array::from(first)) as ArrayRef,
                std::sync::Arc::new(Float32Array::from(second)) as ArrayRef,
            ],
        )
        .expect("a record batch");
        writer.write(&batch).expect("a written batch");
        start = end;
    }
    writer.finish().expect("a finished IPC file");
}
