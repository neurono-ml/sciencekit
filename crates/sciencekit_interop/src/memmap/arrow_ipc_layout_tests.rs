//! Tests for the Arrow IPC footer index (task 3.1, hostile-input hardening).

use sciencekit_common::SKError;

use super::arrow_ipc_layout::sk_add_batch_rows;

#[test]
fn cumulative_row_count_overflow_is_a_typed_error() {
    let error = sk_add_batch_rows(usize::MAX - 1, 2, 7, "table.arrow")
        .expect_err("a cumulative row count that overflows usize is rejected");
    match error {
        SKError::MalformedMappedHeader { path, format, .. } => {
            assert_eq!(path, "table.arrow");
            assert_eq!(format, "arrow-ipc");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn cumulative_row_count_accumulates_without_overflow() {
    assert_eq!(sk_add_batch_rows(10, 5, 1, "table.arrow").unwrap(), 15);
}
