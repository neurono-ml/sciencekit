//! Canonicalization and reversible label-table extraction for Arrow string
//! dictionaries (spec `arrow-columnar-interop`).

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::DataType;
use sciencekit_common::{SKError, SKLabelTable, sk_canonicalize_labels};

pub(super) fn sk_unsupported_dictionary_error(column_name: &str) -> SKError {
    SKError::UnsupportedColumnType {
        column: column_name.to_string(),
        logical_type: "dictionary".to_string(),
        suggestion: "encode the dictionary with string values in the adapter before import"
            .to_string(),
    }
}

pub(super) fn sk_dictionary_conversion_error(column_name: &str, detail: &str) -> SKError {
    SKError::Conversion(format!("column `{column_name}` {detail}"))
}

fn sk_dictionary_slots<'a>(
    column_name: &str,
    values: &'a ArrayRef,
) -> Result<Vec<Option<&'a str>>, SKError> {
    match values.data_type() {
        DataType::Utf8 => values
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|array| array.iter().collect())
            .ok_or_else(|| {
                sk_dictionary_conversion_error(column_name, "failed to downcast utf8 dictionary")
            }),
        DataType::LargeUtf8 => values
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|array| array.iter().collect())
            .ok_or_else(|| {
                sk_dictionary_conversion_error(
                    column_name,
                    "failed to downcast large_utf8 dictionary",
                )
            }),
        _ => Err(sk_unsupported_dictionary_error(column_name)),
    }
}

pub(super) fn sk_canonicalize_dictionary(
    column_name: &str,
    array: &dyn Array,
) -> Result<(Vec<Option<i64>>, SKLabelTable), SKError> {
    let DataType::Dictionary(key_type, value_type) = array.data_type() else {
        return Err(sk_unsupported_dictionary_error(column_name));
    };
    if !key_type.is_integer() || !matches!(**value_type, DataType::Utf8 | DataType::LargeUtf8) {
        return Err(sk_unsupported_dictionary_error(column_name));
    }

    let values = array.as_any_dictionary().values();
    let raw_slots = sk_dictionary_slots(column_name, values)?;
    let present: Vec<&str> = raw_slots.iter().copied().flatten().collect();
    let (canonical_indices, table) = sk_canonicalize_labels(&present);
    let mut canonical_iter = canonical_indices.into_iter();
    let slot_to_code = raw_slots
        .into_iter()
        .map(|slot| {
            slot.map(|_| {
                canonical_iter
                    .next()
                    .expect("every present dictionary slot receives one canonical index")
                    as i64
            })
        })
        .collect();
    Ok((slot_to_code, table))
}

fn sk_dictionary_label_table_for_column(
    column_name: &str,
    array: &dyn Array,
) -> Result<Option<SKLabelTable>, SKError> {
    if !matches!(array.data_type(), DataType::Dictionary(_, _)) {
        return Ok(None);
    }
    let (_, table) = sk_canonicalize_dictionary(column_name, array)?;
    Ok(Some(table))
}

/// Build the reversible label table of a string-dictionary column.
///
/// Unique present dictionary values are canonicalized in first-occurrence
/// order, matching [`sk_column_data_from_arrow_array`](super::sk_column_data_from_arrow_array)
/// so code `i` restores `label_of(i)` even when the Arrow dictionary contains
/// duplicate or unreferenced null value slots. Empty string dictionaries yield
/// an empty label table; non-dictionary arrays return `None`.
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] when the array is a dictionary
/// whose keys are not integers or whose values are not `Utf8`/`LargeUtf8`.
pub fn sk_dictionary_label_table(array: &dyn Array) -> Result<Option<SKLabelTable>, SKError> {
    sk_dictionary_label_table_for_column("dictionary", array)
}

/// The label table of every dictionary column of a record batch, aligned with
/// the batch's columns.
///
/// # Errors
/// Propagates [`SKError::UnsupportedColumnType`] naming the Arrow field when a
/// dictionary column does not carry string values.
pub fn sk_label_tables_from_record_batch(
    record_batch: &RecordBatch,
) -> Result<Vec<Option<SKLabelTable>>, SKError> {
    record_batch
        .schema()
        .fields()
        .iter()
        .zip(record_batch.columns())
        .map(|(field, column)| sk_dictionary_label_table_for_column(field.name(), column.as_ref()))
        .collect()
}
