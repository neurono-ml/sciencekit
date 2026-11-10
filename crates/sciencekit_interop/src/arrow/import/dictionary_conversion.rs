//! Import of Arrow dictionary columns as category codes plus a reversible label
//! table (spec `arrow-columnar-interop`).

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, DictionaryArray, LargeStringArray, RecordBatch, StringArray};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use sciencekit_common::{SKError, SKLabelTable, sk_canonicalize_labels};

use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

fn sk_dictionary_conversion_error(column_name: &str) -> SKError {
    SKError::Conversion(format!(
        "column `{column_name}` is declared as a string dictionary but does not carry one"
    ))
}

fn sk_dictionary_values_as_strings(values: &ArrayRef) -> Vec<String> {
    match values.data_type() {
        DataType::Utf8 => values
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|array| array.iter().flatten().map(str::to_owned).collect())
            .unwrap_or_default(),
        DataType::LargeUtf8 => values
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|array| array.iter().flatten().map(str::to_owned).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Build the reversible label table of a string-dictionary column.
///
/// The table follows the dictionary value order, so code `i` restores
/// `label_of(i)` exactly. Non-dictionary arrays return `None`.
///
/// # Errors
/// Returns [`SKError::Conversion`] when the array is a dictionary whose values
/// are not text.
pub fn sk_dictionary_label_table(array: &dyn Array) -> Result<Option<SKLabelTable>, SKError> {
    if !matches!(array.data_type(), DataType::Dictionary(_, _)) {
        return Ok(None);
    }
    let labels = sk_dictionary_values_as_strings(array.as_any_dictionary().values());
    if labels.is_empty() {
        return Err(sk_dictionary_conversion_error("dictionary"));
    }
    let borrowed: Vec<&str> = labels.iter().map(String::as_str).collect();
    let (_, table) = sk_canonicalize_labels(&borrowed);
    Ok(Some(table))
}

/// The label table of every dictionary column of a record batch, aligned with
/// the batch's columns.
///
/// # Errors
/// Propagates [`SKError::Conversion`] from
/// [`sk_dictionary_label_table`].
pub fn sk_label_tables_from_record_batch(
    record_batch: &RecordBatch,
) -> Result<Vec<Option<SKLabelTable>>, SKError> {
    record_batch
        .columns()
        .iter()
        .map(|column| sk_dictionary_label_table(column.as_ref()))
        .collect()
}

macro_rules! sk_dictionary_codes {
    ($array:expr, $key:ty, $name:expr, $policy:expr) => {{
        let dictionary = $array
            .as_any()
            .downcast_ref::<DictionaryArray<$key>>()
            .ok_or_else(|| sk_dictionary_conversion_error($name))?;
        let codes = dictionary
            .keys()
            .iter()
            .map(|key| match key {
                Some(key) => key
                    .to_i64()
                    .map(Some)
                    .ok_or_else(|| sk_dictionary_conversion_error($name)),
                None => match $policy {
                    SKMissingValuePolicy::Reject => Err(SKError::NullValueInColumn {
                        column: $name.to_string(),
                        policy: $policy.name(),
                    }),
                    _ => Ok(None),
                },
            })
            .collect::<Result<Vec<_>, SKError>>()?;
        Ok(SKColumnData::CategoryCode(codes))
    }};
}

/// Convert a string-dictionary Arrow column into integer category codes.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy and
/// [`SKError::Conversion`] for an unsupported key type.
pub(super) fn sk_dictionary_column(
    name: &str,
    array: &dyn Array,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    match array.data_type() {
        DataType::Dictionary(key, _) => match key.as_ref() {
            DataType::Int8 => sk_dictionary_codes!(array, Int8Type, name, policy),
            DataType::Int16 => sk_dictionary_codes!(array, Int16Type, name, policy),
            DataType::Int32 => sk_dictionary_codes!(array, Int32Type, name, policy),
            DataType::Int64 => sk_dictionary_codes!(array, Int64Type, name, policy),
            DataType::UInt8 => sk_dictionary_codes!(array, UInt8Type, name, policy),
            DataType::UInt16 => sk_dictionary_codes!(array, UInt16Type, name, policy),
            DataType::UInt32 => sk_dictionary_codes!(array, UInt32Type, name, policy),
            DataType::UInt64 => sk_dictionary_codes!(array, UInt64Type, name, policy),
            _ => Err(sk_dictionary_conversion_error(name)),
        },
        _ => Err(sk_dictionary_conversion_error(name)),
    }
}
