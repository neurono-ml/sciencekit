//! Import of Arrow string-dictionary columns into canonical category codes
//! (spec `arrow-columnar-interop`).

use arrow_array::types::{
    ArrowDictionaryKeyType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, DictionaryArray};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use sciencekit_common::SKError;

use super::dictionary_label_table::{
    sk_canonicalize_dictionary, sk_dictionary_conversion_error, sk_unsupported_dictionary_error,
};
use super::null_mapping::sk_validate_constant_fill_policy;
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

fn sk_remap_dictionary_keys<KeyType>(
    name: &str,
    array: &dyn Array,
    slot_to_code: &[Option<i64>],
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError>
where
    KeyType: ArrowDictionaryKeyType,
    KeyType::Native: ArrowNativeType,
{
    let dictionary = array
        .as_any()
        .downcast_ref::<DictionaryArray<KeyType>>()
        .ok_or_else(|| {
            sk_dictionary_conversion_error(name, "failed to downcast dictionary keys")
        })?;
    let codes = dictionary
        .keys()
        .iter()
        .map(|key| match key {
            Some(raw_key) => {
                let slot = raw_key.to_usize().ok_or_else(|| {
                    sk_dictionary_conversion_error(
                        name,
                        &format!("carries negative dictionary key {raw_key:?}"),
                    )
                })?;
                match slot_to_code.get(slot) {
                    Some(Some(code)) => Ok(Some(*code)),
                    Some(None) => Err(sk_dictionary_conversion_error(
                        name,
                        &format!(
                            "carries dictionary key {slot}, which references a null dictionary value"
                        ),
                    )),
                    None => Err(sk_dictionary_conversion_error(
                        name,
                        &format!(
                            "carries dictionary key {slot}, which is out of range for {} dictionary values",
                            slot_to_code.len()
                        ),
                    )),
                }
            }
            None => match policy {
                SKMissingValuePolicy::Reject => Err(SKError::NullValueInColumn {
                    column: name.to_string(),
                    policy: policy.name(),
                }),
                _ => Ok(None),
            },
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    Ok(SKColumnData::CategoryCode(codes))
}

/// Convert a string-dictionary Arrow column into canonical integer category
/// codes consistent with [`super::sk_dictionary_label_table`].
pub(super) fn sk_dictionary_column(
    name: &str,
    array: &dyn Array,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError>
where
{
    sk_validate_constant_fill_policy(name, policy)?;
    let (slot_to_code, _) = sk_canonicalize_dictionary(name, array)?;
    let DataType::Dictionary(key, _) = array.data_type() else {
        return Err(sk_unsupported_dictionary_error(name));
    };
    match key.as_ref() {
        DataType::Int8 => sk_remap_dictionary_keys::<Int8Type>(name, array, &slot_to_code, policy),
        DataType::Int16 => {
            sk_remap_dictionary_keys::<Int16Type>(name, array, &slot_to_code, policy)
        }
        DataType::Int32 => {
            sk_remap_dictionary_keys::<Int32Type>(name, array, &slot_to_code, policy)
        }
        DataType::Int64 => {
            sk_remap_dictionary_keys::<Int64Type>(name, array, &slot_to_code, policy)
        }
        DataType::UInt8 => {
            sk_remap_dictionary_keys::<UInt8Type>(name, array, &slot_to_code, policy)
        }
        DataType::UInt16 => {
            sk_remap_dictionary_keys::<UInt16Type>(name, array, &slot_to_code, policy)
        }
        DataType::UInt32 => {
            sk_remap_dictionary_keys::<UInt32Type>(name, array, &slot_to_code, policy)
        }
        DataType::UInt64 => {
            sk_remap_dictionary_keys::<UInt64Type>(name, array, &slot_to_code, policy)
        }
        _ => Err(sk_unsupported_dictionary_error(name)),
    }
}
