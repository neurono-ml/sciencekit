//! Raw deserialization types for the JSON debug reader (spec `json-debug-format`).

use std::fmt;
use std::marker::PhantomData;

use serde::Deserialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

use super::parameter_deserialization::RawJsonParameter;

fn default_estimator_format_version() -> u32 {
    1
}

/// Raw JSON debug document preserving duplicate map entries for explicit validation.
#[derive(Debug, Deserialize)]
pub(crate) struct RawJsonModelDocument {
    pub(crate) format_version: u32,
    pub(crate) estimator: String,
    #[serde(default = "default_estimator_format_version")]
    pub(crate) estimator_format_version: u32,
    #[serde(rename = "dtype")]
    pub(crate) scalar_data_type: String,
    pub(crate) feature_count: usize,
    pub(crate) class_labels: Option<Vec<String>>,
    #[serde(deserialize_with = "deserialize_ordered_entries")]
    pub(crate) hyperparameters: Vec<(String, RawJsonParameter)>,
    #[serde(deserialize_with = "deserialize_ordered_entries")]
    pub(crate) tensors: Vec<(String, RawJsonTensor)>,
}

/// Raw tensor entry prior to dtype, shape and duplicate-name validation.
#[derive(Debug, Deserialize)]
pub(crate) struct RawJsonTensor {
    #[serde(rename = "dtype")]
    pub(crate) data_type: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<RawJsonScalar>,
}

/// Untagged JSON scalar inside a tensor's flat `values` array.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RawJsonScalar {
    Bool(bool),
    Integer(i64),
    Float(f64),
    Unsupported(&'static str),
}

impl<'de> Deserialize<'de> for RawJsonScalar {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ScalarVisitor;

        impl<'de> Visitor<'de> for ScalarVisitor {
            type Value = RawJsonScalar;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a scalar tensor element")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(RawJsonScalar::Bool(value))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(RawJsonScalar::Integer(value))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(i64::try_from(value).map_or(
                    RawJsonScalar::Unsupported("u64 above i64::MAX"),
                    RawJsonScalar::Integer,
                ))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(RawJsonScalar::Float(value))
            }

            fn visit_str<E: de::Error>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(RawJsonScalar::Unsupported("string"))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(RawJsonScalar::Unsupported("null"))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawJsonScalar::Unsupported("array"))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(RawJsonScalar::Unsupported("object"))
            }
        }

        deserializer.deserialize_any(ScalarVisitor)
    }
}

fn deserialize_ordered_entries<'de, D, V>(deserializer: D) -> Result<Vec<(String, V)>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct EntriesVisitor<V>(PhantomData<V>);

    impl<'de, V: Deserialize<'de>> Visitor<'de> for EntriesVisitor<V> {
        type Value = Vec<(String, V)>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a JSON object")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
            while let Some(pair) = map.next_entry::<String, V>()? {
                entries.push(pair);
            }
            Ok(entries)
        }
    }

    deserializer.deserialize_map(EntriesVisitor(PhantomData))
}
