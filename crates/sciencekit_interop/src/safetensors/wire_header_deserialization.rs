//! Duplicate-preserving `serde` visitors for Safetensors JSON headers
//! (spec `safetensors-format`).

use std::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, Visitor};

#[derive(Debug)]
pub(crate) struct RawTensorHeader {
    pub(crate) data_type: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: [usize; 2],
}

impl<'de> Deserialize<'de> for RawTensorHeader {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct WireTensorFields {
            #[serde(rename = "dtype")]
            data_type: String,
            shape: Vec<usize>,
            data_offsets: [usize; 2],
        }
        let fields = WireTensorFields::deserialize(deserializer)?;
        Ok(Self {
            data_type: fields.data_type,
            shape: fields.shape,
            data_offsets: fields.data_offsets,
        })
    }
}

pub(crate) enum RawHeaderValue {
    Metadata(Vec<(String, String)>),
    Tensor(RawTensorHeader),
}

struct OrderedStringMap(Vec<(String, String)>);

impl<'de> Deserialize<'de> for OrderedStringMap {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StringMapVisitor;

        impl<'de> Visitor<'de> for StringMapVisitor {
            type Value = OrderedStringMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a string-to-string metadata map")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut pairs = Vec::new();
                while let Some(pair) = map.next_entry::<String, String>()? {
                    pairs.push(pair);
                }
                Ok(OrderedStringMap(pairs))
            }
        }

        deserializer.deserialize_map(StringMapVisitor)
    }
}

pub(crate) struct OrderedHeaderEntries(pub(crate) Vec<(String, RawHeaderValue)>);

impl<'de> Deserialize<'de> for OrderedHeaderEntries {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HeaderVisitor;

        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = OrderedHeaderEntries;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a Safetensors JSON header object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    let entry = if key == "__metadata__" {
                        let OrderedStringMap(pairs) = map.next_value::<OrderedStringMap>()?;
                        RawHeaderValue::Metadata(pairs)
                    } else {
                        RawHeaderValue::Tensor(map.next_value::<RawTensorHeader>().map_err(
                            |error| de::Error::custom(format!("invalid tensor `{key}`: {error}")),
                        )?)
                    };
                    entries.push((key, entry));
                }
                Ok(OrderedHeaderEntries(entries))
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}
