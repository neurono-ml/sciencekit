//! Untagged hyperparameter deserialization for the JSON debug reader
//! (spec `json-debug-format`).

use std::fmt;

use sciencekit_common::{SKError, SKParameterValue};
use serde::Deserialize;
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

/// Untagged JSON hyperparameter value prior to validation.
#[derive(Debug, Clone)]
pub(crate) enum RawJsonParameter {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    List(Vec<RawJsonParameter>),
    Unsupported(&'static str),
}

impl RawJsonParameter {
    pub(crate) fn into_parameter_value(self) -> Result<SKParameterValue, SKError> {
        match self {
            RawJsonParameter::Bool(flag) => Ok(SKParameterValue::Bool(flag)),
            RawJsonParameter::Integer(integer) => Ok(SKParameterValue::Integer(integer)),
            RawJsonParameter::Float(float) if float.is_finite() => {
                Ok(SKParameterValue::Float(float))
            }
            RawJsonParameter::Float(_) => Err(SKError::NonFiniteValue { position: 0 }),
            RawJsonParameter::String(text) => Ok(SKParameterValue::String(text)),
            RawJsonParameter::List(items) => items
                .into_iter()
                .map(RawJsonParameter::into_parameter_value)
                .collect::<Result<Vec<_>, _>>()
                .map(SKParameterValue::List),
            RawJsonParameter::Unsupported(kind) => Err(SKError::InvalidHyperparameter {
                name: "hyperparameters",
                reason: format!("hyperparameter value of JSON kind `{kind}` is not supported"),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for RawJsonParameter {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ParameterVisitor;

        impl<'de> Visitor<'de> for ParameterVisitor {
            type Value = RawJsonParameter;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a hyperparameter value")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(RawJsonParameter::Bool(value))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(RawJsonParameter::Integer(value))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(i64::try_from(value).map_or(
                    RawJsonParameter::Unsupported("u64 above i64::MAX"),
                    RawJsonParameter::Integer,
                ))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(RawJsonParameter::Float(value))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(RawJsonParameter::String(value.to_string()))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(RawJsonParameter::String(value))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(RawJsonParameter::Unsupported("null"))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<RawJsonParameter>()? {
                    items.push(item);
                }
                Ok(RawJsonParameter::List(items))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(RawJsonParameter::Unsupported("object"))
            }
        }

        deserializer.deserialize_any(ParameterVisitor)
    }
}
