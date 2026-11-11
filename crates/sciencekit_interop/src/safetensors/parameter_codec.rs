//! Hyperparameter JSON codec for Safetensors `__metadata__` entries
//! (spec `safetensors-format`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sciencekit_common::{SKError, SKParameterValue};
use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};

#[derive(Debug, Clone, Copy)]
struct ParameterValueReference<'state>(&'state SKParameterValue);

impl Serialize for ParameterValueReference<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            SKParameterValue::Bool(flag) => serializer.serialize_bool(*flag),
            SKParameterValue::Integer(integer) => serializer.serialize_i64(*integer),
            SKParameterValue::Float(float) => serializer.serialize_f64(*float),
            SKParameterValue::String(text) => serializer.serialize_str(text),
            SKParameterValue::List(items) => {
                let wrapped: Vec<ParameterValueReference<'_>> =
                    items.iter().map(ParameterValueReference).collect();
                wrapped.serialize(serializer)
            }
            _ => Err(serde::ser::Error::custom(
                "unsupported hyperparameter variant for Safetensors serialization",
            )),
        }
    }
}

fn validate_parameter_finiteness(value: &SKParameterValue) -> Result<(), SKError> {
    match value {
        SKParameterValue::Float(float) if !float.is_finite() => {
            Err(SKError::NonFiniteValue { position: 0 })
        }
        SKParameterValue::List(items) => items.iter().try_for_each(validate_parameter_finiteness),
        _ => Ok(()),
    }
}

/// Encode typed hyperparameters deterministically into a compact JSON object string.
pub(crate) fn encode_parameters_json(
    parameters: &BTreeMap<String, SKParameterValue>,
) -> Result<String, SKError> {
    let mut ordered = BTreeMap::new();
    for (name, value) in parameters {
        validate_parameter_finiteness(value)?;
        ordered.insert(name.as_str(), ParameterValueReference(value));
    }
    serde_json::to_string(&ordered)
        .map_err(|error| SKError::Conversion(format!("failed to encode hyperparameters: {error}")))
}

#[derive(Debug, Clone)]
enum RawParameter {
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    List(Vec<RawParameter>),
    Unsupported(&'static str),
}

impl RawParameter {
    fn into_parameter_value(self) -> Result<SKParameterValue, SKError> {
        match self {
            RawParameter::Bool(flag) => Ok(SKParameterValue::Bool(flag)),
            RawParameter::Integer(integer) => Ok(SKParameterValue::Integer(integer)),
            RawParameter::Float(float) if float.is_finite() => Ok(SKParameterValue::Float(float)),
            RawParameter::Float(_) => Err(SKError::NonFiniteValue { position: 0 }),
            RawParameter::String(text) => Ok(SKParameterValue::String(text)),
            RawParameter::List(items) => items
                .into_iter()
                .map(Self::into_parameter_value)
                .collect::<Result<Vec<_>, _>>()
                .map(SKParameterValue::List),
            RawParameter::Unsupported(kind) => Err(SKError::InvalidHyperparameter {
                name: "parameters",
                reason: format!("hyperparameter value of JSON kind `{kind}` is not supported"),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for RawParameter {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ParameterVisitor;

        impl<'de> Visitor<'de> for ParameterVisitor {
            type Value = RawParameter;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a typed hyperparameter value")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(RawParameter::Bool(value))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(RawParameter::Integer(value))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(i64::try_from(value).map_or(
                    RawParameter::Unsupported("u64 above i64::MAX"),
                    RawParameter::Integer,
                ))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Ok(RawParameter::Float(value))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(RawParameter::String(value.to_string()))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(RawParameter::String(value))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(RawParameter::Unsupported("null"))
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = sequence.next_element::<RawParameter>()? {
                    items.push(item);
                }
                Ok(RawParameter::List(items))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(RawParameter::Unsupported("object"))
            }
        }

        deserializer.deserialize_any(ParameterVisitor)
    }
}

struct OrderedParameters(Vec<(String, RawParameter)>);

impl<'de> Deserialize<'de> for OrderedParameters {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MapVisitor;

        impl<'de> Visitor<'de> for MapVisitor {
            type Value = OrderedParameters;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object of hyperparameters")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some(pair) = map.next_entry::<String, RawParameter>()? {
                    entries.push(pair);
                }
                Ok(OrderedParameters(entries))
            }
        }

        deserializer.deserialize_map(MapVisitor)
    }
}

/// Decode a JSON hyperparameter object string while rejecting duplicate keys.
pub(crate) fn decode_parameters_json(
    encoded: &str,
) -> Result<Vec<(String, SKParameterValue)>, SKError> {
    let OrderedParameters(entries) = serde_json::from_str(encoded).map_err(|error| {
        SKError::Conversion(format!("invalid JSON in `sciencekit.parameters`: {error}"))
    })?;
    let mut seen = BTreeSet::new();
    let mut decoded = Vec::with_capacity(entries.len());
    for (name, raw) in entries {
        if !seen.insert(name.clone()) {
            return Err(SKError::InvalidHyperparameter {
                name: "parameters",
                reason: format!("duplicate hyperparameter `{name}` in metadata"),
            });
        }
        decoded.push((name, raw.into_parameter_value()?));
    }
    Ok(decoded)
}
