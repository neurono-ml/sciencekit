//! Typed hyperparameter values (spec `model-export-contracts`).

/// A hyperparameter value a format adapter can serialize without `serde`.
///
/// The enum is deliberately small: booleans, integers, floats, strings and
/// nested lists cover estimator configuration while keeping
/// `sciencekit_common` dependency-free.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SKParameterValue {
    /// A boolean flag.
    Bool(bool),
    /// A signed 64-bit integer.
    Integer(i64),
    /// A double-precision float.
    Float(f64),
    /// A textual value.
    String(String),
    /// A homogeneous or heterogeneous list of values.
    List(Vec<SKParameterValue>),
}

impl SKParameterValue {
    /// The boolean payload, if this is a [`SKParameterValue::Bool`].
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SKParameterValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// The integer payload, if this is a [`SKParameterValue::Integer`].
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            SKParameterValue::Integer(value) => Some(*value),
            _ => None,
        }
    }

    /// The float payload, if this is a [`SKParameterValue::Float`].
    pub fn as_float(&self) -> Option<f64> {
        match self {
            SKParameterValue::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// The textual payload, if this is a [`SKParameterValue::String`].
    pub fn as_string(&self) -> Option<&str> {
        match self {
            SKParameterValue::String(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// The list payload, if this is a [`SKParameterValue::List`].
    pub fn as_list(&self) -> Option<&[SKParameterValue]> {
        match self {
            SKParameterValue::List(values) => Some(values.as_slice()),
            _ => None,
        }
    }
}

impl From<bool> for SKParameterValue {
    fn from(value: bool) -> Self {
        SKParameterValue::Bool(value)
    }
}

impl From<i64> for SKParameterValue {
    fn from(value: i64) -> Self {
        SKParameterValue::Integer(value)
    }
}

impl From<f64> for SKParameterValue {
    fn from(value: f64) -> Self {
        SKParameterValue::Float(value)
    }
}

impl From<String> for SKParameterValue {
    fn from(value: String) -> Self {
        SKParameterValue::String(value)
    }
}

impl From<&str> for SKParameterValue {
    fn from(value: &str) -> Self {
        SKParameterValue::String(value.to_string())
    }
}

impl From<Vec<SKParameterValue>> for SKParameterValue {
    fn from(value: Vec<SKParameterValue>) -> Self {
        SKParameterValue::List(value)
    }
}
