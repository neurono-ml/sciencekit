//! The target matrix an adapter adapts (spec `lora-adapters`).

use std::fmt;

/// The base parameter block an adapter is allowed to correct.
///
/// The initial coverage is the matrix of coefficients of the library's
/// matrix-parameter models: a linear regressor stores a `1 × features` block
/// (one head), a logistic classifier a `heads × features` block. Tree and
/// neighbour models have no such block and are rejected when attaching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SKLoraTargetModule {
    /// The coefficients of a fitted linear-regression model.
    #[default]
    LinearCoefficients,
    /// The coefficients of a fitted logistic-regression model.
    LogisticCoefficients,
}

impl SKLoraTargetModule {
    /// Every target module, in the documented order.
    pub const ALL: [SKLoraTargetModule; 2] = [
        SKLoraTargetModule::LinearCoefficients,
        SKLoraTargetModule::LogisticCoefficients,
    ];

    /// The stable metadata name of this target.
    pub fn parameter_name(self) -> &'static str {
        match self {
            SKLoraTargetModule::LinearCoefficients => "linear.coefficients",
            SKLoraTargetModule::LogisticCoefficients => "logistic.coefficients",
        }
    }

    /// The stable tensor-name segment of this target.
    ///
    /// The adapter writes its two factors under `lora.<segment>.a` and
    /// `lora.<segment>.b`.
    pub fn tensor_segment(self) -> &'static str {
        match self {
            SKLoraTargetModule::LinearCoefficients => "linear_coefficients",
            SKLoraTargetModule::LogisticCoefficients => "logistic_coefficients",
        }
    }

    /// The human-readable name used in error messages and documentation.
    pub fn display_name(self) -> &'static str {
        match self {
            SKLoraTargetModule::LinearCoefficients => "linear-regression coefficients",
            SKLoraTargetModule::LogisticCoefficients => "logistic-regression coefficients",
        }
    }

    /// Parse a persisted metadata name or tensor segment.
    pub fn parse(value: &str) -> Option<SKLoraTargetModule> {
        SKLoraTargetModule::ALL
            .into_iter()
            .find(|target| target.parameter_name() == value || target.tensor_segment() == value)
    }
}

impl fmt::Display for SKLoraTargetModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.parameter_name())
    }
}
