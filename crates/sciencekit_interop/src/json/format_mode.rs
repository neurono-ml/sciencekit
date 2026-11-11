//! Output formatting modes and schema version constant for the JSON debug
//! representation (spec `json-debug-format`).

use std::fmt;

/// The current major version of the JSON debug envelope.
///
/// Documents carrying another major version are rejected on read as an
/// unsupported representation. Unknown fields within version `1` are ignored
/// for forward compatibility.
pub const SK_JSON_DEBUG_FORMAT_VERSION: u32 = 1;

/// Controls whitespace layout when serializing a model state to JSON.
///
/// [`SKJsonFormatMode::Pretty`] is the default so exported files are directly
/// inspectable by humans and produce clean line-oriented diffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKJsonFormatMode {
    /// Multi-line, two-space indented JSON (the default).
    #[default]
    Pretty,
    /// Single-line JSON with no insignificant whitespace.
    Compact,
}

impl SKJsonFormatMode {
    /// The stable lowercase label recorded in observability spans.
    pub fn name(self) -> &'static str {
        match self {
            SKJsonFormatMode::Pretty => "pretty",
            SKJsonFormatMode::Compact => "compact",
        }
    }
}

impl fmt::Display for SKJsonFormatMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
