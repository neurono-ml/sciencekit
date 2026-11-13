//! Small shared helpers for command argument resolution.

use std::path::Path;

use crate::config::SKEffectiveConfiguration;
use crate::error::SKCliError;
use crate::io::SKOutputFormat;

/// Resolve the output format.
///
/// A format chosen by a flag, the environment or the workflow file wins; only
/// when no layer has an opinion does the output extension decide.
pub fn sk_output_format(
    configuration: &SKEffectiveConfiguration,
    output_path: &Path,
) -> Result<SKOutputFormat, SKCliError> {
    if configuration.output_format_explicit {
        SKOutputFormat::from_name(&configuration.output_format)
    } else {
        Ok(SKOutputFormat::from_path(output_path))
    }
}
