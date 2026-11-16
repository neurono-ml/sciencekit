//! Small shared helpers for command argument resolution.

use std::path::Path;

use crate::config::SKEffectiveConfiguration;
use crate::error::SKCliError;
use crate::io::SKOutputFormat;

/// Resolve the output format, preferring the extension when the format was not
/// set explicitly by a flag or the environment.
pub fn sk_output_format(
    configuration: &SKEffectiveConfiguration,
    explicitly_set: bool,
    output_path: &Path,
) -> Result<SKOutputFormat, SKCliError> {
    let environment_set = std::env::var("SCIENCEKIT_OUTPUT_FORMAT").is_ok();
    if explicitly_set || environment_set {
        SKOutputFormat::from_name(&configuration.output_format)
    } else {
        Ok(SKOutputFormat::from_path(output_path))
    }
}
